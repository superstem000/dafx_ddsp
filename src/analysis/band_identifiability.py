"""Does a level band, on its own, know which candidate is closer to the truth?

Shared by the plate and by diffsynth, for the same reason band_sensitivity is:
a number computed two different ways cannot be compared, and this one exists
precisely to be compared across the two systems.

WHY THIS AND NOT A LANDSCAPE SWEEP. The obvious experiment is to fix a target,
walk a parameter, and look at the shape of each loss. It does not answer the
question. A flat direction is a fact about that DIRECTION, and reporting that
linear is flat where the difference happens to be quiet establishes a
correlation and then asserts a cause. The plate has flat directions that have
nothing to do with level at all -- E, rho, h and nu enter only through
D = E h^3 / 12(1-nu^2) and mu = rho h, so four parameters carry two degrees of
freedom, and that degeneracy holds at every amplitude.

So partition by level FIRST and make loudness the independent variable. Build,
for each dB band separately, the loss restricted to bins in that band, and ask
whether it ranks candidates correctly. Nothing about the shape of a surface is
assumed; the band either votes correctly or it does not.

WHAT IS COMPUTED. For one target theta* and K candidates at known parameter
distances d_k, per band b:

  id_b = P[ L_b(cand_i) < L_b(cand_j)  |  d_i < d_j ]

the concordance of that band's loss with true parameter distance, over every
ordered pair. Ties count a half, which is what makes an all-clamped band read
0.50 rather than 1.00.

  id_b = 1.00   this band alone identifies the parameters
  id_b = 0.50   a coin flip. The band is noise, however much it "moved"
  id_b < 0.50   worse than nothing: it systematically prefers the wrong candidate

Note that comparing a candidate against the TRUTH is not the test and cannot
be -- L_b(theta*) is exactly zero, so truth always wins and every band scores
1.00. Ranking two candidates against each other is what has content.

WHY IT DECIDES HYBRID. A hybrid loss is a linear term plus a log term, and the
log term's gradient is a weighted sum of per-bin votes with weights ~ 1/A. So
adding it helps exactly insofar as the bins whose weight it raises are bins
that vote correctly. Pair id_b with w_log, the share of the log term's total
that lands in band b, and the question becomes arithmetic rather than
rhetorical: log-weighted mean of (2*id_b - 1) is the fraction of the log term's
influence that is signal rather than noise. Negative or near zero means the
extra weight buys noise in proportion to how many quiet bins exist, which is
the plate's 71%-below--120-dB problem stated as a mechanism instead of a
coincidence.
"""

from __future__ import annotations

import torch

from src.analysis.band_sensitivity import DB_BANDS, EPS  # noqa: F401


def _concordance(loss: torch.Tensor, dist: torch.Tensor) -> float:
    """P[loss_i < loss_j | dist_i < dist_j] over ordered pairs, ties at 0.5.

    Ties are counted rather than dropped. A band entirely below a clamp gives
    every candidate the same loss, and dropping ties would report that as an
    empty comparison instead of as the coin flip it is.
    """
    dd = dist[:, None] - dist[None, :]
    ll = loss[:, None] - loss[None, :]
    m = dd < 0
    n = int(m.sum())
    if n == 0:
        return float("nan")
    agree = float(((ll < 0) & m).sum())
    ties = float(((ll == 0) & m).sum())
    return (agree + 0.5 * ties) / n


def probe(A_ref: torch.Tensor, A_cand: torch.Tensor, dist: torch.Tensor,
          eps: float = EPS, bands=DB_BANDS) -> list[dict]:
    """One target: [F,T] reference, [K,F,T] candidates, [K] parameter distances.

    Bands are assigned from the REFERENCE's own peak, so "40-60 dB down" means
    the same thing for every target regardless of its level.
    """
    a = A_ref.flatten().double()
    c = A_cand.reshape(A_cand.shape[0], -1).double()

    db = 20.0 * torch.log10((a / a.max().clamp(min=1e-30)).clamp(min=1e-300))
    # Exact-zero bins log to -inf and would fall outside every band. They are
    # quiet bins and belong in the deepest one, so the scale is clamped just
    # inside its lower edge rather than letting them vanish from the accounting.
    db = db.clamp(min=-(float(bands[-1][1]) - 1e-3))

    lin = (c - a).abs()
    lg = ((c + eps).log() - (a + eps).log()).abs()
    tot_lin = float(lin.sum()) or 1.0
    tot_log = float(lg.sum()) or 1.0

    out = []
    for lo, hi in bands:
        m = (db <= -float(lo)) & (db > -float(hi))
        n = int(m.sum())
        if n == 0:
            out.append(dict(bins=0, id_lin=float("nan"), id_log=float("nan"),
                            w_lin=0.0, w_log=0.0))
            continue
        Ll, Lg = lin[:, m].sum(1), lg[:, m].sum(1)
        out.append(dict(bins=n,
                        id_lin=_concordance(Ll, dist),
                        id_log=_concordance(Lg, dist),
                        w_lin=float(Ll.sum()) / tot_lin,
                        w_log=float(Lg.sum()) / tot_log))
    return out


def accumulate(rows: list[list[dict]], bands=DB_BANDS) -> list[dict]:
    """Mean over targets. id is averaged only over targets where the band exists."""
    out = []
    for i in range(len(bands)):
        cells = [r[i] for r in rows]
        live = [c for c in cells if c["bins"] > 0]
        n = len(live) or 1
        out.append(dict(
            bins=sum(c["bins"] for c in cells) / max(len(cells), 1),
            binfrac=0.0,
            id_lin=sum(c["id_lin"] for c in live) / n if live else float("nan"),
            id_log=sum(c["id_log"] for c in live) / n if live else float("nan"),
            w_lin=sum(c["w_lin"] for c in cells) / max(len(cells), 1),
            w_log=sum(c["w_log"] for c in cells) / max(len(cells), 1),
        ))
    total_bins = sum(o["bins"] for o in out) or 1.0
    for o in out:
        o["binfrac"] = o["bins"] / total_bins
    return out


def report(agg: list[dict], bands=DB_BANDS, title: str = "") -> None:
    """The table. Formatting lives here so no caller reshapes it downstream."""
    if title:
        print(f"\n=== {title}")
    print(f"{'dB below peak':>14}{'bins':>8}{'w_lin':>9}{'w_log':>9}"
          f"{'id_lin':>9}{'id_log':>9}")
    for (lo, hi), o in zip(bands, agg):
        idl = "     -   " if o["id_lin"] != o["id_lin"] else f"{o['id_lin']:>9.3f}"
        idg = "     -   " if o["id_log"] != o["id_log"] else f"{o['id_log']:>9.3f}"
        print(f"{f'{lo}-{hi}':>14}{100*o['binfrac']:>7.1f}%"
              f"{100*o['w_lin']:>8.1f}%{100*o['w_log']:>8.1f}%{idl}{idg}")

    # The one number the whole table is for. Each band's log-domain vote is
    # worth (2*id - 1): +1 for a band that always ranks correctly, 0 for a coin
    # flip, negative for one that prefers the wrong candidate. Weighting by the
    # share of the log term that lands in the band gives what a log term buys
    # with its reweighting, and doing the same with w_lin gives the linear
    # term's baseline. Hybrid can only pay off if the first exceeds the second.
    def payoff(key_w, key_id):
        num = sum(o[key_w] * (2.0 * o[key_id] - 1.0)
                  for o in agg if o[key_id] == o[key_id])
        den = sum(o[key_w] for o in agg if o[key_id] == o[key_id]) or 1.0
        return num / den

    pl, pg = payoff("w_lin", "id_lin"), payoff("w_log", "id_log")
    print(f"\n  weighted vote quality   linear {pl:+.3f}   log {pg:+.3f}"
          f"   (1 = every weighted bin ranks correctly, 0 = coin flip)")
    print("  A log term is worth adding only where its number exceeds the linear")
    print("  one: that is the margin hybrid has to pay for the loud-band")
    print("  reweighting it also forces. Equal or below means the reweighting is")
    print("  buying noise, in proportion to how many quiet bins there are.")
