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


def marginal(A_ref: torch.Tensor, A_cand: torch.Tensor, dist: torch.Tensor,
             eps: float = EPS) -> dict:
    """Is the log term's information NEW, given a linear term already present?

    Hybrid contains the linear term, so what a log term can contribute is only
    the pairs linear gets wrong. Concordance on its own does not distinguish
    "log ranks 62% of pairs correctly" from "log ranks the SAME 62% linear
    already had" -- and those have opposite implications for hybrid, because
    only the first pays for the loud-band reweighting hybrid also forces.

      id_log_given_lin_wrong ~ 0.50   redundant. Adding a log term buys nothing
                                      and costs the reweighting.
      well above 0.50                 complementary. Hybrid should beat both,
                                      even where log alone does not beat linear.

    Computed on the FULL spectrum rather than per band, because that is the
    comparison an actual loss makes.
    """
    a = A_ref.flatten().double()
    c = A_cand.reshape(A_cand.shape[0], -1).double()
    Ll = (c - a).abs().sum(1)
    Lg = ((c + eps).log() - (a + eps).log()).abs().sum(1)

    dd = dist[:, None] - dist[None, :]
    m = dd < 0
    n = int(m.sum())
    if n == 0:
        return {}

    def right(L):
        ll = L[:, None] - L[None, :]
        return (ll < 0) & m, (ll == 0) & m

    rl, tl = right(Ll)
    rg, tg = right(Lg)
    # A tie is half a correct vote, and the two losses tie on different pairs,
    # so "wrong" has to mean "not right and not tied" for the conditionals to
    # partition the same population both ways.
    wl, wg = m & ~rl & ~tl, m & ~rg & ~tg
    nwl, nwg = int(wl.sum()), int(wg.sum())
    return dict(
        n_pairs=n,
        id_lin=(float(rl.sum()) + 0.5 * float(tl.sum())) / n,
        id_log=(float(rg.sum()) + 0.5 * float(tg.sum())) / n,
        lin_wrong=nwl / n,
        log_wrong=nwg / n,
        log_given_lin_wrong=((float((rg & wl).sum()) + 0.5 * float((tg & wl).sum()))
                             / nwl) if nwl else float("nan"),
        lin_given_log_wrong=((float((rl & wg).sum()) + 0.5 * float((tl & wg).sum()))
                             / nwg) if nwg else float("nan"),
    )


def report_marginal(rows: list[dict], title: str = "") -> None:
    live = [r for r in rows if r]
    if not live:
        return

    def mean(k):
        v = [r[k] for r in live if r[k] == r[k]]
        return sum(v) / len(v) if v else float("nan")

    lw, gw = mean("lin_wrong"), mean("log_wrong")
    g_giv_l, l_giv_g = mean("log_given_lin_wrong"), mean("lin_given_log_wrong")
    # THE BASELINE IS NOT 0.5. If the two losses failed independently, log would
    # be right on linear's wrong pairs at its OWN overall rate, 1 - log_wrong.
    # Comparing against a coin flip instead understates redundancy badly and is
    # simply the wrong null: a loss that is right 67% of the time overall and
    # 50% of the time on another loss's failures is already strongly correlated
    # with it, not independent of it.
    ind_g, ind_l = 1.0 - gw, 1.0 - lw
    fixes = lw * g_giv_l                      # linear wrong, log right
    breaks = gw - lw * (1.0 - g_giv_l)        # log wrong, linear right

    print(f"\n=== marginal value of the log term{'   ' + title if title else ''}")
    print(f"  full-spectrum concordance     linear {mean('id_lin'):.3f}   "
          f"log {mean('id_log'):.3f}")
    print(f"  linear misranks               {100*lw:.1f}% of pairs")
    print(f"  log correct on THOSE          {g_giv_l:.3f}   "
          f"(if errors were independent: {ind_g:.3f})")
    print(f"  log misranks                  {100*gw:.1f}% of pairs")
    print(f"  linear correct on THOSE       {l_giv_g:.3f}   "
          f"(independent: {ind_l:.3f})")
    print(f"\n  adding the log term FIXES {100*fixes:.1f}% of pairs and "
          f"BREAKS {100*breaks:.1f}%")
    if g_giv_l < ind_g - 0.02:
        print("  Errors are CORRELATED -- log fails well below its own average on")
        print("  exactly the pairs linear fails. Both losses read the same")
        print("  spectrogram, so a pair whose spectral distance misorders the")
        print("  parameter distance misorders it in every domain. That is a")
        print("  property of the synthesizer's forward map, and no reweighting")
        print("  of the same spectrogram can recover it. Little headroom for any")
        print("  hybrid, whatever weight it puts on each term.")
    else:
        print("  Errors are near-independent or better -- the log term is right")
        print("  about pairs linear is wrong about, which is the headroom a")
        print("  hybrid can actually capture.")


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

    # THE DECOMPOSITION, which is the point of the table. Going from linear to
    # log does two independent things and they are not the same effect:
    #
    #   A = sum w_lin * id_lin    linear as it actually is
    #   B = sum w_log * id_lin    log's WEIGHTING, linear's within-band ranking
    #   C = sum w_log * id_log    log as it actually is
    #
    # B - A is what moving weight into the quiet bands costs. C - B is what
    # comparing in the log domain buys within a band, holding the weighting
    # fixed. Their sum is the whole effect, and they can have opposite signs.
    #
    # This separation is the answer to "when does adding a log term help",
    # and it is not the answer the quiet-bin framing predicted. On the plate
    # the transform is worth about zero and the reweighting costs ~0.16 -- the
    # deep bins are not noise, they vote around 0.59, they are merely much
    # weaker than the loud ones and log hands them ~70% of its weight. On
    # diffsynth the reweighting costs little, because the band profile is flat
    # rather than a cliff, and the transform GAINS within every band.
    #
    # So: hybrid has an edge iff the transform's gain exceeds the reweighting's
    # cost. Both are measurable here, before any training run.
    def wmean(key_w, key_id):
        live = [o for o in agg if o[key_id] == o[key_id]]
        den = sum(o[key_w] for o in live) or 1.0
        return sum(o[key_w] * o[key_id] for o in live) / den

    A = wmean("w_lin", "id_lin")
    B = wmean("w_log", "id_lin")
    C = wmean("w_log", "id_log")
    print(f"\n  weighted concordance    linear {A:.3f}   log {C:.3f}"
          f"   (0.5 = coin flip)")
    print(f"  reweighting  B-A {B - A:+.3f}   moving weight to the quiet bands")
    print(f"  transform    C-B {C - B:+.3f}   comparing in the log domain")
    print(f"  net          C-A {C - A:+.3f}")
    print("\n  A log term is worth adding iff the transform gain exceeds the")
    print("  reweighting cost. They are independent -- a flat band profile makes")
    print("  reweighting cheap regardless of the transform, and a transform can")
    print("  gain within a band whose weight share never changes.")
