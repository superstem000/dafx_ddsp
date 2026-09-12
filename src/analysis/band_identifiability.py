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


def _flatten(A_ref, A_cand, eps):
    """(a, c, w, db, e, ri) for ONE resolution or a SET of them.

    ri is the index of the resolution each bin came from, so a caller can
    partition the flattened spectrum by STFT size as well as by level.

    A_ref/A_cand may be a single [F,T]/[K,F,T] pair, as before, or lists of
    them -- one entry per STFT size. eps may be a scalar or one value per
    resolution. This is what makes the tool answer questions about a
    multi-resolution loss rather than about a spectrogram.

    w IS THE LOSS'S OWN PER-BIN WEIGHT. _make_stft_l1 takes a mean over bins
    within a resolution and then a mean over resolutions, so a bin belonging to
    an R-resolution set with n_r bins carries 1/(R*n_r) and sum(w * |c - a|)
    reproduces the loss value exactly. Concatenating the resolutions unweighted
    would instead let each one vote in proportion to its bin count on top of its
    magnitude, which is a loss nobody trains with.

    db IS PER-RESOLUTION. torch.stft here is unnormalized, so a peak-1 signal's
    bins scale with the window length -- roughly N for tonal content. Measuring
    every resolution against one global peak would drop the whole 512 rung
    ~18 dB purely because its window is shorter, and the band table would report
    that as the short windows living in the quiet bands. Each resolution's bins
    are referred to that resolution's own reference peak instead, so "40-60 dB
    down" means the same thing on every rung.

    A single tensor with a scalar eps gives a uniform w = 1/n, which scales
    every candidate's loss by one constant: concordance is unchanged and w_lin
    / w_log are shares, so every number this module reports for a
    single-resolution call is identical to what it reported before.
    """
    refs = list(A_ref) if isinstance(A_ref, (list, tuple)) else [A_ref]
    cans = list(A_cand) if isinstance(A_cand, (list, tuple)) else [A_cand]
    if len(refs) != len(cans):
        raise ValueError(f"{len(refs)} references against {len(cans)} candidate sets")
    R = len(refs)
    epss = list(eps) if isinstance(eps, (list, tuple)) else [float(eps)] * R
    if len(epss) != R:
        raise ValueError(f"{len(epss)} eps values for {R} resolutions")

    a_l, c_l, w_l, db_l, e_l, r_l = [], [], [], [], [], []
    for ri, (Ar, Ac, ep) in enumerate(zip(refs, cans, epss)):
        a = Ar.flatten().double()
        c = Ac.reshape(Ac.shape[0], -1).double()
        n = a.numel()
        a_l.append(a)
        c_l.append(c)
        w_l.append(torch.full((n,), 1.0 / (R * n), dtype=a.dtype, device=a.device))
        e_l.append(torch.full((n,), float(ep), dtype=a.dtype, device=a.device))
        r_l.append(torch.full((n,), ri, dtype=torch.long, device=a.device))
        db_l.append(20.0 * torch.log10(
            (a / a.max().clamp(min=1e-30)).clamp(min=1e-300)))
    return (torch.cat(a_l), torch.cat(c_l, dim=1), torch.cat(w_l),
            torch.cat(db_l), torch.cat(e_l), torch.cat(r_l))


def marginal(A_ref, A_cand, dist: torch.Tensor,
             eps=EPS, hard_ratio: float | None = None) -> dict:
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
    a, c, w, _db, e, _ri = _flatten(A_ref, A_cand, eps)
    Ll = (w * (c - a).abs()).sum(1)
    Lg = (w * ((c + e).log() - (a + e).log()).abs()).sum(1)

    dd = dist[:, None] - dist[None, :]
    m = dd < 0
    n = int(m.sum())
    if n == 0:
        return {}

    # HARD PAIRS. A candidate at 0.02 against one at 0.28 is ordered correctly
    # by any loss that reacts to the parameter at all, so a concordance over
    # all pairs is mostly a report on how spread the radii were. Restricting to
    # pairs within a factor F of each other asks the question an optimiser near
    # a minimum faces: two guesses about equally wrong, which is closer.
    hard = None
    if hard_ratio is not None:
        lo = torch.minimum(dist[:, None], dist[None, :])
        hi = torch.maximum(dist[:, None], dist[None, :])
        hard = m & (hi <= lo * float(hard_ratio))

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
    extra = {}
    if hard is not None:
        nh = int(hard.sum())
        extra = dict(
            n_hard=nh,
            id_lin_hard=((float((rl & hard).sum())
                          + 0.5 * float((tl & hard).sum())) / nh
                         if nh else float("nan")),
            id_log_hard=((float((rg & hard).sum())
                          + 0.5 * float((tg & hard).sum())) / nh
                         if nh else float("nan")),
        )
    return dict(
        **extra,
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


def _masks(db, bands):
    """Band membership, either by fixed dB edges or by EQUAL BIN COUNT.

    bands as a list of (lo, hi) dB pairs is the original partition: the edges
    mean the same level on every target, and the counts fall where the
    spectrum puts them -- which on the plate is 50% of bins in one band and
    0.8% in another, so six of the seven concordances are measured on a
    sliver and the seventh on half the spectrum.

    bands as an INT n splits by RANK instead: sort by reference level, cut into
    n equal chunks. Every band then carries n_bins/n bins exactly and the dB
    edges move per target. That trades a fixed level axis for a fixed sample
    size, which is the right trade when the question is whether a band ranks
    correctly rather than what happens at a particular loudness.

    Rank, not quantile-of-value: a spectrum with many identical bins -- exact
    zeros clamped to the floor, and the plate has tens of thousands -- gives
    duplicate quantile edges and therefore empty bands. Splitting the sorted
    ORDER cannot.
    """
    if not isinstance(bands, int):
        return [((db <= -float(lo)) & (db > -float(hi))) for lo, hi in bands]
    order = torch.argsort(db, descending=True)
    out = []
    for chunk in torch.chunk(order, bands):
        m = torch.zeros_like(db, dtype=torch.bool)
        m[chunk] = True
        out.append(m)
    return out


def n_bands(bands) -> int:
    return bands if isinstance(bands, int) else len(bands)


def probe(A_ref, A_cand, dist: torch.Tensor,
          eps=EPS, bands=DB_BANDS) -> list[dict]:
    """One target: [F,T] reference, [K,F,T] candidates, [K] parameter distances.

    Bands are assigned from the REFERENCE's own peak, so "40-60 dB down" means
    the same thing for every target regardless of its level.

    A_ref/A_cand may instead be LISTS, one entry per STFT size, with eps a
    scalar or one value per resolution -- see _flatten. The bands are then
    assigned per resolution and every bin is weighted as the multi-resolution
    loss weights it, so w_lin and w_log stay the share of that loss's total
    landing in each band.
    """
    a, c, w, db, e, _ri = _flatten(A_ref, A_cand, eps)

    # Exact-zero bins log to -inf and would fall outside every band. They are
    # quiet bins and belong in the deepest one, so the scale is clamped just
    # inside its lower edge rather than letting them vanish from the accounting.
    db = db.clamp(min=-(float(bands[-1][1]) - 1e-3))

    lin = (c - a).abs() * w
    lg = ((c + e).log() - (a + e).log()).abs() * w
    tot_lin = float(lin.sum()) or 1.0
    tot_log = float(lg.sum()) or 1.0

    out = []
    for m in _masks(db, bands):
        n = int(m.sum())
        if n == 0:
            out.append(dict(bins=0, id_lin=float("nan"), id_log=float("nan"),
                            w_lin=0.0, w_log=0.0,
                            db_lo=float("nan"), db_hi=float("nan")))
            continue
        Ll, Lg = lin[:, m].sum(1), lg[:, m].sum(1)
        sel = db[m]
        out.append(dict(bins=n,
                        id_lin=_concordance(Ll, dist),
                        id_log=_concordance(Lg, dist),
                        w_lin=float(Ll.sum()) / tot_lin,
                        w_log=float(Lg.sum()) / tot_log,
                        db_lo=float(sel.max()), db_hi=float(sel.min())))
    return out


def accumulate(rows: list[list[dict]], bands=DB_BANDS) -> list[dict]:
    """Mean over targets. id is averaged only over targets where the band exists."""
    out = []
    for i in range(n_bands(bands)):
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
            # Under equal-count bands the dB edges move per target, so the
            # band's level range is itself a measured quantity rather than a
            # setting, and printing it is the only way to see where a band sat.
            db_lo=sum(c["db_lo"] for c in live) / n if live else float("nan"),
            db_hi=sum(c["db_hi"] for c in live) / n if live else float("nan"),
        ))
    total_bins = sum(o["bins"] for o in out) or 1.0
    for o in out:
        o["binfrac"] = o["bins"] / total_bins
    return out


def abc(agg: list[dict]) -> tuple[float, float, float]:
    """(A, B, C) -- linear as it is, log's weighting with linear's ranking,
    log as it is. Split out of report() so a per-parameter caller gets the same
    three numbers without reformatting a printed table, which is how one
    decomposition becomes two that disagree.
    """
    def wmean(key_w, key_id):
        live = [o for o in agg if o[key_id] == o[key_id]]
        den = sum(o[key_w] for o in live) or 1.0
        return sum(o[key_w] * o[key_id] for o in live) / den

    return (wmean("w_lin", "id_lin"), wmean("w_log", "id_lin"),
            wmean("w_log", "id_log"))


def report(agg: list[dict], bands=DB_BANDS, title: str = "") -> None:
    """The table. Formatting lives here so no caller reshapes it downstream."""
    if title:
        print(f"\n=== {title}")
    equal = isinstance(bands, int)
    head = "band, equal count" if equal else "dB below peak"
    print(f"{head:>18}{'bins':>8}{'w_lin':>9}{'w_log':>9}"
          f"{'id_lin':>9}{'id_log':>9}")
    for i, o in enumerate(agg):
        idl = "     -   " if o["id_lin"] != o["id_lin"] else f"{o['id_lin']:>9.3f}"
        idg = "     -   " if o["id_log"] != o["id_log"] else f"{o['id_log']:>9.3f}"
        if equal:
            label = f"{i + 1}  {-o['db_lo']:.0f}-{-o['db_hi']:.0f} dB"
        else:
            lo, hi = bands[i]
            label = f"{lo}-{hi}"
        print(f"{label:>18}{100*o['binfrac']:>7.1f}%"
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
    A, B, C = abc(agg)
    print(f"\n  weighted concordance    linear {A:.3f}   log {C:.3f}"
          f"   (0.5 = coin flip)")
    print(f"  reweighting  B-A {B - A:+.3f}   moving weight to the quiet bands")
    print(f"  transform    C-B {C - B:+.3f}   comparing in the log domain")
    print(f"  net          C-A {C - A:+.3f}")
    print("\n  A log term is worth adding iff the transform gain exceeds the")
    print("  reweighting cost. They are independent -- a flat band profile makes")
    print("  reweighting cheap regardless of the transform, and a transform can")
    print("  gain within a band whose weight share never changes.")
