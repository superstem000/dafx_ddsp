"""Where a parameter's signature lives, in bands and in totals.

Shared by every system so the two never drift apart -- that discipline is the
only reason a plate number and a diffsynth number can be read against each
other, and it has already caught one metric being computed two different ways.

Nothing here knows about a synthesizer. A driver renders a reference and a
perturbation; this says where the difference sits and how much of it each kind
of loss can see.
"""

from __future__ import annotations

import torch

DB_BANDS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120), (120, 400)]
EPS = 1e-7


def stft_mag(x: torch.Tensor, n_fft: int, hop: int, normalize: bool) -> torch.Tensor:
    """Peak-normalized or raw magnitude spectrogram, matching diag_gt_floor."""
    if normalize:
        x = x / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    w = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x, n_fft, hop, window=w, return_complex=True).abs()


def decompose(ref: torch.Tensor, per: torch.Tensor, eps: float = EPS):
    """(linear centroid, log centroid, linear bands, log bands, totals).

    Bins are ordered by the REFERENCE magnitude, not the perturbed one, so the
    buckets mean the same thing for every parameter and every step size.
    """
    a, b = ref.flatten(), per.flatten()
    lin = (a - b).abs()
    # eps is the floor of the log measure: two bins both far below it clamp to
    # log(eps) and register no disagreement at all. At the default 1e-7 that
    # floor sits ~160 dB under the reference peak, i.e. forty dB below where
    # this plate's float32 modal sum stops being physics -- so the log column
    # counts arithmetic. --floor-db sets it where the signal actually is.
    log = (torch.log(a + eps) - torch.log(b + eps)).abs()

    # RELATIVE change per bin -- the direct answer to "does this perturbation
    # move the loud parts or the quiet parts", in percent, with no centroid and
    # no share-of-total in the way. A share says how the change is distributed
    # given how many bins are where; this says how much each bin actually moved.
    rel = lin / (a + eps)

    # NEUTRAL FOR THIS FLOOR. A change that is uniform in RELATIVE terms -- every
    # bin scaled by the same factor -- contributes a*delta/(a+eps) per bin, which
    # is flat only while eps is far below every bin. Raise the floor and the
    # quiet bins are suppressed, so a uniform change no longer scores 5.5 and the
    # centroid drifts upward on its own. This computes where neutral actually
    # sits, from the reference and the eps in force, so the log dec column stays
    # readable at any floor. At the default eps it comes out at 5.5.
    w = a / (a + eps)

    order = torch.argsort(a)
    lin_s, log_s, w_s = lin[order], log[order], w[order]
    n = lin_s.numel()
    edges = [round(i * n / 10) for i in range(11)]
    lin_d = torch.tensor([lin_s[edges[i]:edges[i + 1]].sum() for i in range(10)])
    log_d = torch.tensor([log_s[edges[i]:edges[i + 1]].sum() for i in range(10)])
    w_d = torch.tensor([w_s[edges[i]:edges[i + 1]].sum() for i in range(10)])

    def centroid(d):
        t = d.sum()
        # 1 = quietest decile, 10 = loudest. nan when the parameter did nothing
        # at all, which is itself the answer and must not read as decile 1.
        return float((d * torch.arange(1, 11)).sum() / t) if t > 0 else float("nan")

    rel_db = 20.0 * torch.log10((a / a.max().clamp(min=1e-30)).clamp(min=1e-30))
    lin_b, log_b, cnt_b, rel_b = [], [], [], []
    for lo, hi in DB_BANDS:
        m = (-rel_db >= lo) & (-rel_db < hi)
        lin_b.append(float(lin[m].sum()))
        log_b.append(float(log[m].sum()))
        rel_b.append(float(rel[m].mean()) if int(m.sum()) else 0.0)
        # Bin count per band, so the shares above can be read per bin rather
        # than per band. Bands hold wildly different numbers of bins -- the
        # deciles do not, which is why the centroid was legible while this was
        # not -- and a sum over a band is a statement about how many bins are in
        # it as much as about where a parameter's signature lives.
        cnt_b.append(int(m.sum()))
    return (centroid(lin_d), centroid(log_d), lin_b, log_b,
            float(lin.sum()), float(log.sum()), cnt_b, rel_b, centroid(w_d))


