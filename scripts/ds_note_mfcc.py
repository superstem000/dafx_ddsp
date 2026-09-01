"""In-domain MFCC over the NOTE only, beside the same number over the whole clip.

    python scripts/ds_note_mfcc.py --arms pre_var_magx_halfw pre_var_hybridx pre_var_logx_halfw
    python scripts/ds_note_mfcc.py --arms ... --ckpt latest.ckpt --batches 16 --device cuda:3

WHAT IT ANSWERS. On the windowed dataset roughly 40% of every clip is gated
silence, and val_id/mfccdb is logged over all 4 s of it. A dB cepstrum scores a
silent frame against a silent frame as a large term -- the mel energies sit near
the 1e-10 epsilon, a hundred dB under the note's peak -- so an arm can be ahead
or behind on that column for reasons that have nothing to do with how well it
recovered the note. This computes the SAME metric twice, over the whole clip and
over the note alone, so the difference between the two columns IS the silence's
contribution, in the units the result is reported in.

THE WINDOW IS GROUND TRUTH, not a threshold. h2of_var.yaml gates the envelope,
so the saved harmor_amplitudes target is EXACTLY zero outside the note and
nonzero inside it. The note's extent is read off that curve -- first nonzero
frame to last -- rather than estimated from the audio with a dB criterion, so
there is no threshold to tune and no argument about where the note ends.
Interior frames are kept even where the envelope's own noise clamped one to
zero, since the note is contiguous by construction.

MASKS THE CEPSTRUM'S FRAMES rather than cropping the audio. Cropping would
change every clip's length and put an analysis window across a discontinuity;
masking evaluates the same STFT everyone else evaluates and simply declines to
average over frames where the target has no signal. The cepstrum runs 247 frames
against the envelope's 250, so the mask is index-mapped rather than assumed to
line up -- the same arithmetic ds_ood_subset uses.

NORMALISED BY AN UNRELATED PAIR, the batch rolled by one, so the two columns are
comparable to each other despite covering different numbers of frames. Without
it the note-only column would look better for free, since it drops the frames
with the largest raw terms.

THE DEFAULT IS UNFLOORED, top_db None -- the convention the headline result is
written in. --top-db 80 gives the floored variant for comparison.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "external", "diffsynth"))
sys.path.insert(0, HERE)

import torch                                             # noqa: E402
from torch.utils.data import DataLoader                  # noqa: E402

import ds_param_breakdown as pb                          # noqa: E402
import ds_mfcc_check as mc                               # noqa: E402

AMP_KEY = "harmor_amplitudes"


def note_mask(amp: torch.Tensor) -> torch.Tensor:
    """(B, n_frames) True inside the note, from the gated amplitude target.

    Contiguous first-nonzero to last-nonzero rather than a per-frame test: the
    envelope's noise term can clamp an interior frame to exactly zero, and the
    note is one span by construction, so a per-frame test would punch holes in
    it and quietly drop frames that do carry signal.
    """
    a = amp.abs().amax(dim=-1)                      # over oscillators
    nz = a > 0
    idx = torch.arange(a.shape[1], device=a.device)[None, :].expand_as(a)
    big = torch.iinfo(torch.int64).max
    first = torch.where(nz, idx, torch.full_like(idx, big)).amin(dim=1)
    last = torch.where(nz, idx, torch.full_like(idx, -1)).amax(dim=1)
    m = (idx >= first[:, None]) & (idx <= last[:, None])
    return m & (last >= first)[:, None]


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--root", default="results/diffsynth")
    p.add_argument("--ckpt", default="latest.ckpt",
                   help="latest.ckpt is rewritten continuously, so a running "
                        "arm can be measured without waiting for a milestone.")
    p.add_argument("--top-db", type=float, default=None,
                   help="None (default) is the unfloored cepstrum the headline "
                        "result uses; 80 is the floored variant.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--batches", type=int, default=32)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    dev = args.device
    mfcc = mc.make_mfcc(dev, window="hann", log="db", top_db=args.top_db,
                        mel_norm="slaney", mel_scale="slaney")

    print(f"{'arm':<24}{'ckpt':>12}{'kept':>8}{'whole':>10}{'note':>10}"
          f"{'silence share':>15}")
    rows = {}
    for arm in args.arms:
        d = os.path.join(args.root, arm)
        model, cfg, dm, note = pb.load_arm(d, args.ckpt, dev, args.batch_size)
        if model is None:
            print(f"{arm:<24} skipped: {note}")
            continue
        vset, how = pb.val_split(d, dm)
        if vset is None:
            print(f"{arm:<24} skipped: {how}")
            continue
        loader = DataLoader(vset, batch_size=args.batch_size, num_workers=0)
        acc = {"whole": [0.0, 0.0], "note": [0.0, 0.0]}
        kept = [0.0, 0.0]
        for i, batch in enumerate(loader):
            if i >= args.batches:
                break
            batch = {k: (v.to(dev) if torch.is_tensor(v) else
                         {kk: vv.to(dev) for kk, vv in v.items()}
                         if isinstance(v, dict) else v)
                     for k, v in batch.items()}
            tgt = batch["audio"]
            if tgt.shape[0] < 2:
                continue                     # no partner for the denominator
            with torch.no_grad():
                out, _ = model(batch)
            oth = tgt.roll(1, dims=0)
            a, b, c = mfcc(tgt), mfcc(out), mfcc(oth)

            m = note_mask(batch["params"][AMP_KEY])
            if m.shape[-1] != a.shape[-1]:
                j = (torch.arange(a.shape[-1], device=dev)
                     * m.shape[-1] // a.shape[-1])
                m = m[:, j]
            w = m.to(a.dtype)[:, None, :].expand_as(a)
            kept[0] += float(w.sum())
            kept[1] += float(w.numel())

            acc["whole"][0] += float((a - b).abs().sum())
            acc["whole"][1] += float((a - c).abs().sum())
            acc["note"][0] += float(((a - b).abs() * w).sum())
            acc["note"][1] += float(((a - c).abs() * w).sum())

        r = {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in acc.items()}
        rows[arm] = r
        frac = kept[0] / kept[1] if kept[1] else float("nan")
        # How much of the whole-clip score came from frames with no signal.
        # Not a ratio of the two columns -- both are already normalised -- but
        # the share of the whole-clip NUMERATOR that the mask removes.
        share = 1.0 - (acc["note"][0] / acc["whole"][0]) if acc["whole"][0] else float("nan")
        print(f"{arm:<24}{note:>12}{frac:>8.3f}{r['whole']:>10.4f}"
              f"{r['note']:>10.4f}{share:>14.1%}")

    if len(rows) > 1:
        best = lambda k: min(rows, key=lambda a: rows[a][k])
        print(f"\n  whole clip : {best('whole')} first")
        print(f"  note only  : {best('note')} first")
        if best("whole") != best("note"):
            print("  THE ORDERING CHANGES when the silence is excluded, which "
                  "means the\n  whole-clip column is ranking the arms partly on "
                  "how they fill a\n  region the target has no signal in.")
        else:
            print("  Same ordering either way, so the ranking is not an "
                  "artefact of the\n  gated region.")
    print(f"\n  metric: DbCepstrum, Hann, Slaney/Slaney, 10*log10(mel+1e-10), "
          f"top_db {args.top_db}\n"
          f"  kept  = fraction of cepstrum frames inside the note, from the "
          f"GATED\n          harmor_amplitudes target rather than a dB "
          f"threshold\n"
          f"  both columns are arm/saturation against the batch rolled by one, "
          f"so they\n  are comparable to each other despite covering different "
          f"frame counts")


if __name__ == "__main__":
    main()
