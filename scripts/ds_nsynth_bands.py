"""How much of each NSynth category is quiet, and how far down its floor sits.

    python scripts/ds_nsynth_bands.py --dir external/diffsynth/data/nsynth-train
    python scripts/ds_nsynth_bands.py --dir ... --per-group 200 --min-n 10

NO MODEL AND NO GROUND TRUTH. This reads the audio and nothing else, so it can
rank categories before any of them is downloaded in bulk or trained on.

WHAT IT ANSWERS. The one number that separated the plate from diffsynth was the
share of spectrogram bins sitting below -120 dB: 65-71% against 19.4%. That is
what decides how much of a log loss's weight lands on bins carrying nothing,
because log(x + eps) weights a bin by 1/(x + eps) and there are simply more of
them down there. The same count is computable for real recordings.

TWO KINDS OF QUIET, and the floor column is what tells them apart:

  (a) EMPTY. A synthetic render decays into digital silence, so quiet bins hold
      nothing at all. A log term up-weights them and gets noise -- the plate's
      case, where the deep band voted 0.503, a coin flip.
  (b) UNREPRESENTABLE CONTENT. An acoustic recording has a broadband noise
      floor, bow scrape, breath, room tail. Those bins are not empty, and they
      are not reachable either: harmor has no noise source at all. A log term
      weights real structure that no parameter setting can ever fit, so the
      residual is permanent.

(b) is the worse case for a compressed loss, and it shows up as a floor that
stops well above the arithmetic -- around -60 to -70 dB for a room recording
against -100 and below for a clean render. floor_db is the 5th percentile of
bin level, per clip, referenced to that clip's own peak.

MEASURED OVER ACTIVE FRAMES ONLY, and that is not a detail. NSynth notes are
4 s with note-off at 3 s, so a struck or plucked note is over long before the
file is, and a statistic across all frames measures how much of the clip has
ended rather than anything about its spectrum -- the first version of this
script reported floor_dB between -220 and -400 for every single category
because the 5th percentile landed in the silent tail every time.

The distinction matters because trailing silence and a noise floor pull in
OPPOSITE directions for a log loss. Silence clamps on both sides: target and
resynthesis both reach log(eps), the difference is exactly zero, the bin
contributes nothing -- the same reason a hard floor rescued the plate's
compressed arms. A noise floor is content the synth cannot produce at any
parameter setting, so it is a permanent residual weighted by 1/(x+eps). Counted
together they cannot be told apart. --active-db 0 measures whole clips.

WHAT THE COLUMNS MEAN.
  bins%   share of bins in each dB band, peak-referenced per clip. A log term's
          weight is roughly proportional to this when the perturbation is
          relative, so it doubles as where compression spends itself.
  eng%    share of total magnitude in each band -- where a LINEAR loss's weight
          lands. Read the two against each other: the gap between them IS the
          reweighting a log term performs.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import soundfile as sf                                              # noqa: E402
import torch                                                        # noqa: E402

from src.analysis.band_sensitivity import DB_BANDS, stft_mag        # noqa: E402

_NAME = re.compile(r"^(?P<family>[a-z]+(?:_[a-z]+)?)_"
                   r"(?P<source>acoustic|electronic|synthetic)_")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", required=True,
                   help="NSynth directory holding audio/*.wav")
    p.add_argument("--per-group", type=int, default=120,
                   help="Clips per family_source. Bin fractions converge fast; "
                        "this is a distribution over ~130k bins per clip.")
    p.add_argument("--min-n", type=int, default=10)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[40.0, 60.0, 80.0, 100.0])
    p.add_argument("--active-db", type=float, default=40.0, metavar="DB",
                   help="Keep only frames whose energy is within this many dB "
                        "of the clip's loudest frame. WITHOUT IT THE TABLE IS "
                        "ABOUT NOTE LENGTH, NOT SPECTRUM: NSynth notes are 4 s "
                        "with note-off at 3 s, and a struck or plucked note "
                        "decays to exact zeros long before the end, so a "
                        "statistic over all frames is dominated by how much of "
                        "the file is over. That is also why floor_dB read -220 "
                        "to -400 for every category on the first run -- the "
                        "5th percentile landed in the silent tail every time. "
                        "Set 0 to disable and measure whole clips.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "audio", "*.wav")))
    if not files:
        raise SystemExit(f"no audio/*.wav under {args.dir}")

    by_group = defaultdict(list)
    for f in files:
        m = _NAME.match(os.path.basename(f))
        if m:
            by_group[f"{m.group('family')}_{m.group('source')}"].append(f)
    by_group = {k: v for k, v in by_group.items() if len(v) >= args.min_n}
    print(f"{len(files)} files, {len(by_group)} groups with >= {args.min_n}\n")

    dev = torch.device(args.device)
    g = torch.Generator().manual_seed(args.seed)
    rows = {}
    for name in sorted(by_group, key=lambda k: -len(by_group[k])):
        fs = by_group[name]
        take = fs
        if len(fs) > args.per_group:
            idx = torch.randperm(len(fs), generator=g)[: args.per_group]
            take = [fs[i] for i in idx.tolist()]

        cnt = torch.zeros(len(DB_BANDS), dtype=torch.float64)
        eng = torch.zeros(len(DB_BANDS), dtype=torch.float64)
        below = torch.zeros(len(args.thresholds), dtype=torch.float64)
        floors, used, frames_kept = [], 0, 0.0
        for path in take:
            x, sr = sf.read(path, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=-1)
            if sr != args.sr:
                continue
            A = stft_mag(torch.from_numpy(x)[None, :].to(dev),
                          args.n_fft, args.hop, True)[0].double()
            if args.active_db > 0:
                # Frame energy against the loudest frame. Trailing silence and
                # a noise floor have OPPOSITE consequences for a log loss --
                # silence clamps on both sides and contributes exactly nothing,
                # while a noise floor is content the synth cannot produce and
                # so is a permanent weighted residual. A statistic that mixes
                # them cannot say which a category has.
                fe = A.sum(dim=0)
                keep = fe >= fe.max() * 10.0 ** (-args.active_db / 20.0)
                if int(keep.sum()) < 2:
                    continue
                A = A[:, keep]
                frames_kept += float(keep.sum()) / keep.numel()
            a = A.flatten()
            peak = a.max().clamp(min=1e-30)
            # Referenced to the clip's own peak, so level differences between
            # recordings do not masquerade as differences in how quiet a
            # category is. Exact zeros clamp just inside the deepest band
            # rather than falling out of the accounting at -inf.
            db = (20.0 * torch.log10((a / peak).clamp(min=1e-300))
                  ).clamp(min=-(float(DB_BANDS[-1][1]) - 1e-3))
            n = db.numel()
            for i, (lo, hi) in enumerate(DB_BANDS):
                m = (db <= -float(lo)) & (db > -float(hi))
                cnt[i] += float(m.sum()) / n
                eng[i] += float(a[m].sum() / a.sum())
            for i, t in enumerate(args.thresholds):
                below[i] += float((db < -t).sum()) / n
            floors.append(float(torch.quantile(db, 0.05)))
            used += 1
        if not used:
            continue
        rows[name] = dict(n=len(fs), used=used, cnt=cnt / used, eng=eng / used,
                          below=below / used, act=frames_kept / used,
                          floor=sum(floors) / len(floors))
        print(f"  {name:<24}{used:>5} of {len(fs)}"
              + (f"   {100*frames_kept/used:.0f}% of frames active"
                 if args.active_db > 0 else ""))

    print(f"\n=== bins% by dB below peak   (where a LOG term's weight goes)")
    hdr = "".join(f"{f'{lo}-{hi}':>10}" for lo, hi in DB_BANDS)
    print(f"{'group':<24}{'n':>7}{'floor_dB':>10}{hdr}")
    for name, r in sorted(rows.items(), key=lambda kv: kv[1]["floor"]):
        print(f"{name:<24}{r['n']:>7}{r['floor']:>10.1f}"
              + "".join(f"{100*v:>9.1f}%" for v in r["cnt"]))

    print(f"\n=== eng% by dB below peak   (where a LINEAR term's weight goes)")
    print(f"{'group':<24}{'n':>7}{'floor_dB':>10}{hdr}")
    for name, r in sorted(rows.items(), key=lambda kv: kv[1]["floor"]):
        print(f"{name:<24}{r['n']:>7}{r['floor']:>10.1f}"
              + "".join(f"{100*v:>9.1f}%" for v in r["eng"]))

    print(f"\n=== cumulative share of bins below each threshold")
    th = "".join(f"{f'<-{t:g}dB':>10}" for t in args.thresholds)
    print(f"{'group':<24}{'n':>7}{'floor_dB':>10}{th}")
    for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["below"][-1]):
        print(f"{name:<24}{r['n']:>7}{r['floor']:>10.1f}"
              + "".join(f"{100*v:>9.1f}%" for v in r["below"]))

    print("\n  floor_dB is the median of each clip's quietest 5% of bins, "
          "referenced to\n  its own peak. Near -100 or below means the quiet "
          "region is EMPTY -- a log\n  term up-weights nothing. Around -60 to "
          "-70 means it holds real broadband\n  content, which harmor cannot "
          "produce at all (no noise source), so a log\n  term weights a "
          "residual no parameter setting can remove.")
    print("  The gap between bins% and eng% in a band is the reweighting a log "
          "term\n  performs there. A category where the deep bands hold many "
          "bins and almost\n  no energy is the one where compression spends "
          "most and buys least.")


if __name__ == "__main__":
    main()
