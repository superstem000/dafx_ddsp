"""Is a class's energy anchored to k*f0, or spread between the harmonics?

    python scripts/ds_harmonic_share.py --dir external/diffsynth/data/nsynth-train
    python scripts/ds_harmonic_share.py --dir ... --cents 50 --per-group 200

NO MODEL, NO F0 TRACKER, NO GROUND TRUTH BEYOND THE FILENAME. NSynth names are
<family>_<source>_<instr>-<pitch>-<velocity>.wav and the middle number is the
MIDI pitch, so f0 = 440 * 2^((p-69)/12) is EXACT. Every f0-relative statistic
here is therefore free of tracker error, which is the usual reason this kind of
measurement is not trusted.

WHAT IT IS FOR. Spectral flatness and floor_dB both say "flat" for two kinds of
audio that behave oppositely under a compressed loss:

  string_acoustic   bowed: stick-slip motion drives the string PERIODICALLY, so
                    the broadband look comes from jitter and bow scrape as
                    SIDEBANDS around partials that sit at exact k*f0. harmor can
                    approximate a thick harmonic stack.
  reed_acoustic     blown: an air jet through a turbulent constriction. The
                    noise is aerodynamic, spectrally smooth, unrelated to f0,
                    and fills the space BETWEEN harmonics. harmor has no noise
                    source and cannot produce it at any parameter setting.

Those are flatness 0.100 and ~0.09, floor -84.8 and -90.7 -- indistinguishable.
They are also the largest HYBRID win (+1.92 magx-log) and the largest LINEAR
win (-1.23) in the nineteen-class table. If anything computable from the audio
separates them, it has to be this.

THE COLUMNS, and why enrichment rather than raw share. A fixed +-cents window
around every k*f0 covers MORE of the spectrum at low pitch, where harmonics are
close together, so a raw "share of energy near a harmonic" ranks partly by
pitch. Enrichment divides the energy share by the BIN-COUNT share of the same
zones, giving a scale-free number:

  H       energy in harmonic zones / bins in harmonic zones
  I       the same for INTER-harmonic zones, centred at (k+0.5)*f0
  H/I     the discriminator. Large means energy sits on the harmonics and the
          gaps are empty. Near 1 means the spectrum is filled in between --
          content no harmonic oscillator can produce.

White noise gives H = I = 1 and H/I = 1 by construction. A pure harmonic stack
drives I toward 0 and H/I up.

THE WINDOW IS THE ONE PARAMETER THAT CAN FAKE THE ANSWER. Too narrow and a
Hann main lobe falls outside it, so everything looks unharmonic; too wide and
the zones swallow the gaps and everything looks harmonic. Half-width is
max(f * (2^(cents/1200) - 1), --min-bins * bin_width / 2) -- cents-relative
where the resolution allows, floored at a fraction of a bin where it does not,
because at MIDI 24 (32.7 Hz) a 30-cent window is 0.57 Hz against a 3.9 Hz bin.
n_fft defaults to 4096 rather than the 1024 used elsewhere for the same reason:
at 1024 the bins are 15.6 Hz and low notes have harmonics 2 bins apart, which
cannot be resolved from the gaps at all. Report at more than one --cents before
believing a ranking.

A harmonic whose zone would overlap its neighbouring inter-harmonic zone is
dropped, and the count is printed: with no gap left to measure, "between the
harmonics" has no meaning.

ACTIVE FRAMES ONLY, for the reason ds_nsynth_bands documents at length: NSynth
notes are 4 s with note-off at 3 s, so a statistic over all frames measures how
much of the clip has ended.

CAVEAT, stated because it bounds the reading. Piano is inharmonic from string
stiffness -- partial k sits at k*f0*sqrt(1+B*k^2), drifting out of the window as
k grows -- so keyboard_acoustic's H/I is depressed by physics that has nothing
to do with noise. This measures "energy at integer multiples of f0", which is
what harmor can produce, not "energy at the partials this instrument has".
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np                                                  # noqa: E402
import soundfile as sf                                              # noqa: E402
import torch                                                        # noqa: E402

_NAME = re.compile(r"^(?P<family>[a-z]+(?:_[a-z]+)?)_"
                   r"(?P<source>acoustic|electronic|synthetic)_"
                   r"(?P<instr>\d+)-(?P<pitch>\d+)-(?P<vel>\d+)\.wav$")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", required=True, help="holds audio/*.wav")
    p.add_argument("--per-group", type=int, default=120)
    p.add_argument("--min-n", type=int, default=10)
    p.add_argument("--cents", type=float, default=30.0,
                   help="Half-width of a harmonic zone, in cents. The one knob "
                        "that can manufacture the answer -- check a ranking at "
                        "15, 30 and 50 before trusting it.")
    p.add_argument("--min-bins", type=float, default=1.5,
                   help="Floor on the half-width in bins, so a Hann main lobe "
                        "fits at low pitch where a cents window is sub-bin.")
    p.add_argument("--n-fft", type=int, default=4096,
                   help="4096, not the 1024 used elsewhere: at 1024 the bins "
                        "are 15.6 Hz and a low note's harmonics are 2 bins "
                        "apart, so harmonic and inter-harmonic zones are not "
                        "separable at all.")
    p.add_argument("--hop", type=int, default=1024)
    p.add_argument("--f-max", type=float, default=7600.0,
                   help="Stop at the mel range's top, so this and the mel "
                        "metrics cover the same band.")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--active-db", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=0)
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
    if not by_group:
        raise SystemExit("no filenames matched the NSynth naming pattern")
    print(f"{len(files)} files, {len(by_group)} groups   n_fft={args.n_fft} "
          f"(bin {args.sr / args.n_fft:.2f} Hz), +-{args.cents:g} cents, "
          f"min {args.min_bins:g} bins, active-db={args.active_db:g}\n")

    freqs = torch.fft.rfftfreq(args.n_fft, 1.0 / args.sr)
    bin_w = args.sr / args.n_fft
    win = torch.hann_window(args.n_fft)
    g = torch.Generator().manual_seed(args.seed)
    rows = {}

    for name in sorted(by_group, key=lambda k: -len(by_group[k])):
        fs = by_group[name]
        take = fs
        if len(fs) > args.per_group:
            idx = torch.randperm(len(fs), generator=g)[: args.per_group]
            take = [fs[i] for i in idx.tolist()]

        he = hb = ie = ib = 0.0          # energy / bin-count share accumulators
        pitches, used, dropped, ktot = [], 0, 0, 0
        for path in take:
            m = _NAME.match(os.path.basename(path))
            pitch = int(m.group("pitch"))
            f0 = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
            x, sr = sf.read(path, dtype="float32")
            if sr != args.sr:
                continue
            if x.ndim > 1:
                x = x.mean(axis=-1)
            A = torch.stft(torch.from_numpy(x)[None, :], args.n_fft,
                           hop_length=args.hop, window=win, center=True,
                           return_complex=True).abs()[0].double()
            if args.active_db > 0:
                fe = A.sum(dim=0)
                keep = fe >= fe.max() * 10.0 ** (-args.active_db / 20.0)
                if int(keep.sum()) < 2:
                    continue
                A = A[:, keep]
            power = (A ** 2).sum(dim=1)          # over active frames, per bin

            # Zones. Half-width is cents-relative where the resolution allows
            # and floored at a fraction of a bin where it does not.
            hmask = torch.zeros_like(freqs, dtype=torch.bool)
            imask = torch.zeros_like(freqs, dtype=torch.bool)
            k = 1
            while k * f0 <= args.f_max:
                fk = k * f0
                hw = max(fk * (2.0 ** (args.cents / 1200.0) - 1.0),
                         args.min_bins * bin_w / 2.0)
                # The inter-harmonic centre sits half a spacing up. If the two
                # zones would touch there is no gap left to measure, so the
                # harmonic is dropped rather than reported as filled.
                ktot += 1
                if 2.0 * hw >= f0 / 2.0:
                    dropped += 1
                    k += 1
                    continue
                hmask |= (freqs - fk).abs() <= hw
                fi = fk + f0 / 2.0
                if fi <= args.f_max:
                    imask |= (freqs - fi).abs() <= hw
                k += 1
            # A bin claimed by both (possible only at the drop threshold) is
            # given to neither, so the two zones stay disjoint by construction.
            both = hmask & imask
            hmask &= ~both
            imask &= ~both
            if not bool(hmask.any()) or not bool(imask.any()):
                continue

            tot = float(power.sum())
            nb = float(freqs.numel())
            if tot <= 0:
                continue
            he += float(power[hmask].sum()) / tot
            ie += float(power[imask].sum()) / tot
            hb += float(hmask.sum()) / nb
            ib += float(imask.sum()) / nb
            pitches.append(pitch)
            used += 1

        if not used:
            continue
        he, hb, ie, ib = he / used, hb / used, ie / used, ib / used
        H = he / hb if hb else float("nan")
        I = ie / ib if ib else float("nan")
        rows[name] = dict(n=len(fs), used=used, he=he, hb=hb, ie=ie, ib=ib,
                          H=H, I=I, hi=(H / I if I else float("nan")),
                          pitch=float(np.median(pitches)) if pitches else 0.0,
                          drop=(dropped / ktot if ktot else 0.0))
        print(f"  {name:<24}{used:>5} of {len(fs)}   median pitch "
              f"{rows[name]['pitch']:.0f}   {100 * rows[name]['drop']:.0f}% of "
              f"harmonics dropped (no gap)")

    print(f"\n=== harmonic anchoring   (H/I large = energy on k*f0 and gaps "
          f"empty; ~1 = filled in between)")
    print(f"{'group':<24}{'n':>7}{'pitch':>7}{'eng_H%':>9}{'eng_I%':>9}"
          f"{'bins_H%':>9}{'H':>8}{'I':>8}{'H/I':>8}")
    for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["hi"]):
        print(f"{name:<24}{r['n']:>7}{r['pitch']:>7.0f}"
              f"{100 * r['he']:>8.1f}%{100 * r['ie']:>8.1f}%"
              f"{100 * r['hb']:>8.1f}%{r['H']:>8.2f}{r['I']:>8.2f}"
              f"{r['hi']:>8.2f}")

    print("\n  H/I is the number to read. White noise gives exactly 1 -- energy")
    print("  is spread evenly, so both zones are enriched equally. A harmonic")
    print("  stack drives I toward 0 and H/I up. Between them sits the question")
    print("  this exists to answer: bow noise is jitter and scrape AROUND")
    print("  partials at exact k*f0, so it should keep H/I high despite looking")
    print("  flat; breath noise is aerodynamic and unrelated to f0, so it fills")
    print("  the gaps and pushes H/I down.")
    print("  The prediction that makes it worth running: string_acoustic HIGH")
    print("  and reed_acoustic LOW, when flatness (0.100 vs ~0.09) and floor_dB")
    print("  (-84.8 vs -90.7) call them the same. Those two are the largest")
    print("  hybrid win and the largest linear win in the class table, so a")
    print("  statistic that cannot separate them cannot explain either.")
    print("  Re-run at --cents 15 and 50. A ranking that moves is the window's,")
    print("  not the audio's.")
    print("  keyboard_acoustic reads low for a reason that is not noise: piano")
    print("  partials sit at k*f0*sqrt(1+B*k^2) from string stiffness and drift")
    print("  out of the window as k grows. That inharmonicity is equally real as")
    print("  a limit on what harmor can fit, but it is not the same mechanism.")


if __name__ == "__main__":
    main()
