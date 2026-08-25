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

IT SEPARATED THEM THE WRONG WAY. First run, --width cents at 15/30/50, reed
came out HIGH (862 / 1197 / 1436) and string LOWER (315 / 358 / 374), stably
across all three windows -- the reverse of the prediction above. Recorded here
rather than quietly rewritten, because a script that only remembers the
hypotheses it confirmed is worth nothing.

Two things to take from that. First, the whole-spectrum H/I should not be
ranked on at all: eng_I% is under 2% for nearly every class and 0.1% for reed,
so H/I at the top is a ratio over a near-zero denominator and swings 2x between
windows on identical audio (reed_synthetic: 2127 / 3477 / 3986, n=48). Second,
in the region the cents window could reach, NSynth really is overwhelmingly
harmonic, reed included -- so "breath noise fills the gaps" is false BELOW
k ~ 14, whatever is true above it.

And the region above it was not measured, which is the flaw the first run
exposed rather than a finding: a cents window drops 73-92% of harmonics, at a
rate that tracks pitch. See --width.

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

THE WINDOW IS THE ONE PARAMETER THAT CAN FAKE THE ANSWER, and its UNIT decides
which harmonics exist to be measured. A cents half-width grows with k*f0 while
the gap between harmonics stays f0, so the zones close up at k ~ 14 -- about
4 kHz at NSynth's median pitch -- and everything above is dropped. That is
exactly where a broadband bed dominates a harmonic series, so the cents version
is blind in the one place the question is interesting. Default is --width
f0frac, half-width --frac * f0, which is a fixed fraction of the gap and so
stays disjoint at every k with nothing dropped.

Both are floored at --min-bins * bin_width / 2, because at MIDI 24 (32.7 Hz) a
30-cent window is 0.57 Hz against a 3.9 Hz bin and a Hann main lobe would fall
outside it. n_fft defaults to 4096 rather than the 1024 used elsewhere for the
same reason: at 1024 the bins are 15.6 Hz and low notes have harmonics 2 bins
apart, so there is no gap to resolve at all. Report at more than one width
before believing a ranking.

A harmonic whose zone would overlap its neighbouring inter-harmonic zone is
dropped and the count printed: with no gap left, "between the harmonics" has no
meaning. Under f0frac with --frac below 0.25 that never fires.

READ THE PER-BAND TABLE, NOT THE TOTAL. One number over 40-7600 Hz averages
away the thing being looked for -- a reed's low harmonics are clean while its
bed runs to 8 kHz -- and the whole-spectrum denominator is near zero anyway.
Each band is referenced to its own energy, so a band holding little of the clip
is still reported on its own terms.

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
    p.add_argument("--width", default="f0frac", choices=("f0frac", "cents"),
                   help="How a zone's half-width is set, and it decides which "
                        "HARMONICS ARE MEASURABLE AT ALL.\n"
                        "cents scales with k*f0 while the gap between "
                        "harmonics stays f0, so 2*hw >= f0/2 trips at k ~ 14 "
                        "whatever the pitch and everything above roughly 4 kHz "
                        "is dropped -- 73-92% of harmonics on this data, at a "
                        "rate that tracks pitch, so classes end up compared "
                        "over different k ranges. It is also the region where "
                        "broadband noise dominates, which is the thing this "
                        "script exists to find.\n"
                        "f0frac makes the half-width --frac * f0, a fixed "
                        "fraction of the gap, so the zones stay disjoint at "
                        "every k and nothing is dropped. The cost is that a "
                        "partial displaced by more than --frac * f0 falls out: "
                        "piano stiffness puts partial k at k*f0*sqrt(1+B*k^2), "
                        "and string vibrato sweeps partials across a band. "
                        "Both are real limits on what harmor can fit, but "
                        "neither is noise.")
    p.add_argument("--frac", type=float, default=0.2,
                   help="Half-width as a fraction of f0, for --width f0frac. "
                        "Must be under 0.25 or the harmonic and inter-harmonic "
                        "zones touch.")
    p.add_argument("--cents", type=float, default=30.0,
                   help="Half-width in cents, for --width cents.")
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
    if args.width == "f0frac" and args.frac >= 0.25:
        raise SystemExit(f"--frac {args.frac} >= 0.25 makes the harmonic and "
                         f"inter-harmonic zones touch; there is no gap left")
    wdesc = (f"+-{args.frac:g}*f0" if args.width == "f0frac"
             else f"+-{args.cents:g} cents")
    print(f"{len(files)} files, {len(by_group)} groups   n_fft={args.n_fft} "
          f"(bin {args.sr / args.n_fft:.2f} Hz), {wdesc}, "
          f"min {args.min_bins:g} bins, active-db={args.active_db:g}\n")
    # Where the noise is matters as much as whether it is there: a reed's bed
    # runs to 8 kHz while its low harmonics are clean, and one number over the
    # whole range averages that away.
    BANDS = [(0.0, 2000.0), (2000.0, 4000.0), (4000.0, args.f_max)]

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
        bnd = [[0.0, 0.0, 0.0, 0.0] for _ in BANDS]   # he, hb, ie, ib per band
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
                hw = (args.frac * f0 if args.width == "f0frac"
                      else fk * (2.0 ** (args.cents / 1200.0) - 1.0))
                hw = max(hw, args.min_bins * bin_w / 2.0)
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
            for bi, (lo, hi) in enumerate(BANDS):
                inb = (freqs >= lo) & (freqs < hi)
                # Referenced to the BAND's own energy, so a band holding little
                # of the clip is still reported on its own terms rather than
                # vanishing into a whole-spectrum denominator.
                bt = float(power[inb].sum())
                nbb = float(inb.sum())
                if bt <= 0 or nbb <= 0:
                    continue
                bnd[bi][0] += float(power[hmask & inb].sum()) / bt
                bnd[bi][1] += float((hmask & inb).sum()) / nbb
                bnd[bi][2] += float(power[imask & inb].sum()) / bt
                bnd[bi][3] += float((imask & inb).sum()) / nbb
            pitches.append(pitch)
            used += 1

        if not used:
            continue
        he, hb, ie, ib = he / used, hb / used, ie / used, ib / used
        H = he / hb if hb else float("nan")
        I = ie / ib if ib else float("nan")
        bhi = []
        for bh, bhb, bie, bib in bnd:
            bh, bhb, bie, bib = (v / used for v in (bh, bhb, bie, bib))
            Hb = bh / bhb if bhb else float("nan")
            Ib = bie / bib if bib else float("nan")
            bhi.append(Hb / Ib if Ib else float("nan"))
        rows[name] = dict(n=len(fs), used=used, he=he, hb=hb, ie=ie, ib=ib,
                          bhi=bhi,
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

    print(f"\n=== H/I by frequency band   (where the anchoring breaks down)")
    bh = "".join(f"{f'{int(lo/1000)}-{int(hi/1000)}k':>12}" for lo, hi in BANDS)
    print(f"{'group':<24}{'n':>7}{'pitch':>7}{bh}")
    for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["hi"]):
        print(f"{name:<24}{r['n']:>7}{r['pitch']:>7.0f}"
              + "".join(f"{v:>12.1f}" for v in r["bhi"]))
    print("\n  READ THE BANDS, NOT THE TOTAL. eng_I% is under 2% for nearly")
    print("  every class, so the whole-spectrum H/I is a ratio over a near-zero")
    print("  denominator and swings by 2x between windows on the same audio.")
    print("  Per band the denominators are real where the noise actually is.")
    print("  The first version of this measured almost none of that: a cents")
    print("  window grows with k*f0 while the gap stays f0, so it dropped every")
    print("  harmonic above k ~ 14 -- roughly 4 kHz here, 73-92% of them, at a")
    print("  rate that tracked pitch. --width f0frac keeps the zones disjoint")
    print("  at every k and drops nothing.")
    print("\n  H/I is the number to read. White noise gives exactly 1 -- energy")
    print("  is spread evenly, so both zones are enriched equally. A harmonic")
    print("  stack drives I toward 0 and H/I up.")
    print("\n  WHAT THIS MEASURED, and it refuted the prediction it was built")
    print("  for. The hypothesis was string_acoustic HIGH (bow noise anchored")
    print("  to partials) and reed_acoustic LOW (aerodynamic noise filling the")
    print("  gaps), since flatness and floor_dB call them identical while they")
    print("  are the largest hybrid win and the largest linear win. The result")
    print("  is the reverse, in EVERY band and at both widths: reed 377/23/9")
    print("  against string 124/10/4 at --frac 0.2. Not a window artifact.")
    print("\n  The hard fact it did establish: NSynth is 95-99% harmonic by")
    print("  ENERGY in essentially every class -- reed_acoustic is 99.3% within")
    print("  +-0.2*f0 of a harmonic, string_acoustic 97.3%. A reed's visible")
    print("  noise bed carries about 0.2% of its energy and shows up only")
    print("  because a spectrogram panel spans 100 dB. That is the profile")
    print("  where a log term spends most and buys least -- many bins, almost")
    print("  no energy, none of it reachable by a synth with no noise source --")
    print("  but the inter-harmonic SHARE does not order the classes: string")
    print("  has 4x more of it than reed and is the largest hybrid win.")
    print("\n  As a predictor of which loss wins, across the nineteen classes")
    print("  with a margin: r = -0.26 / -0.33 / -0.26 by band, Spearman -0.27 /")
    print("  -0.26 / -0.22, p ~ 0.17 at n=19. The sign is at least the one the")
    print("  mechanism wants -- more anchored, linear wins -- and the top of the")
    print("  4-7k column is four linear wins in a row, but string_acoustic sits")
    print("  fifth. Not significant, and the fifth audio statistic in a row to")
    print("  fail. That NOTHING computable from the audio predicts the outcome")
    print("  is by now a result rather than a gap.")
    print("\n  keyboard_acoustic reads low for a reason that is not noise: piano")
    print("  partials sit at k*f0*sqrt(1+B*k^2) from string stiffness and drift")
    print("  out of the zone as k grows. Real as a limit on what harmor can fit,")
    print("  but a different mechanism. String vibrato does the same thing --")
    print("  a partial swept +-50 cents leaves a +-0.2*f0 zone by k ~ 5 -- so")
    print("  string's low reading may be modulation rather than noise at all.")


if __name__ == "__main__":
    main()
