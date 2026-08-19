"""How much of what an evaluation dB floor discards is masked anyway.

    python scripts/ds_masking.py --selftest
    python scripts/ds_masking.py --limit 50            # quick look
    python scripts/ds_masking.py --out ~/masking       # the whole 2000

WHY. LSD and MFCC are computed on a log whose dynamic-range floor is an
undocumented parameter, and the ranking of two training losses reverses across
it: on the real branch hybrid wins unfloored and at 80 dB below peak, ties at
70, and linear wins at 60/40/20. Choosing a floor by convention decides the
result, so it has to be chosen from psychoacoustics instead. The claim this
supports:

    content more than N dB below peak lies beneath the simultaneous-masking
    threshold of the signal itself, and should therefore not be credited by an
    evaluation metric.

MPEG-1 psychoacoustic model 1, implemented from Painter & Spanias, "Perceptual
Coding of Digital Audio", Proc. IEEE 88(4):451-513, 2000, section V -- not from
a third-party library, so every step is one we can describe in the paper.
Spreading slopes follow Schroeder, Atal & Hall, JASA 66(6):1647-1652, 1979.

Computed on the TARGET. The question is what the metric should credit in the
reference, not what a reconstruction produced.

THREE APPROXIMATIONS, WITH THEIR SIGNS

  Cells are FFT bins, not mel bands. Exact for LSD, which is bin-wise. For MFCC
  the floor is applied to the 40 mel-band energies AFTER pooling, and pooling
  sums, so a band can sit above the floor while its constituent bins sit below
  -- bin-wise S therefore contains cells MFCC keeps. Those cells are by
  construction low-energy and mostly masked, so on the energy-weighted headline
  the effect is second order and errs slightly OPTIMISTIC. On frac_bins_masked
  it is larger. Mel-pooling a masking threshold would be exact and is not a
  standard operation, so it is not done here.

  16 kHz audio puts Nyquist at 8 kHz, which removes the top ~6 Bark bands and
  means the ATH's high-frequency rise never enters -- the estimate excludes the
  region where the absolute threshold does most of its work. CONSERVATIVE:
  understates masking.

  Tonal/noise classification has no fixed sign, which is why --tonal-window
  exists and why both settings should be reported. Misclassifying partials as
  noise moves the offset by 9-24 dB.

THE ANALYSIS MATCHES THE METRIC. n_fft 1024, hop 256, Hann, center=False --
compute_lsd's configuration exactly, asserted at startup, so the cells scored
by the metric and the cells scored here are the same cells. A one-frame
mismatch would silently misalign S against the floor.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "external", "diffsynth")
sys.path.insert(0, DS)
sys.path.insert(0, HERE)

import numpy as np                                       # noqa: E402
import torch                                             # noqa: E402
from torch.utils.data import DataLoader                  # noqa: E402

N_FFT, HOP, SR = 1024, 256, 16000
PN = 90.302          # full scale -> ~90 dB SPL, the +/- 1 bit convention
ISO_BIN_HZ = 44100.0 / 512.0     # the resolution ISO model 1's windows assume

# Zwicker critical bands. Truncated at Nyquist in build_bands().
BARK_EDGES = np.array([
    0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000,
    2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500],
    dtype=float)


def bark(f):
    f = np.asarray(f, dtype=float)
    return 13.0 * np.arctan(7.6e-4 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def ath(f):
    """Terhardt's absolute threshold of hearing, dB SPL.

    f is clamped to 20 Hz: the (f/1000)^-0.8 term diverges at DC and bin 0
    would otherwise carry an infinite threshold.
    """
    k = np.maximum(np.asarray(f, dtype=float), 20.0) / 1000.0
    return (3.64 * k ** -0.8
            - 6.5 * np.exp(-0.6 * (k - 3.3) ** 2)
            + 1e-3 * k ** 4)


def tonal_offsets(f_hz, mode, bin_hz):
    """Bin offsets the 7 dB neighbourhood test is applied over.

    ISO/IEC 11172-3 model 1 specifies j in {2} below 5.5 kHz, {2,3} to 11 kHz
    and {2..6} above -- at a 512-point FFT on 44.1 kHz, i.e. 86.13 Hz bins. Our
    analysis has 15.625 Hz bins, 5.5x finer, so those offsets have to be
    converted before they mean the same thing.

    mode="hz" (default) converts them to frequency and back: the ISO inner edge
    of 2 bins is 172.3 Hz, which is 11 of our bins. The test then runs over the
    CONTIGUOUS span from 2 bins out to that edge rather than at the one
    converted offset, because "exceeds its neighbourhood" reads as a span and a
    single isolated bin is fragile against a neighbouring partial landing
    exactly on it. With a Hann window the main lobe is 4 bins wide, so j >= 2
    is already outside it and a resolved sinusoid clears 7 dB across the whole
    span.

    mode="bins" keeps ISO's literal offsets. At our resolution +/-2 bins is
    +/-31 Hz, inside the main lobe, so the test passes on broadband noise too
    and tonal maskers are massively over-detected. It exists to report the
    sensitivity, not because it is defensible.
    """
    if f_hz < 5500:
        iso = [2]
    elif f_hz < 11000:
        iso = [2, 3]
    else:
        iso = [2, 3, 4, 5, 6]
    if mode == "bins":
        return np.array(sorted(set(iso)), dtype=int)
    outer = int(round(max(iso) * ISO_BIN_HZ / bin_hz))
    return np.arange(2, max(outer, 2) + 1, dtype=int)


def build_bands(freqs):
    """(lo_bin, hi_bin) per critical band, truncated at Nyquist."""
    out = []
    for lo, hi in zip(BARK_EDGES[:-1], BARK_EDGES[1:]):
        idx = np.nonzero((freqs >= lo) & (freqs < hi))[0]
        if idx.size:
            out.append((int(idx[0]), int(idx[-1])))
    return out


def frame_maskers(P, freqs, zb, ath_b, bands, tonal_mode, decimate, bin_hz):
    """Tonal and noise maskers for one frame, after decimation.

    Returns (levels, barks, is_tonal, n_tonal_raw).
    """
    n = P.size
    # --- tonal: local maxima that dominate their neighbourhood by 7 dB
    peaks = np.nonzero((P[1:-1] > P[:-2]) & (P[1:-1] >= P[2:]))[0] + 1
    tonal_k, tonal_P = [], []
    for k in peaks:
        js = tonal_offsets(freqs[k], tonal_mode, bin_hz)
        nb = np.concatenate([k - js, k + js])
        nb = nb[(nb >= 0) & (nb < n)]
        if nb.size and np.all(P[k] - P[nb] >= 7.0):
            lo, hi = max(k - 1, 0), min(k + 2, n)
            tonal_P.append(10.0 * np.log10(np.sum(10.0 ** (0.1 * P[lo:hi]))))
            tonal_k.append(k)
    tonal_k = np.array(tonal_k, dtype=int)
    n_tonal_raw = tonal_k.size

    # --- noise: per band, everything not absorbed into a tonal masker
    claimed = np.zeros(n, dtype=bool)
    for k in tonal_k:
        claimed[max(k - 1, 0):min(k + 2, n)] = True
    noise_k, noise_P = [], []
    for lo, hi in bands:
        ks = np.arange(lo, hi + 1)
        ks = ks[~claimed[ks]]
        ks = ks[ks > 0]                       # DC has no geometric mean
        if ks.size == 0:
            continue
        e = np.sum(10.0 ** (0.1 * P[ks]))
        if e <= 0:
            continue
        kbar = int(round(np.exp(np.mean(np.log(ks.astype(float))))))
        noise_k.append(min(max(kbar, 1), n - 1))
        noise_P.append(10.0 * np.log10(e))

    lev = np.array(list(tonal_P) + list(noise_P), dtype=float)
    kk = np.array(list(tonal_k) + list(noise_k), dtype=int)
    ton = np.array([True] * len(tonal_P) + [False] * len(noise_P), dtype=bool)
    if lev.size == 0:
        return lev, np.array([]), ton, n_tonal_raw

    # --- decimate: below the absolute threshold, then within 0.5 Bark
    keep = lev >= ath_b[kk]
    lev, kk, ton = lev[keep], kk[keep], ton[keep]
    if lev.size == 0:
        return lev, np.array([]), ton, n_tonal_raw
    z = zb[kk]
    order = np.argsort(-lev)
    kept = []
    for i in order:
        # decimate="all" is ISO's own rule and the default; "tonal" restricts
        # it to tonal pairs, which is a deviation and needs saying if used.
        pool = kept if decimate == "all" else [j for j in kept if ton[j]]
        if ton[i] or decimate == "all":
            if any(abs(z[i] - z[j]) < 0.5 for j in pool):
                continue
        kept.append(i)
    kept = np.array(sorted(kept), dtype=int)
    return lev[kept], z[kept], ton[kept], n_tonal_raw


def global_threshold(lev, zm, ton, zb, ath_b):
    """Power-sum of every masker's spread threshold with the ATH."""
    tot = 10.0 ** (0.1 * ath_b)
    if lev.size == 0:
        return 10.0 * np.log10(tot)
    dz = zb[None, :] - zm[:, None]
    # Schroeder et al.: -27 dB/Bark below the masker, and an upper slope that
    # shallows with level -- the upward spread of masking.
    sf = np.where(dz < 0, 27.0 * dz,
                  (-27.0 + 0.37 * np.maximum(lev[:, None] - 40.0, 0.0)) * dz)
    off = np.where(ton, 14.5 + zm, 5.5)
    T = lev[:, None] - off[:, None] + sf
    return 10.0 * np.log10(tot + np.sum(10.0 ** (0.1 * T), axis=0))


def analyse(x, floors, tonal_mode, decimate):
    """One clip -> per-floor fractions, plus masker counts."""
    win = torch.hann_window(N_FFT)
    rem = (x.shape[-1] - N_FFT) % HOP
    if rem:
        x = torch.nn.functional.pad(x, (0, HOP - rem))
    stft = torch.stft(x, N_FFT, hop_length=HOP, window=win, center=False,
                      return_complex=True)
    mag = (2.0 * stft.abs() / win.sum()).numpy()
    P = PN + 20.0 * np.log10(np.maximum(mag, 1e-30))     # [bins, frames]

    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    bin_hz = SR / N_FFT
    zb, ath_b = bark(freqs), ath(freqs)
    bands = build_bands(freqs)

    T = np.empty_like(P)
    n_ton = []
    for t in range(P.shape[1]):
        lev, zm, ton, raw = frame_maskers(P[:, t], freqs, zb, ath_b, bands,
                                          tonal_mode, decimate, bin_hz)
        T[:, t] = global_threshold(lev, zm, ton, zb, ath_b)
        n_ton.append(raw)

    e = 10.0 ** (0.1 * P)
    peak, tot_e = P.max(), e.sum()
    below = P < T
    out = {"frames": P.shape[1], "n_tonal_mean": float(np.mean(n_ton))}
    for F in floors:
        S = P < peak - F
        eS = e[S].sum()
        out[f"frac_energy_masked_{F:g}"] = float(e[S & below].sum() / eS) if eS > 0 else float("nan")
        out[f"frac_bins_masked_{F:g}"] = float(below[S].mean()) if S.any() else float("nan")
        out[f"frac_total_energy_{F:g}"] = float(eS / tot_e)
    return out


# --------------------------------------------------------------------------
def selftest():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    zb, ath_b = bark(freqs), ath(freqs)
    bands = build_bands(freqs)
    bin_hz = SR / N_FFT
    n = np.arange(N_FFT)

    def frame_of(sig):
        w = torch.hann_window(N_FFT).numpy()
        X = np.fft.rfft(w * sig)
        return PN + 20.0 * np.log10(np.maximum(2.0 * np.abs(X) / w.sum(), 1e-30))

    print("=== 1. 1 kHz sine at 80 dB SPL")
    amp = 10.0 ** ((80.0 - PN) / 20.0)
    P = frame_of(amp * np.sin(2 * np.pi * 1000 * n / SR))
    lev, zm, ton, raw = frame_maskers(P, freqs, zb, ath_b, bands, "hz", "all", bin_hz)
    T = global_threshold(lev, zm, ton, zb, ath_b)
    k = int(np.argmin(np.abs(freqs - 1000)))
    tk = np.nonzero(ton)[0]
    print(f"   tonal maskers: {int(ton.sum())} (raw peaks passing 7 dB: {raw})")
    if tk.size:
        i = tk[np.argmax(lev[tk])]
        print(f"   strongest tonal at {zm[i]:.2f} Bark, level {lev[i]:.1f} dB")
        print(f"   offset (14.5 + z_m)      = {14.5 + zm[i]:.1f} dB")
        print(f"   SMR = P(peak) - T(peak)  = {P[k] - T[k]:.1f} dB")
        print("   the two are printed apart on purpose: the canonical ~24 dB is a"
              "\n   tone-masks-noise figure, and this model reaching it via"
              "\n   14.5 + z_m ~ 23 at 1 kHz would otherwise look like agreement"
              "\n   when it is arithmetic.")

    print("\n=== 2. white noise")
    rng = np.random.default_rng(0)
    P = frame_of(0.05 * rng.standard_normal(N_FFT))
    lev, zm, ton, raw = frame_maskers(P, freqs, zb, ath_b, bands, "hz", "all", bin_hz)
    print(f"   tonal maskers: {int(ton.sum())} (expect ~0), noise: {int((~ton).sum())}")

    print("\n=== 3. dense harmonic tone (f0 200 Hz, 20 partials)")
    sig = sum(np.sin(2 * np.pi * 200 * h * n / SR) / h for h in range(1, 21))
    P = frame_of(0.2 * sig / np.max(np.abs(sig)))
    lev, zm, ton, raw = frame_maskers(P, freqs, zb, ath_b, bands, "hz", "all", bin_hz)
    T = global_threshold(lev, zm, ton, zb, ath_b)
    if lev.size:
        dz = zb[None, :] - zm[:, None]
        sf = np.where(dz < 0, 27.0 * dz,
                      (-27.0 + 0.37 * np.maximum(lev[:, None] - 40.0, 0.0)) * dz)
        off = np.where(ton, 14.5 + zm, 5.5)
        single = (lev[:, None] - off[:, None] + sf).max(axis=0)
        lo, hi = 0, int(np.argmin(np.abs(freqs - 4000)))
        print(f"   maskers: {lev.size} ({int(ton.sum())} tonal)")
        print(f"   max(T_global - best single masker) below 4 kHz = "
              f"{np.max(T[lo:hi] - single[lo:hi]):.2f} dB")
        print("   must be clearly positive -- the composite threshold exceeding"
              "\n   any one masker is the effect the whole argument relies on.")

    print("\n=== 4. ATH")
    ff = np.linspace(20, 8000, 4000)
    a = ath(ff)
    print(f"   minimum {a.min():.2f} dB SPL at {ff[np.argmin(a)]:.0f} Hz")
    print("   expect about -5 dB near 3.3 kHz. The canonical statement of the"
          "\n   ATH is '0 dB SPL at 4 kHz', but that is the idealised curve;"
          "\n   Terhardt's fit dips slightly below zero and peaks a little"
          "\n   lower. Not a failure -- the check is that the minimum is in"
          "\n   the 3-4 kHz region and within a few dB of zero.")
    plt.figure(figsize=(6, 3.2))
    plt.plot(ff, a)
    plt.xscale("log"); plt.ylim(-10, 60); plt.grid(alpha=.3)
    plt.xlabel("Hz"); plt.ylabel("dB SPL"); plt.title("Terhardt ATH")
    plt.tight_layout(); plt.savefig("ath.png", dpi=110)
    print("   curve written to ath.png")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="results/diffsynth")
    p.add_argument("--run", default="real_hybridx",
                   help="Any arm -- every run shares one ood_valid membership "
                        "hash, so the split is the same whichever is used. The "
                        "model is built only because the split depends on RNG "
                        "draw order.")
    p.add_argument("--ckpt", default="latest.ckpt")
    p.add_argument("--floors", type=float, nargs="+",
                   default=[80.0, 70.0, 60.0, 40.0, 20.0])
    p.add_argument("--tonal-window", default="hz", choices=("hz", "bins"))
    p.add_argument("--decimate", default="all", choices=("all", "tonal"),
                   help="ISO applies the 0.5 Bark rule to every masker; 'tonal' "
                        "restricts it and is a deviation worth declaring.")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N files, for a quick look")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default="masking")
    p.add_argument("--hist", type=float, default=70.0,
                   help="Floor to histogram frac_energy_masked at")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        selftest()
        return

    import ds_param_breakdown as pb

    d = os.path.join(args.root, args.run)
    model, cfg, dm, note = pb.load_arm(d, args.ckpt, "cpu", args.batch_size)
    if model is None:
        raise SystemExit(f"{args.run}: {note}")
    del model
    vset = dm.ood_datasets["valid"]

    # Filenames, by unwrapping the nested Subset the way split_manifest does.
    idx, base = list(range(len(vset))), vset
    while hasattr(base, "indices"):
        idx = [base.indices[i] for i in idx]
        base = base.dataset
    names = [os.path.basename(base.raw_files[i]) for i in idx]

    os.makedirs(args.out, exist_ok=True)
    rows = []
    loader = DataLoader(vset, batch_size=args.batch_size, num_workers=0)
    done = 0
    print(f"{args.run}: {len(vset)} ood valid files, floors {args.floors}, "
          f"tonal-window={args.tonal_window}, decimate={args.decimate}")
    for batch in loader:
        for j in range(batch["audio"].shape[0]):
            x = batch["audio"][j].cpu()
            if done == 0:
                nf = (x.shape[-1] - N_FFT) // HOP + 1
                assert (x.shape[-1] - N_FFT) % HOP == 0, (
                    "frame grid does not divide evenly; compute_lsd pads "
                    "differently and S would misalign")
                print(f"  {x.shape[-1]} samples -> {nf} frames "
                      f"(compute_lsd's grid)")
            r = analyse(x, args.floors, args.tonal_window, args.decimate)
            r["file"] = names[done]
            r["family"] = names[done].split("_acoustic_")[0] \
                .split("_electronic_")[0].split("_synthetic_")[0]
            rows.append(r)
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(vset)}")
            if args.limit and done >= args.limit:
                break
        if args.limit and done >= args.limit:
            break

    cols = ["file", "family", "frames", "n_tonal_mean"] + [
        f"{m}_{F:g}" for F in args.floors
        for m in ("frac_energy_masked", "frac_bins_masked", "frac_total_energy")]
    path = os.path.join(args.out, "masking.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}  ({len(rows)} files)")

    def q(v, p):
        v = sorted(v)
        return v[max(0, min(len(v) - 1, int(round(p * (len(v) - 1)))))]

    print(f"\nmasker counts: mean tonal per frame = "
          f"{st.mean(r['n_tonal_mean'] for r in rows):.1f} "
          f"(min {min(r['n_tonal_mean'] for r in rows):.1f}, "
          f"max {max(r['n_tonal_mean'] for r in rows):.1f})")
    print("  logged per file so a systematic f0 effect from the +/-172 Hz "
          "window\n  and the 0.5 Bark decimation is visible rather than "
          "discovered in the aggregate.")

    print(f"\n{'floor':>6}{'median':>10}{'IQR':>20}{'>0.9':>8}"
          f"{'med frac_bins':>15}{'med frac_tot_E':>16}")
    for F in args.floors:
        v = [r[f"frac_energy_masked_{F:g}"] for r in rows
             if r[f"frac_energy_masked_{F:g}"] == r[f"frac_energy_masked_{F:g}"]]
        b = [r[f"frac_bins_masked_{F:g}"] for r in rows]
        t = [r[f"frac_total_energy_{F:g}"] for r in rows]
        if not v:
            continue
        print(f"{F:>6.0f}{st.median(v):>10.4f}"
              f"{f'[{q(v, .25):.4f}, {q(v, .75):.4f}]':>20}"
              f"{sum(x > 0.9 for x in v) / len(v):>8.2f}"
              f"{st.median(b):>15.4f}{st.median(t):>16.6f}")

    fams = sorted({r["family"] for r in rows})
    print(f"\nby family, frac_energy_masked median")
    print(f"{'family':<14}{'n':>5}" + "".join(f"{F:>10.0f}" for F in args.floors))
    for fam in fams:
        sub = [r for r in rows if r["family"] == fam]
        cells = "".join(
            f"{st.median([r[f'frac_energy_masked_{F:g}'] for r in sub]):>10.4f}"
            for F in args.floors)
        print(f"{fam:<14}{len(sub):>5}{cells}")

    if args.hist:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        v = [r[f"frac_energy_masked_{args.hist:g}"] for r in rows]
        plt.figure(figsize=(6, 3.4))
        plt.hist(v, bins=40, range=(0, 1))
        plt.xlabel(f"frac_energy_masked at {args.hist:g} dB below peak")
        plt.ylabel("files"); plt.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"hist_{args.hist:g}.png"), dpi=110)
        print(f"\nhistogram -> {args.out}/hist_{args.hist:g}.png")


if __name__ == "__main__":
    main()
