"""Loss-combination assay with ordered loss pairs and lambda weighting.

For each ordered pair (main_loss, sub_loss) where main != sub (n-permutation-2),
and each lambda in --lambdas, this script evaluates combined loss:

    L_combined = L_main + lambda * L_sub

across ELA-style metrics, IR lengths, and n_fft settings.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path

import numpy as np
import torch

from src.diagnostics.loss_assay import (
    CANDIDATE_LOSSES,
    SAMPLE_RATE,
    BatchedModalPlateTorch,
    build_candidate_loss,
    generate_sobol_targets,
    measure_autocorrelation_and_entropy,
    measure_fdc,
    measure_metamodel,
    measure_monotonicity,
    measure_y_distribution,
    physical_to_plate14,
    sample_and_evaluate,
)


def _sample_and_evaluate_combined(
    synth,
    main_loss_fn,
    sub_loss_fn,
    lam,
    gt_params,
    target_ir_t,
    duration,
    device,
    dtype,
    n_samples,
    batch_size,
):
    # Adapted from loss_assay.sample_and_evaluate; preserves sampling semantics.
    from src.diagnostics.loss_assay import BOUNDS_LO, BOUNDS_RANGE, normalize_6d, normalize_7d

    gt = gt_params.reshape(1, -1)
    samples = BOUNDS_LO + np.random.rand(n_samples, 7) * BOUNDS_RANGE
    dist_7d = normalize_7d(samples, gt)
    dist_6d = normalize_6d(samples, gt)

    losses = []
    for i in range(0, n_samples, batch_size):
        batch = samples[i : i + batch_size]
        plate14 = physical_to_plate14(batch, device, dtype)
        with torch.no_grad():
            audios = synth(plate14, duration)
            target_exp = target_ir_t.expand(audios.shape[0], -1)
            main_vals = main_loss_fn(audios, target_exp)
            sub_vals = sub_loss_fn(audios, target_exp)
            vals = main_vals + float(lam) * sub_vals
        arr = vals.detach().cpu().numpy()
        losses.append(np.nan_to_num(arr, nan=1e6, posinf=1e6, neginf=1e6))

    return samples, dist_7d, dist_6d, np.concatenate(losses)


def _measure_autocorr_entropy_combined(
    synth,
    main_loss_fn,
    sub_loss_fn,
    lam,
    gt_params,
    target_ir_t,
    duration,
    device,
    dtype,
    n_walks,
    walk_length,
):
    # Wrap combined fn into one callable compatible with shared helper.
    def comb_loss(candidate, target):
        return main_loss_fn(candidate, target) + float(lam) * sub_loss_fn(candidate, target)

    return measure_autocorrelation_and_entropy(
        synth,
        comb_loss,
        gt_params,
        target_ir_t,
        duration,
        device,
        dtype,
        n_walks,
        walk_length,
    )


def _measure_monotonicity_combined(
    synth,
    main_loss_fn,
    sub_loss_fn,
    lam,
    gt_params,
    target_ir_t,
    duration,
    device,
    dtype,
    n_starts,
    n_steps,
):
    def comb_loss(candidate, target):
        return main_loss_fn(candidate, target) + float(lam) * sub_loss_fn(candidate, target)

    return measure_monotonicity(
        synth,
        comb_loss,
        gt_params,
        target_ir_t,
        duration,
        device,
        dtype,
        n_starts,
        n_steps,
    )


def run(args):
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float64":
        dtype = torch.float64
    else:
        dtype = torch.float32

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    losses = args.losses
    lambdas = args.lambdas
    lengths = args.lengths
    nffts = args.n_ffts

    ordered_pairs = [(a, b) for a, b in itertools.permutations(losses, 2)]

    print("=" * 72)
    print(" LOSS ASSAY (COMBINED)")
    print("=" * 72)
    print(f"Device: {device}, dtype={dtype}")
    print(f"Targets: {args.n_targets}")
    print(f"Losses: {losses}")
    print(f"Ordered pairs (nP2): {len(ordered_pairs)}")
    print(f"Lambdas: {lambdas}")
    print(f"Lengths: {lengths}")
    print(f"n_ffts: {nffts}")

    synth = BatchedModalPlateTorch(sample_rate=SAMPLE_RATE, device=device, dtype=dtype)
    gt_targets = generate_sobol_targets(args.n_targets, seed=args.seed)

    summary_rows = []
    t0_all = time.time()

    for main_loss, sub_loss in ordered_pairs:
        for length in lengths:
            # Keep n_fft schedule tied to main loss semantics, matching base assay style.
            if main_loss in ("waveform_l1", "waveform_l2"):
                fft_choices = [None]
            elif main_loss == "fft":
                fft_choices = [int(length)]
            else:
                fft_choices = [int(nf) for nf in nffts]

            for n_fft in fft_choices:
                for lam in lambdas:
                    label = (
                        f"main={main_loss} sub={sub_loss} lambda={lam} "
                        f"len={length} n_fft={n_fft}"
                    )
                    print(f"\n--- {label} ---")

                    try:
                        main_fn = build_candidate_loss(main_loss, SAMPLE_RATE, n_fft, length)
                        sub_fn = build_candidate_loss(sub_loss, SAMPLE_RATE, n_fft, length)
                    except Exception as e:
                        print(f"  skip: cannot build loss pair ({e})")
                        continue

                    all_fdc7, all_fdc6 = [], []
                    all_ac, all_ent = [], []
                    all_r2, all_slope = [], []
                    all_skew, all_kurt = [], []
                    all_mono = []

                    for ti in range(args.n_targets):
                        gt = gt_targets[ti]
                        duration = float(length) / SAMPLE_RATE

                        plate14 = physical_to_plate14(gt.reshape(1, -1), device, dtype)
                        with torch.no_grad():
                            target_ir = synth(plate14, duration).squeeze(0)
                        target_t = target_ir.unsqueeze(0)

                        np.random.seed(args.seed + 10_000 * ti)
                        _, d7, d6, losses_vals = _sample_and_evaluate_combined(
                            synth=synth,
                            main_loss_fn=main_fn,
                            sub_loss_fn=sub_fn,
                            lam=lam,
                            gt_params=gt,
                            target_ir_t=target_t,
                            duration=duration,
                            device=device,
                            dtype=dtype,
                            n_samples=args.n_samples,
                            batch_size=args.batch_size,
                        )

                        f7, f6 = measure_fdc(d7, d6, losses_vals, n_bins=args.n_bins)
                        all_fdc7.append(f7)
                        all_fdc6.append(f6)

                        r2, slope = measure_metamodel(d7, losses_vals)
                        all_r2.append(r2)
                        all_slope.append(slope)

                        skew, kurt = measure_y_distribution(losses_vals)
                        all_skew.append(skew)
                        all_kurt.append(kurt)

                        np.random.seed(args.seed + 10_000 * ti + 1)
                        ac, ent = _measure_autocorr_entropy_combined(
                            synth=synth,
                            main_loss_fn=main_fn,
                            sub_loss_fn=sub_fn,
                            lam=lam,
                            gt_params=gt,
                            target_ir_t=target_t,
                            duration=duration,
                            device=device,
                            dtype=dtype,
                            n_walks=args.n_walks,
                            walk_length=args.walk_length,
                        )
                        all_ac.append(ac)
                        all_ent.append(ent)

                        np.random.seed(args.seed + 10_000 * ti + 2)
                        mono = _measure_monotonicity_combined(
                            synth=synth,
                            main_loss_fn=main_fn,
                            sub_loss_fn=sub_fn,
                            lam=lam,
                            gt_params=gt,
                            target_ir_t=target_t,
                            duration=duration,
                            device=device,
                            dtype=dtype,
                            n_starts=args.n_mono_starts,
                            n_steps=args.n_mono_steps,
                        )
                        all_mono.append(mono)

                    if not all_fdc7:
                        continue

                    fdc7_med = np.nanmedian(np.asarray(all_fdc7, dtype=np.float64), axis=0)
                    fdc6_med = np.nanmedian(np.asarray(all_fdc6, dtype=np.float64), axis=0)

                    out = {
                        "main_loss": main_loss,
                        "sub_loss": sub_loss,
                        "lambda": float(lam),
                        "input_len": int(length),
                        "n_fft": int(n_fft) if n_fft is not None else "NA",
                        "n_targets": len(all_mono),
                        "monotonicity": float(np.nanmedian(all_mono)),
                        "autocorr_length": float(np.nanmedian(all_ac)),
                        "entropy": float(np.nanmedian(all_ent)),
                        "metamodel_r2": float(np.nanmedian(all_r2)),
                        "metamodel_slope": float(np.nanmedian(all_slope)),
                        "skewness": float(np.nanmedian(all_skew)),
                        "kurtosis": float(np.nanmedian(all_kurt)),
                        "fdc_7d_global": float(np.nanmean(fdc7_med)),
                        "fdc_6d_global": float(np.nanmean(fdc6_med)),
                    }
                    for b in range(args.n_bins):
                        out[f"fdc_7d_q{b}"] = float(fdc7_med[b]) if b < len(fdc7_med) else np.nan
                        out[f"fdc_6d_q{b}"] = float(fdc6_med[b]) if b < len(fdc6_med) else np.nan

                    summary_rows.append(out)
                    print(
                        f"  mono={out['monotonicity']:.3f} ac={out['autocorr_length']:.2f} "
                        f"ent={out['entropy']:.3f} fdc7={out['fdc_7d_global']:.3f}"
                    )

    if not summary_rows:
        raise RuntimeError("No combinations produced results")

    out_csv = out_dir / "loss_assay_comb_summary.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved: {out_csv}")
    print(f"Total time: {time.time() - t0_all:.1f}s")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output_dir", type=str, default="results/diagnostics/loss_assay_comb")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "float64"])

    p.add_argument("--n_targets", type=int, default=20)
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--n_bins", type=int, default=5)

    p.add_argument("--n_walks", type=int, default=10)
    p.add_argument("--walk_length", type=int, default=500)
    p.add_argument("--n_mono_starts", type=int, default=20)
    p.add_argument("--n_mono_steps", type=int, default=100)

    p.add_argument("--losses", nargs="+", default=CANDIDATE_LOSSES)
    p.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--lengths", nargs="+", type=int, default=[11025, 22050, 44100])
    p.add_argument("--n_ffts", nargs="+", type=int, default=[512, 1024, 2048, 4096])

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
