"""STFT-only loss assay for fixed multi-resolution L1-STFT configurations.

This variant evaluates a single loss at a time:
    MultiResolutionL1STFT(n_ffts=config)
with no linear combinations between different losses.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from src.diagnostics.loss_assay import (
    SAMPLE_RATE,
    BatchedModalPlateTorch,
    generate_sobol_targets,
    measure_autocorrelation_and_entropy,
    measure_fdc,
    measure_metamodel,
    measure_monotonicity,
    measure_y_distribution,
    physical_to_plate14,
    sample_and_evaluate,
)
from src.loss.candidate_losses import make_multi_resolution_l1_stft_loss


def _parse_fft_config(s: str) -> tuple[int, ...]:
    vals = [v.strip() for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError(f"Empty FFT configuration: {s}")
    return tuple(int(v) for v in vals)


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

    fft_configs = [_parse_fft_config(s) for s in args.fft_configs]
    lengths = [int(v) for v in args.lengths]

    print("=" * 72)
    print(" STFT LOSS ASSAY (FIXED MULTI-RESOLUTION)")
    print("=" * 72)
    print(f"Device: {device}, dtype={dtype}")
    print(f"Targets: {args.n_targets}")
    print(f"FFT configs: {fft_configs}")
    print(f"Lengths: {lengths}")
    print(f"Reduction: {args.reduction}")

    synth = BatchedModalPlateTorch(sample_rate=SAMPLE_RATE, device=device, dtype=dtype)
    gt_targets = generate_sobol_targets(args.n_targets, seed=args.seed)

    summary_rows = []
    t0_all = time.time()

    for length in lengths:
        duration = float(length) / SAMPLE_RATE

        for cfg in fft_configs:
            label = f"len={length} fft_config={cfg}"
            print(f"\n--- {label} ---")

            loss_fn = make_multi_resolution_l1_stft_loss(
                n_ffts=cfg,
                hop_lengths=tuple(max(1, int(nf) // 4) for nf in cfg),
                reduction=args.reduction,
            )

            all_fdc7, all_fdc6 = [], []
            all_ac, all_ent = [], []
            all_r2, all_slope = [], []
            all_skew, all_kurt = [], []
            all_mono = []

            for ti in range(args.n_targets):
                gt = gt_targets[ti]

                plate14 = physical_to_plate14(gt.reshape(1, -1), device, dtype)
                with torch.no_grad():
                    target_ir = synth(plate14, duration).squeeze(0)
                target_t = target_ir.unsqueeze(0)

                np.random.seed(args.seed + 10_000 * ti)
                _, d7, d6, losses_vals = sample_and_evaluate(
                    synth=synth,
                    loss_fn=loss_fn,
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
                ac, ent = measure_autocorrelation_and_entropy(
                    synth=synth,
                    loss_fn=loss_fn,
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
                mono = measure_monotonicity(
                    synth=synth,
                    loss_fn=loss_fn,
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
                "loss": "multi_resolution_l1_stft",
                "fft_config": ",".join(str(v) for v in cfg),
                "reduction": args.reduction,
                "input_len": int(length),
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
        raise RuntimeError("No configurations produced results")

    out_csv = out_dir / "loss_assay_stft_fixed_summary.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved: {out_csv}")
    print(f"Total time: {time.time() - t0_all:.1f}s")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output_dir", type=str, default="results/diagnostics/loss_assay_stft")
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

    p.add_argument(
        "--fft_configs",
        nargs="+",
        type=str,
        default=["512,1024,2048", "1024,2048,4096", "512,1024,2048,4096"],
        help="Each config is a comma-separated FFT list, e.g. '512,1024,2048'",
    )
    p.add_argument("--reduction", type=str, default="mean", choices=["mean", "sum"])
    p.add_argument("--lengths", nargs="+", type=int, default=[11025, 22050, 44100])

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
