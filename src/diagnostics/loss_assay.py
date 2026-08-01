"""Loss assay for candidate losses across IR length and spectral settings.

This is a diagnostics-focused adaptation of ela_results/ela_landscape.py.
It evaluates many ELA-style metrics for combinations of:
- candidate loss family
- input IR length (in samples)
- n_fft (for applicable losses)

Output is numeric tables (CSV) only.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from scipy.stats.qmc import Sobol
import torch

from src.loss.candidate_losses import make_loss_factory
from src.plate.SevenParamPlate import BatchedModalPlateTorch

SAMPLE_RATE = 44100
NU = 0.25

PARAM_KEYS = ["E", "rho", "h", "Ly", "T0", "op_x", "op_y"]
PARAM_BOUNDS = {
    "E": (6.7e10, 2.2e11),
    "rho": (2430.0, 21230.0),
    "h": (0.001, 0.005),
    "Ly": (1.1, 4.0),
    "T0": (0.01, 1000.0),
    "op_x": (0.51, 1.0),
    "op_y": (0.51, 1.0),
}

BOUNDS_LO = np.array([PARAM_BOUNDS[k][0] for k in PARAM_KEYS], dtype=np.float64)
BOUNDS_HI = np.array([PARAM_BOUNDS[k][1] for k in PARAM_KEYS], dtype=np.float64)
BOUNDS_RANGE = BOUNDS_HI - BOUNDS_LO

COMPOSITE_KEYS = ["mu", "D_mu", "T0_mu", "Ly", "op_x", "op_y"]
COMPOSITE_BOUNDS = {
    "mu": (2.43, 106.15),
    "D_mu": (0.28052546, 201.188843),
    "T0_mu": (0.000094206, 411.522634),
    "Ly": (1.1, 4.0),
    "op_x": (0.51, 1.0),
    "op_y": (0.51, 1.0),
}

FIXED_PLATE_PARAMS = {
    "Lx": 1.0,
    "nu": 0.25,
    "T60_DC": 6.0,
    "T60_F1": 2.0,
    "loss_F1": 500.0,
    "fp_x": 0.335,
    "fp_y": 0.467,
}

CANDIDATE_LOSSES = [
    "l1_stft",
    "l1_cqt",
    "waveform_l2",
    "waveform_l1",
    "group_delay",
    "gammatone",
    "fft",
]


def physical_to_plate14(phys_batch_np: np.ndarray, device: str | torch.device, dtype: torch.dtype) -> torch.Tensor:
    E, rho, h, Ly, T0, op_x, op_y = [phys_batch_np[:, i] for i in range(7)]
    cols = np.stack(
        [
            np.full_like(E, FIXED_PLATE_PARAMS["Lx"]),
            Ly,
            h,
            T0,
            rho,
            E,
            np.full_like(E, FIXED_PLATE_PARAMS["nu"]),
            np.full_like(E, FIXED_PLATE_PARAMS["T60_DC"]),
            np.full_like(E, FIXED_PLATE_PARAMS["T60_F1"]),
            np.full_like(E, FIXED_PLATE_PARAMS["loss_F1"]),
            np.full_like(E, FIXED_PLATE_PARAMS["fp_x"]),
            np.full_like(E, FIXED_PLATE_PARAMS["fp_y"]),
            op_x,
            op_y,
        ],
        axis=1,
    )
    return torch.tensor(cols, dtype=dtype, device=device)


def normalize_7d(params: np.ndarray, gt: np.ndarray) -> np.ndarray:
    p_norm = (params - BOUNDS_LO) / BOUNDS_RANGE
    g_norm = (gt - BOUNDS_LO) / BOUNDS_RANGE
    return np.sqrt(np.sum((p_norm - g_norm) ** 2, axis=-1))


def to_composite_np(params_7d: np.ndarray) -> np.ndarray:
    if params_7d.ndim == 1:
        params_7d = params_7d.reshape(1, -1)
    E, rho, h, Ly, T0, op_x, op_y = [params_7d[:, i] for i in range(7)]
    mu = rho * h
    D = (E * h**3) / (12 * (1 - NU**2))
    return np.stack([mu, D / mu, T0 / mu, Ly, op_x, op_y], axis=1)


def normalize_6d(params_7d: np.ndarray, gt_7d: np.ndarray) -> np.ndarray:
    comp = to_composite_np(params_7d)
    comp_gt = to_composite_np(gt_7d)
    blo = np.array([COMPOSITE_BOUNDS[k][0] for k in COMPOSITE_KEYS], dtype=np.float64)
    bhi = np.array([COMPOSITE_BOUNDS[k][1] for k in COMPOSITE_KEYS], dtype=np.float64)
    p_norm = (comp - blo) / (bhi - blo)
    g_norm = (comp_gt - blo) / (bhi - blo)
    return np.sqrt(np.sum((p_norm - g_norm) ** 2, axis=-1))


def build_candidate_loss(loss_name: str, sample_rate: int, n_fft: int | None, length: int):
    key = loss_name.lower()
    if key == "fft":
        return make_loss_factory("fft", n_fft=int(length))
    if key == "l1_stft":
        return make_loss_factory("l1_stft", n_fft=int(n_fft), hop_length=max(1, int(n_fft) // 4))
    if key == "group_delay":
        return make_loss_factory("group_delay", n_fft=int(n_fft), hop_length=max(1, int(n_fft) // 4))
    if key == "l1_cqt":
        return make_loss_factory("l1_cqt", sample_rate=sample_rate, hop_length=max(1, int(n_fft) // 4))
    if key == "gammatone":
        return make_loss_factory(
            "gammatone",
            sample_rate=sample_rate,
            n_fft=int(n_fft),
            hop_length=max(1, int(n_fft) // 4),
        )
    if key == "waveform_l1":
        return make_loss_factory("waveform_l1")
    if key == "waveform_l2":
        return make_loss_factory("waveform_l2")
    raise ValueError(f"Unknown candidate loss: {loss_name}")


def generate_sobol_targets(n_targets: int, seed: int = 42) -> np.ndarray:
    sampler = Sobol(d=7, scramble=True, seed=seed)
    n_pow2 = 1
    while n_pow2 < n_targets + 1:
        n_pow2 *= 2
    raw = sampler.random(n_pow2)
    raw = np.clip(raw[1 : n_targets + 1], 0.05, 0.95)
    return BOUNDS_LO + raw * BOUNDS_RANGE


def evaluate_batch(loss_fn, target_t: torch.Tensor, candidate_batch: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        vals = loss_fn(candidate_batch, target_t.expand(candidate_batch.shape[0], -1))
    arr = vals.detach().cpu().numpy()
    return np.nan_to_num(arr, nan=1e6, posinf=1e6, neginf=1e6)


def sample_and_evaluate(
    synth,
    loss_fn,
    gt_params,
    target_ir_t,
    duration,
    device,
    dtype,
    n_samples,
    batch_size,
):
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
        losses.append(evaluate_batch(loss_fn, target_ir_t, audios))

    return samples, dist_7d, dist_6d, np.concatenate(losses)


def measure_fdc(dist_7d, dist_6d, loss_vals, n_bins=5):
    def _corr_bins(d):
        pct = np.percentile(d, np.linspace(0, 100, n_bins + 1))
        out = []
        for b in range(n_bins):
            lo, hi = pct[b], pct[b + 1]
            mask = (d >= lo) & (d <= hi if b == n_bins - 1 else d < hi)
            if mask.sum() < 20:
                out.append(np.nan)
            else:
                r, _ = scipy_stats.spearmanr(d[mask], loss_vals[mask])
                out.append(r)
        return out

    f7 = _corr_bins(dist_7d)
    f6 = _corr_bins(dist_6d)
    return f7, f6


def measure_metamodel(dist_7d, loss_vals):
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(dist_7d, loss_vals)
    del intercept, p_value, std_err
    return r_value**2, slope


def measure_y_distribution(loss_vals):
    return float(scipy_stats.skew(loss_vals)), float(scipy_stats.kurtosis(loss_vals))


def measure_autocorrelation_and_entropy(
    synth,
    loss_fn,
    gt_params,
    target_ir_t,
    duration,
    device,
    dtype,
    n_walks,
    walk_length,
    step_frac=0.05,
):
    step_size = BOUNDS_RANGE * step_frac
    ac_lengths = []
    entropies = []

    for _ in range(n_walks):
        pos = BOUNDS_LO + np.random.rand(7) * BOUNDS_RANGE
        walk_points = [pos.copy()]
        for _k in range(walk_length - 1):
            d = np.random.randn(7)
            d /= max(np.linalg.norm(d), 1e-15)
            pos = np.clip(pos + d * step_size, BOUNDS_LO, BOUNDS_HI)
            walk_points.append(pos.copy())
        walk_points = np.asarray(walk_points)

        seq_parts = []
        for i in range(0, walk_length, 50):
            batch = walk_points[i : i + 50]
            plate14 = physical_to_plate14(batch, device, dtype)
            with torch.no_grad():
                audios = synth(plate14, duration)
            seq_parts.append(evaluate_batch(loss_fn, target_ir_t, audios))
        seq = np.concatenate(seq_parts)

        centered = seq - np.mean(seq)
        var = np.var(seq)
        if var < 1e-15:
            ac_lengths.append(float(walk_length))
        else:
            max_lag = min(50, len(seq) // 2)
            ac = np.array(
                [np.mean(centered[: len(seq) - lag] * centered[lag:]) / var for lag in range(1, max_lag + 1)]
            )
            below = np.where(ac < 1.0 / np.e)[0]
            ac_lengths.append(float(below[0] + 1 if len(below) > 0 else max_lag))

        if len(seq) >= 3:
            diffs = np.diff(seq)
            eps = np.std(seq) * 0.01 if np.std(seq) > 0 else 1e-10
            labels = np.where(diffs > eps, 1, np.where(diffs < -eps, -1, 0))
            counts = np.array([np.sum(labels == v) for v in (-1, 0, 1)], dtype=float) + 1e-10
            probs = counts / counts.sum()
            entropies.append(float(-np.sum(probs * np.log2(probs))))

    ac_med = float(np.nanmedian(ac_lengths)) if ac_lengths else np.nan
    ent_med = float(np.nanmedian(entropies)) if entropies else np.nan
    return ac_med, ent_med


def measure_monotonicity(
    synth,
    loss_fn,
    gt_params,
    target_ir_t,
    duration,
    device,
    dtype,
    n_starts,
    n_steps,
):
    gt = gt_params.reshape(1, -1)
    monos = []

    for _ in range(n_starts):
        start = BOUNDS_LO + np.random.rand(7) * BOUNDS_RANGE
        alphas = np.linspace(0.0, 1.0, n_steps)
        path = np.array([start + a * (gt.flatten() - start) for a in alphas])

        seq_parts = []
        for i in range(0, n_steps, 50):
            batch = path[i : i + 50]
            plate14 = physical_to_plate14(batch, device, dtype)
            with torch.no_grad():
                audios = synth(plate14, duration)
            seq_parts.append(evaluate_batch(loss_fn, target_ir_t, audios))
        seq = np.concatenate(seq_parts)
        if len(seq) >= 3:
            monos.append(float(np.mean(np.diff(seq) < 0)))

    return float(np.nanmedian(monos)) if monos else np.nan


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
    lengths = args.lengths
    nffts = args.n_ffts

    print("=" * 70)
    print(" LOSS ASSAY")
    print("=" * 70)
    print(f"Device: {device}, dtype={dtype}")
    print(f"Targets: {args.n_targets}")
    print(f"Losses: {losses}")
    print(f"Lengths: {lengths}")
    print(f"n_ffts: {nffts}")

    synth = BatchedModalPlateTorch(sample_rate=SAMPLE_RATE, device=device, dtype=dtype)
    gt_targets = generate_sobol_targets(args.n_targets, seed=args.seed)

    summary_rows = []
    t0_all = time.time()

    for loss_name in losses:
        for length in lengths:
            combos = []
            if loss_name in ("waveform_l1", "waveform_l2"):
                combos = [None]
            elif loss_name == "fft":
                combos = [int(length)]
            else:
                combos = [int(nf) for nf in nffts]

            for n_fft in combos:
                combo_label = f"loss={loss_name}, len={length}, n_fft={n_fft}"
                print(f"\n--- {combo_label} ---")

                try:
                    loss_fn = build_candidate_loss(loss_name, SAMPLE_RATE, n_fft, length)
                except Exception as e:
                    print(f"  skip: cannot build loss ({e})")
                    continue

                all_fdc7 = []
                all_fdc6 = []
                all_ac = []
                all_ent = []
                all_r2 = []
                all_slope = []
                all_skew = []
                all_kurt = []
                all_mono = []

                for ti in range(args.n_targets):
                    gt = gt_targets[ti]

                    duration = float(length) / SAMPLE_RATE
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
                    "loss": loss_name,
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
                    f"  mono={out['monotonicity']:.3f} ac={out['autocorr_length']:.2f} ent={out['entropy']:.3f} "
                    f"fdc7={out['fdc_7d_global']:.3f}"
                )

    if not summary_rows:
        raise RuntimeError("No combinations produced results")

    out_csv = out_dir / "loss_assay_summary.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved: {out_csv}")
    print(f"Total time: {time.time() - t0_all:.1f}s")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output_dir", type=str, default="results/diagnostics/loss_assay")
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
    p.add_argument("--lengths", nargs="+", type=int, default=[4096, 8192, 16384])
    p.add_argument("--n_ffts", nargs="+", type=int, default=[512, 1024, 2048, 4096])

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
