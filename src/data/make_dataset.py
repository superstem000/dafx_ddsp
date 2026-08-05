"""Generate an IR dataset in one step, with torch-rendered targets.

This replaces the two-step process:

    python -m ModalPlate.DatasetGen --number 200 --duration 0.25 --seed 0
    python gen_torch_targets_200.py        # overwrite the numpy IRs with torch ones

The numpy renders produced by step one were only ever placeholders: step two
overwrote every one of them so that targets and fitter share a synthesis code
path exactly. That agreement is what makes NMSE of 1e-14 reachable at all -- a
numpy-rendered target leaves a model-mismatch floor far above it. Doing it in
two passes means rendering everything twice and leaving a directory that is
correct only if the second script was actually run against it.

Parameter sampling reproduces ModalPlate/DatasetGen.py exactly: the same
np.random.seed, the same iteration over ParamRange.params, the same skipping of
fixed parameters so the random stream stays aligned. So

    python -m src.data.make_dataset --number 200 --duration 0.25 --seed 0

produces the same parameter sets as the existing random-IR-200-0.2s, and
--verify-against will check that against a directory rather than asking you to
take it on trust.

Output matches what the loaders expect (ternary_mu.load_target_ir_from_npz and
graddescent._read_params_csv):

    random_IR_XXXX.npz         ir (unnormalized), sample_rate, duration_s,
                               normalization_factor
    random_IR_params_XXXX.csv  all 14 plate parameters

The IR in the npz is unnormalized on purpose. Its amplitude carries mu = rho*h,
and peak-normalizing would discard the only thing that identifies it.

Usage:
    python -m src.data.make_dataset --number 20000 --duration 0.25 --seed 0 \
        --output-dir data/train-20000-0.25s
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from ModalPlate.ParamRange import params as PLATE_PARAM_RANGES
from src.plate.SevenParamPlate import BatchedModalPlateTorch as SevenParamPlate

SAMPLE_RATE = 44100
PARAM_ORDER = list(SevenParamPlate.PARAM_ORDER)


def sample_parameters(num: int, seed: int) -> List[Dict[str, float]]:
    """Draw parameter sets, reproducing DatasetGen.generate_random_parameters.

    Fixed parameters are assigned without touching the RNG, exactly as there, so
    the random stream stays aligned and a given seed yields identical sets.
    """
    np.random.seed(seed)
    out = []
    for _ in range(num):
        p = {}
        for name, rng in PLATE_PARAM_RANGES.items():
            if rng.low == rng.high:
                p[name] = rng.low
            else:
                p[name] = float(np.random.uniform(rng.low, rng.high))
        out.append(p)
    return out


def render(plate: SevenParamPlate, batch: List[Dict[str, float]], duration: float) -> np.ndarray:
    """Historical path: plate14 straight from the parameter dict.

    Kept to reproduce the existing datasets. Do not use for new ones -- see
    render_training_path for why.
    """
    rows = torch.tensor(
        [[float(p[k]) for k in PARAM_ORDER] for p in batch],
        dtype=torch.float32,
        device=plate.device,
    )
    with torch.no_grad():
        return plate(rows, duration=duration, normalize=False).cpu().numpy()


def render_training_path(space, batch: List[Dict[str, float]], duration: float) -> np.ndarray:
    """Render exactly as training synthesizes: through the float32 z the encoder emits.

    Training never sees physical parameters. It emits z in [-1,1], stores it
    float32, and maps back through norm_to_physical_torch. That quantises each
    parameter by its *range* rather than its value: T0 spans (0.01, 1000), so
    its float32-z quantum is ~6e-5 -- 0.6% of a T0 near the bottom of the range
    -- and rho lands at ~2.2e-4 relative. Through
    om^2 = (T0/mu)*g1 + (D/mu)*g2 that is a ~1e-4 relative frequency error: a
    fraction of a bin at 10.8 Hz.

    A linear loss cannot see it. Measured on the val set, the loss at true
    parameters as a fraction of what unrelated IRs score:

        L1_STFT   0.0002%      L1_STFT_c2   0.0076%
        pow(0.3)  2.71%        log(x+1e-7)  19.84%

    with the quietest magnitude decile carrying 27.4% of the log disagreement
    against 3.7% of the linear. A compression sweep run on targets rendered the
    other way would report our own target/synthesis disagreement, concentrated
    in exactly the bins log weights most, as if it were what compression does
    to the terrain.

    Rendering here through the same z closes it: both sides derive z from the
    same CSV values by the same float64 formula and cast to float32
    deterministically, so target and training synthesis agree bit-for-bit up to
    the batch-composition term (measured at 0.027% of saturation).
    """
    from src.cmaes.fit_7param_norm_es import PARAM_KEYS

    z = np.stack([space.gt_z({k: float(p[k]) for k in PARAM_KEYS}) for p in batch])
    z_t = torch.as_tensor(z, dtype=torch.float32, device=space.device)
    with torch.no_grad():
        return space.forward(z_t, None, duration).float().cpu().numpy()


def verify_against(generated: List[Dict[str, float]], ref_dir: Path) -> None:
    """Check the generated parameters match an existing dataset's CSVs."""
    csvs = sorted(ref_dir.glob("random_IR_params_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No random_IR_params_*.csv in {ref_dir}")
    n = min(len(csvs), len(generated))
    worst, worst_key = 0.0, ""
    for i in range(n):
        ref = pd.read_csv(csvs[i]).iloc[0].to_dict()
        for k, v in generated[i].items():
            if k not in ref:
                continue
            denom = max(abs(float(ref[k])), 1e-30)
            rel = abs(float(v) - float(ref[k])) / denom
            if rel > worst:
                worst, worst_key = rel, f"{k} (IR {i+1})"
    print(f"verify: compared {n} parameter sets against {ref_dir}")
    print(f"  worst relative difference {worst:.3e} on {worst_key or 'n/a'}")
    print("  " + ("MATCH" if worst < 1e-9 else "MISMATCH -- sampling has diverged"))


def run(args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = args.output_dir or Path(f"data/random-IR-torch-{args.number}-{args.duration:g}s")
    out_dir = Path(out_dir)

    if args.rerender_from is not None:
        # Re-render an existing dataset's parameters without re-sampling. The
        # seed that produced a directory is not always recoverable, and any
        # re-sample risks silently shifting the parameters a finished run was
        # trained against; reading the CSVs keeps them bit-identical and only
        # the audio changes.
        csvs = sorted(Path(args.rerender_from).glob("random_IR_params_*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No random_IR_params_*.csv in {args.rerender_from}")
        param_sets = [pd.read_csv(c).iloc[0].to_dict() for c in csvs]
        indices = [c.stem.split("_")[-1] for c in csvs]
        args.number = len(param_sets)
        print(f"Re-rendering {args.number} parameter sets from {args.rerender_from}")
    else:
        print(f"Sampling {args.number} parameter sets (seed {args.seed})")
        param_sets = sample_parameters(args.number, args.seed)
        indices = [f"{i + 1:04d}" for i in range(args.number)]

    if args.verify_against is not None:
        verify_against(param_sets, args.verify_against)
        if args.verify_only:
            return

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.render_path == "training":
        # Same object the fitter and the encoder synthesize through, configured
        # with the same plate flags, so generation and training are one call.
        from src.gd.graddescent import Raw7Space

        space = Raw7Space(device, torch.float32, normalize=False)
        space.configure_plate(
            args.chunk_elems, False, args.batched_plate, args.compile_plate, args.mode_bucket
        )
        renderer = lambda chunk: render_training_path(space, chunk, args.duration)
        print(f"Rendering on {device} into {out_dir} via the training path "
              f"(batched={args.batched_plate}, compile={args.compile_plate}, "
              f"chunk={args.chunk_elems}, bucket={args.mode_bucket})")
    else:
        plate = SevenParamPlate(
            sample_rate=args.sample_rate,
            device=device,
            dtype=torch.float32,
            drop_sub_20hz_modes=False,
            compile_modal_sum=args.compile_plate,
        )
        renderer = lambda chunk: render(plate, chunk, args.duration)
        print(f"Rendering on {device} into {out_dir} via the historical direct path")

    t0 = time.time()
    for start in range(0, args.number, args.batch_size):
        chunk = param_sets[start : start + args.batch_size]
        audio = renderer(chunk)
        for j, p in enumerate(chunk):
            idx = indices[start + j]
            ir = audio[j].astype(np.float32)
            peak = float(np.max(np.abs(ir)))
            # Unnormalized IR; normalization_factor is recorded, never applied.
            np.savez(
                out_dir / f"random_IR_{idx}.npz",
                ir=ir,
                sample_rate=np.int64(args.sample_rate),
                duration_s=np.float64(args.duration),
                normalization_factor=np.float64(peak if peak > 0 else 1.0),
            )
            pd.DataFrame([p]).to_csv(out_dir / f"random_IR_params_{idx}.csv", index=False)

        done = min(start + args.batch_size, args.number)
        if done % max(args.batch_size, args.number // 10 or 1) < args.batch_size:
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {done}/{args.number}  ({rate:.1f} IR/s, eta {(args.number-done)/max(rate,1e-9):.0f}s)")

    elapsed = time.time() - t0
    (out_dir / "generation_summary.txt").write_text(
        f"generator: src.data.make_dataset\n"
        f"number: {args.number}\nduration_s: {args.duration}\nsample_rate: {args.sample_rate}\n"
        f"seed: {args.seed}\nrenderer: SevenParamPlate (torch, float32, normalize=False)\n"
        f"elapsed_s: {elapsed:.1f}\n"
    )
    print(f"Done: {args.number} IRs in {elapsed:.0f}s -> {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sample plate parameters and render torch targets in one pass",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--number", type=int, default=200)
    p.add_argument("--duration", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--compile-plate", action="store_true")
    p.add_argument(
        "--render-path", choices=["training", "direct"], default="training",
        help="training: render through the float32 z the encoder emits, so targets and "
             "training synthesis agree bit-for-bit. direct: the historical path that "
             "builds plate14 from the CSV -- reproduces existing datasets, but leaves a "
             "loss floor at true parameters worth 19.8%% of saturation for log(x+1e-7).",
    )
    p.add_argument("--batched-plate", action="store_true", help="Match training's modal-sum path")
    p.add_argument("--chunk-elems", type=int, default=20_000_000, help="Match training's")
    p.add_argument("--mode-bucket", type=int, default=1024, help="Match training's")
    p.add_argument(
        "--verify-against", type=Path, default=None,
        help="Compare sampled parameters to an existing dataset's CSVs",
    )
    p.add_argument("--verify-only", action="store_true", help="Verify and exit without rendering")
    p.add_argument(
        "--rerender-from", type=Path, default=None,
        help="Re-render an existing dataset's parameters, read from its CSVs, instead "
             "of sampling. Keeps parameters bit-identical while changing only how the "
             "audio was synthesized -- which is what --render-path training fixes.",
    )
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
