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
from src.cmaes.fit_7param_norm_es import (
    FIXED_PLATE_PARAMS,
    PARAM_KEYS,
    PARAM_SPACE,
    PRODUCT_PLATE_PARAMS,
    norm_to_physical,
)
from src.plate.SevenParamPlate import BatchedModalPlateTorch as SevenParamPlate

SAMPLE_RATE = 44100
PARAM_ORDER = list(SevenParamPlate.PARAM_ORDER)


def parse_mode_grid(text: str):
    """Parse "DDX,DDY" for --fixed-mode-grid.

    Defined here rather than imported from train_encoder: a dataset generator
    should not pull in matplotlib and the encoder stack to parse two integers.
    """
    try:
        x, y = (int(v) for v in text.split(","))
    except Exception:
        raise argparse.ArgumentTypeError(f"expected DDX,DDY (e.g. 116,340), got {text!r}")
    if x < 1 or y < 1:
        raise argparse.ArgumentTypeError("both grid dimensions must be positive")
    return (x, y)


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


def sample_space_parameters(num: int, seed: int) -> List[Dict[str, float]]:
    """Draw parameter sets for a non-raw7 space, uniformly in its z.

    Not the DatasetGen stream: that iterates ModalPlate's fourteen ParamRanges,
    which is the wrong distribution for a space whose searched set is different
    and whose bounds are its own. Uniform in z is the distribution the encoder's
    output coordinate is uniform over, and for a log-scaled parameter that is
    log-uniform in physical units -- which is exactly how the sensitivity sweep
    that chose these ranges sampled T0.

    Every row carries all fourteen plate parameters (searched, derived and
    pinned) plus any non-plate coordinate such as T60_ratio, so the CSVs stay
    readable by everything that reads a dataset today and the pinned half is
    recorded rather than implied.
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, size=(num, len(PARAM_KEYS)))
    phys = norm_to_physical(z)
    out = []
    for row in phys:
        p = {k: float(v) for k, v in zip(PARAM_KEYS, row)}
        for name, (a, b) in PRODUCT_PLATE_PARAMS.items():
            p[name] = p[a] * p[b]
        for name, v in FIXED_PLATE_PARAMS.items():
            p.setdefault(name, float(v))
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
    z = np.stack([space.gt_z({k: float(p[k]) for k in PARAM_KEYS}) for p in batch])
    z_t = torch.as_tensor(z, dtype=torch.float32, device=space.device)
    with torch.no_grad():
        return space.forward(z_t, None, duration).float().cpu().numpy()


def mode_grid_of(param_sets: List[Dict[str, float]], sample_rate: int):
    """(DDx, DDy) each parameter set needs, in closed form -- no synthesis."""
    plate = SevenParamPlate(sample_rate=sample_rate, device="cpu", dtype=torch.float32,
                            drop_sub_20hz_modes=False)
    a = np.array([[p["E"], p["rho"], p["h"], p["Ly"], p["T0"], p["nu"]] for p in param_sets],
                 dtype=np.float64)
    E, rho, h, Ly, T0, nu = (a[:, i] for i in range(6))
    D = E * h ** 3 / (12.0 * (1.0 - nu ** 2))
    inner = np.sqrt(np.maximum(T0 ** 2 + 4.0 * (plate.max_omega ** 2) * rho * h * D, 0.0))
    disc = np.maximum((-T0 + inner) / (2.0 * D), 0.0)
    return (np.floor(1.0 / np.pi * np.sqrt(disc)), np.floor(Ly / np.pi * np.sqrt(disc)))


def report_mode_grid(param_sets: List[Dict[str, float]], sample_rate: int) -> None:
    """Largest modal grid any of these parameter sets needs.

    Closed form, no synthesis, so it runs over the whole set rather than a
    sample -- which matters, because --fixed-mode-grid must be at least this
    large. Truncating a *prediction* is acceptable and no different from the
    max_omega cut the plate already applies; truncating a *true* parameter set
    silently makes its target something other than the plate's output there,
    which is exactly the class of error the pin exists to remove.
    """
    ddx, ddy = mode_grid_of(param_sets, sample_rate)
    print(f"mode grid over {len(param_sets)} parameter sets:")
    print(f"  DDx  median {np.median(ddx):.0f}  p99 {np.percentile(ddx,99):.0f}  max {ddx.max():.0f}")
    print(f"  DDy  median {np.median(ddy):.0f}  p99 {np.percentile(ddy,99):.0f}  max {ddy.max():.0f}")
    # The cost of pinning is against what the batched plate already pays, which
    # is the *batch* maximum, not the median individual IR -- those differ by
    # nearly 5x here, so comparing to the per-IR median overstates the cost.
    rng = np.random.default_rng(0)
    B = 64
    batch_modes = [ddx[i].max() * ddy[i].max()
                   for i in (rng.integers(0, len(ddx), size=(200, B)))]
    typical = float(np.median(batch_modes))
    pin = float(ddx.max() * ddy.max())
    print(f"  modes  per-IR median {np.median(ddx*ddy):,.0f}   "
          f"batch-of-{B} max, median {typical:,.0f}   pin {pin:,.0f}")
    print(f"  pinning costs {pin/max(typical,1):.2f}x what the batched plate pays now")
    px, py = int(np.percentile(ddx, 99)), int(np.percentile(ddy, 99))
    drop = int(((ddx > px) | (ddy > py)).sum())
    print(f"\n  --fixed-mode-grid {int(ddx.max())},{int(ddy.max())}   (keeps every IR, "
          f"{ddx.max()*ddy.max()/max(typical,1):.2f}x)")
    print(f"  --fixed-mode-grid {px},{py}   (drops {drop} of {len(ddx)}, "
          f"{100*drop/len(ddx):.2f}%, at {px*py/max(typical,1):.2f}x)")
    print("\n  The max is priced by a handful of outliers and is paid on every step of")
    print("  every run; the p99 grid usually costs about what an unpinned batch already")
    print("  did. Generating with --fixed-mode-grid drops the sets that exceed it, so a")
    print("  dataset never contains an IR its own pin would truncate.")


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


def check_pin(ref_dir: Path) -> None:
    """Check the pinned half against the IR the sensitivity sweep held fixed.

    diag_param_sensitivity --vary builds its family by varying the named
    parameters around the FIRST parameter set in a directory and pinning
    everything else, so the ranges it reports describe one particular plate.
    quiet3's pin is that plate, copied into PARAM_SPACES by hand. If the
    directory the sweep read is not the one those numbers came from -- or if
    that IR was among the few the p99 grid excluded, which shifts sorted()[0]
    to a different plate -- then the family being generated is a legitimate
    task but not the one that was measured. Cheap to check, so check it.
    """
    csvs = sorted(ref_dir.glob("random_IR_params_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No random_IR_params_*.csv in {ref_dir}")
    ref = pd.read_csv(csvs[0]).iloc[0].to_dict()
    print(f"pin check: {PARAM_SPACE} pinned parameters against {csvs[0].name}")
    worst = 0.0
    for k, v in sorted(FIXED_PLATE_PARAMS.items()):
        if k not in ref:
            print(f"  {k:<10} {v:<24.10g} not in the reference CSV")
            continue
        rel = abs(v - float(ref[k])) / max(abs(float(ref[k])), 1e-30)
        worst = max(worst, rel)
        print(f"  {k:<10} {v:<24.10g} ref {float(ref[k]):<24.10g} rel {rel:.2e}")
    print("  " + ("MATCH -- this is the plate the sweep measured" if worst < 1e-9
                  else "MISMATCH -- the ranges in PARAM_SPACES were measured at a "
                       "different plate"))


def run(args) -> None:
    if args.check_pin is not None:
        check_pin(args.check_pin)
        return

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = args.output_dir or Path(f"data/random-IR-torch-{args.number}-{args.duration:g}s")
    out_dir = Path(out_dir)

    if args.params_csv is not None:
        # One combined CSV instead of per-IR files. val-1000-0.25s could not be
        # reproduced from any seed we tried -- it predates the current sampler --
        # so its parameters are the irreplaceable part and travel as a file,
        # while its 51 MB of audio does not. Rendering is deterministic given
        # the parameters and the flags, and must be redone per machine anyway
        # since a different GPU reduces float32 in a different order.
        df = pd.read_csv(args.params_csv)
        if "idx" not in df.columns:
            raise ValueError(f"{args.params_csv} has no 'idx' column")
        df = df.sort_values("idx", key=lambda c: c.astype(int))
        indices = [f"{int(v):04d}" for v in df["idx"]]
        param_sets = df.drop(columns=["idx"]).to_dict("records")
        args.number = len(param_sets)
        print(f"Rendering {args.number} parameter sets from {args.params_csv}")
    elif args.rerender_from is not None:
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
    elif PARAM_SPACE != "raw7":
        print(f"Sampling {args.number} parameter sets in space {PARAM_SPACE} "
              f"({', '.join(PARAM_KEYS)}, seed {args.seed})")
        param_sets = sample_space_parameters(args.number, args.seed)
        indices = [f"{i + 1:04d}" for i in range(args.number)]
    else:
        print(f"Sampling {args.number} parameter sets (seed {args.seed})")
        param_sets = sample_parameters(args.number, args.seed)
        indices = [f"{i + 1:04d}" for i in range(args.number)]

    if args.report_grid:
        report_mode_grid(param_sets, args.sample_rate)
        return

    if args.verify_against is not None:
        verify_against(param_sets, args.verify_against)
        if args.verify_only:
            return

    if args.fixed_mode_grid is not None:
        # A target rendered under a pin that truncates it is not the plate's
        # output at those parameters, so such sets are excluded rather than
        # silently mis-rendered. The pin therefore defines the dataset, and the
        # plate's truncation warning can never fire on data made this way.
        ddx, ddy = mode_grid_of(param_sets, args.sample_rate)
        keep = (ddx <= args.fixed_mode_grid[0]) & (ddy <= args.fixed_mode_grid[1])
        dropped = int((~keep).sum())
        if dropped:
            print(f"  pin {args.fixed_mode_grid[0]},{args.fixed_mode_grid[1]} excludes "
                  f"{dropped} of {len(keep)} parameter sets ({100*dropped/len(keep):.2f}%) "
                  f"that need a finer grid")
        param_sets = [p for p, k in zip(param_sets, keep) if k]
        indices = [i for i, k in zip(indices, keep) if k]
        args.number = len(param_sets)

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.render_path == "training":
        # Same object the fitter and the encoder synthesize through, configured
        # with the same plate flags, so generation and training are one call.
        from src.gd.graddescent import Raw7Space

        space = Raw7Space(device, torch.float32, normalize=False)
        space.configure_plate(
            args.chunk_elems, False, args.batched_plate, args.compile_plate,
            args.mode_bucket, args.fixed_mode_grid,
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
        f"seed: {args.seed}\nparam_space: {PARAM_SPACE} ({', '.join(PARAM_KEYS)})\n"
        f"renderer: SevenParamPlate (torch, float32, normalize=False)\n"
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
        "--fixed-mode-grid", type=parse_mode_grid, default=None, metavar="DDX,DDY",
        help="Pin the modal grid; must match the training run's exactly, or targets "
             "and training synthesis disagree again.",
    )
    p.add_argument(
        "--verify-against", type=Path, default=None,
        help="Compare sampled parameters to an existing dataset's CSVs",
    )
    p.add_argument("--verify-only", action="store_true", help="Verify and exit without rendering")
    p.add_argument(
        "--check-pin", type=Path, default=None, metavar="DIR",
        help="Compare this space's PINNED parameters against the first parameter CSV "
             "in DIR, and exit. For a space whose ranges came from a "
             "diag_param_sensitivity --vary run, DIR is the directory that run read: "
             "the sweep pins everything it does not vary at that IR, so this says "
             "whether the family being generated is the one that was measured.",
    )
    p.add_argument(
        "--report-grid", action="store_true",
        help="Print the largest modal grid these parameters need, and exit. The value "
             "to pass to --fixed-mode-grid.",
    )
    p.add_argument(
        "--params-csv", type=Path, default=None,
        help="Render from a single combined parameter CSV with an 'idx' column, as written "
             "by docs/DATASETS.md. Lets a dataset move between machines as a file of "
             "parameters rather than gigabytes of regenerable audio.",
    )
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
