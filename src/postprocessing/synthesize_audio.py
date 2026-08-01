"""Synthesize unnormalized experiment audio from best-params CSV files.

The script scans a folder for `best_params_{file_id}.csv`, synthesizes a
5-second impulse response for each file using the GPU-backed six-parameter
plate model, and writes the result back into the same folder as
`best_audio_{file_id}.npz`.

Each `.npz` file contains:
    - ir: the unnormalized synthesized impulse response
    - sample_rate: 44100
    - duration_s: 5.0
    - normalization_factor: the max absolute amplitude of the IR

The six-column submission CSVs are interpreted directly as
    mu, D_div_mu, T0_div_mu, Ly, op_x, op_y
and are synthesized on CUDA using `src.plate.SixParamPlate`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch


SAMPLE_RATE = 44100
DURATION_S = 5.0
DEVICE = "cuda"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.append(str(_workspace_root()))

from src.plate.SixParamPlate import BatchedModalPlateTorch  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthesize unnormalized audio from best_params CSV files in a folder."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing best_params_{file_id}.csv files and the output npz files.",
    )
    return parser


def _read_single_row(csv_path: Path) -> dict[str, float]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader, None)
        if row is None:
            raise ValueError(f"{csv_path.name} is empty")
        extra = next(reader, None)
        if extra is not None:
            raise ValueError(f"{csv_path.name} must contain exactly one row")

    key_aliases = {
        "mu": ["mu"],
        "D_div_mu": ["D_div_mu", "D_mu"],
        "T0_div_mu": ["T0_div_mu", "T0_mu"],
        "Ly": ["Ly"],
        "op_x": ["op_x"],
        "op_y": ["op_y"],
    }

    missing = [name for name, aliases in key_aliases.items() if not any(alias in row for alias in aliases)]
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {', '.join(missing)}")

    resolved = {}
    for canonical, aliases in key_aliases.items():
        for alias in aliases:
            if alias in row:
                resolved[canonical] = float(row[alias])
                break
    return resolved


def _synthesize_from_csv(csv_path: Path) -> Path:
    row = _read_single_row(csv_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for synthesis but torch.cuda.is_available() is False")

    model = BatchedModalPlateTorch(
        sample_rate=SAMPLE_RATE,
        device=DEVICE,
        drop_sub_20hz_modes=False,
        dtype=torch.float64,
    )
    param_tensor = BatchedModalPlateTorch.params_dicts_to_tensor(
        [row],
        device=DEVICE,
        dtype=torch.float64,
    )
    with torch.no_grad():
        ir_tensor = model(param_tensor, duration=DURATION_S, vel_calc=False, normalize=False)
    ir = ir_tensor[0].detach().cpu().numpy().astype(np.float64, copy=False)
    normalization_factor = float(np.abs(ir).max())

    file_id = csv_path.stem.split("_")[-1]
    output_path = csv_path.with_name(f"best_audio_{file_id}.npz")
    np.savez(
        output_path,
        ir=ir,
        sample_rate=np.int32(SAMPLE_RATE),
        duration_s=np.float64(DURATION_S),
        normalization_factor=np.float64(normalization_factor),
    )
    return output_path


def synthesize_folder(folder: Path) -> list[Path]:
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    outputs: list[Path] = []
    for csv_path in sorted(folder.glob("best_params_*.csv")):
        outputs.append(_synthesize_from_csv(csv_path))
    return outputs


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    outputs = synthesize_folder(args.folder)
    print(f"Wrote {len(outputs)} npz files to {args.folder}")


if __name__ == "__main__":
    main()