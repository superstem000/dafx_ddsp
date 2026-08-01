"""Congregate phase summaries into a unified experiment summary.

This script joins the phase 1 summary with the phase 2 refinement summary and
writes a single CSV named `experiment_summary.csv` into the requested output
directory.

The output columns match the reference experiment summary format:

    target_file, target_index, duration, optimization_time, best_loss, iterations

Rules:
* `target_file` and `target_index` come from the phase 1 filename.
* `duration` is fixed at `0.25` for every row.
* `optimization_time` is `runtime + elapsed_s`.
* `best_loss` is taken from `loss_refined` in the phase 2 file.
* `iterations` is `total_evaluations` from phase 1 plus `total_evaluations` from phase 2.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUTPUT_FILENAME = "experiment_summary.csv"
OUTPUT_COLUMNS = ["target_file", "target_index", "duration", "optimization_time", "best_loss", "iterations"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join phase summaries into a single experiment summary CSV."
    )
    parser.add_argument(
        "summary_csv",
        type=Path,
        help="Path to the phase 1 summary CSV.",
    )
    parser.add_argument(
        "mu_refined_csv",
        type=Path,
        help="Path to the phase 2 refined summary CSV.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where experiment_summary.csv will be written.",
    )
    return parser


def _require_columns(fieldnames: list[str] | None, required: set[str], source_name: str) -> None:
    missing = sorted(required - set(fieldnames or []))
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {', '.join(missing)}")


def _load_rows(csv_path: Path, required_columns: set[str]) -> tuple[list[dict[str, str]], list[str]]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, required_columns, csv_path.name)

        rows: list[dict[str, str]] = []
        order: list[str] = []
        seen: set[str] = set()
        for row in reader:
            filename = row["filename"]
            if filename in seen:
                raise ValueError(f"Duplicate filename in {csv_path.name}: {filename}")
            seen.add(filename)
            rows.append(row)
            order.append(filename)

    return rows, order


def _extract_target_index(filename: str) -> str:
    stem = Path(filename).stem
    suffix = stem.split("_")[-1]
    if not suffix.isdigit():
        raise ValueError(f"Could not derive numeric target index from filename: {filename}")
    return suffix


def congregate_summary(summary_csv: Path, mu_refined_csv: Path, output_dir: Path) -> Path:
    summary_rows, summary_order = _load_rows(
        summary_csv, {"filename", "runtime", "total_evaluations"}
    )
    refined_rows, refined_order = _load_rows(
        mu_refined_csv, {"filename", "loss_refined", "elapsed_s", "total_evaluations"}
    )

    summary_by_name = {row["filename"]: row for row in summary_rows}
    refined_by_name = {row["filename"]: row for row in refined_rows}

    summary_names = set(summary_by_name)
    refined_names = set(refined_by_name)
    if summary_names != refined_names:
        missing_in_refined = sorted(summary_names - refined_names)
        missing_in_summary = sorted(refined_names - summary_names)
        problems: list[str] = []
        if missing_in_refined:
            problems.append(
                f"missing in {mu_refined_csv.name}: {', '.join(missing_in_refined[:5])}"
            )
        if missing_in_summary:
            problems.append(f"missing in {summary_csv.name}: {', '.join(missing_in_summary[:5])}")
        raise ValueError("Filename mismatch between summaries (" + "; ".join(problems) + ")")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for filename in summary_order:
            summary_row = summary_by_name[filename]
            refined_row = refined_by_name[filename]
            writer.writerow(
                {
                    "target_file": filename,
                    "target_index": _extract_target_index(filename),
                    "duration": "0.25",
                    "optimization_time": float(summary_row["runtime"]) + float(refined_row["elapsed_s"]),
                    "best_loss": refined_row["loss_refined"],
                    "iterations": int(summary_row["total_evaluations"]) + int(refined_row["total_evaluations"]),
                }
            )

    return output_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    congregate_summary(args.summary_csv, args.mu_refined_csv, args.output_dir)


if __name__ == "__main__":
    main()