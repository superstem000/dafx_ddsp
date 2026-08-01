"""Join phase summaries into per-IR submission CSVs.

The first summary provides the per-IR parameter values for
`D_mu`, `T0_mu`, `Ly`, `op_x`, and `op_y`.
The second summary provides the refined `mu` value.

For each matched `filename`, this script writes a one-row CSV named
`<filename stem>.csv` into the requested output directory with exactly
these columns in order:

    mu, D_mu, T0_mu, Ly, op_x, op_y
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUTPUT_COLUMNS = ["mu", "D_mu", "T0_mu", "Ly", "op_x", "op_y"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join two summaries and write one submission-style CSV per IR."
    )
    parser.add_argument(
        "summary_csv",
        type=Path,
        help="Path to the phase 1 summary CSV containing D_mu, T0_mu, Ly, op_x, and op_y.",
    )
    parser.add_argument(
        "mu_refined_csv",
        type=Path,
        help="Path to the phase 2 summary CSV containing refined_mu.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where per-IR CSV files will be written.",
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


def join_and_write(summary_csv: Path, mu_refined_csv: Path, output_dir: Path) -> None:
    summary_rows, summary_order = _load_rows(
        summary_csv, {"filename", *OUTPUT_COLUMNS[1:]}
    )
    refined_rows, refined_order = _load_rows(mu_refined_csv, {"filename", "refined_mu"})

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

    for filename in summary_order:
        summary_row = summary_by_name[filename]
        refined_row = refined_by_name[filename]
        ir_suffix = Path(filename).stem.split("_")[-1]
        output_name = f"best_params_{ir_suffix}.csv"
        output_path = output_dir / output_name
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerow(
                {
                    "mu": refined_row["refined_mu"],
                    "D_mu": summary_row["D_mu"],
                    "T0_mu": summary_row["T0_mu"],
                    "Ly": summary_row["Ly"],
                    "op_x": summary_row["op_x"],
                    "op_y": summary_row["op_y"],
                }
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    join_and_write(args.summary_csv, args.mu_refined_csv, args.output_dir)


if __name__ == "__main__":
    main()