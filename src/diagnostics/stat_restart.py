#!/usr/bin/env python3
"""Compute restart statistics from a CMA-ES/optimizer summary CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _pick_restart_column(fieldnames: list[str]) -> str:
    candidates = [
        "n_restarts_run",
        "n_restarts",
        "restarts",
        "num_restarts",
        "restart_count",
    ]
    for col in candidates:
        if col in fieldnames:
            return col

    lowered = {c.lower(): c for c in fieldnames}
    for key, original in lowered.items():
        if "restart" in key:
            return original

    raise ValueError(
        "Could not find a restart-count column. "
        f"Available columns: {fieldnames}"
    )


def _quantile_linear(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    n = len(sorted_values)
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count total restarts and report mean/variance from summary.csv"
    )
    parser.add_argument("summary_csv", type=Path, help="Path to summary.csv")
    parser.add_argument(
        "--ddof",
        type=int,
        default=0,
        choices=[0, 1],
        help="Variance ddof: 0=population variance (default), 1=sample variance",
    )
    args = parser.parse_args()

    if not args.summary_csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.summary_csv}")

    with args.summary_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        col = _pick_restart_column(fieldnames)
        values = []
        for row in reader:
            raw = row.get(col, "")
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue

    if not values:
        raise ValueError(f"Column '{col}' has no numeric values.")

    n = len(values)
    total = float(sum(values))
    mean = total / n
    if args.ddof == 1 and n < 2:
        variance = float("nan")
    else:
        denom = (n - args.ddof)
        variance = sum((v - mean) ** 2 for v in values) / denom
    sorted_values = sorted(values)
    q1 = _quantile_linear(sorted_values, 0.25)
    q2 = _quantile_linear(sorted_values, 0.50)
    q3 = _quantile_linear(sorted_values, 0.75)

    print(f"summary_csv: {args.summary_csv}")
    print(f"restart_column: {col}")
    print(f"n_rows_used: {n}")
    print(f"total_restarts: {int(total) if total.is_integer() else total}")
    print(f"mean_restarts: {mean:.6f}")
    if math.isnan(variance):
        print("variance_restarts: nan")
    else:
        print(f"variance_restarts: {variance:.6f}")
    print(f"q1_restarts: {q1:.6f}")
    print(f"median_restarts: {q2:.6f}")
    print(f"q3_restarts: {q3:.6f}")


if __name__ == "__main__":
    main()
