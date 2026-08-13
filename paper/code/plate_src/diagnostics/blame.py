#!/usr/bin/env python3
"""Analyze per-variable responsibility for 6D NMSE from a summary CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

COMPOSITE_KEYS = ["mu", "D_mu", "T0_mu", "Ly", "op_x", "op_y"]
PARAM_KEYS = ["E", "rho", "h", "Ly", "T0", "op_x", "op_y"]
NU = 0.25

COMPOSITE_BOUNDS = {
    "mu": (2.43, 106.15),
    "D_mu": (0.28052546, 201.188843),
    "T0_mu": (0.000094206, 411.522634),
    "Ly": (1.1, 4.0),
    "op_x": (0.51, 1.0),
    "op_y": (0.51, 1.0),
}


def _f(v):
    try:
        if v is None or v == "":
            return None
        out = float(v)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def to_composite(raw):
    rho = raw["rho"]
    h = raw["h"]
    E = raw["E"]
    T0 = raw["T0"]
    mu = rho * h
    D = (E * (h**3)) / (12.0 * (1.0 - NU**2))
    return {
        "mu": mu,
        "D_mu": D / mu,
        "T0_mu": T0 / mu,
        "Ly": raw["Ly"],
        "op_x": raw["op_x"],
        "op_y": raw["op_y"],
    }


def extract_pair(row):
    # Preferred for mu_refined_summary.csv: post-refinement composite vs GT composite.
    refined_comp_keys = {
        "mu": "refined_mu",
        "D_mu": "refined_D_div_mu",
        "T0_mu": "refined_T0_div_mu",
        "Ly": "refined_Ly",
        "op_x": "refined_op_x",
        "op_y": "refined_op_y",
    }
    gt_comp_keys = {
        "mu": "gt_mu",
        "D_mu": "gt_D_div_mu",
        "T0_mu": "gt_T0_div_mu",
        "Ly": "gt_Ly",
        "op_x": "gt_op_x",
        "op_y": "gt_op_y",
    }
    if all(_f(row.get(refined_comp_keys[k])) is not None for k in COMPOSITE_KEYS) and all(
        _f(row.get(gt_comp_keys[k])) is not None for k in COMPOSITE_KEYS
    ):
        est_c = {k: _f(row[refined_comp_keys[k]]) for k in COMPOSITE_KEYS}
        gt_c = {k: _f(row[gt_comp_keys[k]]) for k in COMPOSITE_KEYS}
        return est_c, gt_c, "refined_composite_vs_gt_composite"

    # Next preference: pre-refinement composite estimate vs GT composite.
    est_comp_keys = {
        "mu": "est_mu",
        "D_mu": "est_D_div_mu",
        "T0_mu": "est_T0_div_mu",
        "Ly": "est_Ly",
        "op_x": "est_op_x",
        "op_y": "est_op_y",
    }
    if all(_f(row.get(est_comp_keys[k])) is not None for k in COMPOSITE_KEYS) and all(
        _f(row.get(gt_comp_keys[k])) is not None for k in COMPOSITE_KEYS
    ):
        est_c = {k: _f(row[est_comp_keys[k]]) for k in COMPOSITE_KEYS}
        gt_c = {k: _f(row[gt_comp_keys[k]]) for k in COMPOSITE_KEYS}
        return est_c, gt_c, "est_composite_vs_gt_composite"

    # Next preference: est_* raw7 vs gt_* raw7.
    est_raw_keys = {k: f"est_{k}" for k in PARAM_KEYS}
    gt_raw_keys = {k: f"gt_{k}" for k in PARAM_KEYS}
    if all(_f(row.get(est_raw_keys[k])) is not None for k in PARAM_KEYS) and all(
        _f(row.get(gt_raw_keys[k])) is not None for k in PARAM_KEYS
    ):
        est_raw = {k: _f(row[est_raw_keys[k]]) for k in PARAM_KEYS}
        gt_raw = {k: _f(row[gt_raw_keys[k]]) for k in PARAM_KEYS}
        return to_composite(est_raw), to_composite(gt_raw), "est_raw7_vs_gt_raw7"

    if all(_f(row.get(k)) is not None for k in PARAM_KEYS) and all(_f(row.get(f"gt_{k}")) is not None for k in PARAM_KEYS):
        est_raw = {k: _f(row[k]) for k in PARAM_KEYS}
        gt_raw = {k: _f(row[f"gt_{k}"]) for k in PARAM_KEYS}
        return to_composite(est_raw), to_composite(gt_raw), "raw7_vs_gt_raw7"

    direct_pairs = {}
    for k in COMPOSITE_KEYS:
        est = _f(row.get(k))
        gt = _f(row.get(f"gt_{k}"))
        if est is None or gt is None:
            return None, None, ""
        direct_pairs[k] = (est, gt)
    est_c = {k: direct_pairs[k][0] for k in COMPOSITE_KEYS}
    gt_c = {k: direct_pairs[k][1] for k in COMPOSITE_KEYS}
    return est_c, gt_c, "composite_vs_gt_composite"


def extract_mu_refined_only(row, use_mu_orig: bool = False):
    """Fallback for mu_refined_summary.csv-style rows.

    These rows typically contain:
    - mu_refined, mu_gt (or mu_orig, mu_gt)
    but not the full 7D or 6D estimate/GT columns.
    """
    mu_est = _f(row.get("mu_orig")) if use_mu_orig else _f(row.get("mu_refined"))
    if mu_est is None:
        return None, ""
    mu_gt = _f(row.get("mu_gt"))
    if mu_est is None or mu_gt is None:
        return None

    est_c = {k: 0.0 for k in COMPOSITE_KEYS}
    gt_c = {k: 0.0 for k in COMPOSITE_KEYS}
    est_c["mu"] = mu_est
    gt_c["mu"] = mu_gt

    # Keep denominator scales valid for non-mu axes by assigning bound minima;
    # their contribution is forced to 0 since est==gt on those axes.
    for k in COMPOSITE_KEYS:
        if k == "mu":
            continue
        lo, _ = COMPOSITE_BOUNDS[k]
        est_c[k] = lo
        gt_c[k] = lo
    return est_c, gt_c, ("mu_orig_vs_mu_gt" if use_mu_orig else "mu_refined_vs_mu_gt")


def component_errors(est_c, gt_c):
    errs = {}
    for k in COMPOSITE_KEYS:
        lo, hi = COMPOSITE_BOUNDS[k]
        e = (est_c[k] - lo) / (hi - lo)
        g = (gt_c[k] - lo) / (hi - lo)
        errs[k] = (e - g) ** 2
    return errs


def main():
    parser = argparse.ArgumentParser(description="Analyze contribution percentages of 6 variables to NMSE.")
    parser.add_argument("summary_csv", type=Path, help="Input summary.csv")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <summary_dir>/blame",
    )
    parser.add_argument(
        "--use_mu_orig",
        action="store_true",
        help="For mu_refined_summary-style files, use mu_orig instead of mu_refined.",
    )
    args = parser.parse_args()

    summary = args.summary_csv.expanduser().resolve()
    if not summary.exists():
        raise FileNotFoundError(f"summary csv not found: {summary}")

    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else (summary.parent / "blame")
    out_dir.mkdir(parents=True, exist_ok=True)

    with summary.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("summary csv has no rows")

    per_ir_rows = []
    sum_components = {k: 0.0 for k in COMPOSITE_KEYS}
    used = 0

    for i, row in enumerate(rows):
        est_c, gt_c, source = extract_pair(row)
        mode = "full_or_composite"
        if est_c is None:
            fallback = extract_mu_refined_only(row, use_mu_orig=args.use_mu_orig)
            if fallback is not None:
                est_c, gt_c, source = fallback
                mode = "mu_refined_only"
        if est_c is None:
            continue
        errs = component_errors(est_c, gt_c)
        nmse = sum(errs.values()) / len(COMPOSITE_KEYS)
        total_err = sum(errs.values())
        if total_err <= 0:
            pct = {k: 0.0 for k in COMPOSITE_KEYS}
        else:
            pct = {k: (errs[k] / total_err) * 100.0 for k in COMPOSITE_KEYS}

        for k in COMPOSITE_KEYS:
            sum_components[k] += errs[k]

        used += 1
        per_ir_rows.append(
            {
                "index": i,
                "filename": row.get("filename", f"row_{i:04d}"),
                "mode": mode,
                "source": source,
                "mu_source": ("mu_orig" if args.use_mu_orig else "mu_refined") if mode == "mu_refined_only" else "",
                "nmse_orig_csv": _f(row.get("nmse_orig")),
                "nmse_refined_csv": _f(row.get("nmse_refined")),
                "nmse_recomputed": nmse,
                **{f"sqerr_{k}": errs[k] for k in COMPOSITE_KEYS},
                **{f"pct_{k}": pct[k] for k in COMPOSITE_KEYS},
            }
        )

    if used == 0:
        raise ValueError("No usable rows with both estimate and ground-truth values were found.")

    total_err_all = sum(sum_components.values())
    if total_err_all <= 0:
        agg_pct = {k: 0.0 for k in COMPOSITE_KEYS}
    else:
        agg_pct = {k: (sum_components[k] / total_err_all) * 100.0 for k in COMPOSITE_KEYS}

    agg_mean_pct = {}
    for k in COMPOSITE_KEYS:
        vals = [r[f"pct_{k}"] for r in per_ir_rows]
        agg_mean_pct[k] = sum(vals) / len(vals)

    per_ir_csv = out_dir / "blame_per_ir.csv"
    agg_csv = out_dir / "blame_summary.csv"

    per_ir_fields = [
        "index",
        "filename",
        "mode",
        "source",
        "mu_source",
        "nmse_orig_csv",
        "nmse_refined_csv",
        "nmse_recomputed",
    ] + [f"sqerr_{k}" for k in COMPOSITE_KEYS] + [f"pct_{k}" for k in COMPOSITE_KEYS]
    with per_ir_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=per_ir_fields)
        w.writeheader()
        for r in per_ir_rows:
            w.writerow(r)

    agg_rows = []
    for k in COMPOSITE_KEYS:
        agg_rows.append(
            {
                "variable": k,
                "sum_sqerr": sum_components[k],
                "pct_of_total_sqerr": agg_pct[k],
                "mean_pct_per_ir": agg_mean_pct[k],
            }
        )
    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variable", "sum_sqerr", "pct_of_total_sqerr", "mean_pct_per_ir"])
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)

    print(f"input_csv: {summary}")
    print(f"rows_total: {len(rows)}")
    print(f"rows_used: {used}")
    print("aggregate_percent_responsibility:")
    for k in COMPOSITE_KEYS:
        print(f"  {k:6s}: {agg_pct[k]:9.4f}%   (mean per-IR: {agg_mean_pct[k]:9.4f}%)")
    print(f"wrote_per_ir: {per_ir_csv}")
    print(f"wrote_summary: {agg_csv}")


if __name__ == "__main__":
    main()
