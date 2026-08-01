import argparse
import csv
import math
from pathlib import Path

RUNTIME_CANDIDATES = ("runtime", "runtime_s", "elapsed_s")


def _format_hms(total_seconds: float) -> str:
    secs = int(round(total_seconds))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sum runtime from a summary.csv and print total time spent."
    )
    parser.add_argument("summary_csv", type=Path, help="Path to summary.csv")
    args = parser.parse_args()

    csv_path = args.summary_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        runtime_col = next((c for c in RUNTIME_CANDIDATES if c in reader.fieldnames), None)
        if runtime_col is None:
            cols = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Could not find runtime column. Expected one of {RUNTIME_CANDIDATES}. Found: {cols}"
            )

        values = []
        for row in reader:
            val = (row.get(runtime_col) or "").strip()
            if not val:
                continue
            try:
                values.append(float(val))
            except ValueError:
                continue

    n = len(values)
    total = sum(values)
    mean = (total / n) if n > 0 else float("nan")
    if n >= 2:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = float("nan")

    print(f"rows_counted: {n}")
    print(f"runtime_column: {runtime_col}")
    print(f"total_seconds: {total:.6f}")
    print(f"total_minutes: {total / 60.0:.6f}")
    print(f"total_hours: {total / 3600.0:.6f}")
    print(f"total_hms: {_format_hms(total)}")
    print(f"mean_seconds: {mean:.6f}")
    print(f"std_seconds: {std:.6f}")


if __name__ == "__main__":
    main()
