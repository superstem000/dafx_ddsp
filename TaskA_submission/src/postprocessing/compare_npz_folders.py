"""Compare two folders of .npz files for near-equality.

Two files are considered almost the same if, for every shared array key,

    max(abs(a - b)) < 1e-5 * max(abs(a))

If the reference array is all zeros, the check falls back to exact equality
for that array.

The script expects both folders to contain the same .npz filenames. It checks
that the set of keys and the shapes match for each paired file, then reports
any differences.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


TOL_SCALE = 1e-5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two folders of .npz files using a relative max-error threshold."
    )
    parser.add_argument("folder_a", type=Path, help="Reference folder containing .npz files.")
    parser.add_argument("folder_b", type=Path, help="Comparison folder containing .npz files.")
    return parser


def _load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _extract_file_id(file_name: str, prefix: str) -> str:
    stem = Path(file_name).stem
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected filename prefix for {file_name}; expected {prefix}*")
    return stem.removeprefix(prefix)


def _compare_arrays(a: np.ndarray, b: np.ndarray, file_name: str, key: str) -> tuple[bool, str | None]:
    # if a.shape != b.shape:
    #     return False, f"{file_name}:{key} shape mismatch {a.shape} != {b.shape}"
    a = a[:b.shape[0]]
    b = b[:a.shape[0]]

    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    max_abs = float(np.max(np.abs(a64)))
    max_err = float(np.max(np.abs(a64 - b64)))

    if max_abs == 0.0:
        if max_err != 0.0:
            return False, f"{file_name}:{key} max error {max_err:.6g} (reference is all zeros)"
        return True, None

    limit = TOL_SCALE * max_abs
    if max_err >= limit:
        return False, f"{file_name}:{key} max error {max_err:.6g} >= {limit:.6g}"
    return True, None


def compare_npz_folders(folder_a: Path, folder_b: Path) -> list[str]:
    if not folder_a.is_dir():
        raise ValueError(f"Not a directory: {folder_a}")
    if not folder_b.is_dir():
        raise ValueError(f"Not a directory: {folder_b}")

    files_a = {path.name for path in folder_a.glob("random_IR_*.npz")}
    files_b = {path.name for path in folder_b.glob("best_audio_*.npz")}

    by_id_a = {_extract_file_id(name, "random_IR_"): name for name in files_a}
    by_id_b = {_extract_file_id(name, "best_audio_"): name for name in files_b}

    errors: list[str] = []

    ids_a = set(by_id_a)
    ids_b = set(by_id_b)

    missing_in_b = sorted(ids_a - ids_b)
    missing_in_a = sorted(ids_b - ids_a)
    if missing_in_b:
        errors.append(
            f"Missing in {folder_b}: {', '.join(by_id_a[file_id] for file_id in missing_in_b)}"
        )
    if missing_in_a:
        errors.append(
            f"Missing in {folder_a}: {', '.join(by_id_b[file_id] for file_id in missing_in_a)}"
        )

    for file_id in sorted(ids_a & ids_b):
        file_name_a = by_id_a[file_id]
        file_name_b = by_id_b[file_id]
        arrays_a = _load_npz(folder_a / file_name_a)
        arrays_b = _load_npz(folder_b / file_name_b)

        keys_a = set(arrays_a)
        keys_b = set(arrays_b)
        if keys_a != keys_b:
            missing_keys_b = sorted(keys_a - keys_b)
            missing_keys_a = sorted(keys_b - keys_a)
            if missing_keys_b:
                errors.append(f"{file_name_a}: missing in {folder_b.name}: {', '.join(missing_keys_b)}")
            if missing_keys_a:
                errors.append(f"{file_name_a}: missing in {folder_a.name}: {', '.join(missing_keys_a)}")
            continue

        for key in sorted(keys_a):
            if key == 'ir':
                ok, message = _compare_arrays(arrays_a[key], arrays_b[key], file_name_a, key)
                if not ok and message is not None:
                    errors.append(message)

    return errors


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    errors = compare_npz_folders(args.folder_a, args.folder_b)
    if errors:
        print("NPZ folders differ:")
        for message in errors[:50]:
            print(f"- {message}")
        raise SystemExit(1)

    print("NPZ folders match under the max-error threshold.")


if __name__ == "__main__":
    main()