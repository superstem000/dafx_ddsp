"""Constant-predictor error per parameter group, for normalising Param.

    python scripts/ds_param_baseline.py

Writes results/diffsynth/param_baseline.json, which monitor_diffsynth.py reads
to report each group as a fraction of the error you would get by always
guessing the dataset mean. 1.0 there means an arm learned nothing about that
parameter; 0.0 means it recovered it exactly.

The six raw L1s are not comparable to each other -- f0_hz lands near 0.004 and
osc_mix near 0.13 -- so param_loss's unweighted mean is about half osc_mix and
q by magnitude alone, and f0_hz contributes under 1% of it whatever any arm
does. Normalising also reverses at least one reading: amplitudes has the
smallest baseline of the six, so its small raw L1 was hiding that it is the
LEAST well recovered group, not the best.

This is a property of the dataset, not of any run or split, so it is computed
over the id dataset as a whole rather than over one arm's validation split. The
validation split is a random 10% of the same files, and the difference is far
below the precision anyone would quote.

The mean is taken per feature channel over clips and frames, so a time-varying
parameter is scored against one constant rather than against its own mean
envelope -- the conservative choice, and the one that matches how the plate
work quotes a constant-predictor floor.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "external", "diffsynth")
sys.path.insert(0, DS)

import torch                                    # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir",
                   default=os.path.join(DS, "data", "diffsynth_5-6", "harmor_2oscfree"))
    p.add_argument("--n", type=int, default=2000,
                   help="Parameter files to read; the baseline is a dataset "
                        "statistic and converges quickly")
    p.add_argument("--out", default="results/diffsynth/param_baseline.json")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "param", "*.pt")))
    if not files:
        raise SystemExit(f"no parameter files under {args.data_dir}/param")
    files = files[:args.n]
    print(f"{len(files)} parameter files from {args.data_dir}")

    stack: dict[str, list] = {}
    for f in files:
        for k, v in torch.load(f, weights_only=False).items():
            if torch.is_tensor(v) and v.numel():
                stack.setdefault(k, []).append(v.float().unsqueeze(0))

    out = {}
    for k, vs in stack.items():
        try:
            x = torch.cat(vs, dim=0)
        except RuntimeError:                    # ragged frame counts
            print(f"  {k:<24} skipped (inconsistent shapes)")
            continue
        # Per feature channel, over clips and frames.
        dims = tuple(range(x.dim() - 1)) if x.dim() > 1 else (0,)
        out[k] = (x - x.mean(dim=dims, keepdim=True)).abs().mean().item()
        print(f"  {k:<24} {out[k]:.4f}   shape {tuple(x.shape)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"n_files": len(files), "data_dir": args.data_dir,
                   "baseline_l1": out}, fh, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
