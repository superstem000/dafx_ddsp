"""Split the Param column into its six parameter groups, per arm.

    python scripts/ds_param_breakdown.py
    python scripts/ds_param_breakdown.py --only '^synth_' --device cuda:0

model.py:33 sums an L1 over every saved parameter group and divides by the
count, so `val_id/param` is a six-way mean and the breakdown is thrown away
before it is logged. This recomputes it from each arm's checkpoint -- a forward
pass over the in-domain validation split, no training.

WHY IT DISCRIMINATES. Two different mechanisms both predict that the linear arms
lose Param accuracy once param_w reaches 0, and they predict different
breakdowns:

  compression   log up-weights quiet bins, so parameters whose only evidence is
                in the quiet tail -- harmor_cutoff, harmor_q -- should favour
                the log arms, while the rest favour linear.

  scale balance the linear half of the loss is top-heavy across FFT sizes
                (ds_scale_balance.py measures 40-52% at n_fft=2048 against an
                equal-weight 16.7%, and 4.7-10% for the two shortest windows
                against the log half's 18-37%). A 128ms window cannot resolve
                an envelope on a 16ms grid, so the TIME-VARYING parameters --
                harmor_amplitudes from enva, harmor_cutoff from envc -- should
                be where the linear arms lose, and f0_hz / f0_mult / osc_mix
                should be at parity or better.

harmor_cutoff is the one group both mechanisms claim, which is why the others
decide it: a deficit concentrated in amplitudes+cutoff with f0 at parity is
scale balance; one in cutoff+q with amplitudes at parity is compression. An
even spread across all six is neither.

THE SPLIT. The validation set is drawn by random_split from the global RNG at
setup() time (data.py:83), so it is reproduced here by seeding exactly as
train.py does. That is an assumption, not a guarantee, so it is checked against
the split_manifest.json each run wrote -- the same hash SplitManifest computes.
A mismatch means these numbers are being measured on different files than the
run validated on, and the script says so rather than printing a table anyway.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "external", "diffsynth")
sys.path.insert(0, DS)

import torch                                            # noqa: E402
import torch.nn.functional as F                         # noqa: E402
import hydra                                            # noqa: E402
import pytorch_lightning as pl                          # noqa: E402
from omegaconf import OmegaConf                          # noqa: E402

from diffsynth import util                              # noqa: E402
from diffsynth.model import EstimatorSynth              # noqa: E402

sys.path.insert(0, os.path.join(DS))
from split_manifest import _files                       # noqa: E402


def param_l1(model, synth_output, param_dict):
    """model.py:33, but keeping the per-group terms instead of summing them."""
    out = {}
    for k, target in param_dict.items():
        output_name = model.synth.dag_summary[k]
        if output_name in model.synth.fixed_param_names:
            continue
        if target.numel() == 0:
            continue
        x = synth_output[output_name]
        if target.shape[1] > 1:
            x = util.resample_frames(x, target.shape[1])
        out[k] = F.l1_loss(x, target).item()
    return out


def load_arm(run_dir: str, ckpt_name: str, device: str):
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    ck_path = os.path.join(run_dir, "tb_logs", "checkpoints", ckpt_name)
    if not os.path.exists(cfg_path) or not os.path.exists(ck_path):
        return None, None, f"missing {'config' if not os.path.exists(cfg_path) else ckpt_name}"
    cfg = OmegaConf.load(cfg_path)
    try:
        # weights_only=False for the same reason train.py needs it: the
        # checkpoint carries the hydra DictConfig via save_hyperparameters().
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    except Exception as e:                              # a half-written latest.ckpt
        return None, None, f"unreadable checkpoint ({type(e).__name__})"
    model = EstimatorSynth(cfg.model, cfg.synth, cfg.schedule)
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, cfg, f"epoch {ck.get('epoch', '?')}"


def build_val_loader(cfg, batch_size: int):
    """The id 'valid' split, reproduced the way train.py produces it."""
    pl.seed_everything(0, workers=True)
    dcfg = OmegaConf.to_container(cfg.data, resolve=True)
    dcfg["batch_size"] = batch_size
    dcfg["num_workers"] = 0
    dm = hydra.utils.instantiate(dcfg)
    dm.setup(None)
    return dm


def split_sha1(ds) -> str:
    return hashlib.sha1("\n".join(sorted(_files(ds))).encode()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="results/diffsynth")
    p.add_argument("--only", default=None, metavar="REGEX")
    p.add_argument("--ckpt", default="latest.ckpt")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--batches", type=int, default=8,
                   help="Validation batches to average over; 8x16 = 128 clips")
    p.add_argument("--device", default="cpu",
                   help="cpu by default so this never competes with the queue "
                        "for a card. cuda:N is much faster if one is free.")
    args = p.parse_args()

    runs = [d for d in sorted(glob.glob(os.path.join(args.root, "*")))
            if os.path.isdir(d) and (not args.only or re.search(args.only, Path(d).name))]
    if not runs:
        print(f"no runs under {args.root}")
        return

    rows, groups = {}, []
    for d in runs:
        name = Path(d).name
        model, cfg, note = load_arm(d, args.ckpt, args.device)
        if model is None:
            print(f"{name:<20} skipped: {note}")
            continue

        dm = build_val_loader(cfg, args.batch_size)
        # The split is reproduced, not read. Check it against what the run
        # itself recorded before trusting a single number below.
        man = os.path.join(d, "split_manifest.json")
        if os.path.exists(man):
            want = json.load(open(man)).get("id_valid", {}).get("sha1")
            got = split_sha1(dm.id_datasets["valid"])
            if want and want != got:
                print(f"{name:<20} SPLIT MISMATCH -- manifest {want[:8]}, "
                      f"reproduced {got[:8]}; skipping rather than reporting "
                      f"numbers from the wrong files")
                continue
        else:
            print(f"{name:<20} note: no split_manifest.json, split unverified")

        loader = dm.val_dataloader()[0]
        acc, n = {}, 0
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= args.batches:
                    break
                batch = {k: (v.to(args.device) if torch.is_tensor(v) else
                             {kk: vv.to(args.device) for kk, vv in v.items()})
                         for k, v in batch.items()}
                _audio, out = model(batch)
                for k, v in param_l1(model, out, batch["params"]).items():
                    acc[k] = acc.get(k, 0.0) + v
                n += 1
        if not n:
            print(f"{name:<20} skipped: no validation batches")
            continue
        rows[name] = {k: v / n for k, v in acc.items()}
        if not groups:
            groups = list(rows[name])
        print(f"{name:<20} {note}, {n} batches of {args.batch_size}")

    if not rows:
        return

    short = [g.replace("harmor_", "") for g in groups]
    w = max(10, max(len(s) for s in short) + 2)
    print("\n=== per-group L1 on the in-domain validation split")
    print("val_id/param is the mean of these (model.py:44 divides by the group "
          "count),\nso the 'mean' column should match the monitor's Param.\n")
    print(f"{'run':<20}" + "".join(f"{s:>{w}}" for s in short) + f"{'mean':>10}")
    for name, r in rows.items():
        vals = [r.get(g, float('nan')) for g in groups]
        mean = sum(v for v in vals if v == v) / len(vals)
        print(f"{name:<20}" + "".join(f"{v:>{w}.4f}" for v in vals) + f"{mean:>10.4f}")

    # Ratios against the first arm, which is what the prediction is about: a
    # deficit concentrated in particular groups, not a uniform one.
    ref = next(iter(rows))
    print(f"\n=== ratio to {ref}  (>1 is worse than {ref})")
    print(f"{'run':<20}" + "".join(f"{s:>{w}}" for s in short))
    for name, r in rows.items():
        if name == ref:
            continue
        cells = []
        for g in groups:
            a, b = r.get(g, float("nan")), rows[ref].get(g, float("nan"))
            cells.append(f"{a / b:>{w}.3f}" if b else f"{'-':>{w}}")
        print(f"{name:<20}" + "".join(cells))


if __name__ == "__main__":
    main()
