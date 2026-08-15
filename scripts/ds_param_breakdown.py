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

THE SPLIT. The validation set is drawn by random_split off the global RNG at
setup() time (data.py:83), so which files are in it depends on every draw made
before that point -- including the estimator's weight init. Reproducing it means
reproducing train.py's construction order exactly, which this does and which the
first version of this script did not: seeding and setting up immediately gave a
different split, and the manifest check caught it.

Preferably it is not reproduced at all. SplitManifest now records the membership
itself, so for any run written after that the split is read rather than derived,
and the fragile ordering dependency does not apply. Runs from before it fall
back to reproduction plus the hash check -- and if that hash disagrees the arm
is skipped, because a hash can say the split is wrong but not what the right one
was.
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
from torch.utils.data import DataLoader, Subset          # noqa: E402
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


def baseline_l1(targets):
    """L1 of a constant predictor: the error from always guessing the mean.

    The raw per-group L1s are not comparable to each other -- f0_hz lands
    around 0.004 and osc_mix around 0.13, so the six-way mean param_loss takes
    is roughly half osc_mix and q by magnitude alone, and f0_hz contributes
    under 1% whatever any arm does to it. Equal units would not fix that:
    equal units do not imply equal difficulty.

    Dividing by this baseline does. 1.0 means an arm did no better than
    guessing the dataset mean for that parameter; 0.0 means it recovered it
    exactly. The ratio then says how much of the LEARNABLE signal each arm
    captured, which is comparable across groups and is the same normalisation
    the plate work reports against (train_encoder.py's constant-predictor
    floor).

    The mean is taken per feature channel over clips and frames, so a
    time-varying parameter is scored against one constant, not against its own
    mean envelope. That is the conservative choice: a mean-envelope baseline
    would be stronger and would make every arm look worse, but it is also a
    less standard thing to quote.
    """
    x = torch.cat(targets, dim=0)
    return (x - x.mean(dim=(0, 1), keepdim=True)).abs().mean().item()


def load_arm(run_dir: str, ckpt_name: str, device: str, batch_size: int):
    """Model and datamodule, built in train.py's order.

    The order is the point. train.py seeds, constructs EstimatorSynth, and only
    then sets up the datamodule -- and create_split calls random_split off the
    global generator, so every draw the estimator's weight init makes shifts the
    permutation. Seeding and setting up immediately gives a different validation
    split, which is exactly what the manifest check caught the first time this
    ran.
    """
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    ck_path = os.path.join(run_dir, "tb_logs", "checkpoints", ckpt_name)
    if not os.path.exists(cfg_path) or not os.path.exists(ck_path):
        return None, None, None, f"missing {'config' if not os.path.exists(cfg_path) else ckpt_name}"
    cfg = OmegaConf.load(cfg_path)
    try:
        # weights_only=False for the same reason train.py needs it: the
        # checkpoint carries the hydra DictConfig via save_hyperparameters().
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    except Exception as e:                              # a half-written latest.ckpt
        return None, None, None, f"unreadable checkpoint ({type(e).__name__})"

    pl.seed_everything(0, workers=True)
    model = EstimatorSynth(cfg.model, cfg.synth, cfg.schedule)
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)

    dcfg = OmegaConf.to_container(cfg.data, resolve=True)
    dcfg["batch_size"] = batch_size
    dcfg["num_workers"] = 0
    dm = hydra.utils.instantiate(dcfg)
    dm.setup(None)
    return model, cfg, dm, f"epoch {ck.get('epoch', '?')}"


def split_sha1(ds) -> str:
    return hashlib.sha1("\n".join(sorted(_files(ds))).encode()).hexdigest()


def underlying(ds):
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds


def val_split(run_dir: str, dm):
    """The arm's own id/valid split: from the manifest if it recorded one.

    Returns (dataset, how). Reading the membership beats reproducing it --
    a hash can only tell you that you got the wrong split, never what the right
    one was. Runs written before the manifest carried 'files' fall back to the
    reproduction plus the hash check.
    """
    repro = dm.id_datasets["valid"]
    man = os.path.join(run_dir, "split_manifest.json")
    if not os.path.exists(man):
        return repro, "split unverified (no manifest)"
    rec = json.load(open(man)).get("id_valid", {})
    names = rec.get("files")
    if names:
        base = underlying(repro)
        idx = {os.path.basename(f): i for i, f in enumerate(base.raw_files)}
        missing = [n for n in names if n not in idx]
        if missing:
            return None, f"manifest lists {len(missing)} files not in {base.audio_dir}"
        return Subset(base, [idx[n] for n in names]), "split read from manifest"
    want, got = rec.get("sha1"), split_sha1(repro)
    if want and want != got:
        return None, (f"SPLIT MISMATCH -- manifest {want[:8]}, reproduced {got[:8]}; "
                      f"this run predates the manifest recording membership, so "
                      f"the right split cannot be recovered")
    return repro, "split reproduced, hash matches manifest"


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

    rows, base, groups = {}, {}, []
    for d in runs:
        name = Path(d).name
        model, cfg, dm, note = load_arm(d, args.ckpt, args.device, args.batch_size)
        if model is None:
            print(f"{name:<20} skipped: {note}")
            continue

        vset, how = val_split(d, dm)
        if vset is None:
            print(f"{name:<20} skipped: {how}")
            continue

        loader = DataLoader(vset, batch_size=args.batch_size, num_workers=0)
        acc, tgt, n = {}, {}, 0
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
                    tgt.setdefault(k, []).append(batch["params"][k].detach().float().cpu())
                n += 1
        if not n:
            print(f"{name:<20} skipped: no validation batches")
            continue
        rows[name] = {k: v / n for k, v in acc.items()}
        base[name] = {k: baseline_l1(v) for k, v in tgt.items()}
        if not groups:
            groups = list(rows[name])
        print(f"{name:<20} {note}, {n} batches of {args.batch_size} -- {how}")

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

    # Normalised by the constant-predictor baseline: 1.0 = no better than
    # guessing the dataset mean, 0.0 = recovered exactly. This is the table to
    # read across groups; the raw one above is not comparable between columns.
    ref_base = next(iter(base.values()))
    spread = {g: max(b.get(g, float("nan")) for b in base.values())
                 - min(b.get(g, float("nan")) for b in base.values())
              for g in groups}
    bad = [g for g in groups if spread[g] == spread[g] and ref_base.get(g)
           and spread[g] / ref_base[g] > 1e-6]
    if bad:
        print(f"\nWARNING: the constant-predictor baseline differs between arms "
              f"for {', '.join(bad)} -- the arms are not being scored on the "
              f"same validation files.")
    print("\n=== fraction of the constant-predictor error remaining "
          "(1.0 = learned nothing)")
    print(f"{'run':<20}" + "".join(f"{s_:>{w}}" for s_ in short) + f"{'mean':>10}")
    print(f"{'(baseline L1)':<20}"
          + "".join(f"{ref_base.get(g, float('nan')):>{w}.4f}" for g in groups))
    for name, r in rows.items():
        vals = [r.get(g, float("nan")) / base[name][g] if base[name].get(g)
                else float("nan") for g in groups]
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
