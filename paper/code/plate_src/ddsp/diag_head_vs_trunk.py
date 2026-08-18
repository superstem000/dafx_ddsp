"""Is the head the bottleneck, or was it already over before the head?

A collapsed encoder shows the same symptom either way: every input maps to one
prediction. The 1/3-of-range per-parameter error says that constant sits at the
edge of the box, but it does NOT say tanh put it there. With an audio-only loss
a collapsed head settles wherever audio error is lowest, which need not be the
parameter-space mean -- so a large |z| is equally consistent with "saturation
caused the collapse" and with "the collapse happened upstream and z drifted
afterwards".

Those two have different consequences. If the trunk still produces
input-dependent features and only the head is pinned, a non-saturating output
map fixes it and is worth the compute. If the trunk's own features have
collapsed, no output map rescues anything -- and that is the stronger result,
because it says the loss provided no usable input-dependent gradient at all.

Reads checkpoints, so it needs no training and no GPU time worth counting.

    python -m src.ddsp.diag_head_vs_trunk --runs results/ddsp/eps_ladder/*
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from src.ddsp.train_encoder import Encoder, load_dataset
from src.cmaes.fit_7param_norm_es import PARAM_KEYS
from src.gd.graddescent import Raw7Space


def probe(ckpt: Path, x: torch.Tensor, device) -> dict | None:
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    a = ck.get("args", {})
    if not a:
        return None
    model = Encoder(
        n_out=len(PARAM_KEYS), width=a.get("width", 32), n_fft=a.get("n_fft", 2048),
        hop=a.get("hop", 512), n_blocks=a.get("n_blocks", 5),
        max_ch=a.get("max_ch", 256), input_mode=a.get("input_mode", "norm_amp"),
        norm=a.get("norm", "group"), head_bound=a.get("head_bound", "tanh"),
        head_grad_floor=a.get("head_grad_floor", 0.05),
        head_cap=a.get("head_cap", 3.0),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    with torch.no_grad():
        feat = model.features(x, float(ck.get("scale", 1.0)))
        # h is exactly what the head sees: trunk output, flattened, pre-Linear.
        h = model.flatten(model.net(feat))
        z = model.head(h)
        y = model.from_features(feat)

    h = h.float().cpu().numpy()
    z = z.float().cpu().numpy()
    y = y.float().cpu().numpy()

    # Variance ACROSS the batch, per feature, then averaged. This is the
    # question "do two different IRs produce two different feature vectors" --
    # not "is there variation within one vector", which says nothing.
    h_sd = float(h.std(axis=0).mean())
    h_sd_rel = h_sd / (float(np.abs(h).mean()) + 1e-12)
    return {
        "step": ck.get("step", -1),
        "loss": a.get("loss", "?"),
        "norm": a.get("norm", "?"),
        "head": a.get("head_bound", "tanh"),
        "trunk_sd": h_sd,
        "trunk_sd_rel": h_sd_rel,
        "z_absmean": float(np.abs(z).mean()),
        "z_absmax": float(np.abs(z).max()),
        "z_sat": float((np.abs(z) > 2.5).mean()),
        "y_sd": float(y.std(axis=0).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", nargs="+", default=None,
                   help="Run directories; default is every dir under eps_ladder")
    p.add_argument("--ckpt", default="encoder_last.pt")
    p.add_argument("--val-data-dir", type=Path, default=Path("data/val-p99"))
    p.add_argument("--n-val", type=int, default=64)
    p.add_argument("--duration", type=float, default=0.25)
    p.add_argument("--chunk-elems", type=int, default=20_000_000)
    p.add_argument("--mode-bucket", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    space = Raw7Space(dev, torch.float32, normalize=False)
    space.configure_plate(args.chunk_elems, False, True, False, args.mode_bucket, None)
    _z, x = load_dataset(space, args.val_data_dir, args.duration, dev, args.n_val)
    print(f"{x.shape[0]} val IRs from {args.val_data_dir}\n")

    runs = args.runs or sorted(glob.glob("results/ddsp/eps_ladder/*"))
    print(f"{'run':<18}{'step':>7}{'norm':>7}{'head':>9}"
          f"{'trunk_sd':>10}{'rel':>8}{'|z|mean':>9}{'|z|max':>8}{'sat':>7}{'y_sd':>8}")
    rows = []
    for r in runs:
        c = Path(r) / args.ckpt
        if not c.exists():
            continue
        d = probe(c, x, dev)
        if d is None:
            print(f"{Path(r).name:<18}  checkpoint predates args saving"); continue
        rows.append((Path(r).name, d))
        print(f"{Path(r).name:<18}{d['step']:>7}{d['norm']:>7}{d['head']:>9}"
              f"{d['trunk_sd']:>10.4f}{d['trunk_sd_rel']:>8.3f}{d['z_absmean']:>9.3f}"
              f"{d['z_absmax']:>8.2f}{d['z_sat']:>7.2f}{d['y_sd']:>8.4f}")

    if not rows:
        print("\nno usable checkpoints"); return

    ref = next((d for n, d in rows if d["loss"] == "L1_STFT"), None)
    print("\nreading, against the linear arm as reference:")
    for n, d in rows:
        if ref and d is ref:
            print(f"  {n:<18} reference (this one works)"); continue
        trunk_alive = ref and d["trunk_sd_rel"] > 0.3 * ref["trunk_sd_rel"]
        pinned = d["z_sat"] > 0.2
        if trunk_alive and pinned:
            v = "HEAD is the bottleneck -- trunk still varies, head pinned. A " \
                "non-saturating output map should fix it."
        elif not trunk_alive:
            v = "TRUNK has collapsed -- the head is innocent and no output map " \
                "rescues this. The loss gave no input-dependent gradient."
        elif pinned:
            v = "head pinned but trunk reference unavailable -- rerun with the " \
                "linear arm included"
        else:
            v = "neither pinned nor collapsed -- look at the training curve instead"
        print(f"  {n:<18} {v}")


if __name__ == "__main__":
    main()
