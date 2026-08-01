"""DDSP-style encoder trained purely by resynthesis loss.

    IR -> encoder -> parameters -> differentiable plate -> IR' -> loss(IR, IR')

The only training signal is the audio loss. There is no parameter supervision:
the point of the experiment is whether the loss's terrain is benign enough for
the gradient through the synthesizer to train an encoder at all, and a
parameter loss would answer a different question.

Everything about the terrain is shared with src.gd.graddescent -- the same
Raw7Space, so the same bounds, the same linear [-1,1] map, the same plate, and
no peak normalization -- so a result here is directly comparable to the per-IR
gradient-descent results on the same loss.

What per-IR fitting cannot tell you, and this can
------------------------------------------------
Per-IR descent commits to one starting point: a start whose gradient is
misleading parks at the loss's saturation floor and never recovers. An encoder
holds one weight vector against a whole batch of targets, so per-target errors
that disagree partly cancel while any component consistent across targets
survives and is amplified by sqrt(batch). Whether such a consistent component
exists at initialization is exactly what this measures.

Reading the training loss tells you which regime you are in, and quickly:

    stuck near the saturation floor  -- the gradient is uninformative here; the
                                        encoder is not learning the mapping
    descending well below it         -- the coarse mapping is being learned
    approaching gt_loss              -- most examples are in-basin

The saturation floor and gt_loss are both reported at startup so the numbers can
be read against something rather than in the abstract.

Targets are synthesized with the same plate that closes the training loop, so
target and model share a code path exactly, as gen_torch_targets_200.py does for
the fitting datasets.

Usage:
    python -m src.ddsp.train_encoder --loss L1_STFT --steps 20000 --compile-plate
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.cmaes.fit_7param_norm_es import BOUNDS_HI_NP, BOUNDS_LO_NP, PARAM_KEYS
from src.gd.graddescent import (
    SAMPLE_RATE,
    Raw7Space,
    _read_params_csv,
    nmse_7d,
    verify_mapping_matches_cmaes,
)
from src.loss.loss_selector import select_loss_function
from src.mu_optimization.ternary_mu import load_target_ir_from_npz, nmse_6d, seven_to_six


class Encoder(nn.Module):
    """Deliberately ordinary CNN over the magnitude spectrogram.

    Kept unremarkable on purpose: the experiment is about the loss's terrain, and
    an unusual architecture would invite the result being attributed to it.
    Outputs tanh-bounded coordinates in [-1,1], the same normalized space the
    fitter searches, so predictions are in-bounds by construction.
    """

    def __init__(
        self,
        n_out: int = 7,
        width: int = 32,
        n_fft: int = 2048,
        hop: int = 512,
        n_blocks: int = 5,
        max_ch: int = 256,
    ):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft))

        # Frequency position is the signal here: the parameters are fixed by
        # where the modes sit, via om^2 = (T0/mu)*g1 + (D/mu)*g2. Pooling the
        # frequency axis away would leave the encoder able to see how many peaks
        # there are and how loud they are, but not where -- so it could only ever
        # recover the coordinates that follow from gross spectral statistics.
        # Time is pooled instead: the IR is a sum of stationary damped sinusoids,
        # so its time structure is an exponential envelope and little else.
        blocks, ch_in, n_freq = [], 1, n_fft // 2 + 1
        for i in range(n_blocks):
            ch_out = min(width * (2 ** i), max_ch)
            # Stride time only while there are frames left to spend.
            stride = (2, 2) if i < 3 else (2, 1)
            blocks += [
                nn.Conv2d(ch_in, ch_out, 3, stride=stride, padding=1),
                nn.GroupNorm(min(8, ch_out), ch_out),
                nn.GELU(),
            ]
            ch_in, n_freq = ch_out, (n_freq + 1) // 2
        blocks.append(nn.AdaptiveAvgPool2d((None, 1)))  # pool time, keep frequency
        self.net = nn.Sequential(*blocks)

        feat = ch_in * n_freq
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(feat, 256), nn.GELU(), nn.Linear(256, n_out)
        )

    def features(self, x: torch.Tensor, scale: float, log_input: bool) -> torch.Tensor:
        spec = torch.stft(
            x, self.n_fft, self.hop, window=self.window, return_complex=True
        ).abs()
        # A fixed constant, never a per-example norm: dividing each example by its
        # own peak would erase the absolute amplitude that identifies mu.
        spec = spec / scale
        if log_input:
            spec = torch.log(spec + 1e-6)
        return spec.unsqueeze(1)

    def forward(self, x: torch.Tensor, scale: float, log_input: bool) -> torch.Tensor:
        return torch.tanh(self.head(self.net(self.features(x, scale, log_input))))


@torch.no_grad()
def synth_dataset(
    space: Raw7Space, n: int, duration: float, seed: int, batch: int, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample parameters uniformly in [-1,1]^7 and render them with the training plate.

    Uniform in the normalized raw-7 box is uniform in the physical box, which is
    how ModalPlate/DatasetGen.py draws its parameters, so the training
    distribution matches the datasets the fitter is evaluated on.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = (torch.rand((n, len(PARAM_KEYS)), generator=g) * 2.0 - 1.0).to(device)
    outs = []
    for i in range(0, n, batch):
        outs.append(space.forward(z[i : i + batch], None, duration).float())
    return z, torch.cat(outs, dim=0)


def load_dataset(
    space: Raw7Space, data_dir: Path, duration: float, device, limit: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a dataset written by src.data.make_dataset (or the two-step process).

    Used so the encoder can be validated on the very IRs the per-IR fitting runs
    were measured on, rather than only on freshly sampled ones.
    """
    csvs = sorted(data_dir.glob("random_IR_params_*.csv"))
    if limit is not None:
        csvs = csvs[:limit]
    if not csvs:
        raise FileNotFoundError(f"No random_IR_params_*.csv in {data_dir}")

    want = int(duration * SAMPLE_RATE)
    zs, irs = [], []
    for c in csvs:
        rid = c.stem.split("_")[-1]
        npz = data_dir / f"random_IR_{rid}.npz"
        if not npz.exists():
            continue
        ir = load_target_ir_from_npz(npz, duration, SAMPLE_RATE)
        if ir.shape[0] != want:
            raise ValueError(
                f"{npz.name} has {ir.shape[0]} samples, expected {want} at "
                f"--duration {duration}; regenerate or pick a shorter duration"
            )
        zs.append(space.gt_z(_read_params_csv(c)))
        irs.append(ir)

    z = torch.as_tensor(np.asarray(zs), dtype=torch.float32, device=device)
    x = torch.as_tensor(np.asarray(irs), dtype=torch.float32, device=device)
    return z, x


def z_to_dicts(z: np.ndarray) -> list:
    phys = BOUNDS_LO_NP + ((z + 1.0) / 2.0) * (BOUNDS_HI_NP - BOUNDS_LO_NP)
    return [{k: float(v) for k, v in zip(PARAM_KEYS, row)} for row in phys]


@torch.no_grad()
def evaluate(model, space, z_val, x_val, args, loss_fn, scale) -> Dict[str, float]:
    model.eval()
    losses, preds = [], []
    for i in range(0, x_val.shape[0], args.batch_size):
        xb = x_val[i : i + args.batch_size]
        zp = model(xb, scale, args.log_input)
        pred = space.forward(zp, None, args.duration)
        losses.append(loss_fn(xb, pred).detach())
        preds.append(zp.detach())
    model.train()

    zp = torch.cat(preds).cpu().numpy()
    zt = z_val.cpu().numpy()
    est, gt = z_to_dicts(zp), z_to_dicts(zt)
    n6 = [nmse_6d(seven_to_six(e), seven_to_six(g)) for e, g in zip(est, gt)]
    n7 = [nmse_7d(e, g) for e, g in zip(est, gt)]

    out = {
        "val_loss": float(torch.cat(losses).mean()),
        "val_nmse_6d": float(np.median(n6)),
        "val_nmse_7d": float(np.median(n7)),
    }
    # Averaged NMSE hides which coordinates are being learned. Correlation says
    # whether a coordinate is tracked at all; the spread ratio separates "wrong"
    # from "collapsed", since an encoder ignoring its input emits a near-constant
    # prediction and scores about what predicting the mean would.
    def _corr(a, b):
        if float(a.std()) > 1e-12 and float(b.std()) > 1e-12:
            return float(np.corrcoef(a, b)[0, 1])
        return 0.0

    for i, k in enumerate(PARAM_KEYS):
        pred_sd, true_sd = float(zp[:, i].std()), float(zt[:, i].std())
        out[f"corr_{k}"] = _corr(zp[:, i], zt[:, i])
        out[f"spread_{k}"] = pred_sd / max(true_sd, 1e-12)

    # The raw-7 correlations above are only meaningful for Ly, op_x and op_y.
    # E, rho and h are individually unidentifiable from audio -- the synthesis
    # depends on them solely through mu = rho*h and D/mu, and the symmetry
    # (E, rho, h) -> (c^3 E, c rho, h/c) leaves the IR untouched -- so their
    # correlations track drift along a direction the loss cannot see. These are
    # the ones that say whether the mapping is being learned. Compared in log
    # space for mu, D_mu and T0_mu, which span decades.
    est6 = [seven_to_six(e) for e in est]
    gt6 = [seven_to_six(g) for g in gt]
    for k in ("mu", "D_div_mu", "T0_div_mu", "Ly", "op_x", "op_y"):
        a = np.array([e[k] for e in est6], dtype=np.float64)
        b = np.array([g[k] for g in gt6], dtype=np.float64)
        if k in ("mu", "D_div_mu", "T0_div_mu"):
            a, b = np.log(np.maximum(a, 1e-300)), np.log(np.maximum(b, 1e-300))
        out[f"c6_{k}"] = _corr(a, b)
    return out


def constant_predictor_nmse(z_train: torch.Tensor, z_val: torch.Tensor) -> Tuple[float, float]:
    """NMSE of ignoring the input entirely and always emitting the training mean.

    The floor any encoder beats for free. Without it an NMSE of 4e-2 reads as a
    result rather than as "about what predicting the marginal distribution gives".
    """
    zc = np.repeat(z_train.mean(0, keepdim=True).cpu().numpy(), z_val.shape[0], axis=0)
    est, gt = z_to_dicts(zc), z_to_dicts(z_val.cpu().numpy())
    n6 = [nmse_6d(seven_to_six(e), seven_to_six(g)) for e, g in zip(est, gt)]
    n7 = [nmse_7d(e, g) for e, g in zip(est, gt)]
    return float(np.median(n6)), float(np.median(n7))


def run(args) -> None:
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    verify_mapping_matches_cmaes(device)

    space = Raw7Space(device, torch.float32, normalize=False)
    space.configure_plate(args.chunk_elems, True, False, args.compile_plate, args.mode_bucket)
    loss_fn = select_loss_function(args.loss, sample_rate=SAMPLE_RATE, device=device)

    print(f"Device {device} | loss {args.loss} | duration {args.duration}s")
    t0 = time.time()
    if args.data_dir is not None:
        print(f"Loading train targets from {args.data_dir}")
        z_tr, x_tr = load_dataset(space, args.data_dir, args.duration, device, args.n_train)
    else:
        print(f"Generating {args.n_train} train targets...")
        z_tr, x_tr = synth_dataset(space, args.n_train, args.duration, args.seed, args.batch_size, device)

    if args.val_data_dir is not None:
        print(f"Loading val targets from {args.val_data_dir}")
        z_va, x_va = load_dataset(space, args.val_data_dir, args.duration, device, args.n_val)
    else:
        z_va, x_va = synth_dataset(space, args.n_val, args.duration, args.seed + 1, args.batch_size, device)
    print(
        f"  {x_tr.shape[0]} train / {x_va.shape[0]} val in {time.time() - t0:.0f}s   "
        f"train tensor {x_tr.numel() * 4 / 1e9:.2f} GB"
    )

    # Fixed input scale from the training set; constant, so relative amplitude
    # between examples survives and mu stays recoverable.
    scale = float(x_tr.abs().max())

    # Two reference levels, so the training curve can be read against something.
    # gt_loss is the floor: the loss at the true parameters. The saturation level
    # is what unrelated IRs score, i.e. where an uninformative gradient parks.
    with torch.no_grad():
        gt_loss = float(loss_fn(x_va[: args.batch_size], space.forward(z_va[: args.batch_size], None, args.duration)).mean())
        perm = torch.randperm(x_va.shape[0])[: args.batch_size]
        sat = float(loss_fn(x_va[: args.batch_size], x_va[perm]).mean())
    const6, const7 = constant_predictor_nmse(z_tr, z_va)
    print(f"reference levels:  gt_loss {gt_loss:.4e}   saturation (unrelated IRs) {sat:.4e}")
    print(f"constant-predictor NMSE: 6d {const6:.3e}  7d {const7:.3e}  (the floor to beat)")
    print("training loss stuck near saturation = gradient uninformative; well below = learning\n")

    model = Encoder(
        n_out=len(PARAM_KEYS), width=args.width, n_fft=args.n_fft, hop=args.hop,
        n_blocks=args.n_blocks, max_ch=args.max_ch,
    ).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.adam_eps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    print(f"encoder: {n_par/1e6:.2f}M params, width {args.width}\n")

    # Same trick as the fitter: a constant divisor keeps the objective at O(1)
    # for any loss in the registry without moving its optimum.
    loss_scale: Optional[float] = None
    hist = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, x_tr.shape[0], (args.batch_size,), device=device)
        xb = x_tr[idx]
        zp = model(xb, scale, args.log_input)
        pred = space.forward(zp, None, args.duration)
        loss = loss_fn(xb, pred)
        finite = torch.isfinite(loss)
        obj = torch.where(finite, loss, torch.zeros_like(loss)).mean()

        if loss_scale is None:
            loss_scale = max(float(obj.detach()), 1e-30)
            print(f"loss scale (fixed): {loss_scale:.4e}")

        opt.zero_grad(set_to_none=True)
        (obj / loss_scale).backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        opt.step()
        sched.step()

        if step % args.log_every == 0 or step == 1:
            tr = float(obj.detach())
            row = {
                "step": step,
                "train_loss": tr,
                "grad_norm": gnorm,
                "clipped": bool(gnorm > args.grad_clip),
                "elapsed_s": time.time() - t0,
            }
            if step % args.eval_every == 0 or step == 1:
                row.update(evaluate(model, space, z_va, x_va, args, loss_fn, scale))
                corr = "  ".join(
                    f"{k}={row[f'c6_{k}']:+.2f}"
                    for k in ("mu", "D_div_mu", "T0_div_mu", "Ly", "op_x", "op_y")
                )
                spread = np.mean([row[f"spread_{k}"] for k in PARAM_KEYS])
                print(
                    f"step {step:6d}  train {tr:.4e}  val {row['val_loss']:.4e}  "
                    f"NMSE_6d {row['val_nmse_6d']:.3e} (const {const6:.3e})  "
                    f"NMSE_7d {row['val_nmse_7d']:.3e}  |g| {gnorm:.2e}  "
                    f"[{row['elapsed_s']:.0f}s]"
                )
                print(f"           corr(identifiable)  {corr}   mean spread/GT {spread:.2f}")
            else:
                print(f"step {step:6d}  train {tr:.4e}  |g| {gnorm:.2e}  [{row['elapsed_s']:.0f}s]")
            hist.append(row)
            with (out_dir / "history.json").open("w") as f:
                json.dump(
                    {
                        "gt_loss": gt_loss,
                        "saturation": sat,
                        "const_nmse_6d": const6,
                        "const_nmse_7d": const7,
                        "history": hist,
                    },
                    f,
                    indent=2,
                )

    torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "encoder.pt")

    steps = [h["step"] for h in hist]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].semilogy(steps, [h["train_loss"] for h in hist], label="train")
    ev = [h for h in hist if "val_loss" in h]
    axes[0].semilogy([h["step"] for h in ev], [h["val_loss"] for h in ev], label="val")
    axes[0].axhline(sat, color="r", ls="--", lw=0.8, label="saturation")
    axes[0].axhline(gt_loss, color="g", ls="--", lw=0.8, label="gt_loss")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("loss"); axes[0].legend(fontsize=8)
    axes[0].set_title(f"{args.loss}: resynthesis loss"); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy([h["step"] for h in ev], [h["val_nmse_6d"] for h in ev], marker="o", label="NMSE_6d")
    axes[1].semilogy([h["step"] for h in ev], [h["val_nmse_7d"] for h in ev], marker="s", label="NMSE_7d")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("median val NMSE"); axes[1].legend(fontsize=8)
    axes[1].set_title("parameter recovery (never trained on)"); axes[1].grid(True, alpha=0.3)
    plt.suptitle(f"DDSP encoder | {args.loss} | resynthesis loss only", fontweight="bold")
    plt.tight_layout(); plt.savefig(out_dir / "training.png", dpi=140); plt.close(fig)
    print(f"\nDone. Outputs written to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train an audio-to-parameter encoder through the differentiable plate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", type=Path, default=Path("results/ddsp/encoder"))
    p.add_argument("--loss", type=str, default="L1_STFT")
    p.add_argument("--duration", type=float, default=0.25)
    p.add_argument(
        "--data-dir", type=Path, default=None,
        help="Load training targets from a dataset directory instead of generating them",
    )
    p.add_argument(
        "--val-data-dir", type=Path, default=None,
        help="Validate on a fixed dataset, e.g. the IRs the fitting runs were measured on",
    )
    p.add_argument("--n-train", type=int, default=8192)
    p.add_argument("--n-val", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--adam-eps", type=float, default=1e-16)
    p.add_argument(
        "--grad-clip", type=float, default=10.0,
        help="Clip is on the objective *after* loss scaling; at 1.0 it bound on "
             "essentially every step, making it a constant rather than an outlier guard",
    )
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--n-blocks", type=int, default=5)
    p.add_argument("--max-ch", type=int, default=256)
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--hop", type=int, default=512)
    p.add_argument(
        "--log-input", action="store_true",
        help="Log-compress the input spectrogram. This is a representation choice, "
             "independent of compression in the loss; state it separately in writeups.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--compile-plate", action="store_true")
    p.add_argument("--mode-bucket", type=int, default=1024)
    p.add_argument("--chunk-elems", type=int, default=8_000_000)
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
