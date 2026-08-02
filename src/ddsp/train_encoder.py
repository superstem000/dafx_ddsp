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
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.cmaes.fit_7param_norm_es import BOUNDS_HI_NP, BOUNDS_LO_NP, NU, PARAM_KEYS
from src.gd.graddescent import (
    SAMPLE_RATE,
    Raw7Space,
    _read_params_csv,
    nmse_7d,
    verify_mapping_matches_cmaes,
)
from src.loss.loss_selector import select_loss_function
from src.mu_optimization.ternary_mu import (
    COMPOSITE_BOUNDS,
    load_target_ir_from_npz,
    nmse_6d,
    seven_to_six,
)


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
        input_mode: str = "norm_amp",
        in_ch: Optional[int] = None,
        n_extra: int = 0,
    ):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.input_mode = input_mode
        self.register_buffer("window", torch.hann_window(n_fft))

        # Frequency position is the signal here: the parameters are fixed by
        # where the modes sit, via om^2 = (T0/mu)*g1 + (D/mu)*g2. Pooling the
        # frequency axis away would leave the encoder able to see how many peaks
        # there are and how loud they are, but not where -- so it could only ever
        # recover the coordinates that follow from gross spectral statistics.
        # Time is pooled instead: the IR is a sum of stationary damped sinusoids,
        # so its time structure is an exponential envelope and little else.
        first_ch = in_ch if in_ch is not None else (2 if input_mode == "norm_amp" else 1)
        blocks, ch_in, n_freq = [], first_ch, n_fft // 2 + 1
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

        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(ch_in * n_freq + n_extra, 256), nn.GELU(), nn.Linear(256, n_out)
        )
        # Start the output layer near zero so tanh begins in its linear region.
        # At default init the pre-activations are large enough that tanh can pin
        # at +-1 within a few hundred steps; its derivative is then zero and the
        # network is permanently stuck emitting a constant, whatever the input.
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def features(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        """Spectrogram, conditioned so the network can actually see the modes.

        Plate spectra span a huge dynamic range: the low modes dominate and the
        high ones are orders of magnitude weaker, while the overall level varies
        with 1/mu across the dataset. Under a single global linear scale the
        first convolution sees a few loud low-frequency peaks and near-zero
        everywhere else -- so mode positions, which is where D/mu, T0/mu and Ly
        live, are effectively invisible.

        Compressing the *input* is unrelated to compressing the *loss*: one
        changes what the network can see, the other changes which errors get
        penalized. Only the latter bears on whether a loss is a good estimation
        objective.

        norm_amp keeps both halves of the information separately: a
        peak-normalized spectrogram carrying the shape, plus log peak as a
        second channel carrying the absolute level that identifies mu. Neither
        is discarded, and each is on a sane numeric scale.
        """
        spec = torch.stft(
            x, self.n_fft, self.hop, window=self.window, return_complex=True
        ).abs()

        if self.input_mode == "linear":
            return (spec / scale).unsqueeze(1)
        if self.input_mode == "log":
            return torch.log(spec / scale + 1e-8).unsqueeze(1)

        peak = spec.amax(dim=(1, 2), keepdim=True).clamp(min=1e-30)
        shape = torch.log(spec / peak + 1e-8).unsqueeze(1)
        level = torch.log10(peak).unsqueeze(1).expand_as(shape)
        return torch.cat([shape, level], dim=1)

    def from_features(self, feat: torch.Tensor, extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Trunk plus head, with optional non-spatial inputs joined at the head.

        The conditioning vector is six scalars with no spatial extent, so it is
        concatenated after pooling rather than broadcast into constant input
        planes -- convolution can do nothing useful with a plane that is the same
        value everywhere.
        """
        h = self.flatten(self.net(feat))
        if extra is not None:
            h = torch.cat([h, extra], dim=1)
        return torch.tanh(self.head(h))

    def forward(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        return self.from_features(self.features(x, scale))


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


def _batch(x: torch.Tensor, idx: torch.Tensor, device) -> torch.Tensor:
    """Index a possibly-CPU-resident dataset and place the batch on the device.

    Holding the training set in host memory costs one ~700 KB transfer per step,
    which is nothing beside a synthesis, and lifts the dataset size limit from
    VRAM to RAM.
    """
    out = x[idx]
    return out if out.device == device else out.to(device, non_blocking=True)


class CompositeConditioner:
    """Maps a raw-7 prediction to the six normalized composites, for conditioning.

    The refiner is told where it is starting from, but only in terms the sound
    actually depends on. The raw seven carry one dimension that provably does not
    matter -- (E, rho, h) -> (c^3 E, c rho, h/c) leaves the IR identical -- so
    feeding all seven would hand the refiner six meaningful numbers plus one that
    is pure drift. Log-scaled for mu, D_mu and T0_mu, which span 1.6, 2.9 and 6.6
    decades respectively.
    """

    KEYS = ("mu", "D_div_mu", "T0_div_mu", "Ly", "op_x", "op_y")
    LOG = (True, True, True, False, False, False)

    def __init__(self, device, dtype=torch.float32):
        lo = np.array([COMPOSITE_BOUNDS[k][0] for k in self.KEYS], dtype=np.float64)
        hi = np.array([COMPOSITE_BOUNDS[k][1] for k in self.KEYS], dtype=np.float64)
        self.lo = torch.as_tensor(lo, device=device, dtype=dtype)
        self.hi = torch.as_tensor(hi, device=device, dtype=dtype)
        self.log_lo = torch.as_tensor(np.log(lo), device=device, dtype=dtype)
        self.log_hi = torch.as_tensor(np.log(hi), device=device, dtype=dtype)
        self.is_log = torch.as_tensor(np.array(self.LOG), device=device)
        self.b_lo = torch.as_tensor(BOUNDS_LO_NP, device=device, dtype=dtype)
        self.b_hi = torch.as_tensor(BOUNDS_HI_NP, device=device, dtype=dtype)

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        phys = self.b_lo + ((z + 1.0) / 2.0) * (self.b_hi - self.b_lo)
        E, rho, h, Ly, T0, op_x, op_y = [phys[:, i] for i in range(7)]
        mu = rho * h
        D = E * h.pow(3) / (12.0 * (1.0 - NU ** 2))
        six = torch.stack([mu, D / mu, T0 / mu, Ly, op_x, op_y], dim=1).clamp(min=1e-30)
        lin = (six - self.lo) / (self.hi - self.lo)
        log = (torch.log(six) - self.log_lo) / (self.log_hi - self.log_lo)
        return torch.where(self.is_log, log, lin).clamp(0.0, 1.0)


def two_stage_forward(enc0, refiner, cond, space, x, scale, args, two_stage: bool):
    """Stage 0, one synthesis, then a correction conditioned on the residual.

    Returns (z0, x0, z1, x1); z1/x1 are None until the refiner is active.
    """
    fx = enc0.features(x, scale)
    z0 = enc0.from_features(fx)
    x0 = space.forward(z0, None, args.duration)
    if not two_stage:
        return z0, x0, None, None

    f0 = enc0.features(x0, scale)
    c0 = cond(z0)
    if not args.refine_attach:
        # The gradient that matters already reaches z0 through z1 = z0 + a*dz.
        # Leaving the residual path attached mostly teaches stage 0 to make
        # errors that are convenient for stage 1 rather than errors that are small.
        f0, c0 = f0.detach(), c0.detach()
    dz = refiner.from_features(torch.cat([fx, f0, fx - f0], dim=1), c0)
    # Detaching z0 here splits the objectives: stage 0 is then trained only by
    # its own loss, and the refiner is the only thing optimizing the final
    # output. Left attached, stage 0 receives gradient from the final loss too,
    # by a direct unscaled path against the refiner's path through
    # refine_scale -- so anything the refiner might correct, stage 0 corrects
    # first, and the refiner's contribution converges to nothing.
    base = z0.detach() if args.detach_stage0 else z0
    z1 = torch.clamp(base + args.refine_scale * dz, -1.0, 1.0)
    return z0, x0, z1, space.forward(z1, None, args.duration)


def z_to_dicts(z: np.ndarray) -> list:
    phys = BOUNDS_LO_NP + ((z + 1.0) / 2.0) * (BOUNDS_HI_NP - BOUNDS_LO_NP)
    return [{k: float(v) for k, v in zip(PARAM_KEYS, row)} for row in phys]


@torch.no_grad()
def evaluate(model, space, z_val, x_val, args, loss_fn, scale,
             refiner=None, cond=None, two_stage=False) -> Dict[str, float]:
    model.eval()
    if refiner is not None:
        refiner.eval()
    losses, preds, preds0 = [], [], []
    for i in range(0, x_val.shape[0], args.batch_size):
        xb = x_val[i : i + args.batch_size]
        if str(xb.device) != str(space.device):
            xb = xb.to(space.device, non_blocking=True)
        z0, x0, z1, x1 = two_stage_forward(model, refiner, cond, space, xb, scale, args, two_stage)
        zp, pred = (z1, x1) if two_stage else (z0, x0)
        losses.append(loss_fn(xb, pred).detach())
        preds.append(zp.detach())
        preds0.append(z0.detach())
    model.train()
    if refiner is not None:
        refiner.train()

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

    # With the refiner active, report stage 0 separately so "did the correction
    # help" is visible rather than inferred from a single combined number.
    if two_stage:
        est0 = z_to_dicts(torch.cat(preds0).cpu().numpy())
        out["val_nmse_6d_stage0"] = float(
            np.median([nmse_6d(seven_to_six(e), seven_to_six(g)) for e, g in zip(est0, gt)])
        )
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
    space.configure_plate(
        args.chunk_elems, not args.no_grad_checkpoint, args.batched_plate,
        args.compile_plate, args.mode_bucket,
    )
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
    if args.data_device == "cpu":
        x_tr = x_tr.cpu()
        print("  training set held in host memory; batches transferred per step")

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
        n_blocks=args.n_blocks, max_ch=args.max_ch, input_mode=args.input_mode,
    ).to(device)
    cond = CompositeConditioner(device)
    refiner = None
    if args.stage1_start_step > 0:
        # Three images in (target, attempt, residual), each two channels, plus the
        # six normalized composites joined at the head.
        refiner = Encoder(
            n_out=len(PARAM_KEYS), width=args.width, n_fft=args.n_fft, hop=args.hop,
            n_blocks=args.n_blocks, max_ch=args.max_ch, input_mode=args.input_mode,
            in_ch=6, n_extra=len(CompositeConditioner.KEYS),
        ).to(device)

    params = list(model.parameters()) + (list(refiner.parameters()) if refiner else [])
    n_par = sum(p.numel() for p in params)

    # Separate parameter groups, because the two networks are at different points
    # in their training when the refiner appears. Sharing one cosine would drop a
    # randomly initialized refiner onto whatever learning rate stage 0 had
    # annealed to by then -- 65% of peak and falling at step 100k of 250k -- while
    # a converged stage 0 wants the small rate the same curve is giving it.
    groups = [{"params": list(model.parameters()), "lr": args.lr}]
    if refiner is not None:
        groups.append({"params": list(refiner.parameters()), "lr": args.lr})
    opt = torch.optim.Adam(groups, lr=args.lr, eps=args.adam_eps)
    def cosine(step: int, start: int, end: int) -> float:
        """Linear warmup then cosine decay, measured from this stage's own start.

        The warmup is what makes a raised learning rate safe: the first steps out
        of a random initialization are the ones that can drive the head's
        pre-activations far enough to saturate tanh, after which the gradient is
        zero and the network cannot recover.
        """
        if step < start:
            return 0.0
        t = step - start
        if t < args.warmup_steps:
            return (t + 1) / max(1, args.warmup_steps)
        frac = (t - args.warmup_steps) / max(1, end - start - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(frac, 1.0)))

    # Stage 0 anneals across the whole run; the refiner gets its own full cosine
    # over its own lifetime, so it starts at peak rate when it joins.
    # Stage 0 can complete its own cosine before the run ends, so that resuming
    # to train a refiner does not restart or stretch a schedule stage 0 was
    # already partway through. Once its cosine reaches zero it is frozen by the
    # schedule rather than by a flag.
    s0_end = args.stage0_end_step or args.steps

    def stage0_lr(st: int) -> float:
        # Before the handoff, stage 0 runs its own cosine to completion. After it,
        # stage 0 tracks the refiner's schedule at a fixed fraction, so it keeps
        # some room to adapt while the refiner owns the correction -- rather than
        # sitting at exactly zero, which would make it a frozen feature extractor.
        if refiner is not None and st >= args.stage1_start_step:
            return args.stage0_lr_mult * cosine(st, args.stage1_start_step, args.steps)
        return cosine(st, 0, s0_end)

    lambdas = [stage0_lr]
    if refiner is not None:
        lambdas.append(lambda st: cosine(st, args.stage1_start_step, args.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambdas)
    print(f"encoder{'+refiner' if refiner else ''}: {n_par/1e6:.2f}M params, width {args.width}")
    if refiner:
        print(
            f"refiner joins at step {args.stage1_start_step}, correction scale "
            f"{args.refine_scale}, deep supervision {args.deep_supervision}\n"
            f"  lr schedules: stage0 cosine 0..{s0_end}"
            f"{f', then {args.stage0_lr_mult}x the refiner' if args.stage0_lr_mult != 1.0 else ''}, "
            f"refiner cosine {args.stage1_start_step}..{args.steps} "
            f"(warmup {args.warmup_steps})\n"
        )
    else:
        print()

    # Same trick as the fitter: a constant divisor keeps the objective at O(1)
    # for any loss in the registry without moving its optimum.
    loss_scale: Optional[float] = None
    hist = []
    best_nmse = float("inf")
    start_step = 1

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if refiner is not None and ck.get("refiner") is not None and not args.reset_refiner:
            refiner.load_state_dict(ck["refiner"])
            print("  refiner weights restored")
        start_step = int(ck["step"]) + 1
        scale = float(ck.get("scale", scale))
        # The objective is divided by a constant fixed on the first step. Letting
        # it be recomputed after a resume would rescale every gradient and
        # silently change the effective learning rate mid-run.
        if ck.get("loss_scale") is not None:
            loss_scale = float(ck["loss_scale"])
            print(f"  loss scale restored: {loss_scale:.4e}")
        else:
            print("  WARNING: checkpoint predates loss_scale saving; it will be "
                  "recomputed, which rescales the objective relative to the original run")
        if not args.reset_optimizer:
            try:
                opt.load_state_dict(ck["optimizer"])
                print("  optimizer state restored")
            except (ValueError, KeyError) as e:
                print(f"  WARNING: optimizer state not loadable ({e}); starting fresh. "
                      "This happens when the parameter groups differ, e.g. resuming a "
                      "single-stage checkpoint into a two-stage run.")
        hp = out_dir / "history.json"
        if hp.exists():
            try:
                hist = [r for r in json.load(hp.open())["history"] if r["step"] < start_step]
                done = [r["val_nmse_6d"] for r in hist if "val_nmse_6d" in r]
                best_nmse = min(done) if done else float("inf")
                print(f"  history continued from {len(hist)} rows, best so far {best_nmse:.4e}")
            except Exception:
                pass
        print(f"Resumed from {args.resume} at step {ck['step']}, continuing to {args.steps}")

    # The schedule is a pure function of step, so replaying it costs nothing and
    # avoids having to serialize scheduler state.
    for _ in range(start_step - 1):
        sched.step()

    t0 = time.time()

    def save(name: str, step: int) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "refiner": refiner.state_dict() if refiner is not None else None,
                "optimizer": opt.state_dict(),
                "step": step,
                "scale": scale,
                "loss_scale": loss_scale,
                # str() the Paths: torch>=2.6 loads with weights_only=True by
                # default, which refuses to unpickle PosixPath and makes the
                # checkpoint unreadable without opting out of the safety check.
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            },
            out_dir / name,
        )

    for step in range(start_step, args.steps + 1):
        idx = torch.randint(0, x_tr.shape[0], (args.batch_size,), device=x_tr.device)
        xb = _batch(x_tr, idx, device)
        two_stage = refiner is not None and step >= args.stage1_start_step
        if args.freeze_stage0 and two_stage:
            for prm in model.parameters():
                prm.requires_grad_(False)

        z0, x0, z1, x1 = two_stage_forward(model, refiner, cond, space, xb, scale, args, two_stage)
        loss0 = loss_fn(xb, x0)
        if two_stage:
            # Deep supervision on stage 0. Without it stage 0 drifts toward
            # "emit anything, the refiner will fix it", the residual stops being
            # an error signal, and the cascade collapses to one deeper stage.
            loss = loss_fn(xb, x1) + args.deep_supervision * loss0
        else:
            loss = loss0
        finite = torch.isfinite(loss)
        obj = torch.where(finite, loss, torch.zeros_like(loss)).mean()

        if loss_scale is None:
            loss_scale = max(float(obj.detach()), 1e-30)
            print(f"loss scale (fixed): {loss_scale:.4e}")

        opt.zero_grad(set_to_none=True)
        (obj / loss_scale).backward()
        # Clip each network separately: one shared norm would let stage 0's much
        # larger gradient decide how much the refiner's gets scaled.
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        if refiner is not None:
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), args.grad_clip)
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
                row.update(
                    evaluate(
                        model, space, z_va, x_va, args, loss_fn, scale,
                        refiner=refiner, cond=cond, two_stage=two_stage,
                    )
                )
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
                if spread < 0.05:
                    print(
                        f"           WARNING: prediction spread {spread:.3f} of ground truth -- "
                        f"the output has collapsed to a constant (saturated tanh); "
                        f"lower --lr or raise --warmup-steps and restart"
                    )
                stage0 = row.get("val_nmse_6d_stage0")
                extra = f"   stage0 6d {stage0:.3e}" if stage0 is not None else ""
                print(f"           corr(identifiable)  {corr}   mean spread/GT {spread:.2f}{extra}")
            else:
                print(f"step {step:6d}  train {tr:.4e}  |g| {gnorm:.2e}  [{row['elapsed_s']:.0f}s]")
            hist.append(row)
            if "val_nmse_6d" in row and row["val_nmse_6d"] < best_nmse:
                best_nmse = row["val_nmse_6d"]
                save("encoder_best.pt", step)
            if step % args.ckpt_every == 0:
                save("encoder_last.pt", step)
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

    save("encoder_last.pt", args.steps)
    print(f"best val NMSE_6d {best_nmse:.4e} (encoder_best.pt)")

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
        "--warmup-steps", type=int, default=2000,
        help="Linear warmup at the start of each stage, counted from that stage's "
             "own start step. Guards the tanh output against saturating early.",
    )
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
        "--input-mode", type=str, default="norm_amp", choices=["linear", "log", "norm_amp"],
        help="Input conditioning. norm_amp = log peak-normalized spectrogram plus log "
             "level as a second channel. This is a representation choice, independent "
             "of compression in the loss; state it separately in writeups.",
    )
    p.add_argument("--log-input", action="store_true", help="Deprecated alias for --input-mode log")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--data-device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Where the training set lives; cpu lifts the size limit from VRAM to RAM",
    )
    p.add_argument("--ckpt-every", type=int, default=5000, help="Steps between checkpoint writes")
    p.add_argument(
        "--stage1-start-step", type=int, default=0,
        help="Step at which the refiner joins; 0 disables it. Training stage 0 first "
             "matters: with the refiner present from the start its residual input is "
             "noise, and it learns to ignore it.",
    )
    p.add_argument(
        "--refine-scale", type=float, default=0.25,
        help="Correction magnitude. Unconstrained, the refiner re-predicts from "
             "scratch and the cascade becomes one deeper stage.",
    )
    p.add_argument("--deep-supervision", type=float, default=0.5, help="Weight on the stage-0 loss")
    p.add_argument(
        "--refine-attach", action="store_true",
        help="Backprop into stage 0 through the residual too (default: detached)",
    )
    p.add_argument(
        "--detach-stage0", action="store_true",
        help="Cut the final loss's gradient path into stage 0, so stage 0 trains only "
             "on its own loss and the refiner owns the correction. Without this the "
             "two stages optimize the same objective and stage 0, having a direct "
             "unscaled path, absorbs the improvement.",
    )
    p.add_argument("--resume", type=Path, default=None, help="Checkpoint to continue from")
    p.add_argument("--reset-optimizer", action="store_true", help="Ignore the saved optimizer state")
    p.add_argument("--reset-refiner", action="store_true", help="Reinitialize the refiner on resume")
    p.add_argument(
        "--stage0-end-step", type=int, default=0,
        help="Step at which stage 0's cosine reaches zero; 0 means --steps. Set this to "
             "the original run's --steps when resuming to train a refiner, so stage 0 "
             "finishes the schedule it was on instead of having it stretched.",
    )
    p.add_argument(
        "--stage0-lr-mult", type=float, default=1.0,
        help="Multiplier on stage 0's learning rate once the refiner joins. 0 freezes it, "
             "small values keep it nearly fixed while still letting it adapt.",
    )
    p.add_argument(
        "--freeze-stage0", action="store_true",
        help="Freeze stage 0 entirely once the refiner joins. Sharper test of what "
             "residual conditioning adds, at the cost of stage 0's further progress.",
    )
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--compile-plate", action="store_true")
    p.add_argument(
        "--batched-plate", action="store_true",
        help="Sum modes for the whole batch in one kernel instead of looping over the "
             "batch. Measured 3.0x at batch 96: the per-example loop is launch-bound "
             "and its throughput is flat in batch size.",
    )
    p.add_argument("--mode-bucket", type=int, default=1024)
    p.add_argument(
        "--chunk-elems", type=int, default=8_000_000,
        help="Time-chunk budget for the modal sum. Sized for the unfused path; with "
             "--compile-plate the intermediates never materialize, so a much larger "
             "value cuts the chunk count and the launch overhead that goes with it.",
    )
    p.add_argument(
        "--no-grad-checkpoint", action="store_true",
        help="Checkpointing makes backward recompute the forward. It bounds memory that "
             "fusion already bounds, so with --compile-plate it is mostly pure cost.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.log_input:
        args.input_mode = "log"
    run(args)


if __name__ == "__main__":
    main()
