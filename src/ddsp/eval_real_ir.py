"""Run trained plate encoders on REAL impulse responses and score the fit.

    python -m src.ddsp.eval_real_ir --list results/ddsp
    python -m src.ddsp.eval_real_ir --wav-dir data/EMT-140 \
        --arms results/ddsp/gamma_ppre/L1_STFT \
               results/ddsp/gamma_ppre/L1_STFT_hyb1e2 \
               results/ddsp/gamma_ppre/L1_STFT_eps1e7 \
        --render-duration 4.0 --prior 10

gamma_ppre is the pretrained family in the original parameter space
(train-p99): L1_STFT / hyb1e2 / eps1e7 are linear / hybrid / log off a shared
parameter-only base, the plate's counterpart to diffsynth's pre_base ->
magx_halfw / hybridx / logx_halfw. The plate's linear has always been on
MAGNITUDE (losses.py: torch.abs(torch.stft(...))), so L1_STFT is the magx
equivalent rather than a power loss, and g1 is the power one. Use --list to
find the rest; an arm called L1_STFT exists in six sweeps across two parameter
spaces and the name records neither.

Every plate number so far is against a SYNTHETIC target drawn from the same
seven-parameter model the encoder inverts, so the target is reachable by
construction and the only question is whether the encoder finds it. A real
EMT-140 plate is not reachable at all: it is a physical plate with a real
suspension, real damping plates, tube electronics and a room, described here by
seven numbers. This says how far off that is, and lets it be listened to.

WHAT IT DOES NOT DO. It does not fit. There is no optimisation here -- each
encoder makes one forward pass and the seven parameters it emits are rendered
as-is. A CMA-ES fit against the same audio would be a different and much
stronger test of the MODEL; this is a test of the ENCODERS, which is what the
losses under dispute produced.

ENCODE LENGTH AND RENDER LENGTH ARE SEPARATE, and only the first is fixed by
the model. The encoders were trained at 0.25 s and their conv stack expects
that many frames, so --duration defaults to the checkpoint's own value and
should stay there. But the seven parameters describe modes and decay rates,
so the RENDERER can produce any length from them: --render-duration 4.0
predicts from the first 0.25 s and renders four seconds.

That matters for listening more than for scoring. The first 0.25 s of a real
plate IR is mostly the strike -- a broadband crash the model has least to say
about -- while the modal ring the seven parameters actually describe is in the
tail. Judging the fit on 0.25 s judges it on the part it was never going to
get.

--prior N is the control for "it sounds nothing like a plate": N draws from the
synthetic prior, rendered, with no encoder and no real audio involved. If those
do not sound like a plate either, the encoders are not what is being heard and
neither are the losses.

TWO THINGS TO EXPECT, so they are not read as bugs:

  The encoder was trained on IRs from the synthetic prior. A real plate is out
  of that distribution in level, in decay shape, in the noise floor and in
  everything the model has no term for. Large errors are the expected result;
  the question is only whether they order the arms the same way the synthetic
  numbers did.

  mu = rho*h sets the absolute level and a real recording's gain is arbitrary,
  so both signals are peak-normalised before scoring. A dB-domain metric is
  otherwise dominated by its zeroth cepstral coefficient, which would be
  measuring the recording's fader position.

THE METRIC IS THE PLATE'S OWN, and its numbers are NOT comparable to the
diffsynth tables. losses.py::loss_mfcc is what every other plate result is
reported in: librosa's mel filterbank at 44.1 kHz, 128 mels, n_fft 2048,
hop 512, power -> 10*log10 -> orthonormal DCT-II, 20 coefficients. That is the
standard librosa/torchaudio pipeline and it has NO top_db floor.

mfcc_db80 is the same thing with the 80 dB floor the diffsynth headline metric
applies, reported beside it because the floor is not a detail: it decides
whether the quiet region counts at all, and a real plate's noise floor is
exactly the kind of content that sits down there. The two columns disagreeing
is informative rather than a problem.

Neither is comparable to a 16 kHz diffsynth number -- different rate, different
band, different mel count. Read down a column.

linmag is reported beside it for the reason it always is: mfcc rewards a
log-domain loss for optimising something structurally like what it measures,
and without an uncompressed control there is no way to tell that apart from a
better fit.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.ddsp.train_encoder import (                       # noqa: E402
    Encoder, CompositeConditioner, two_stage_forward,
)
from src.emt.band import brickwall_lowpass                  # noqa: E402
from src.gd.graddescent import (                           # noqa: E402
    SAMPLE_RATE, Raw7Space, norm_to_physical_torch, physical_to_plate14_torch,
)
# The class is BatchedModalPlateTorch; graddescent aliases it and every other
# caller uses the alias, so match them rather than inventing a third spelling.
from src.plate.SevenParamPlate import (                    # noqa: E402
    BatchedModalPlateTorch as SevenParamPlate,
)
from src.cmaes.fit_7param_norm_es import (                 # noqa: E402
    PARAM_BOUNDS, PARAM_KEYS,
)
# The PLATE's own mel filterbank and DCT, not ds_mfcc_check's. That module
# imports torchaudio, which dsenv has and dafxenv does not -- and there is no
# reason to add a dependency when losses.py already carries a librosa mel bank
# and an orthonormal DCT-II verified against scipy. Using the plate's own
# machinery also keeps this metric identical to loss_mfcc, which is what every
# other plate table reports.
from src.loss.losses import (                              # noqa: E402
    _get_dct, _get_mel_fb, _stft_mag, configure_loss_runtime,
)


P14 = list(SevenParamPlate.PARAM_ORDER)


def parse_fix(items):
    """--fix k=v pairs into {column index: value}, validated against PARAM_ORDER."""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--fix wants key=value, got {it!r}")
        k, v = it.split("=", 1)
        k = k.strip()
        if k not in P14:
            raise SystemExit(f"--fix {k!r} is not a plate parameter. The 14 are: "
                             + ", ".join(P14))
        out[P14.index(k)] = float(v)
    return out


def render(space, z, duration, fixes):
    """Render z, overriding named plate columns after packing.

    Applied to the packed 14-vector rather than to FIXED_PLATE_PARAMS, because
    that dict is consulted only for columns the space does NOT search --
    physical_to_plate14_torch takes `named[k] if k in named else FIXED[k]` -- so
    an override of a searched parameter (Ly, E, rho, h, T0, op_x, op_y in raw7)
    would be silently ignored there. Here every one of the fourteen can be
    pinned the same way.
    """
    phys = norm_to_physical_torch(z, space._lo, space._hi)
    p14 = physical_to_plate14_torch(phys)
    if fixes:
        p14 = p14.clone()
        for i, v in fixes.items():
            p14[:, i] = v
    return space.plate(p14, duration=duration, normalize=space.normalize)


class _Args:
    """ckpt['args'] as attributes, so two_stage_forward can read it back."""

    def __init__(self, d: dict):
        self.__dict__.update(d)


def load_wav(path: str, duration: float, sr: int,
             res_type: str | None) -> np.ndarray | None:
    """Mono, resampled to `sr`, first `duration` seconds, zero-padded.

    RESAMPLING IS SAFE HERE AND REINTERPRETING THE RATE IS NOT, which is the
    distinction an earlier version of this got wrong by refusing both. A
    bandlimited resampler preserves every partial's frequency in Hz -- a
    440 Hz mode is still at 440 Hz afterwards -- and mode POSITION is the whole
    signal the encoder reads. Simply relabelling 48 kHz samples as 44.1 kHz
    would shift every mode by 8.8%, which is what must never happen.

    --resample none restores the refusal, for checking that a rate conversion
    is not doing something to a result.
    """
    x, file_sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if file_sr != sr:
        if not res_type:
            print(f"  SKIP {os.path.basename(path)}: {file_sr} Hz, expected {sr}")
            return None
        import librosa
        x = librosa.resample(x, orig_sr=file_sr, target_sr=sr,
                             res_type=res_type).astype(np.float32)
    want = int(round(duration * sr))
    if x.shape[0] < want:
        x = np.pad(x, (0, want - x.shape[0]))
    return x[:want].copy()


def load_arm(run_dir: str, ckpt_name: str, device):
    """Encoder, renderer and the training-set input scale, from a checkpoint."""
    ck_path = os.path.join(run_dir, ckpt_name)
    if not os.path.exists(ck_path):
        return None, f"missing {ckpt_name}"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    a = _Args(ck["args"])
    model = Encoder(
        n_out=len(PARAM_KEYS), width=a.width, n_fft=a.n_fft, hop=a.hop,
        n_blocks=a.n_blocks, max_ch=a.max_ch, input_mode=a.input_mode,
        norm=a.norm, head_bound=a.head_bound,
        head_grad_floor=a.head_grad_floor, head_cap=a.head_cap,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    refiner = None
    if ck.get("refiner") is not None:
        refiner = Encoder(
            n_out=len(PARAM_KEYS), width=a.width, n_fft=a.n_fft, hop=a.hop,
            n_blocks=a.n_blocks, max_ch=a.max_ch, input_mode=a.input_mode,
            in_ch=6, n_extra=len(CompositeConditioner.KEYS), norm=a.norm,
            head_bound=a.head_bound, head_grad_floor=a.head_grad_floor,
            head_cap=a.head_cap,
        ).to(device)
        refiner.load_state_dict(ck["refiner"])
        refiner.eval()
    space = Raw7Space(device, torch.float32, normalize=False)
    # fmax comes from the checkpoint like every other numeric, and it has to:
    # emt7 trains at 12000 while the renderer's constructor default is 10000, so
    # omitting it renders a DIFFERENT PLATE from the one the encoder learned --
    # silently, and in exactly the octave a real EMT-140 is judged on. getattr
    # because raw7 checkpoints predate the flag; None there keeps their 10 kHz.
    space.configure_plate(a.chunk_elems, not a.no_grad_checkpoint,
                          a.batched_plate, a.compile_plate, a.mode_bucket,
                          a.fixed_mode_grid, getattr(a, "fmax", None))
    return (model, refiner, space, CompositeConditioner(device), a,
            float(ck["scale"]), int(ck.get("step", -1))), "ok"


def list_runs(root: str) -> None:
    """Every run directory with a checkpoint, and what it actually is.

    Reads each checkpoint's stored args rather than guessing from the path.
    An arm called L1_STFT exists in at least six different sweeps here, in two
    different parameter spaces, with and without parameter pretraining -- and
    comparing across those is not a comparison. torch.load with weights_only
    off is needed because train_encoder stores the args dict.
    """
    ck_paths = sorted(Path(root).rglob("encoder_*.pt"))
    if not ck_paths:
        print(f"no encoder_*.pt under {root}")
        return
    seen = {}
    for q in ck_paths:
        seen.setdefault(str(q.parent), []).append(q.name)
    print(f"{len(seen)} run dir(s) with checkpoints under {root}\n")
    print(f"{'step':>8}  {'train data':<22}{'loss':<18}{'ppre':<6}"
          f"{'ckpt':<6}  dir")
    rows = []
    for d, names in seen.items():
        pick = "encoder_last.pt" if "encoder_last.pt" in names else names[0]
        try:
            ck = torch.load(os.path.join(d, pick), map_location="cpu",
                            weights_only=False)
        except Exception as e:
            print(f"{'?':>8}  {'?':<22}{'unreadable':<18}{'?':<6}{'?':<6}"
                  f"  {d}   ({type(e).__name__})")
            continue
        a = ck.get("args", {})
        # param_w > 0 with a hold fraction is the parameter pretraining; the
        # run names do not record it and two sweeps here differ by only that.
        ppre = "yes" if float(a.get("param_w", 0) or 0) > 0 else "no"
        dd = a.get("data_dir")
        rows.append((int(ck.get("step", -1)),
                     os.path.basename(str(dd)) if dd else "synthetic",
                     str(a.get("loss", "?")), ppre,
                     ("last" if "encoder_last.pt" in names else "best"), d))
    for r in sorted(rows, key=lambda r: r[5]):
        print(f"{r[0]:>8}  {r[1]:<22}{r[2]:<18}{r[3]:<6}{r[4]:<6}  {r[5]}")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--list", nargs="?", const="results", default=None,
                   metavar="ROOT",
                   help="List every run directory under ROOT holding a "
                        "checkpoint, with its step, parameter space, loss and "
                        "whether it was parameter-pretrained -- then exit. "
                        "Run names alone do not say which parameter space an "
                        "arm belongs to, and mixing spaces is not a comparison; "
                        "this reads it out of each checkpoint's own args. The "
                        "space is identified by the TRAINING SET, since it is "
                        "set by the PLATE_PARAM_SPACE environment variable at "
                        "import time and never lands in the args dict -- "
                        "data/train-quiet7 is the quiet7 space, anything else "
                        "is the original.")
    p.add_argument("--wav-dir", help="a directory of .wav IRs")
    p.add_argument("--arms", nargs="+", default=None,
                   help="run directories holding the checkpoints. Required "
                        "unless --list, which only enumerates them.")
    p.add_argument("--ckpt", default="encoder_last.pt",
                   help="encoder_last.pt is this project's checkpoint: every "
                        "--resume in the jobs files reads it, so it is the "
                        "state every downstream arm was branched from.\n"
                        "encoder_best.pt is selected on VALIDATION and lands at "
                        "very different steps per arm -- 6000 for quiet7's "
                        "hyb1e4 and eps1e4 against 40000 for L1_STFT -- so it "
                        "compares each arm at its own peak rather than at equal "
                        "training. Useful as a cross-check, misleading as a "
                        "default.")
    p.add_argument("--duration", type=float, default=None,
                   help="Seconds of IR the ENCODER sees. Defaults to the "
                        "checkpoint's own training duration and should stay "
                        "there: the conv stack was trained on that many frames "
                        "and its input statistics change with the length.")
    p.add_argument("--render-duration", type=float, default=None,
                   help="Seconds to RENDER and score, from the parameters the "
                        "encoder predicted. Defaults to --duration.\n"
                        "These are separable because only the encoder is length-"
                        "bound: the seven parameters describe modes and decay "
                        "rates, so the renderer can produce any length from "
                        "them. Predicting from 0.25 s and rendering 4 s is the "
                        "honest way to hear the tail without feeding the "
                        "encoder something it was never trained on -- and the "
                        "first 0.25 s of a real plate IR is mostly the strike, "
                        "which is the part the model has least to say about.")
    p.add_argument("--fix", nargs="+", default=None, metavar="KEY=VALUE",
                   help="Pin plate parameters to these values for every render, "
                        "searched or not. Any of the fourteen: "
                        "Lx Ly h T0 rho E nu T60_DC T60_F1 loss_F1 fp_x fp_y "
                        "op_x op_y.\n"
                        "Applied to the packed 14-vector, so it overrides a "
                        "SEARCHED parameter too -- setting FIXED_PLATE_PARAMS "
                        "would not, since that dict is only read for columns "
                        "the space does not search.\n"
                        "Pinning a searched parameter makes the encoder's "
                        "prediction for it inert; with --prior it simply "
                        "narrows what is drawn.")
    p.add_argument("--prior", type=int, default=0, metavar="N",
                   help="Also write N IRs drawn from the synthetic PRIOR -- no "
                        "encoder, no real audio, just the parameter space the "
                        "encoders invert, rendered.\n"
                        "This is the control for 'the output sounds nothing "
                        "like a plate'. If the prior itself does not sound like "
                        "one, no encoder can, and the losses are not what is "
                        "being heard. Written as prior_NN.wav.")
    p.add_argument("--prior-seed", type=int, default=0,
                   help="Seed for the prior draw. Fixed by default so two "
                        "runs with different --fix values sample the SAME "
                        "seven parameters and the only audible difference is "
                        "the pinning.")
    p.add_argument("--resample", default="soxr_hq",
                   help="Resampler for files not already at the model's rate; "
                        "'none' skips them instead. A bandlimited resample "
                        "preserves mode frequencies in Hz, so it is safe; "
                        "relabelling the rate would shift every mode by 8.8% "
                        "for 48k -> 44.1k and is what this must never do.")
    p.add_argument(
        "--lowpass-input", type=float, default=None, metavar="HZ",
        help="Band-limit the audio the ENCODER READS to this, normally the "
             "checkpoint's --fmax. The encoder is on a linear axis of 1025 bins "
             "over 22.05 kHz, and 467 of them -- 46%% -- sit above a 12 kHz "
             "ceiling. Every training render is silent there, so across nearly "
             "half the input width the network only ever saw the log floor, "
             "while a real 44.1 kHz IR puts live signal in all of it. If that "
             "shift is what collapses the predictions to a constant, this fixes "
             "it with no retraining, and the PREDICTED PARAMETERS table is the "
             "place it shows: spread appearing where there was none.")
    p.add_argument(
        "--lowpass-score", type=float, default=None, metavar="HZ",
        help="Band-limit the TARGET before scoring and before writing it, "
             "normally the checkpoint's --fmax. Roughly 19 of loss_mfcc's 128 "
             "mel bands sit above a 12 kHz ceiling: floored in every render, "
             "live in both targets of the saturation reference, so every "
             "arm/saturation figure carries a constant penalty for a band "
             "nothing can reach. SEPARATE from --lowpass-input on purpose -- "
             "they change different things, and one flag doing both would make "
             "any movement in the numbers unattributable.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="real_ir_eval")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-fft", type=int, default=2048,
                   help="Metric FFT, matching loss_mfcc. At --duration 0.25 "
                        "that is 11025 samples, about 21 frames at hop 512 -- "
                        "few, but the model only renders 0.25 s.")
    p.add_argument("--hop", type=int, default=512)
    p.add_argument("--n-mels", type=int, default=128,
                   help="128, matching losses.py::loss_mfcc, not the 40 the "
                        "diffsynth metric uses at 16 kHz.")
    p.add_argument("--n-mfcc", type=int, default=20)
    p.add_argument("--no-tar", action="store_true")
    args = p.parse_args()

    if args.list:
        list_runs(args.list)
        return
    missing = [f for f, v in (("--wav-dir", args.wav_dir), ("--arms", args.arms))
               if not v]
    if missing:
        p.error(f"{' and '.join(missing)} required (or use --list alone)")

    dev = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    wavs = sorted(str(q) for q in Path(args.wav_dir).rglob("*.wav"))
    if not wavs:
        raise SystemExit(f"no .wav under {args.wav_dir}")
    if args.limit:
        wavs = wavs[: args.limit]
    print(f"{len(wavs)} wav(s) under {args.wav_dir}")

    configure_loss_runtime(SAMPLE_RATE, dev)
    win = torch.hann_window(args.n_fft, device=dev)

    def mfcc(x, top_db=None):
        """loss_mfcc's pipeline, with the dB floor optional.

        Kept as one function with a flag rather than two copies, so the floored
        and unfloored columns cannot drift into being different transforms with
        different mel banks.
        """
        fb = _get_mel_fb(args.n_fft, args.n_mels)
        dct = _get_dct(args.n_mfcc, args.n_mels)
        mel = torch.matmul(fb.unsqueeze(0), _stft_mag(x, args.n_fft, args.hop) ** 2)
        db = 10.0 * torch.log10(mel + 1e-10)
        if top_db is not None:
            db = torch.maximum(db, db.amax(dim=(-2, -1), keepdim=True) - top_db)
        return torch.matmul(dct.unsqueeze(0), db)

    def linmag(x):
        return torch.stft(x, args.n_fft, hop_length=args.hop, window=win,
                          center=True, return_complex=True).abs()

    def peak_norm(x):
        # mu sets the absolute level and a recording's gain is arbitrary, so a
        # dB metric would otherwise be reporting the fader position through its
        # zeroth cepstral coefficient.
        return x / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)

    loaded, target = {}, None
    for arm in args.arms:
        got, note = load_arm(arm, args.ckpt, dev)
        if got is None:
            print(f"{Path(arm).name:<26} skipped: {note}")
            continue
        loaded[Path(arm).name] = got
        print(f"{Path(arm).name:<26} step {got[6]}   input scale {got[5]:.4g}")
    steps = {n: g[6] for n, g in loaded.items()}
    if len(set(steps.values())) > 1:
        print(f"\n  NOTE: the arms are at different steps -- "
              f"{', '.join(f'{n} {s}' for n, s in steps.items())}.")
        print("  With --ckpt encoder_last.pt this should not happen: the arms")
        print("  run the same number of steps. If it does, an arm died early.")
        print("  With encoder_best.pt it is expected -- best is selected on")
        print("  validation, so an early best means the arm peaked and got")
        print("  worse, which is a real property but not equal training.\n")
    if not loaded:
        raise SystemExit("no arm loaded")

    dur = args.duration
    if dur is None:
        dur = float(next(iter(loaded.values()))[4].duration)
        print(f"encode duration {dur}s (from the checkpoint)")
    # Say out loud which plate is being rendered. Every one of these is silent
    # when wrong -- a mismatched ceiling or pin produces plausible audio at the
    # wrong parameters, and there is no error anywhere.
    _a = next(iter(loaded.values()))[4]
    _fmax = getattr(_a, "fmax", None) or 10000.0
    print(f"renderer: fmax {_fmax}  "
          f"pin {getattr(_a, 'fixed_mode_grid', None)}  "
          f"space {os.environ.get('PLATE_PARAM_SPACE', 'raw7')}")
    print(f"lowpass:  input {args.lowpass_input or 'off'}  "
          f"score {args.lowpass_score or 'off'}")
    for _nm, _v in (("--lowpass-input", args.lowpass_input),
                    ("--lowpass-score", args.lowpass_score)):
        if _v and abs(_v - _fmax) > 1.0:
            print(f"  NOTE {_nm} {_v:.0f} != the renderer's fmax {_fmax:.0f}. "
                  f"The point of both is to match the band the renders occupy.")
    rdur = args.render_duration if args.render_duration else dur
    if rdur != dur:
        print(f"render/score duration {rdur}s -- parameters predicted from the "
              f"first {dur}s, rendered out to {rdur}s")

    fixes = parse_fix(args.fix)
    if fixes:
        print("  pinned: " + ", ".join(f"{P14[i]}={v:g}"
                                       for i, v in sorted(fixes.items())))

    if args.prior:
        _m, _r, space, _c, _a, _s, _st = next(iter(loaded.values()))
        # synth_dataset's own sampling, so the draws are the training
        # distribution, but rendered through the override path.
        g = torch.Generator(device="cpu").manual_seed(args.prior_seed)
        z = (torch.rand((args.prior, len(PARAM_KEYS)), generator=g)
             * 2.0 - 1.0).to(dev)
        with torch.no_grad():
            x = torch.cat([render(space, z[i:i + 8], rdur, fixes).float()
                           for i in range(0, args.prior, 8)], dim=0)
        for i in range(x.shape[0]):
            sf.write(os.path.join(args.out, f"prior_{i:02d}.wav"),
                     peak_norm(x[i:i + 1])[0].cpu().numpy(), SAMPLE_RATE)
        print(f"wrote {args.prior} prior draws (no encoder, no real audio) "
              f"to {args.out}/prior_*.wav")

    rows, names, tgts, preds = {}, [], {}, {}
    for wi, w in enumerate(wavs):
        rt = None if args.resample.lower() == "none" else args.resample
        x = load_wav(w, dur, SAMPLE_RATE, rt)
        if x is None:
            continue
        # The window the encoder reads and the window everything is scored and
        # written over are the same array only when the two durations agree.
        xr = x if rdur == dur else load_wav(w, rdur, SAMPLE_RATE, rt)
        stem = Path(w).stem
        names.append(stem)
        tgt = torch.from_numpy(x)[None, :].to(dev)          # encoder input
        tgt_r = torch.from_numpy(xr)[None, :].to(dev)       # scored + written
        tgt = brickwall_lowpass(tgt, args.lowpass_input, SAMPLE_RATE)
        tgt_r = brickwall_lowpass(tgt_r, args.lowpass_score, SAMPLE_RATE)
        # Peak AFTER band-limiting: the level the metric sees should be the
        # level of the band being compared, not of a transient partly removed.
        tgts[stem] = peak_norm(tgt_r).float()
        sf.write(os.path.join(args.out, f"{stem}__target.wav"),
                 tgts[stem][0].cpu().numpy(), SAMPLE_RATE)
        for name, (model, refiner, space, cond, a, scale, _s) in loaded.items():
            with torch.no_grad():
                two = refiner is not None
                z0, x0, z1, _x1 = two_stage_forward(
                    model, refiner, cond, space, tgt, scale, a, two)
                # Render from the PREDICTED PARAMETERS at the scoring length,
                # rather than reusing two_stage_forward's own render, which is
                # fixed at the encoder's training duration.
                z = z1 if two else z0
                pred = (render(space, z, rdur, fixes)
                        if (fixes or rdur != dur) else (_x1 if two else x0))
            tn, pn = tgts[stem], peak_norm(pred).float()
            l1 = torch.nn.functional.l1_loss
            m = {"linmag": l1(linmag(tn), linmag(pn)).item(),
                 "mfcc": l1(mfcc(tn), mfcc(pn)).item(),
                 "mfcc_db80": l1(mfcc(tn, 80.0), mfcc(pn, 80.0)).item()}
            rows.setdefault(name, {})[stem] = m
            # The predicted PHYSICAL parameters, which are what the tilt and
            # decay tables are ultimately about: a render is dark because the
            # encoder chose a low loss_F1, not because the renderer is dark.
            preds.setdefault(name, {})[stem] = dict(zip(
                PARAM_KEYS,
                norm_to_physical_torch(z, space._lo, space._hi)[0].cpu().tolist()))
            sf.write(os.path.join(args.out, f"{stem}__{name}.wav"),
                     pn[0].cpu().numpy(), SAMPLE_RATE)
        print(f"  [{wi + 1}/{len(wavs)}] {stem}")

    if not names:
        raise SystemExit("nothing scored -- check the sample rate lines above; "
                         "--resample none refuses files at another rate")

    # THE ONLY THING THAT MAKES THESE NUMBERS READABLE. A distance of 60 on a
    # dB-domain cepstrum means nothing on its own -- it could be a total failure
    # or the ordinary scale of this metric on this audio. So score every target
    # against ANOTHER REAL IR, which is what an encoder conveying nothing would
    # amount to. An arm at the saturation level has told us nothing; an arm well
    # below it has, however large its absolute number looks.
    #
    # Rolled by one rather than random pairs: the set is 15 clips of the same
    # plate at three brightnesses, so consecutive names are usually the same
    # brightness -- a HARDER reference than random pairing, and the conservative
    # direction for a claim that an arm beats it.
    l1 = torch.nn.functional.l1_loss
    sat = {"linmag": [], "mfcc": [], "mfcc_db80": []}
    for i, stem in enumerate(names):
        a, b = tgts[stem], tgts[names[(i + 1) % len(names)]]
        sat["linmag"].append(l1(linmag(a), linmag(b)).item())
        sat["mfcc"].append(l1(mfcc(a), mfcc(b)).item())
        sat["mfcc_db80"].append(l1(mfcc(a, 80.0), mfcc(b, 80.0)).item())

    for key in ("mfcc", "mfcc_db80", "linmag"):
        print(f"\n=== {key}   (peak-normalised both sides; lower is better)")
        hdr = "".join(f"{n[:22]:>24}" for n in loaded)
        print(f"{'ir':<34}{hdr}")
        for stem in names:
            print(f"{stem[:34]:<34}"
                  + "".join(f"{rows[n][stem][key]:>24.4f}" for n in loaded))
        print(f"{'MEAN':<34}"
              + "".join(f"{np.mean([rows[n][s][key] for s in names]):>24.4f}"
                        for n in loaded))
        sm = float(np.mean(sat[key]))
        print(f"{'SATURATION (another real IR)':<34}{sm:>24.4f}")
        print(f"{'  arm / saturation':<34}"
              + "".join(f"{np.mean([rows[n][s][key] for s in names]) / sm:>24.3f}"
                        for n in loaded))

    print(f"\nwrote {args.out}: {len(names)} target + "
          f"{len(names) * len(loaded)} resynthesis wavs, all peak-normalised")
    if not args.no_tar:
        base = os.path.basename(os.path.normpath(args.out))
        tar = os.path.normpath(args.out) + ".tar.gz"
        subprocess.run(["tar", "czf", tar, "-C",
                        os.path.dirname(os.path.abspath(args.out)) or ".", base],
                       check=True)
        print(f"bundled: {tar}  ({os.path.getsize(tar) / 1024 / 1024:.1f} MB)")

    # The predicted parameters, which is where an audible difference between
    # two arms on the same audio actually lives. A render is dark because the
    # encoder picked a low loss_F1, not because the renderer is dark, and no
    # spectral table says which parameter moved.
    print("\n=== PREDICTED PARAMETERS   median over IRs, and the spread")
    print("  The bounds are the space's own, so a median pinned at a bound means")
    print("  the encoder wanted to leave the space and could not.")
    lo = {k: PARAM_BOUNDS[k][0] for k in PARAM_KEYS}
    hi = {k: PARAM_BOUNDS[k][1] for k in PARAM_KEYS}
    print(f"  {'param':>10}{'bounds':>22}" + "".join(f"{n:>22}" for n in rows))
    for k in PARAM_KEYS:
        cells = ""
        for n in rows:
            v = np.array([preds[n][st][k] for st in names], dtype=float)
            cells += f"{np.median(v):>13.4g}{'':1}[{v.min():.3g},{v.max():.3g}]".rjust(22)
        print(f"  {k:>10}{f'[{lo[k]:.3g},{hi[k]:.3g}]':>22}" + cells)
    csvp = os.path.join(args.out, "predicted_params.csv")
    with open(csvp, "w") as fh:
        fh.write("arm,ir," + ",".join(PARAM_KEYS) + "\n")
        for n in rows:
            for st in names:
                fh.write(f"{n},{st}," +
                         ",".join(f"{preds[n][st][k]:.8g}" for k in PARAM_KEYS) + "\n")
    print(f"  per-IR values: {csvp}")

    print("\n  READ arm/saturation, NOT the absolute number. Saturation is the")
    print("  distance between two different real IRs -- what an encoder that")
    print("  conveyed nothing about the target would score. Near 1.0 means the")
    print("  arm is telling us nothing on this audio however large or small its")
    print("  raw value looks; well below 1.0 means it is, whatever the scale of")
    print("  the metric happens to be.")
    print("\n  ONE FORWARD PASS PER ARM, no fitting. These are the encoders'")
    print("  answers, not the seven-parameter model's best case -- a CMA-ES fit")
    print("  against the same audio would be a much stronger test OF THE MODEL")
    print("  and a different question from the one the losses are being judged")
    print("  on. Expect large errors either way: a physical plate has a")
    print("  suspension, damping plates, tube electronics and a room, none of")
    print("  which exists in seven parameters. What is worth reading is whether")
    print("  the arms ORDER the same way they do on synthetic targets.")


if __name__ == "__main__":
    main()
