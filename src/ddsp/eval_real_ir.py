"""Run trained plate encoders on REAL impulse responses and score the fit.

    python -m src.ddsp.eval_real_ir --wav-dir data/EMT-140 \
        --arms results/ddsp/quiet7/L1_STFT_hyb1e4 \
               results/ddsp/quiet7/L1_STFT_eps1e4 \
               results/ddsp/quiet7/L1_STFT
    python -m src.ddsp.eval_real_ir --wav-dir data/EMT-140 --arms ... --limit 8

The quiet7 family is the pretrained one: quiet7_pre/L1_STFT branches into
quiet7/<arm>, so hyb1e4 / eps1e4 / L1_STFT are hybrid / log / linear off a
shared base -- the plate's counterpart to diffsynth's pre_base -> hybridx /
logx_halfw / magx_halfw. The plate's linear has always been on MAGNITUDE
(losses.py: torch.abs(torch.stft(...))), so L1_STFT is the magx equivalent
rather than a power loss. quiet7_floor/ holds the same three at a -40 dB hard
floor.

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

ONLY THE FIRST --duration SECONDS ARE SCORED, and that is the model's limit,
not a choice made here. The encoders were trained at 0.25 s, so the renderer
produces 0.25 s and the comparison covers the onset and the first part of the
decay of a 2-6 s recording -- not the tail, which is where a plate's character
largely lives. The default reads the value out of the checkpoint so it cannot
silently disagree with what the encoder expects.

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
from src.gd.graddescent import SAMPLE_RATE, Raw7Space      # noqa: E402
from src.cmaes.fit_7param_norm_es import PARAM_KEYS        # noqa: E402
# The PLATE's own mel filterbank and DCT, not ds_mfcc_check's. That module
# imports torchaudio, which dsenv has and dafxenv does not -- and there is no
# reason to add a dependency when losses.py already carries a librosa mel bank
# and an orthonormal DCT-II verified against scipy. Using the plate's own
# machinery also keeps this metric identical to loss_mfcc, which is what every
# other plate table reports.
from src.loss.losses import (                              # noqa: E402
    _get_dct, _get_mel_fb, _stft_mag, configure_loss_runtime,
)


class _Args:
    """ckpt['args'] as attributes, so two_stage_forward can read it back."""

    def __init__(self, d: dict):
        self.__dict__.update(d)


def load_wav(path: str, duration: float, sr: int) -> np.ndarray | None:
    """Mono, native rate only, first `duration` seconds, zero-padded.

    Deliberately refuses to resample. An EMT-140 IR set is normally 44.1 kHz
    already, and silently resampling a file that is not would change the mode
    frequencies -- which are the entire signal the encoder reads.
    """
    x, file_sr = sf.read(path, dtype="float32", always_2d=True)
    if file_sr != sr:
        print(f"  SKIP {os.path.basename(path)}: {file_sr} Hz, expected {sr}")
        return None
    x = x.mean(axis=1)
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
    space.configure_plate(a.chunk_elems, not a.no_grad_checkpoint,
                          a.batched_plate, a.compile_plate, a.mode_bucket,
                          a.fixed_mode_grid)
    return (model, refiner, space, CompositeConditioner(device), a,
            float(ck["scale"]), int(ck.get("step", -1))), "ok"


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--wav-dir", required=True, help="a directory of .wav IRs")
    p.add_argument("--arms", nargs="+", required=True,
                   help="run directories holding the checkpoints")
    p.add_argument("--ckpt", default="encoder_best.pt",
                   help="encoder_best.pt or encoder_last.pt -- those are the "
                        "two names train_encoder writes.")
    p.add_argument("--duration", type=float, default=None,
                   help="Seconds of IR to model. Defaults to whatever the "
                        "checkpoint was trained at, which is what the encoder "
                        "and the renderer both assume.")
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
    if not loaded:
        raise SystemExit("no arm loaded")

    dur = args.duration
    if dur is None:
        dur = float(next(iter(loaded.values()))[4].duration)
        print(f"duration {dur}s (from the checkpoint)")

    rows, names = {}, []
    for wi, w in enumerate(wavs):
        x = load_wav(w, dur, SAMPLE_RATE)
        if x is None:
            continue
        stem = Path(w).stem
        names.append(stem)
        tgt = torch.from_numpy(x)[None, :].to(dev)
        sf.write(os.path.join(args.out, f"{stem}__target.wav"),
                 peak_norm(tgt)[0].cpu().numpy(), SAMPLE_RATE)
        for name, (model, refiner, space, cond, a, scale, _s) in loaded.items():
            with torch.no_grad():
                two = refiner is not None
                _z0, x0, z1, x1 = two_stage_forward(
                    model, refiner, cond, space, tgt, scale, a, two)
                pred = x1 if two else x0
            tn, pn = peak_norm(tgt).float(), peak_norm(pred).float()
            l1 = torch.nn.functional.l1_loss
            m = {"linmag": l1(linmag(tn), linmag(pn)).item(),
                 "mfcc": l1(mfcc(tn), mfcc(pn)).item(),
                 "mfcc_db80": l1(mfcc(tn, 80.0), mfcc(pn, 80.0)).item()}
            rows.setdefault(name, {})[stem] = m
            sf.write(os.path.join(args.out, f"{stem}__{name}.wav"),
                     pn[0].cpu().numpy(), SAMPLE_RATE)
        print(f"  [{wi + 1}/{len(wavs)}] {stem}")

    if not names:
        raise SystemExit("nothing scored -- check the sample rate line above")

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

    print(f"\nwrote {args.out}: {len(names)} target + "
          f"{len(names) * len(loaded)} resynthesis wavs, all peak-normalised")
    if not args.no_tar:
        base = os.path.basename(os.path.normpath(args.out))
        tar = os.path.normpath(args.out) + ".tar.gz"
        subprocess.run(["tar", "czf", tar, "-C",
                        os.path.dirname(os.path.abspath(args.out)) or ".", base],
                       check=True)
        print(f"bundled: {tar}  ({os.path.getsize(tar) / 1024 / 1024:.1f} MB)")

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
