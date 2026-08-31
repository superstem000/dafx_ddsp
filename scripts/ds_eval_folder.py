"""Score an arbitrary folder of real audio with several arms, no datamodule.

    python scripts/ds_eval_folder.py --dirs data/juno/open_pad data/juno/pulse_bass \
        --arms synth_magx_halfw synth_hybridx synth_logx_halfw --n 50

    python scripts/ds_eval_folder.py --dirs data/juno/* --arms ... \
        --metrics mfcc db80 g0.3 --device cuda:0

WHY THIS EXISTS RATHER THAN ds_ood_subset. Every other evaluator here reaches
its audio through IdOodDataModule, and two things in it make a small folder
impossible to score:

  data.py:92 is `np.random.choice(len(ood_dat), len(id_dat), replace=False)`,
  so the out-of-domain directory must hold AT LEAST as many files as the
  in-domain one -- 20000 for the paper's runs. A 150-clip sample pack raises
  there before the first batch, with a numpy error that says nothing about
  what is wrong.

  Even past that, create_split puts 80% of the subsample in `train` and only
  10% in `valid`, and `valid` is what the evaluators read. Scoring 150 clips
  would report 15 of them.

So this loads the checkpoints the same way -- pl.seed_everything(0), construct
EstimatorSynth from the run's own .hydra/config.yaml, load_state_dict -- and
then feeds them audio it reads itself. Nothing about the split RNG matters here
because there is no split: every file selected is scored.

THE METRIC IS THE UNFLOORED STANDARD MFCC, and that is the default because it
is what the headline result is written in. DbCepstrum with top_db=None: Hann
window, Slaney mel scale with Slaney filterbank normalisation, n_fft 1024 /
hop 256, 40 mels over 40-7600 Hz, 10*log10(mel + 1e-10) with NO clamp, 20 ortho
DCT coefficients. The floored variants are available as db80/db70 and the
power-cepstrum rung as g0.3, but they are extras -- pass --metrics to add them.

CLIPS SHORTER THAN --length ARE ZERO-PADDED, where WaveParamDataset asserts.
data.py:49 is `assert audio.shape[0] == self.length * self.sample_rate`, which
is true of NSynth and false of almost any sample library: a 0.5 s bass hit
raises. Padding is the right call -- the synthetic training set contains clips
whose ADSR reaches zero early and sits in silence for the rest of the window,
so short-note-plus-silence is in distribution -- but it is not free, and the
ACTIVE column reports what fraction of each folder's window carries signal. A
folder at 15% active is being measured mostly in silence, which is precisely
the region the arms are supposed to differ in. Read the scores next to it.

LEVELS ARE NOT NORMALISED, matching WaveParamDataset, which passes librosa.load
straight through. Amplitude is a parameter the estimator predicts, so
normalising here would hand it information the training data never had. PEAK
reports each folder's median so a pack recorded 20 dB down is visible rather
than mysterious.

NORMALISED BY AN UNRELATED PAIR FROM THE SAME FOLDER, the same `saturation`
denominator ds_ood_subset and the plate work use: the batch rolled by one, so
the partner is another clip of the same kind. A raw L1 on MFCCs sits on
whatever scale that material's cepstrum occupies, and a folder with more
spectral variation scores worse at equal relative fit. Dividing by the distance
between two unrelated clips of the same folder gives "fraction of the distance
between two clips of this kind": 0 is exact, 1 is no better than picking
another clip of the pack at random. --norm mean gives the raw per-coefficient
L1 instead, which is comparable across arms but not across folders.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "external", "diffsynth")
sys.path.insert(0, DS)
sys.path.insert(0, HERE)

import numpy as np                                       # noqa: E402
import torch                                             # noqa: E402
import pytorch_lightning as pl                           # noqa: E402
from omegaconf import OmegaConf                          # noqa: E402

from diffsynth.model import EstimatorSynth               # noqa: E402
import ds_mfcc_check as mc                               # noqa: E402

AUDIO_EXT = (".wav", ".aiff", ".aif", ".flac", ".ogg", ".mp3")


def audio_files(d: str) -> list[str]:
    """Every audio file under d, recursively, in a stable order.

    Recursive because sample packs arrive with the audio one level down, and
    because `<dir>/audio/` is where WaveParamDataset would have looked -- so a
    directory prepared for the datamodule works here unchanged.
    """
    out = []
    for root, _dirs, names in os.walk(d):
        out += [os.path.join(root, n) for n in names
                if n.lower().endswith(AUDIO_EXT)]
    return sorted(out)


def load_clip(path: str, sr: int, length: float):
    """One mono clip at sr, exactly length seconds, plus what was there before.

    Returns (audio, active_fraction, peak). `active` is the fraction of the
    padded window whose sample magnitude is within 60 dB of the clip's own
    peak, measured BEFORE padding is counted as signal -- the number that says
    whether a score is about the note or about the silence after it.
    """
    import librosa
    n = int(round(length * sr))
    y, _ = librosa.load(path, sr=sr, mono=True, duration=length)
    peak = float(np.abs(y).max()) if y.size else 0.0
    if peak > 0:
        active = float((np.abs(y) >= peak * 10.0 ** (-60.0 / 20.0)).sum()) / n
    else:
        active = 0.0
    if y.shape[0] < n:
        y = np.pad(y, (0, n - y.shape[0]))
    return y[:n], active, peak


def load_model(run_dir: str, ckpt_name: str, device: str):
    """EstimatorSynth from a run directory, in train.py's construction order.

    The datamodule is deliberately absent -- see the module docstring -- so the
    seed here only has to make the architecture and its weight shapes right,
    not reproduce any split.
    """
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    ck_path = os.path.join(run_dir, "tb_logs", "checkpoints", ckpt_name)
    if not os.path.exists(cfg_path):
        return None, f"missing .hydra/config.yaml"
    if not os.path.exists(ck_path):
        return None, f"missing tb_logs/checkpoints/{ckpt_name}"
    cfg = OmegaConf.load(cfg_path)
    try:
        # weights_only=False for the same reason train.py needs it: the
        # checkpoint carries the hydra DictConfig via save_hyperparameters().
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return None, f"unreadable checkpoint ({type(e).__name__})"
    pl.seed_everything(0, workers=True)
    model = EstimatorSynth(cfg.model, cfg.synth, cfg.schedule)
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, f"epoch {ck.get('epoch', '?')}"


def build_metrics(names: list[str], device: str, sr: int):
    """name -> callable(audio) -> cepstrum, all on this run's frame grid."""
    table = {
        # THE DEFAULT. Unfloored: DbCepstrum with top_db=None applies no clamp
        # at all, so nothing below the peak is discarded and the quiet region
        # -- the whole subject of the comparison -- is measured rather than
        # thrown away. Every floored variant below is a deliberate extra.
        "mfcc": lambda: mc.make_mfcc(device, window="hann", log="db",
                                     top_db=None, mel_norm="slaney",
                                     mel_scale="slaney", sr=sr),
        "db80": lambda: mc.make_mfcc(device, window="hann", log="db",
                                     top_db=80.0, mel_norm="slaney",
                                     mel_scale="slaney", sr=sr),
        "db70": lambda: mc.make_mfcc(device, window="hann", log="db",
                                     top_db=70.0, mel_norm="slaney",
                                     mel_scale="slaney", sr=sr),
        "g0.3": lambda: mc.make_mfcc(device, window="hann", log="pow",
                                     gamma=0.3, mel_norm="slaney",
                                     mel_scale="slaney", sr=sr),
    }
    bad = [m for m in names if m not in table]
    if bad:
        raise SystemExit(f"unknown metric(s) {', '.join(bad)}; "
                         f"available: {', '.join(table)}")
    return {m: table[m]() for m in names}


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dirs", nargs="+", required=True, metavar="DIR",
                   help="Audio folders, scored and reported separately. The "
                        "basename names the row group.")
    p.add_argument("--arms", nargs="+", required=True,
                   help="Run directory names under --root.")
    p.add_argument("--root", default="results/diffsynth")
    p.add_argument("--ckpt", default="latest.ckpt")
    p.add_argument("--n", type=int, default=50,
                   help="Clips sampled per folder. 0 takes all of them.")
    p.add_argument("--seed", type=int, default=0,
                   help="Selects WHICH clips, and the same ones for every arm "
                        "-- the sample is drawn once, before any model loads.")
    p.add_argument("--metrics", nargs="+", default=["mfcc"],
                   help="Default is the unfloored standard MFCC, which is what "
                        "the headline result is written in. db80/db70/g0.3 add "
                        "floored and power-cepstrum rungs.")
    p.add_argument("--norm", default="sat", choices=("sat", "mean"),
                   help="sat divides by the distance between two unrelated "
                        "clips of the same folder; mean is the raw "
                        "per-coefficient L1.")
    p.add_argument("--length", type=float, default=4.0,
                   help="Window in seconds. MUST match the training data's "
                        "`length` -- 4.0 for every run in this project -- "
                        "because the estimator's frame count follows it.")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--csv", default=None, metavar="PATH",
                   help="Also write the per-clip scores.")
    args = p.parse_args()

    dev = args.device
    metrics = build_metrics(args.metrics, dev, args.sr)

    # THE SAMPLE IS DRAWN ONCE, before any model is loaded, so every arm sees
    # exactly the same clips. Drawing per arm would make the comparison between
    # different audio, which is the mistake ds_compare_audio exists to avoid.
    rng = random.Random(args.seed)
    groups: dict[str, list[str]] = {}
    for d in args.dirs:
        files = audio_files(d)
        if not files:
            raise SystemExit(f"no audio files under {d} "
                             f"(looked for {', '.join(AUDIO_EXT)})")
        if args.n and len(files) > args.n:
            files = sorted(rng.sample(files, args.n))
        groups[os.path.basename(os.path.normpath(d))] = files

    # Read the audio once too, for the same reason.
    print(f"{'folder':<28}{'clips':>7}{'active':>9}{'peak':>9}")
    audio: dict[str, torch.Tensor] = {}
    for g, files in groups.items():
        ys, acts, peaks = [], [], []
        for f in files:
            y, a, pk = load_clip(f, args.sr, args.length)
            ys.append(y)
            acts.append(a)
            peaks.append(pk)
        audio[g] = torch.tensor(np.stack(ys), dtype=torch.float32)
        print(f"{g:<28}{len(files):>7}{np.median(acts):>9.3f}"
              f"{np.median(peaks):>9.3f}")
    print("  active = fraction of the window within 60 dB of the clip's own\n"
          "  peak. A low value means the score is mostly about the silence.\n")

    rows = []
    per_arm: dict[str, dict[str, dict[str, float]]] = {}
    for arm in args.arms:
        model, note = load_model(os.path.join(args.root, arm), args.ckpt, dev)
        if model is None:
            print(f"{arm:<28} skipped: {note}")
            continue
        print(f"{arm:<28} {note}")
        acc: dict[str, dict[str, list[float]]] = {}
        for g, x in audio.items():
            e = {m: [0.0, 0.0, 0.0] for m in metrics}
            # A trailing batch of ONE has no partner for the saturation
            # denominator, so merge it back rather than silently dropping a
            # clip -- at --n 49 and batch 16 that would be 16/16/16/1.
            edges = list(range(0, x.shape[0], args.batch_size)) + [x.shape[0]]
            if len(edges) > 2 and edges[-1] - edges[-2] == 1:
                edges.pop(-2)
            for lo, hi in zip(edges, edges[1:]):
                tgt = x[lo:hi].to(dev)
                if tgt.shape[0] < 2:
                    continue          # a one-clip folder; nothing to divide by
                with torch.no_grad():
                    out, _ = model({"audio": tgt})
                oth = tgt.roll(1, dims=0)
                for mname, fn in metrics.items():
                    a, b, c = fn(tgt), fn(out), fn(oth)
                    e[mname][0] += float((a - b).abs().sum())
                    e[mname][1] += float((a - c).abs().sum())
                    e[mname][2] += a.numel()
            acc[g] = e
            for mname, (num, den, k) in e.items():
                rows.append((arm, g, mname, num, den, k))
        per_arm[arm] = {
            g: {m: ((num / den if den else float("nan")) if args.norm == "sat"
                    else (num / k if k else float("nan")))
                for m, (num, den, k) in e.items()}
            for g, e in acc.items()}

    if not per_arm:
        raise SystemExit("no arm produced results")

    arms = list(per_arm)
    for mname in metrics:
        label = ("arm / saturation" if args.norm == "sat"
                 else "mean |dMFCC| per coefficient")
        print(f"\n=== {mname}   ({label}; lower is better)")
        print(f"{'folder':<28}" + "".join(f"{a:>24}" for a in arms))
        totals = {a: [0.0, 0.0] for a in arms}
        for g in groups:
            cells = []
            for a in arms:
                v = per_arm[a].get(g, {}).get(mname, float("nan"))
                cells.append(f"{v:>24.4f}")
                if v == v:
                    totals[a][0] += v
                    totals[a][1] += 1
            print(f"{g:<28}" + "".join(cells))
        print(f"{'MEAN over folders':<28}"
              + "".join(f"{(totals[a][0] / totals[a][1]):>24.4f}"
                        if totals[a][1] else f"{'nan':>24}" for a in arms))

    if args.norm == "sat":
        print("\n  READ arm/saturation, NOT the absolute number. Saturation is\n"
              "  the distance between two unrelated clips of the same folder --\n"
              "  what a model conveying nothing about the target would score.\n"
              "  Near 1.0 means the arm is telling us nothing on this material\n"
              "  however large or small the raw value looks.")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["arm", "folder", "metric", "num", "den", "n"])
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
