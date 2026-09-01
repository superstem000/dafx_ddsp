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


_WIN: dict[str, "torch.Tensor"] = {}


def _linmag(x, device: str):
    """Per-bin magnitude on the same 1024/256 grid ds_ood_subset masks from."""
    if device not in _WIN:
        _WIN[device] = torch.hann_window(1024, device=device)
    return torch.stft(x, 1024, hop_length=256, window=_WIN[device],
                      center=True, return_complex=True).abs()


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


def load_clip(path: str, sr: int, length: float, peak_to: float | None = None):
    """One mono clip at sr, exactly length seconds, plus what was there before.

    Returns (audio, active_fraction, peak, raw_seconds).

    RESAMPLING IS librosa's, the same call WaveParamDataset makes: load(sr=...)
    runs soxr_hq with a proper anti-alias filter, so a 44.1 kHz source loses
    everything above 8 kHz rather than folding it back. NSynth is natively 16
    kHz, so the two sides of any comparison differ in having been resampled --
    which removes content from the sample pack and adds nothing to it.

    LENGTH IS ENFORCED IN BOTH DIRECTIONS. librosa's duration= truncates a long
    clip to the FIRST `length` seconds -- so an 11 s pad keeps its attack and
    loses its release, which is the right half to keep since the estimator
    predicts an amplitude curve and an onset is what constrains it. Short clips
    are zero-padded at the END, where WaveParamDataset asserts instead
    (data.py:49). `raw_seconds` reports what was actually there so truncation
    and padding are both visible.

    `active` is the fraction of the FULL padded window whose sample magnitude is
    within 60 dB of the clip's own peak -- computed before padding, so the pad
    counts as silence, which is the point.
    """
    import librosa
    n = int(round(length * sr))
    y, _ = librosa.load(path, sr=sr, mono=True, duration=length)
    raw = y.shape[0] / float(sr)
    peak = float(np.abs(y).max()) if y.size else 0.0
    if peak_to is not None and peak > 0:
        y = y * (peak_to / peak)
        peak = peak_to
    if peak > 0:
        active = float((np.abs(y) >= peak * 10.0 ** (-60.0 / 20.0)).sum()) / n
    else:
        active = 0.0
    if y.shape[0] < n:
        y = np.pad(y, (0, n - y.shape[0]))
    return y[:n], active, peak, raw


def load_model(run_dir: str, ckpt_name: str, device: str):
    """EstimatorSynth from a run directory, in train.py's construction order.

    The datamodule is deliberately absent -- see the module docstring -- so the
    seed here only has to make the architecture and its weight shapes right,
    not reproduce any split.
    """
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    ck_path = os.path.join(run_dir, "tb_logs", "checkpoints", ckpt_name)
    if not os.path.exists(cfg_path):
        return None, None, f"missing .hydra/config.yaml"
    if not os.path.exists(ck_path):
        return None, None, f"missing tb_logs/checkpoints/{ckpt_name}"
    cfg = OmegaConf.load(cfg_path)
    try:
        # weights_only=False for the same reason train.py needs it: the
        # checkpoint carries the hydra DictConfig via save_hyperparameters().
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return None, None, f"unreadable checkpoint ({type(e).__name__})"
    pl.seed_everything(0, workers=True)
    model = EstimatorSynth(cfg.model, cfg.synth, cfg.schedule)
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, cfg, f"epoch {ck.get('epoch', '?')}"


def describe(cfg) -> dict:
    """Each arm's loss, from its OWN resolved config rather than from a jobs file.

    THE MISTAKE THIS CATCHES has no error message. `synth_magx_halfw` is a
    directory name, and a directory name is a claim about what was trained, not
    a record of it. Two things in this family are easy to get backwards:

      `x` is the DOMAIN. power=1 is magnitude, i.e. compression exponent 0.5 on
      the power spectrogram, against the published power=2. Not to be confused
      with sw_loss.gamma, which is a separate exponent knob and is null in all
      of these.

      `halfw` is the WEIGHT, schedule.sw_w.end_v=0.5, and it exists to make the
      comparison single-variable rather than to change the loss. The hybrid has
      TWO terms over 6 FFT sizes, so at sw_w 1.0 each carries 1/12. A one-term
      arm at sw_w 1.0 would carry 1/6 -- twice the hybrid's per-term weight --
      so magx_halfw and logx_halfw run 0.5 to land on 1/12 as well. If the
      per-term column below is not identical across the arms, the comparison is
      confounded and nothing downstream means what it looks like.
    """
    s = cfg.model.sw_loss
    n_scales = len(s.fft_sizes)
    terms = int(float(s.get("mag_w", 0)) > 0) + int(float(s.get("log_mag_w", 0)) > 0)
    end_v = float(cfg.schedule.sw_w.end_v)
    return {
        "domain": "mag" if int(s.get("power", 2)) == 1 else "pow",
        "mag_w": float(s.get("mag_w", 0)),
        "log_w": float(s.get("log_mag_w", 0)),
        "eps": float(s.get("log_eps_v", 0)),
        "gamma": s.get("gamma", None),
        "sw_w": end_v,
        "per_term": end_v / (terms * n_scales) if terms else 0.0,
        "train": cfg.data.train_type,
        "sr": int(cfg.data.sample_rate),
        "len": float(cfg.data.length),
    }


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
    p.add_argument("--active-db", type=float, default=0.0, metavar="DB",
                   help="Score only frames whose energy is within this many dB "
                        "of the TARGET's loudest frame; 0 scores whole clips. "
                        "A silent frame is silent in target and resynthesis "
                        "alike and clamps to the same floor, so it adds nothing "
                        "to the numerator while still counting in the mean -- "
                        "which ranks partly by note length. The synthetic "
                        "training set measures 0.995 active at 60 dB and the "
                        "Juno bass packs 0.19, so without this the comparison "
                        "is between clips that are 80% silence and clips that "
                        "are none, and silence is exactly where a linear loss "
                        "has no gradient and a log one has all of it. 60 is the "
                        "generous setting that matches the ACTIVE column's own "
                        "criterion; 40 and below start discarding real decay.")
    p.add_argument("--active-max", type=float, default=1.0, metavar="F",
                   help="Keep only clips whose ACTIVE fraction is at most this. "
                        "The synthetic set's MEDIAN occupancy is 0.995, but "
                        "sus_level is drawn uniform on [0,1], so its low tail "
                        "may already contain the sparse clips the Juno packs "
                        "are made of. Scoring the arms on that tail tests "
                        "whether occupancy is the mechanism WITHOUT "
                        "regenerating anything -- and if the in-domain ordering "
                        "survives there, it is not, and a redistributed "
                        "dataset would buy nothing.")
    p.add_argument("--active-min", type=float, default=0.0, metavar="F",
                   help="The other end, for the matching control: the same "
                        "folder's DENSE clips, so the comparison is within one "
                        "dataset rather than against another.")
    p.add_argument("--scan", type=int, default=4000, metavar="N",
                   help="With an active filter on, how many files to examine "
                        "before giving up on reaching --n. Loading is the cost; "
                        "the whole 20000-clip set is a few minutes.")
    p.add_argument("--trim-pad", action="store_true",
                   help="Drop the zero padding THIS SCRIPT added, and nothing "
                        "else. The folder is truncated to its longest clip's "
                        "own original duration, rounded up to a whole STFT "
                        "frame -- so every clip keeps all of its real content "
                        "and the only samples removed are ones that were never "
                        "in the file. No threshold, no free parameter, and "
                        "nothing to tune after seeing the answer, which is what "
                        "--trim-db cannot say for itself. On a folder read at "
                        "its full --length this is a no-op, so the in-domain "
                        "set is an exact control.")
    p.add_argument("--trim-db", type=float, default=0.0, metavar="DB",
                   help="Cut the trailing silence off the AUDIO before the "
                        "model sees it, rather than masking it out of the score "
                        "afterwards. The folder is truncated to the longest "
                        "clip's last sample within this many dB of its own "
                        "peak, rounded up to a whole STFT frame, so no clip "
                        "loses content and the batch stays rectangular. 0 is "
                        "off.\n"
                        "DIFFERENT FROM --active-db, and the difference is the "
                        "point. Masking still hands the estimator 3 s of "
                        "silence and merely declines to score what it does "
                        "there; trimming means it never sees the silence at "
                        "all. Those answer different questions -- 'is the "
                        "score about the note' versus 'is the silence itself "
                        "throwing the model off' -- and the second is the one "
                        "worth asking once an arm is audibly generating sound "
                        "in the padding.\n"
                        "IT ALSO LEAVES THE TRAINING LENGTH, deliberately. "
                        "Every run here trained on 4.0 s windows. The estimator "
                        "is a conv stack into a GRU over frames and "
                        "EstimatorSynth takes its render length from "
                        "conditioning['audio'].shape[1] (model.py:148), so a "
                        "shorter clip runs correctly -- but it is off the "
                        "training distribution in a second way while removing "
                        "the first, and any result under it has to say so.")
    p.add_argument("--folder-peak", type=float, default=None, metavar="P",
                   help="ONE gain per folder, putting that folder's MEDIAN peak "
                        "at P. This is the level match to use: it removes the "
                        "systematic offset between a sample pack and the "
                        "training set while leaving the clip-to-clip level "
                        "spread intact, so the saturation denominator keeps "
                        "meaning what it meant. The synthetic set's median peak "
                        "is 0.496, so --folder-peak 0.5 matches it.")
    p.add_argument("--peak", type=float, default=None, metavar="P",
                   help="Rescale every clip to this peak. OFF by default, "
                        "matching WaveParamDataset, which passes librosa.load "
                        "straight through -- amplitude is a parameter the "
                        "estimator predicts, so normalising hands it "
                        "information the training data never had. But the "
                        "estimator's input normalisation is BatchNorm2d with "
                        "affine=False, which in eval mode subtracts the "
                        "TRAINING set's running mean rather than each clip's "
                        "own, so absolute level does reach the network and a "
                        "pack mastered hotter than the synthetic data is a "
                        "systematic offset. Run it both ways: if the ranking "
                        "moves, the result is about level, not about the loss. "
                        "CONFOUNDED WITH THE DENOMINATOR under --norm sat: "
                        "giving every clip an identical peak removes the "
                        "level difference BETWEEN clips too, which shrinks "
                        "saturation and raises every ratio for a reason that "
                        "has nothing to do with the model. Prefer "
                        "--folder-peak, which does not.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--csv", default=None, metavar="PATH",
                   help="Also write the per-clip scores.")
    p.add_argument("--render", type=int, default=0, metavar="N",
                   help="Write the first N clips of each folder as target plus "
                        "one resynthesis per arm, so the numbers can be checked "
                        "by ear. THE SAME CLIPS FOR EVERY ARM, since the sample "
                        "was drawn once before any model loaded.")
    p.add_argument("--render-out", default="folder_audio", metavar="DIR")
    args = p.parse_args()

    dev = args.device
    metrics = build_metrics(args.metrics, dev, args.sr)

    # THE SAMPLE IS DRAWN ONCE, before any model is loaded, so every arm sees
    # exactly the same clips. Drawing per arm would make the comparison between
    # different audio, which is the mistake ds_compare_audio exists to avoid.
    rng = random.Random(args.seed)
    filtering = args.active_min > 0.0 or args.active_max < 1.0
    groups: dict[str, list[str]] = {}
    scan_order: dict[str, list[str]] = {}
    for d in args.dirs:
        files = audio_files(d)
        if not files:
            raise SystemExit(f"no audio files under {d} "
                             f"(looked for {', '.join(AUDIO_EXT)})")
        g = os.path.basename(os.path.normpath(d))
        if filtering:
            # A clip's ACTIVE fraction is only known once it is read, so the
            # selection cannot be made up front: walk a shuffled order and keep
            # what passes. Deliberately a SEPARATE path from the unfiltered
            # one, which still draws with rng.sample -- the two consume the RNG
            # differently, and reusing this path unfiltered would silently
            # change which clips every earlier run was computed on.
            order = files[:]
            rng.shuffle(order)
            scan_order[g] = order
            groups[g] = []
        else:
            groups[g] = (sorted(rng.sample(files, args.n))
                         if args.n and len(files) > args.n else files)

    # Read the audio once too, for the same reason.
    print(f"{'folder':<28}{'clips':>7}{'raw_s':>8}{'win_s':>8}{'trunc':>7}"
          f"{'pad':>6}{'act_p10':>9}{'active':>9}{'act_p90':>9}{'peak':>9}"
          f"{'gain':>8}")
    audio: dict[str, torch.Tensor] = {}
    for g in list(groups):
        ys, acts, peaks, raws, keep_f = [], [], [], [], []
        scanned = 0
        for f in (scan_order[g] if filtering else groups[g]):
            if args.n and len(keep_f) >= args.n:
                break
            if filtering and scanned >= args.scan:
                break
            scanned += 1
            y, a, pk, raw = load_clip(f, args.sr, args.length, args.peak)
            if filtering and not (args.active_min <= a <= args.active_max):
                continue
            keep_f.append(f)
            ys.append(y)
            acts.append(a)
            peaks.append(pk)
            raws.append(raw)
        if filtering:
            # Sorted so the roll-by-one saturation partner is a neighbouring
            # file rather than a shuffle artefact, matching the unfiltered path.
            o = sorted(range(len(keep_f)), key=lambda i: keep_f[i])
            keep_f = [keep_f[i] for i in o]
            ys = [ys[i] for i in o]
            acts = [acts[i] for i in o]
            peaks = [peaks[i] for i in o]
            raws = [raws[i] for i in o]
            groups[g] = keep_f
            print(f"  {g}: scanned {scanned}, kept {len(keep_f)} with active "
                  f"in [{args.active_min:g}, {args.active_max:g}]")
            if len(keep_f) < 2:
                raise SystemExit(
                    f"{g}: only {len(keep_f)} clip(s) passed the active filter "
                    f"in {scanned} scanned. Saturation needs a partner, so "
                    f"there is nothing to divide by -- widen the band or raise "
                    f"--scan.")
        files = groups[g]
        x = np.stack(ys)
        # ONE gain for the folder, applied after every clip is read, so the
        # clip-to-clip level spread -- and with it the saturation denominator --
        # survives while the pack's systematic offset from the training set
        # does not. Per-clip normalisation would flatten both; see --peak.
        gain = 1.0
        if args.folder_peak is not None:
            med = float(np.median(peaks))
            if med > 0:
                gain = args.folder_peak / med
                x = x * gain
                peaks = [p * gain for p in peaks]
        keep = None
        if args.trim_pad:
            # What the files actually contained. `raws` is recorded by
            # load_clip before any padding, so this needs no threshold: the
            # samples removed are exactly the ones the script invented.
            keep = int(np.ceil(max(raws) * args.sr / 256.0)) * 256
        elif args.trim_db > 0:
            # The folder's longest active clip sets the window, so nothing is
            # cut off any clip -- only the tail that every clip has already
            # fallen silent in. Rounded up to a whole hop, and never below one
            # FFT, or the STFT has no frames to give.
            thr = 10.0 ** (-args.trim_db / 20.0)
            ends = []
            for row in x:
                pk = float(np.abs(row).max())
                nz = np.nonzero(np.abs(row) >= pk * thr)[0] if pk > 0 else []
                ends.append(int(nz[-1]) + 1 if len(nz) else 0)
            keep = int(np.ceil(max(ends) / 256.0)) * 256
        if keep is not None:
            keep = min(x.shape[1], max(1024, keep))
            x = x[:, :keep]
            # `active` has to be recomputed against the window that is actually
            # scored, or the column describes a clip nobody evaluated.
            acts = []
            for row in x:
                pk = float(np.abs(row).max())
                acts.append(float((np.abs(row) >= pk * 10.0 ** (-60.0 / 20.0)
                                   ).sum()) / row.shape[0] if pk > 0 else 0.0)
            if args.trim_pad and keep < int(round(max(raws) * args.sr)):
                print(f"  NOTE: {g} rounded {max(raws):.3f}s down to "
                      f"{keep / args.sr:.3f}s to reach a whole frame")
        audio[g] = torch.tensor(x, dtype=torch.float32)
        # A clip that came back at exactly --length was cut there by librosa's
        # duration=; anything shorter got padded. Both counts, because the two
        # failure modes read very differently in the scores.
        trunc = sum(1 for r in raws if r >= args.length - 1e-6)
        print(f"{g:<28}{len(files):>7}{np.median(raws):>8.2f}"
              f"{audio[g].shape[1] / args.sr:>8.2f}{trunc:>7}"
              f"{len(files) - trunc:>6}{np.percentile(acts, 10):>9.3f}"
              f"{np.median(acts):>9.3f}{np.percentile(acts, 90):>9.3f}"
              f"{np.median(peaks):>9.3f}{gain:>8.2f}")
    print("  raw_s  median seconds actually read, before padding\n"
          "  win_s  the window actually scored, after any --trim-db\n"
          "  trunc  clips at least --length long, cut to the first --length s\n"
          "  pad    clips shorter than --length, zero-padded at the end\n"
          "  active fraction of the window within 60 dB of the clip's own peak;\n"
          "         low means the score is mostly about the silence. The p10 and\n"
          "         p90 columns are there because the MEDIAN hid the question:\n"
          "         a set at 0.995 median can still have a sparse decile, and\n"
          "         whether it does decides if occupancy can be tested without\n"
          "         regenerating anything\n"
          "  peak   median AFTER gain; unnormalised unless --peak/--folder-peak\n"
          "  gain   the single per-folder gain --folder-peak applied\n")
    if args.active_db > 0:
        print(f"  scoring only frames within {args.active_db:g} dB of each "
              f"TARGET's loudest frame.\n  The mask comes from the target for "
              f"both sides: taking it from an arm's own\n  output would grade "
              f"an arm that under-synthesises on fewer frames,\n  biasing the "
              f"comparison in exactly the direction under dispute.\n")

    rows = []
    per_arm: dict[str, dict[str, dict[str, float]]] = {}
    renders: dict[tuple[str, str], "np.ndarray"] = {}
    shown = False
    for arm in args.arms:
        model, cfg, note = load_model(os.path.join(args.root, arm),
                                      args.ckpt, dev)
        if model is None:
            print(f"{arm:<28} skipped: {note}")
            continue
        d = describe(cfg)
        if not shown:
            print(f"\n{'arm':<24}{'dom':>5}{'mag_w':>7}{'log_w':>7}{'eps':>8}"
                  f"{'gamma':>7}{'sw_w':>6}{'per_term':>10}{'train':>7}"
                  f"{'sr':>7}{'len':>6}  ckpt")
            shown = True
        print(f"{arm:<24}{d['domain']:>5}{d['mag_w']:>7.2f}{d['log_w']:>7.2f}"
              f"{d['eps']:>8.0e}{str(d['gamma']):>7}{d['sw_w']:>6.2f}"
              f"{d['per_term']:>10.4f}{d['train']:>7}{d['sr']:>7}"
              f"{d['len']:>6.1f}  {note}")
        if d["sr"] != args.sr or abs(d["len"] - args.length) > 1e-6:
            raise SystemExit(
                f"{arm} was trained at sample_rate {d['sr']} / length "
                f"{d['len']}s but this run reads {args.sr} / {args.length}s. "
                f"The estimator's frame count follows both, so the scores "
                f"would be computed on a different grid than it was trained "
                f"on. Pass --sr and --length to match.")
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
                m = None
                if args.active_db > 0:
                    fe = _linmag(tgt, dev).sum(dim=1)
                    m = fe >= fe.amax(dim=1, keepdim=True) * 10.0 ** (
                        -args.active_db / 20.0)
                for mname, fn in metrics.items():
                    a, b, c = fn(tgt), fn(out), fn(oth)
                    mm = m
                    if mm is not None and mm.shape[-1] != a.shape[-1]:
                        # The cepstrum and the raw spectrogram can differ by a
                        # frame of padding; index rather than assume they match.
                        j = (torch.arange(a.shape[-1], device=dev)
                             * mm.shape[-1] // a.shape[-1])
                        mm = mm[:, j]
                    if mm is None:
                        e[mname][0] += float((a - b).abs().sum())
                        e[mname][1] += float((a - c).abs().sum())
                        e[mname][2] += a.numel()
                    else:
                        w = mm[:, None, :].expand_as(a)
                        e[mname][0] += float(((a - b).abs() * w).sum())
                        e[mname][1] += float(((a - c).abs() * w).sum())
                        e[mname][2] += float(w.sum())
            acc[g] = e
            for mname, (num, den, k) in e.items():
                rows.append((arm, g, mname, num, den, k))
            if args.render:
                # A separate forward on the first N clips rather than keeping
                # every batch's output: the scoring loop runs under no_grad and
                # discards, and holding 50 clips x 3 arms x 4 folders of audio
                # to render 3 of them is not worth the memory.
                with torch.no_grad():
                    o, _ = model({"audio": x[:args.render].to(dev)})
                renders[(g, arm)] = o.cpu().numpy()
        per_arm[arm] = {
            g: {m: ((num / den if den else float("nan")) if args.norm == "sat"
                    else (num / k if k else float("nan")))
                for m, (num, den, k) in e.items()}
            for g, e in acc.items()}

    if not per_arm:
        raise SystemExit("no arm produced results")

    # THE SINGLE-VARIABLE CHECK, stated rather than assumed. If the arms do not
    # share a per-term weight the comparison is between two things at once, and
    # every table below is unreadable -- so say so where it cannot be missed.
    pts = {a: round(describe(OmegaConf.load(
        os.path.join(args.root, a, ".hydra", "config.yaml")))["per_term"], 6)
        for a in per_arm}
    if len(set(pts.values())) > 1:
        print(f"\n  WARNING: per-term weights differ across arms {pts}. "
              f"halfw exists to equalise them at 1/12; an arm missing it "
              f"carries twice the weight and the comparison is confounded.")

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

    if args.render and renders:
        import soundfile as sf
        os.makedirs(args.render_out, exist_ok=True)
        # ONE headroom gain across every file written, not per-file limiting.
        # An arm that gets the overall level wrong should sound like it does --
        # that is a real difference between these arms and normalising it away
        # would hide it -- so the only scaling is whatever it takes to keep the
        # loudest single file inside full scale, applied to all of them.
        pk = 0.0
        for g in groups:
            pk = max(pk, float(np.abs(audio[g][:args.render].numpy()).max()))
        for v in renders.values():
            pk = max(pk, float(np.abs(v).max()))
        head = min(1.0, 0.99 / pk) if pk > 0 else 1.0
        n = 0
        for g, files in groups.items():
            for i, src in enumerate(files[:args.render]):
                stem = os.path.splitext(os.path.basename(src))[0][:60]
                base = os.path.join(args.render_out, f"{g[:24]}__{stem}")
                sf.write(f"{base}__target.wav",
                         audio[g][i].numpy() * head, args.sr)
                n += 1
                for arm in per_arm:
                    v = renders.get((g, arm))
                    if v is None or i >= v.shape[0]:
                        continue
                    sf.write(f"{base}__{arm}.wav", v[i] * head, args.sr)
                    n += 1
        print(f"\nwrote {n} wav(s) to {args.render_out}  "
              f"(one common gain {head:.3f}, levels otherwise as scored"
              + (f", after the --folder-peak match" if args.folder_peak
                 else ", unnormalised") + ")")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["arm", "folder", "metric", "num", "den", "n"])
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
