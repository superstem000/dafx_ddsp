import numpy as np
import torch
import torch.nn as nn
import functools

# torchcrepe is only needed by the f0 estimation path -- compute_f0 and
# process_f0 below, reached when data.f0 is true. The paper's three training
# procedures all run with f0: false (configs/data) and a null perc_model, so an
# unconditional import here was the single thing standing between a fresh
# environment and a working one: estimator.py imports FMIN/FMAX from this
# module, which are plain constants, and that pulled in the whole dependency.
#
# Deferred rather than removed: if the f0 path IS used, the error should say so
# precisely rather than fail somewhere downstream.
try:
    import torchcrepe
except ImportError:  # pragma: no cover
    torchcrepe = None


def _require_torchcrepe():
    if torchcrepe is None:
        raise ImportError(
            "torchcrepe is required for f0 estimation (data.f0=true or a crepe "
            "perceptual model). Install it with `pip install torchcrepe`; the "
            "paper's P-loss / Synth / Real procedures do not need it."
        )

FMIN = 32
FMAX = 2000

def process_f0(f0_hz, periodicity):
    # Shape [1, 1 + int(time // hop_length,]
    # Postprocessing on f0_hz
    # replace unvoiced regions with NaN
    # win_length = 3
    # periodicity = torchcrepe.filter.mean(periodicity, win_length)
    _require_torchcrepe()
    threshold = 1e-3
    # if all noisy, do not perform thresholding
    if (periodicity > threshold).any():
        f0_hz = torchcrepe.threshold.At(1e-3)(f0_hz, periodicity)
    # f0_hz = torchcrepe.filter.mean(f0_hz, win_length)
    f0_hz = f0_hz[0]
    # interpolate Nans
    # https://stackoverflow.com/questions/9537543/replace-nans-in-numpy-array-with-closest-non-nan-value
    f0_hz = f0_hz.numpy()
    mask = np.isnan(f0_hz)
    f0_hz[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), f0_hz[~mask])
    return torch.from_numpy(f0_hz)# Shape [1 + int(time // hop_length,]

def compute_f0(audio, sample_rate):
    """ For preprocessing
    Args:
        audio: torch.Tensor of single audio example. Shape [audio_length,].
        sample_rate: Sample rate in Hz.

    Returns:
        f0_hz: Fundamental frequency in Hz. Shape [1, 1 + int(time // hop_length,]
        periodicity: Basically, confidence of pitch value. Shape [1, 1 + int(time // hop_length,]
    """
    _require_torchcrepe()
    audio = audio.unsqueeze(0)

    # Compute f0 with torchcrepe.
    # uses viterbi by default
    # pad=False is probably center=False
    # [output_shape=(1, 1 + int(time // hop_length))]
    f0_hz, periodicity = torchcrepe.predict(audio, sample_rate, hop_length=128, pad=False, device='cuda', batch_size=2048, model='full', fmin=FMIN, fmax=FMAX, return_periodicity=True)
    return f0_hz, periodicity

def write_f0(audiofile, f0_dir, duration, overwrite):
    basename = os.path.basename(audiofile)
    f0_file = os.path.join(f0_dir, basename[:-4]+'.pt')
    if overwrite or not os.path.exists(f0_file):
        audio, _sr = librosa.load(audiofile, sr=16000, duration=duration)
        f0, periodicity = compute_f0(torch.from_numpy(audio), 16000)
        torch.save((f0, periodicity), f0_file)


if __name__ == "__main__":
    import argparse, os, glob
    import librosa
    import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir',  type=str,   help='')
    parser.add_argument('--duration',  type=float, default=4.0, help='')
    parser.add_argument('--overwrite',  action='store_true')
    args = parser.parse_args()
    
    audio_dir = os.path.join(args.base_dir, 'audio')
    f0_dir = os.path.join(args.base_dir, 'f0')
    os.makedirs(f0_dir, exist_ok=True)
    raw_files = sorted(glob.glob(os.path.join(audio_dir, '*.wav')))

    pool = torch.multiprocessing.Pool(processes=2)
    func = functools.partial(write_f0, f0_dir=f0_dir, duration=args.duration, overwrite=args.overwrite)
    with tqdm.tqdm(total=len(raw_files)) as t:
        for _ in pool.imap_unordered(func, raw_files):
            t.update(1)