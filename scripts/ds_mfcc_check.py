"""Recompute the MFCC distance with a window, and with library conventions.

    python scripts/ds_mfcc_check.py --only '^(synth|real)_'
    python scripts/ds_mfcc_check.py --only '^synth_magx' --ckpt 'ep*.ckpt'

WHY. model.py's Mfcc reaches torch.stft through MelSpec.forward, which calls
spectrogram(audio, n_fft, hop_length, power, center) -- five positional
arguments, so `window` keeps its None default and the transform runs with a
RECTANGULAR window. compute_lsd, two functions away, passes a Hann window
explicitly. So the two audio metrics do not share an analysis window, and the
MFCC one has full spectral leakage smearing every strong partial into its
neighbouring bands. That is a defect rather than a convention, and it weakens
the argument MFCC is being used for -- leakage is not perceptual.

Three variants are reported so the effect is visible rather than asserted:

  as_logged   exactly what val_id/mfcc contains: rectangular window, HTK mel
              scale, unnormalised filterbank, natural log with eps 1e-6, no dB
              conversion and no top_db floor, 40 mels -> 20 DCT coefficients
  hann        the same with a Hann window, changing one thing
  standard    Hann, Slaney mel scale and Slaney filterbank normalisation, dB
              (10*log10) with an 80 dB floor -- torchaudio/librosa conventions.
              Expect roughly 4.343x the natural-log values from the log base
              alone, so read it against itself, not against the other columns.

Then a POWER-CEPSTRUM ladder, g1 / g0.6 / g0.3 / g0.15 / g0: `standard` with its
dB step replaced by mel^gamma and nothing else changed, so moving along it
changes the metric's compression and only that.

  g0 is the ladder's OWN log rung, and it is not as_logged. Box-Cox says
  (x^g - 1)/g -> ln x, so the log is the limit of this ladder rather than a
  different metric -- but only when it shares the ladder's conventions.
  as_logged is rectangular-windowed and HTK; hann is HTK and unnormalised;
  standard puts an 80 dB floor on top. g0 is Hann + Slaney + Slaney with eps
  1e-12, GammaCepstrum's own, because for a log metric the eps IS the floor and
  1e-6 would sit six decades higher. So g1 -> g0 is one knob moving, end to end.

  Every audio metric here measures in the log domain -- MFCC logs mel-band
  totals, LSD logs per bin -- and a log-trained model is optimised for exactly
  that domain, so ranking losses by them is close to circular. g1 is a linear
  metric and the linear-trained arms win it by construction; as_logged is the
  log one and they lose it. So a crossover exists, and where it sits is the
  question. Stevens' law puts loudness at I^0.3, which makes g0.3 the
  perceptually calibrated rung -- pure linear and pure log are both wrong, on
  opposite sides, and MFCC and LSD sit far past perception. If the crossover is
  below 0.3, the linear loss wins where hearing actually is.

  Quoting a stated ladder with a reference fixed by psychoacoustics is what
  separates this from picking whichever metric happens to be flattering.

With --mask, a MASKING row is added on top of the dB ladder: `mask`, whose
floor is the target's own per-bin MPEG-1 psychoacoustic threshold rather than a
constant offset from its peak, and `mask+80 ... mask+20` crossing it with each
dB rung. Both are floors on the same dB quantity, so the cross clamps to the
higher of the two -- at 80 the mask dominates, at 20 the fixed floor does, and
the row is an interpolation between the two mechanisms.

  The dB ladder always invites "why that floor". The mask has no free
  parameter to argue about: it is what the signal itself renders inaudible,
  computed by ds_masking.py, whose four self-tests check it against the
  canonical 1 kHz SMR, white noise, superposition, and the Terhardt ATH. If
  the arm ordering is the same at `mask` as at some dB rung, the rung was a
  fair stand-in; if it is not, the mask is the one to quote.

  Thresholds come from the TARGET for both members of a pair. An arm that
  under-synthesises quiet content must not get a lower floor for having done
  so. --shared-peak extends the same discipline to the dB floors, which
  otherwise keep librosa's per-signal peak: a resynthesis 3 dB quiet gets a
  3 dB lower floor and keeps relatively more low-level detail than the target,
  putting a gain term into a distance that is supposed to measure spectral
  accuracy. Off by default so nothing already reported moves silently; if the
  columns agree with and without it, no arm here has a gain offset worth
  worrying about, which is itself worth knowing.

The ORDERING across arms is what matters. If it survives all three log variants
the result does not depend on the metric's implementation details; if it moves,
that is worth knowing before the number goes in a table. Read every column
DOWN -- dB is ~4.34x natural log by base alone, and each gamma is a different
power of the same numbers, so nothing compares across columns.

Only checkpointed epochs can be recomputed -- MFCC was never logged any other
way. --ckpt takes a glob, so 'ep*.ckpt' gives the 50-epoch trajectory for runs
that had trainer.checkpoint_every_n_epochs set, and the default latest.ckpt
gives the final epoch for everything.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "external", "diffsynth")
sys.path.insert(0, DS)
sys.path.insert(0, HERE)

import torch                                            # noqa: E402
import torch.nn.functional as F                         # noqa: E402
from torch.utils.data import DataLoader                 # noqa: E402
from torchaudio.transforms import MelScale              # noqa: E402
from torchaudio.functional import create_dct            # noqa: E402
from diffsynth.spectral import compute_lsd              # noqa: E402

import ds_param_breakdown as pb                         # noqa: E402
import ds_masking as dm_mask                            # noqa: E402


class MaskCache:
    """Per-bin masking thresholds, precomputed by ds_masking.py --dump-thresholds.

    The MPEG-1 model is a per-frame sweep over maskers with a 0.5 Bark
    decimation that does not vectorise cleanly, so it is not recomputed here.
    It is computed once by the numpy implementation the four self-tests
    validate, and this loads the result. One implementation of the physics
    means the threshold justifying the floor in the text and the threshold
    setting the floor in the table are the same number, which is the only
    version of this worth publishing.

    Keyed by a content hash of the target audio, so a moved split raises
    instead of silently pairing a clip with someone else's threshold.
    """

    def __init__(self, path, device):
        import numpy as np
        z = np.load(path, allow_pickle=False)
        self.T = z["T"]                        # [N, bins, frames], dB SPL
        self.key = {k: i for i, k in enumerate(z["keys"].tolist())}
        # The SPL offset the dump was built with. Carried in the file rather
        # than re-derived: ds_masking normalises by the window sum and adds
        # PN, and reconstructing that here from memory would misplace every
        # threshold by tens of dB while still looking entirely plausible.
        self.C = float(z["C"])
        self.note = (f"{self.T.shape[0]} clips, tonal-window="
                     f"{z['tonal_window']}, decimate={z['decimate']}")
        self.device = device
        self._np = np

    def floor(self, ref):
        """Target audio [B, samples] -> floor in 10*log10(power) units."""
        import torch as _t
        rows = []
        for i in range(ref.shape[0]):
            h = dm_mask.audio_key(ref[i])
            j = self.key.get(h)
            if j is None:
                raise SystemExit(
                    "a target clip is not in the threshold cache. The cache "
                    "was dumped over one split; this run is scoring a "
                    "different one. Re-dump with matching --domain/--family, "
                    f"or drop --mask.\n  cache: {self.note}")
            rows.append(self.T[j])
        t = _t.from_numpy(self._np.stack(rows)).float().to(self.device)
        return t - self.C


def make_mfcc(device, window="rect", log="nat", top_db=None, gamma=0.3,
              mel_norm=None, mel_scale="htk", sr=16000, n_fft=1024, hop=256,
              n_mels=40, n_mfcc=20, f_min=40.0, f_max=7600.0,
              mask=None, pre_mel=False, shared_peak=False, eps=None):
    """A callable (audio, ref) -> MFCC, mirroring spectral.py but with the knobs open.

    `ref` is the TARGET for both members of a pair. Only the masked columns
    read it: the threshold has to come from the target for the resynthesis
    too, since an arm that under-synthesises quiet content would otherwise be
    graded against a floor its own quietness lowered -- biasing the comparison
    in exactly the direction under dispute.

    `pre_mel` moves the dB clamp ahead of the filterbank. The mask has no
    choice: a masking threshold is defined per FFT bin, and a mel band at 6 kHz
    spans several critical bands with no single threshold to speak of. The dB
    floors move with it so the table is one table rather than two conventions
    side by side.
    """
    if log == "pow":
        # The gamma columns come from diffsynth's own GammaCepstrum, which
        # model.py logs as mfcc03 every epoch. Single implementation, so the
        # number computed here from a checkpoint and the number logged during
        # training cannot drift apart -- and they have to agree, because this
        # script is the only way a run predating that metric gets the column.
        from diffsynth.spectral import GammaCepstrum
        gc = GammaCepstrum(gamma=gamma, n_fft=n_fft, hop_length=hop,
                           n_mels=n_mels, n_mfcc=n_mfcc, sample_rate=sr,
                           f_min=f_min, f_max=f_max).to(device)
        return lambda audio, ref=None: gc(audio)
    if (log == "db" and window == "hann" and mel_norm == "slaney"
            and mel_scale == "slaney" and mask is None and not pre_mel
            and not shared_peak and eps is None):
        # The plain headline configuration, which model.py now logs live as
        # mfccdb. Same delegation the log == "pow" branch above does, for the
        # same reason: the logged curve and this column have to be one
        # implementation or the table mixes two quantities under one name --
        # which is exactly the bug that made val_ood/mfcc unreadable against
        # it. The masked / pre_mel / shared_peak variants fall through to the
        # general path below; DbCepstrum deliberately has none of those knobs.
        from diffsynth.spectral import DbCepstrum
        dc = DbCepstrum(top_db=top_db, n_fft=n_fft, hop_length=hop,
                        n_mels=n_mels, n_mfcc=n_mfcc, sample_rate=sr,
                        f_min=f_min, f_max=f_max).to(device)
        return lambda audio, ref=None: dc(audio)
    win = None if window == "rect" else torch.hann_window(n_fft, device=device)
    ms = MelScale(n_mels, sr, f_min, f_max, n_fft // 2 + 1, mel_norm,
                  mel_scale).to(device)
    dct = create_dct(n_mfcc, n_mels, "ortho").to(device)

    def spec(audio):
        # MelSpec.pad_end: zero-pad up to a whole number of hops.
        rem = (audio.shape[-1] - n_fft) % hop
        if rem:
            audio = F.pad(audio, (0, hop - rem), "constant")
        st = torch.stft(audio, n_fft, hop_length=hop, window=win,
                        center=False, return_complex=True)
        return st.real ** 2 + st.imag ** 2

    def f(audio, ref=None):
        power = spec(audio)
        # The spectrum the dB FLOOR is taken from. Per-signal by default, which
        # is librosa's convention and what every dB number reported so far
        # used. shared_peak takes it from the target for both members of the
        # pair instead -- see build_variants.
        rpow = (spec(ref) if (shared_peak and ref is not None
                              and ref is not audio) else power)
        if mask is not None or (top_db is not None and pre_mel):
            pdb = 10.0 * torch.log10(power + 1e-30)
            fl = None
            if mask is not None:
                fl = mask.floor(ref if ref is not None else audio)
                if fl.shape[-2:] != pdb.shape[-2:]:
                    raise SystemExit(
                        f"threshold cache is {tuple(fl.shape[-2:])} but this "
                        f"STFT is {tuple(pdb.shape[-2:])}; the dump used a "
                        f"different frame grid")
            if top_db is not None and pre_mel:
                rdb = pdb if rpow is power else 10.0 * torch.log10(rpow + 1e-30)
                pk = rdb.amax(dim=(-2, -1), keepdim=True) - top_db
                fl = pk if fl is None else torch.maximum(fl, pk)
            power = 10.0 ** (0.1 * torch.maximum(pdb, fl))
        mel = ms(power)
        if log == "nat":
            lm = torch.log(mel + (1e-6 if eps is None else eps))
        else:
            lm = 10.0 * torch.log10(mel + (1e-10 if eps is None else eps))
            if top_db is not None and not pre_mel:
                rlm = (lm if rpow is power
                       else 10.0 * torch.log10(ms(rpow) + 1e-10))
                lm = torch.maximum(
                    lm, rlm.amax(dim=(-2, -1), keepdim=True) - top_db)
        return torch.matmul(lm.transpose(1, 2), dct).transpose(1, 2)

    return f


_FIXED = (
    ("as_logged", dict(window="rect", log="nat")),
    ("hann", dict(window="hann", log="nat")),
    ("standard", dict(window="hann", log="db", top_db=80.0,
                      mel_norm="slaney", mel_scale="slaney")),
)


def _se(vals):
    """Standard error of the mean across batches, or nan for a single batch."""
    if len(vals) < 2:
        return float("nan")
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return (var / len(vals)) ** 0.5


def build_variants(gammas, top_dbs=(), mask=None, pre_mel=None,
                   shared_peak=False):
    """The three log variants, a power-cepstrum column per gamma, a dB-floor
    column per top_db.

    The two ladders attack the same question from opposite directions. gamma
    changes HOW MUCH a quiet bin counts -- continuously, from linear at 1 to
    log at 0. top_db changes WHETHER it counts at all: everything more than N
    dB below the clip's peak is clamped flat, so it contributes nothing. Both
    are `standard` with one thing altered.

    A floor is the easier claim to defend in a paper -- "energy 40 dB below the
    peak is inaudible here" is a statement anyone can evaluate, where an
    exponent needs Stevens' law to license it. If the arm ordering moves the
    same way along both ladders, the result does not depend on which knob was
    chosen.

    NOTE the clamp is against the peak of the WHOLE clip (amax over mels and
    time), which is librosa's convention for power_to_db(top_db=...). For a
    decaying note that means the tail is clamped wholesale once it falls far
    enough, rather than each frame being floored against its own peak. That is
    the standard behaviour, not an oversight, but it is why a tight floor
    removes decay tails specifically.

    The gamma columns are `standard` with its dB step replaced by mel^gamma and
    nothing else changed -- same Hann window, same Slaney mel scale and
    filterbank normalisation -- so moving down the gamma list changes the
    metric's COMPRESSION and only that.

    Why this is worth having. Every audio metric in this project measures in
    the log domain: MFCC logs mel-band totals, LSD logs per bin. A log-trained
    model is optimised for that domain, so ranking losses by them is close to
    circular. gamma=1 is a linear metric and the linear-trained arms must win
    it by construction; gamma->0 is the log metric and they must lose. There is
    therefore a crossover, and where it sits is the actual question.

    Stevens' law puts loudness at I^0.3, so gamma=0.3 is the perceptually
    calibrated rung -- both pure linear and pure log are wrong, on opposite
    sides, and MFCC/LSD sit far past perception at gamma->0. If the crossover
    is below 0.3 the linear loss wins where hearing actually is; if above, it
    does not. Reading the answer off a stated ladder with a reference point
    fixed by psychoacoustics is what keeps this from being metric-shopping.
    """
    std = dict(window="hann", log="db", mel_norm="slaney", mel_scale="slaney",
               shared_peak=shared_peak)
    # The mask forces pre-mel, but pre-mel is useful on its own: post-mel a band
    # holding one strong partial sits above the floor and shelters every quiet
    # bin in it, so the clamp barely bites and the ladder understates its own
    # effect. --pre-mel runs the ladder without needing a threshold cache.
    if pre_mel is None:
        pre_mel = mask is not None
    out = _FIXED + tuple(
        (f"g{g:g}", dict(window="hann", log="pow", gamma=g,
                         mel_norm="slaney", mel_scale="slaney"))
        for g in gammas
    ) + ((
        # The ladder's own gamma -> 0 rung. Box-Cox: (x^g - 1)/g -> ln x, so the
        # log IS the limit of this ladder rather than a separate metric -- but
        # only if it shares the ladder's conventions. as_logged and hann do not
        # (rectangular window for one, HTK and an unnormalised filterbank for
        # both), and standard adds an 80 dB floor on top. This is Hann + Slaney
        # + Slaney with eps 1e-12, GammaCepstrum's own eps, because for a log
        # metric the eps IS the floor and 1e-6 would put it six decades up.
        ("g0", dict(window="hann", log="nat", eps=1e-12,
                    mel_norm="slaney", mel_scale="slaney")),
    ) if gammas else ()) + tuple(
        (f"db{t:g}", dict(std, top_db=float(t), pre_mel=pre_mel))
        for t in top_dbs
    )
    # The bridge back to the post-mel convention every number in sections 08-11
    # was computed under. Needed whenever the ladder moved, mask or no mask.
    if pre_mel and top_dbs:
        out = out + (("db70post", dict(std, top_db=70.0, pre_mel=False)),)
    if mask is None:
        return out
    # The masked row, and the cross. Both floors clamp the same dB quantity,
    # so applying both means taking the higher of them -- which makes the row
    # an interpolation rather than two effects piled up. At 80 the mask
    # dominates (it sits above peak-80 for ~99.9% of the bins below that
    # floor); at 20 the fixed floor does. If the arm ordering flips somewhere
    # in between, the flip point says which mechanism owns the disagreement.
    #
    # db70post reproduces the OLD post-mel convention, unchanged, so this
    # table can be tied back to every db70 already reported. Without it the
    # move to a pre-mel clamp would silently restate published numbers.
    return out + (("db70post", dict(std, top_db=70.0, pre_mel=False)),
                  ("mask", dict(std, mask=mask, pre_mel=True))) + tuple(
        (f"mask+{t:g}", dict(std, mask=mask, top_db=float(t), pre_mel=True))
        for t in top_dbs)


def main() -> None:
    # Line-buffer stdout. Lightning's "Seed set to 0" goes to stderr and shows
    # up at once, while every progress line here is a print() -- so piped into
    # tee the two separate and the log looks stalled for minutes at a time with
    # checkpoints already finished. Same fix as gpu_queue.py needed.
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="results/diffsynth")
    p.add_argument("--only", default=None, metavar="REGEX")
    p.add_argument("--ckpt", default="latest.ckpt",
                   help="Checkpoint file name or glob within the run's "
                        "checkpoints/ directory, e.g. 'ep*.ckpt'")
    p.add_argument("--domain", default="id", choices=("id", "ood"))
    p.add_argument(
        "--gamma", type=float, nargs="*", default=[1.0, 0.6, 0.3, 0.15],
        metavar="G",
        help="Power-cepstrum exponents to add as extra columns: mel^gamma in "
             "place of the log, everything else held at the `standard` "
             "conventions. 1.0 is a linear metric, ->0 approaches the log one, "
             "and 0.3 is Stevens' law (loudness ~ I^0.3) -- the perceptually "
             "calibrated rung, which MFCC and LSD both sit far past. The "
             "default ladder brackets it on both sides so the crossover is "
             "visible rather than assumed. Pass no values to skip them.",
    )
    p.add_argument(
        "--family", default=None, metavar="PREFIX",
        help="Score against one NSynth instrument family instead of the ood "
             "valid split -- e.g. --family synth_lead. NSynth filenames lead "
             "with the family (bass_synthetic_..., synth_lead_synthetic_...), "
             "so this is a basename prefix match over the WHOLE ood pool.\n\n"
             "Why the whole pool and not the valid split: a family is a few "
             "percent of NSynth, so intersecting it with a 2000-file valid "
             "split leaves ~47 clips. That is only sound for arms which never "
             "trained on ood at all -- the synth_* branch, which trains on "
             "harmor throughout. For real_* the whole pool IS its training "
             "set, so this refuses to run on them unless --allow-seen.\n\n"
             "The point is a model-mismatch ladder. harmor is 2 oscillators "
             "morphing saw<->square through a 2nd-order lowpass, which is what "
             "a synth lead is, and is nothing like a mallet or a plucked "
             "string. Scoring one arm on harmor (no mismatch), on synth_lead "
             "(mild) and on all of NSynth (severe) separates 'this loss fits "
             "better' from 'this loss distributes irreducible bias better'.",
    )
    p.add_argument(
        "--allow-seen", action="store_true",
        help="Score --family on real_* arms anyway. They trained on 16000 of "
             "the 20000-file ood pool, so most of the family is in their "
             "training set and the number is not held out.",
    )
    p.add_argument(
        "--crepe", action="store_true",
        help="Add an L1 distance between CREPE embeddings. Needs torchcrepe "
             "(pip install torchcrepe; it bundles its own weights).\n\n"
             "The point is independence. Every other column here -- MFCC in any "
             "variant, LSD, the gamma ladder -- is a spectral distance under "
             "some compression, so a reviewer can answer the gamma result with "
             "'you changed the compression to suit yourself'. A learned "
             "embedding makes no compression choice at all, so if it agrees "
             "with the gamma ladder that is a second, structurally different "
             "axis rather than the same one re-scaled.\n\n"
             "Read it for what it is, though: CREPE is a PITCH model, so this "
             "mostly measures whether the resynthesis has the right pitch, not "
             "whether it sounds the same. It is a check on the gamma result, "
             "not a general perceptual metric. OpenL3 would be that, and needs "
             "downloaded weights.\n\n"
             "Slow on CPU -- ~124 frames per 4 s clip through the model, per "
             "arm, per checkpoint. Use --device cuda, or cut --batches.",
    )
    p.add_argument(
        "--openl3", action="store_true",
        help="Add a cosine distance between OpenL3 embeddings, music-trained, "
             "512-d. Needs torchopenl3 (pip install torchopenl3); it fetches "
             "its own weights via ol3.core.load_audio_embedding_model, so "
             "there are no model files to place.\n\n"
             "This is the one that carries a general audio-matching claim. "
             "CREPE is a pitch model and scores pitch agreement; OpenL3 is a "
             "general music-audio representation, and it is standard enough to "
             "cite without arguing for it. Neither makes a spectral "
             "compression choice, which is the whole point against 'you moved "
             "the compression to suit yourself'.\n\n"
             "Heavier than CREPE: clips are resampled 16k -> 48k and cut into "
             "1 s windows, so a 4 s clip is 4 forward passes. Use --device "
             "cuda or cut --batches.",
    )
    p.add_argument(
        "--saturation", action="store_true",
        help="Also score UNRELATED pairs -- each target against a shuffled "
             "target from the same batch -- and report it as a row per "
             "column.\n\n"
             "Without it none of these numbers has a scale. An OpenL3 cosine "
             "distance of 0.042 against 0.047 could be eight near-perfect "
             "models 12%% apart, or a metric with no dynamic range reporting "
             "noise, and the table cannot tell you which. Saturation is what "
             "an arm that learned nothing about THIS target would score, so "
             "value/saturation says how much of the available range is "
             "actually being used. Same discipline as gt_loss and saturation "
             "on the plate side.\n\n"
             "Model-independent, so it is computed once on the first arm and "
             "reused. Roughly doubles the metric cost, which matters for "
             "--crepe and --openl3.",
    )
    p.add_argument(
        "--top-db", type=float, nargs="*", default=[80.0, 60.0, 40.0, 20.0],
        metavar="DB",
        help="dB floors to add as extra columns: `standard` with everything "
             "more than N dB below the clip's peak clamped flat, so those bins "
             "contribute nothing. The complement of --gamma -- that changes "
             "how much a quiet bin counts, this changes whether it counts at "
             "all. db80 reproduces `standard` exactly and is the wiring check. "
             "Pass no values to skip them.",
    )
    p.add_argument(
        "--mask", default=None, metavar="NPZ",
        help="Threshold cache from ds_masking.py --dump-thresholds. Adds a "
             "`mask` column whose floor is the target's own per-bin MPEG-1 "
             "masking threshold instead of a constant offset from its peak, "
             "and a mask+N cross against each --top-db rung. The point is a "
             "floor with NO free parameter: nobody can ask why 70. Supplying "
             "this also moves the plain dbN clamps ahead of the mel "
             "filterbank, since the mask has to be there and a table cannot "
             "mix conventions -- db70post is added to bridge back to the "
             "post-mel numbers already reported.")
    p.add_argument(
        "--shared-peak", action="store_true",
        help="Take the dB floor's reference peak from the TARGET for both "
             "members of a pair, instead of each signal using its own. "
             "librosa's convention is per-signal and every dB number reported "
             "so far used it, but for a COMPARISON it is wrong: a resynthesis "
             "3 dB quiet gets a 3 dB lower floor and keeps relatively more "
             "low-level detail than the target does, so the distance picks up "
             "a gain term that has nothing to do with spectral accuracy. The "
             "mask column has always been target-derived for exactly this "
             "reason; this makes the dB columns agree. Off by default so "
             "nothing already reported moves without being asked.")
    p.add_argument(
        "--pre-mel", action="store_true",
        help="Clamp the dB floors before the mel filterbank, without needing a "
             "threshold cache. Post-mel, a band holding one strong partial "
             "sits above the floor and shelters every quiet bin inside it, so "
             "the ladder barely bites; pre-mel those bins are raised "
             "individually. Implied by --mask, which has no choice about it. "
             "Adds db70post either way, since this changes what dbN means and "
             "sections 08-11 were computed under the post-mel convention.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--batches", type=int, default=32)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    mask = None
    if args.mask:
        mask = MaskCache(args.mask, args.device)
        print(f"masking thresholds: {args.mask} ({mask.note})")
        print("  dbN columns are now clamped PRE-mel so the table is one "
              "convention; db70post keeps the old post-mel one for continuity")
        print("  so `standard` vs `db80` and `db70post` vs `db70` are two "
              "pre/post-mel bridge pairs -- if each pair agrees, the stage "
              "change never mattered and the old numbers still stand")
    if args.pre_mel and not args.mask:
        print("dB floors clamped PRE-mel; db70post carries the post-mel "
              "convention sections 08-11 used, as a bridge")
    if args.shared_peak:
        print("dB floors take their peak from the TARGET for both signals; a "
              "gain offset no longer moves an arm's own floor")
    variants = build_variants(args.gamma, args.top_db, mask,
                              True if args.pre_mel else None, args.shared_peak)
    fns = {n: make_mfcc(args.device, **kw) for n, kw in variants}

    # Imported only when asked for, so torchcrepe stays an optional dependency
    # and every existing invocation keeps working without it.
    crepe = None
    if args.crepe:
        from diffsynth.perceptual.crepe import CREPELoss
        crepe = CREPELoss().to(args.device).eval()
    # Column order after the cepstral ones: LSD first since it is the metric
    # the paper already quotes, then the embedding distance.
    openl3 = None
    if args.openl3:
        from diffsynth.perceptual.openl3 import load_openl3_model, openl3_loss
        # base_dir is unused -- the manual .pth.tar path in that function is
        # commented out and it goes through torchopenl3's own loader.
        _ol3 = load_openl3_model("", input_repr="mel256", data="music",
                                 embed_size=512).to(args.device)
        def openl3(t, r):
            return float(openl3_loss(_ol3, t, r))
    extra = (["lsd"] + (["crepe"] if crepe is not None else [])
             + (["openl3"] if openl3 is not None else []))

    runs = [d for d in sorted(glob.glob(os.path.join(args.root, "*")))
            if os.path.isdir(d) and (not args.only or re.search(args.only, Path(d).name))]
    rows = []
    # Model-independent -- unrelated real audio against unrelated real audio --
    # so it is measured once on the first arm and reused for every row.
    sat, sat_per = {}, {k: [] for k in (
        [n for n, _ in variants] + ["lsd"]
        + (["crepe"] if crepe is not None else [])
        + (["openl3"] if openl3 is not None else []))}
    for d in runs:
        name = Path(d).name
        cdir = os.path.join(d, "tb_logs", "checkpoints")
        for ck in sorted(glob.glob(os.path.join(cdir, args.ckpt))):
            model, cfg, dm, note = pb.load_arm(d, os.path.basename(ck),
                                               args.device, args.batch_size)
            if model is None:
                print(f"{name:<20} {os.path.basename(ck):<14} skipped: {note}")
                continue
            if args.family:
                if name.startswith("real_") and not args.allow_seen:
                    print(f"{name:<20} skipped: trains on ood, the family is "
                          f"not held out for it (--allow-seen to override)")
                    continue
                from diffsynth.data import WaveParamDataset
                # params=False: the ood side has no ground-truth parameters,
                # which is how IdOodDataModule builds it too.
                ds = WaveParamDataset(cfg.data.ood_dir, cfg.data.sample_rate,
                                      cfg.data.length, False, False)
                ds.raw_files = [f for f in ds.raw_files
                                if os.path.basename(f).startswith(args.family)]
                if not ds.raw_files:
                    print(f"{name:<20} skipped: no files matching "
                          f"'{args.family}*' under {cfg.data.ood_dir}/audio")
                    continue
                vset = ds
                how = f"{len(ds.raw_files)} {args.family} files, whole ood pool"
            elif args.domain == "id":
                vset, how = pb.val_split(d, dm)
            else:
                vset, how = dm.ood_datasets["valid"], "ood split reproduced"
            if vset is None:
                print(f"{name:<20} skipped: {how}")
                continue
            loader = DataLoader(vset, batch_size=args.batch_size, num_workers=0)
            # LSD alongside, because it is the other audio number the paper
            # quotes and it disagrees with MFCC often enough that reading one
            # without the other has already been misleading once.
            cols = list(fns) + extra
            # Per-batch values rather than a running sum: the spread across
            # batches is the only estimate of noise available here, and several
            # of these columns differ between arms by less than it.
            per = {k: [] for k in cols}
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    if i >= args.batches:
                        break
                    batch = {k: (v.to(args.device) if torch.is_tensor(v) else
                                 {kk: vv.to(args.device) for kk, vv in v.items()})
                             for k, v in batch.items()}
                    resyn, _out = model(batch)
                    tgt = batch["audio"]
                    for k, fn in fns.items():
                        per[k].append(
                            F.l1_loss(fn(tgt, tgt), fn(resyn, tgt)).item())
                    per["lsd"].append(float(compute_lsd(tgt, resyn)))
                    if crepe is not None:
                        per["crepe"].append(
                            float(crepe.perceptual_loss(tgt, resyn)))
                    if openl3 is not None:
                        per["openl3"].append(openl3(tgt, resyn))
                    # Unrelated pairs, for the saturation row. Shuffled within
                    # the batch, so it is real audio against other real audio
                    # and shares every property of the eval set but identity.
                    if args.saturation and not sat:
                        sh = tgt[torch.randperm(tgt.shape[0], device=tgt.device)]
                        for k, fn in fns.items():
                            sat_per[k].append(
                                F.l1_loss(fn(tgt, tgt), fn(sh, tgt)).item())
                        sat_per["lsd"].append(float(compute_lsd(tgt, sh)))
                        if crepe is not None:
                            sat_per["crepe"].append(
                                float(crepe.perceptual_loss(tgt, sh)))
                        if openl3 is not None:
                            sat_per["openl3"].append(openl3(tgt, sh))
            n = len(per["lsd"])
            if not n:
                continue
            if args.saturation and not sat and sat_per["lsd"]:
                sat = {k: sum(v) / len(v) for k, v in sat_per.items()}
            ep = int(note.split()[-1]) if note.split()[-1].isdigit() else -1
            rows.append((name, ep, {k: sum(v) / len(v) for k, v in per.items()},
                         {k: _se(v) for k, v in per.items()}))
            print(f"{name:<20} {note:<12} {n} batches -- {how}")

    if not rows:
        return
    w = max(20, max(len(r[0]) for r in rows) + 2)
    where = (f"the {args.family} family" if args.family
             else f"the {args.domain} validation split")
    print(f"\n=== MFCC on {where}, up to "
          f"{args.batches * args.batch_size} clips")
    print("Every column has its own scale -- dB is ~4.34x natural log by base "
          "alone, and each\ngamma is a different power of the same numbers. "
          "Read each column DOWN, never across.\nWhat carries information is "
          "the ORDER of the arms within a column, and how that order\nchanges "
          "as the metric's compression moves from g1 (linear) toward "
          "as_logged (log).\ng0.3 is Stevens' law and is the rung to quote.\n")
    cols = [n for n, _ in variants] + extra
    print(f"{'run':<{w}}{'epoch':>7}" + "".join(f"{k:>12}" for k in cols))
    for name, ep, v, _e in rows:
        print(f"{name:<{w}}{ep:>7}" + "".join(f"{v[k]:>12.4f}" for k in cols))

    # Two references, without which none of the above has a scale.
    #
    # 2*se is the batch noise: the standard error of each arm's own mean across
    # batches, doubled, taken as the median over arms. An arm-to-arm difference
    # smaller than this is not a difference.
    #
    # saturation is what unrelated audio scores -- the value an arm that
    # learned nothing about THIS target would reach. spread/sat says how much
    # of the available range the eight arms occupy: a metric where every arm
    # sits at a few percent of saturation and they differ by a hair is not
    # resolving them, however clean the ordering looks.
    print()
    med = lambda xs: sorted(xs)[len(xs) // 2]
    print(f"{'2*se (median over arms)':<{w}}{'':>7}"
          + "".join(f"{2 * med([r[3][k] for r in rows]):>12.4f}" for k in cols))
    lo = {k: min(r[2][k] for r in rows) for k in cols}
    hi = {k: max(r[2][k] for r in rows) for k in cols}
    print(f"{'spread (max-min)':<{w}}{'':>7}"
          + "".join(f"{hi[k] - lo[k]:>12.4f}" for k in cols))
    if sat:
        print(f"{'SATURATION (unrelated)':<{w}}{'':>7}"
              + "".join(f"{sat[k]:>12.4f}" for k in cols))
        print(f"{'best as % of saturation':<{w}}{'':>7}"
              + "".join(f"{100 * lo[k] / sat[k]:>11.1f}%" for k in cols))
        print(f"{'spread as % of saturation':<{w}}{'':>7}"
              + "".join(f"{100 * (hi[k] - lo[k]) / sat[k]:>11.1f}%" for k in cols))
    else:
        print("\n  (no saturation reference -- pass --saturation. Without it "
              "a small\n   arm-to-arm difference cannot be told from a metric "
              "with no dynamic range.)")


if __name__ == "__main__":
    main()
