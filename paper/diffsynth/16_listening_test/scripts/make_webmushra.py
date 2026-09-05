"""Turn an eval_real_ir output directory into a webMUSHRA test, loudness-matched.

    python scripts/make_webmushra.py --dir emt11_listen_last --id emt11
    python scripts/make_webmushra.py --dir emt12_listen_last --id emt12 --pick 2,3,4

Then, on the machine that holds webMUSHRA:

    cp -r mushra_emt11/configs/* ~/webMUSHRA/configs/
    python scripts/mushra_server.py --root ~/webMUSHRA
    # ssh -L 8000:localhost:8000 <user>@<host>, then
    # http://localhost:8000/?config=emt11.yaml

WHY webMUSHRA RATHER THAN scripts/make_mushra.py. That one is a good single
listener self-test and says so in its own docstring: one seed baked into the
page, so every listener sees the same blinding, and results land in a textarea.
Neither survives contact with a second listener. webMUSHRA gives per-session
condition randomisation (randomize: true), both BS.1534 anchors from a boolean,
and CSV collection, and it is citable -- Schoeffler et al., JORS 2018 -- which
removes the "how do you know your MUSHRA is correct" objection that a homebrew
page invites. Keep make_mushra.py for checking stimuli before a panel hears
them.

WHAT THIS SCRIPT ACTUALLY DOES, since webMUSHRA does none of it:

  LOUDNESS MATCHING, and it is the reason the script exists. eval_real_ir
  peak-normalises. In a listening test that is a confound: a darker render
  carries less energy, plays quieter, and listeners rate quieter as worse. The
  whole question here is whether the arms differ in TIMBRE -- emt11's arms sit
  at f11 65, 82 and 43 Hz -- so letting level ride along would let a level
  difference be scored as the thing under test. Every stimulus is matched to a
  common LUFS by ITU-R BS.1770.

  ONE CAVEAT ON BS.1770 AND IMPULSE RESPONSES. The gate drops blocks below -70
  LUFS absolute and -10 LU relative, which on a decaying IR discards the tail
  and effectively measures the first few hundred ms. That is defensible -- it is
  roughly what the ear weights -- but it is not the same as matching total
  energy, and an arm whose tail is long gets no credit for it in the gain. Stated
  rather than hidden; --match rms is there if you want the other convention.

  A COMMON HEADROOM GAIN afterwards, not per-file limiting. If matching pushes
  anything past full scale, ONE gain is applied to every stimulus in the test so
  the match is preserved exactly. Per-file limiting would undo it.

WHAT IT DELIBERATELY DOES NOT DO.

  IT PRESENTS BARE IMPULSE RESPONSES. Nobody listens to a plate this way -- a
  plate is a send effect and what reaches a listener is dry material convolved
  with the IR. A MUSHRA on convolved drums or voice would be closer to the
  application and probably more sensitive, since transients expose the onset
  dispersion these arms differ in. Bare IRs are what eval_real_ir writes and
  what every number in this project is computed on, so they are what makes the
  listening test comparable to the metrics -- but "the arms are distinguishable
  on bare IRs" is a weaker claim than "on music", and only the first is tested
  here.

  IT DOES NOT SCREEN HEADPHONES. With two listeners on one machine that is what
  the room is for; over the network it would matter and the hidden-reference
  post-screen becomes the defence.

FIVE CONDITIONS PER TRIAL: the three arms, the hidden reference, one anchor.

  THE HIDDEN REFERENCE IS webMUSHRA'S. Its manual documents `reference` and
  `stimuli` separately and never says whether the reference is also inserted
  among the rated conditions. It is -- measured in the browser: listing it in
  `stimuli` as well gave seven sliders rather than six. So `stimuli` holds the
  arms only.

  ONE ANCHOR, not BS.1534's two. The 7 kHz mid anchor calibrates the top of the
  scale where codec conditions crowd near transparent; here nothing is near
  transparent (1.6-2.0 x saturation) and the source is a male voice with almost
  nothing above 7 kHz, so a 7 kHz lowpass of it is very nearly the reference.
  It would sit beside the hidden reference, calibrate nothing, and cost a
  slider and a listen on all nine trials.
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def discover(d: Path):
    """{trial: {condition: path}} from either renderer's naming.

    TWO LAYOUTS, one rule. eval_real_ir writes '<ir>__<arm>.wav' and
    ds_eval_folder writes '<folder>__<stem>__<arm>.wav'. Splitting on the LAST
    '__' handles both: everything before it names the trial, the tail names the
    condition. The old non-greedy '(.+?)__(.+)' took the FIRST separator
    instead, which on a diffsynth render made every clip in a folder collapse
    into one trial called 'moog-minitaur' whose conditions were
    'doubles-48-1__target', 'doubles-49-2__target' and so on -- nine hundred
    conditions in one trial rather than nine trials of five.

    Neither renderer puts '__' inside an arm name, so the last separator is
    unambiguous; a stem may contain one, which is why the split is from the
    right and not the left.
    """
    clips: dict = defaultdict(dict)
    for p in sorted(d.glob("*__*.wav")):
        trial, _, cond = p.stem.rpartition("__")
        if trial:
            clips[trial][cond] = p
    return {k: v for k, v in clips.items() if "target" in v}


def spread_balanced(clips: dict, spread_re: str, balance_re: str | None,
                    n_trials: int, order_seed: int, distinct: int = 0):
    """n_trials clips spread along one axis, balanced across a second.

    WHY BOTH AXES. Selecting on the MIDI note alone gave nine Moog trials whose
    velocities were 32, 32, 32, 32, 32, 95, 95, 95, 95 -- every low note quiet
    and every high note loud, because within a note the stems sort by file id
    and the pack is laid out note-major. Velocity is not a nuisance here: on
    the Minitaur it drives the filter envelope, which is exactly what cutoff
    and Q are being asked to recover, and the one within-note contrast in that
    test moved the ratings by 30-47 points. Confounded with pitch, neither
    effect can be read.

    So the primary axis is spread by rank and the secondary is cycled, giving
    each velocity an equal share of the trials at spread-out notes.

    THE ORDER IS SHUFFLED, deterministically. webMUSHRA's page order is fixed
    (a nested `random` block is unsupported -- see the note further down), so
    without this the listener walks up the keyboard, and any drift in how they
    use the scale over a 20-minute sitting is aliased onto pitch. A fixed seed
    keeps every listener on the same order, which is what makes their ratings
    poolable.
    """
    import random as _random

    axis: dict = defaultdict(list)
    for s in sorted(clips):
        m = re.search(spread_re, s)
        if not m:
            continue
        b = re.search(balance_re, s) if balance_re else None
        axis[float(m.group(1))].append((float(b.group(1)) if b else 0.0, s))
    if not axis:
        raise SystemExit(f"--spread-re {spread_re!r} matched no trial name; "
                         f"first is {sorted(clips)[0]!r}")

    keys = sorted(axis)
    vals = sorted({v for lst in axis.values() for v, _s in lst})
    # Slots even in RANK over the primary axis, not in value: the pack's notes
    # are unequally populated and value spacing would cluster where files are.
    #
    # WITH MORE TRIALS THAN NOTES, every note appears once and the surplus is
    # spread by rank over the range. Not "pick N notes and repeat them": with
    # 12 trials and 8 notes, rank-spacing 12 slots would have excluded two
    # interior semitones, and a reader has no way to tell a mechanical rule
    # from a convenient one when the result is "these two notes are missing".
    # Covering every note removes the question, and which notes are heard
    # TWICE is a far weaker lever than which are heard at all.
    if n_trials >= len(keys):
        extra = n_trials - len(keys)
        more = [keys[round(i * (len(keys) - 1) / max(extra - 1, 1))]
                for i in range(extra)] if extra else []
        slots = sorted(keys + more)
    else:
        slots = [keys[round(i * (len(keys) - 1) / (n_trials - 1))]
                 for i in range(n_trials)]

    # The assignment is BALANCED THEN SHUFFLED, not cycled. Walking
    # 32, 64, 95, 32, 64, 95 up the keyboard keeps the counts equal but makes
    # velocity a deterministic function of note rank -- a periodic confound
    # instead of a monotonic one, and just as unreadable. Shuffling a balanced
    # multiset keeps three of each while breaking the correspondence.
    rnd = _random.Random(order_seed)
    plan = [vals[i % len(vals)] for i in range(len(slots))]
    rnd.shuffle(plan)

    chosen, used = [], set()
    for want, k in zip(plan, slots):
        # Fall through to the other values when a note lacks the planned one or
        # its stem is already taken, so a hole in the pack costs balance rather
        # than dropping the trial.
        order = [want] + [v for v in vals if v != want]
        for w in order:
            hit = next((s for v, s in axis[k] if v == w and s not in used),
                       None)
            if hit:
                chosen.append(hit)
                used.add(hit)
                break
        else:
            hit = next((s for _v, s in axis[k] if s not in used), None)
            if hit:
                chosen.append(hit)
                used.add(hit)

    if balance_re:
        got: dict = defaultdict(int)
        for s in chosen:
            m = re.search(balance_re, s)
            got[m.group(1) if m else "?"] += 1
        print("  balanced on --balance-re: "
              + ", ".join(f"{k}x{n}" for k, n in sorted(got.items())))
    rnd.shuffle(chosen)
    return chosen


def select(clips: dict, groups: str, pick: str, spread_re: str | None,
           n_trials: int):
    """The trials to run, chosen deterministically -- never per session.

    WHY NOT RANDOM PER SESSION. With eight listeners, drawing a fresh subset
    each sitting gives every clip one or two ratings, and then between-listener
    and between-stimulus variance are inseparable: no per-trial mean exists to
    compare arms on, and no per-listener offset can be fitted out. Everyone
    rating the SAME trials is what makes "arm A beat arm B over N listeners" a
    statement about the arms. Randomisation in MUSHRA belongs at the condition
    level -- which slider holds which arm -- and webMUSHRA already does that
    per session with `randomize: true`.

    --spread-re picks n_trials clips spaced evenly along a number in the trial
    name, rather than n_trials at random. On the Moog that number is the MIDI
    note, so the register is covered by construction instead of by luck, which
    on eight draws is not a risk worth taking.
    """
    want = [f"{g}_{n}" for g in groups.split(",") for n in pick.split(",")]
    # eval_real_ir stems carry a prefix (emt_140_bright_2); match on the tail so
    # the same --pick works whatever the recordings are called.
    chosen = [s for s in sorted(clips) if any(s.endswith(w) for w in want)]
    missing = [w for w in want if not any(s.endswith(w) for s in chosen)]
    if missing:
        print(f"  not found, skipped: {', '.join(missing)}")
    if not chosen:
        raise SystemExit(f"none of {want} under the render directory")
    return chosen


def energy_above(x: np.ndarray, sr: int, fc: float) -> float:
    """Fraction of this signal's energy above fc, as a percentage.

    Reported per trial so the anchor can be checked rather than assumed. This
    is the number that was wrong in the plate test: BS.1534-3's low anchor is a
    3.5 kHz lowpass of the reference, which removes an obvious amount from
    broadband programme material, but the EMT-140's darker settings roll off
    well below that. Lowpassing them at 3.5 kHz removed nearly nothing, so the
    anchor reached the listener sounding like the reference. An anchor that is
    not audibly worse than the reference calibrates nothing -- it adds a slider
    that gets rated near 100 and compresses the range the arms sit in.

    If this column reads a fraction of a per cent on any trial, the anchor is
    inaudible there and --anchor-hz needs to come down further.
    """
    p = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.shape[0], 1.0 / sr)
    tot = float(p.sum())
    return 100.0 * float(p[f > fc].sum()) / tot if tot > 0 else 0.0


def lowpass(x: np.ndarray, sr: int, fc: float, width_oct: float = 1.0):
    """Lowpass at fc with a raised-cosine skirt over `width_oct` above it.

    Not a brickwall: zeroing bins outright gives a sinc in the time domain,
    and on an impulse response the pre-ringing that produces is audible as a
    distinct artefact rather than as the dullness an anchor is meant to be.
    The skirt costs nothing and keeps the degradation spectral.
    """
    n = x.shape[0]
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    hi = fc * 2.0 ** width_oct
    g = np.ones_like(f)
    band = (f > fc) & (f < hi)
    g[band] = 0.5 * (1.0 + np.cos(np.pi * np.log2(f[band] / fc) / width_oct))
    g[f >= hi] = 0.0
    return np.fft.irfft(X * g, n)


def loudness(x: np.ndarray, sr: int, mode: str) -> float:
    """Return the measured level in dB, by BS.1770 or plain RMS."""
    if mode == "rms":
        return 20.0 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-12))
    import pyloudnorm  # noqa: PLC0415  -- optional, only for --match lufs
    return float(pyloudnorm.Meter(sr).integrated_loudness(x))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", type=Path, required=True,
                   help="an eval_real_ir output directory")
    p.add_argument("--id", required=True,
                   help="testId and config filename stem, e.g. emt11")
    p.add_argument("--out", type=Path, default=None,
                   help="default mushra_<id>")
    p.add_argument("--pick", default="2,3,4",
                   help="eval_real_ir layout: which numbered file of each "
                        "group, e.g. 2,3,4. Ignored when --spread-re is given.")
    p.add_argument("--groups", default="bright,dark,medium",
                   help="eval_real_ir layout only.")
    p.add_argument("--spread-re", default=None, metavar="REGEX",
                   help="ds_eval_folder layout: choose --trials clips spaced "
                        "evenly along a number captured from the trial name, "
                        r"e.g. 'doubles-(\d+)-' for the Moog's MIDI note. "
                        "Deterministic and identical for every listener -- "
                        "MUSHRA randomises which SLIDER holds which arm, not "
                        "which trials you get, and with a handful of listeners "
                        "a per-session subset leaves each clip with one or two "
                        "ratings and nothing to average.")
    p.add_argument("--balance-re", default=None, metavar="REGEX",
                   help="A SECOND axis to spread evenly across while "
                        r"--spread-re walks the first, e.g. '-\d+-(\d+)-' for "
                        "the Moog's velocity. Without it, selecting on note "
                        "alone gave nine trials at velocity 32,32,32,32,32,"
                        "95,95,95,95 -- velocity perfectly confounded with "
                        "pitch, and neither effect readable.")
    p.add_argument("--order-seed", type=int, default=0, metavar="N",
                   help="Shuffles the PAGE order, deterministically. "
                        "webMUSHRA cannot randomise pages per session, so "
                        "without this the listener walks up the keyboard and "
                        "any drift in their use of the scale is aliased onto "
                        "pitch. Same seed for every listener keeps ratings "
                        "poolable.")
    p.add_argument("--trials", type=int, default=9, metavar="N",
                   help="How many trials --spread-re selects. Nine matches "
                        "the plate test, so the two sittings are comparable "
                        "in length.")
    p.add_argument("--match", choices=("lufs", "rms"), default="lufs",
                   help="lufs is BS.1770 and needs pyloudnorm; rms is the "
                        "total-energy convention and needs nothing.")
    p.add_argument("--target-db", type=float, default=-23.0,
                   help="LUFS (or dBFS RMS) every stimulus is matched to")
    p.add_argument("--mono", action="store_true",
                   help="write single-channel. The default duplicates the "
                        "channel because webMUSHRA plays a 1-channel buffer "
                        "in the left ear only.")
    p.add_argument("--anchor-hz", type=float, default=1500.0, metavar="HZ",
                   help="Low-anchor cutoff, ONE value for every trial. "
                        "BS.1534-3 specifies 3.5 kHz, which is right for "
                        "broadband programme material and wrong here: the dark "
                        "EMT-140 settings have almost no energy above it, so "
                        "the anchor arrives sounding like the reference and "
                        "calibrates nothing on exactly those trials. Lowering "
                        "the number is a stated, application-specific "
                        "deviation and keeps the anchor identical across "
                        "trials. Pass 3500 for the standard value.")
    p.add_argument("--material", default="EMT-140 plate",
                   help="Names the test in its title bar. Pass e.g. "
                        "'Moog Minitaur' for the diffsynth version.")
    p.add_argument("--reference-is", default="a real EMT-140 plate reverb",
                   metavar="TEXT",
                   help="How the welcome page describes the reference. The "
                        "listener is told what they are comparing against, so "
                        "this has to match the material or the instructions "
                        "are wrong on their face.")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    out = args.out or Path(f"mushra_{args.id}")
    clips = discover(args.dir)
    if not clips:
        raise SystemExit(f"no <ir>__<arm>.wav under {args.dir}")

    if args.spread_re:
        chosen = spread_balanced(clips, args.spread_re, args.balance_re,
                                 args.trials, args.order_seed)
    else:
        chosen = select(clips, args.groups, args.pick, None, args.trials)

    arms = sorted({a for s in chosen for a in clips[s] if a != "target"})
    # Every trial must offer every arm, or listeners get a different number of
    # sliders on different pages and the per-arm means are over different
    # trials. Better to drop the trial than to publish a ragged test.
    ragged = [s for s in chosen if any(a not in clips[s] for a in arms)]
    if ragged:
        print(f"  dropped, missing an arm: {len(ragged)} of {len(chosen)}")
        for s in ragged[:4]:
            miss = ", ".join(a for a in arms if a not in clips[s])
            print(f"    {s}\n      has {', '.join(sorted(clips[s]))}"
                  f"\n      missing {miss}")
        chosen = [s for s in chosen if s not in ragged]
    if not chosen:
        # Almost always a render directory holding two different runs: `arms`
        # is the UNION over trials, so a stale set of clips rendered from other
        # checkpoints contributes arm names no current trial has, and every
        # trial then looks incomplete. Re-render to a fresh --render-out rather
        # than adding to one.
        raise SystemExit(
            f"no trial has all of: {', '.join(arms)}\n"
            f"That set is the union over trials, so this usually means "
            f"{args.dir} holds renders from more than one run. Render to an "
            f"empty directory and point --dir at that.")
    print(f"{len(chosen)} trials x {len(arms)} arms + hidden ref + 1 anchor")
    print(f"  arms: {', '.join(arms)}")
    print(f"  trials: {', '.join(chosen)}")

    audio_rel = Path("configs/resources/audio") / args.id
    audio_dir = out / audio_rel
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: measure everything, so the headroom gain can be common.
    loaded, peak_after = {}, 0.0
    above = {}
    for stem in chosen:
        for tag in ["target"] + arms:
            x, sr = sf.read(clips[stem][tag], dtype="float64", always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=1)
            g = 10.0 ** ((args.target_db - loudness(x, sr, args.match)) / 20.0)
            loaded[(stem, tag)] = (x * g, sr)
            peak_after = max(peak_after, float(np.abs(x * g).max()))
            if tag != "target":
                continue
            # THE ANCHOR IS BUILT HERE, not by webMUSHRA, purely so the cutoff
            # is ours to set. It is loudness-matched with everything else --
            # a lowpass removes energy, and an anchor that also plays quieter
            # would be rated worse for the wrong reason.
            above[stem] = energy_above(x, sr, args.anchor_hz)
            a = lowpass(x, sr, args.anchor_hz)
            ga = 10.0 ** ((args.target_db - loudness(a, sr, args.match)) / 20.0)
            loaded[(stem, "anchor")] = (a * ga, sr)
            peak_after = max(peak_after, float(np.abs(a * ga).max()))

    print(f"  low anchor: {args.anchor_hz:g} Hz lowpass of each reference"
          + ("  [BS.1534-3 value]" if abs(args.anchor_hz - 3500) < 1
             else "  [lowered from BS.1534-3's 3500 Hz]"))
    print(f"{'':<4}{'trial':<28}{'energy above cutoff':>21}")
    for stem in chosen:
        flag = "   <-- anchor barely differs" if above[stem] < 1.0 else ""
        print(f"{'':<4}{stem:<28}{above[stem]:>20.2f}%{flag}")

    # ONE gain for the whole test if anything would clip. Per-file limiting
    # would undo the match this script exists to apply.
    head = min(1.0, 0.98 / peak_after) if peak_after > 0.98 else 1.0
    if head < 1.0:
        print(f"  peak after matching {peak_after:.3f} -- applying a common "
              f"{20 * np.log10(head):+.2f} dB to every stimulus")

    # DUAL MONO, and it is not cosmetic. The plate has one pickup so every
    # stimulus is genuinely one channel, but webMUSHRA's Web Audio graph routes
    # a 1-channel buffer to output channel 0 only -- measured: it plays in the
    # LEFT EAR ALONE. A listener judging a monaural signal presented to one ear
    # is not judging what the metric measured, and the anchors webMUSHRA builds
    # itself would come out the same way. Duplicating the channel puts it back
    # in the middle of the head; it adds no spatial information, which is the
    # right call anyway since a real EMT-140 has two pickups and stereo width
    # would be a cue that has nothing to do with what the losses differ on.
    # --mono writes single-channel if some other player needs it.
    for (stem, tag), (x, sr) in loaded.items():
        y = (x * head).astype(np.float32)
        if not args.mono:
            y = np.stack([y, y], axis=-1)
        sf.write(audio_dir / f"{stem}__{tag}.wav", y, sr, subtype="PCM_24")
    ch = "mono" if args.mono else "dual-mono (both ears)"
    print(f"  wrote {len(loaded)} stimuli to {audio_dir}  [{ch}]")

    # --- the config -------------------------------------------------------
    title = args.title or f"{args.material} resynthesis -- {args.id}"
    L = [
        f'testname: "{title}"',
        f"testId: {args.id}",
        "bufferSize: 2048",
        "stopOnErrors: true",
        "showButtonPreviousPage: true",
        "remoteService: service/write.php",
        "",
        "pages:",
        "  - type: generic",
        "    id: welcome",
        "    name: Welcome",
        "    content: >",
        f"      You will hear a reference recording of {args.reference_is}",
        "      and several versions of it. Rate each version by how",
        "      different it sounds from the reference, on one scale, whatever",
        "      the cause of the difference. Use headphones in a quiet room.",
        "      One of the versions IS the reference and one is a lowpassed",
        "      anchor; neither is labelled.",
    ]
    # PAGE ORDER IS FIXED, and that is a measured decision rather than an
    # oversight. The manual documents `random` as the first element of a pages
    # array; a NESTED array beginning with `random`, which would shuffle only
    # the trials and leave the welcome and finish pages in place, is not
    # supported -- webMUSHRA reads the inner list as a page, finds no `type`,
    # and dies with "Error: Type not specified". Putting `random` at the top
    # level instead would shuffle the instructions and the finish page into the
    # middle of the test, which is worse than a fixed order.
    #
    # What is lost is small. Page order guards against fatigue and learning
    # effects across trials, which matter for a panel; CONDITION order within a
    # trial is what guards against a listener learning that slot C is always
    # the log arm, and that is `randomize: true` below, per page, documented,
    # and kept. If you later need page order shuffled for a real panel, the
    # robust way is one config file per listener with `chosen` pre-shuffled --
    # deterministic, recorded, and not dependent on an undocumented feature.
    for stem in chosen:
        L += [
            "  - type: mushra",
            f"    id: {stem}",
            f"    name: {stem.replace('_', ' ')}",
            "    showWaveform: true",
            "    enableLooping: true",
            # strict forces every slider to be moved, so an untouched slider
            # cannot be silently exported as a genuine 0 -- which is the flaw
            # in make_mushra.py's sliders starting at 0.
            "    strict: true",
            "    randomize: true",
            # ONE ANCHOR, not two. BS.1534 specifies a 3.5 kHz low anchor and a
            # 7 kHz mid anchor, but the mid anchor exists to calibrate the top
            # of the scale where codec conditions cluster near transparent, and
            # neither half of that applies here. These conditions sit at 1.6-2.0
            # x saturation, nowhere near the reference, and the source is a male
            # voice with almost no energy above 7 kHz -- so a 7 kHz lowpass of
            # it is very nearly the reference itself. It would sit next to the
            # hidden reference, calibrate nothing, and cost a slider and a
            # listen on every trial. make_mushra.py drops it for diffsynth on
            # the same grounds, arrived at from Nyquist rather than from the
            # source's bandwidth.
            # createAnchor35 is OFF: the anchor is written by this script at
            # --anchor-hz and listed among the stimuli below. webMUSHRA's own
            # anchor is fixed at 3.5 kHz with no way to change it, and on the
            # dark EMT-140 references that removed so little that the anchor
            # was indistinguishable from the reference -- the one flaw the
            # first plate test had.
            "    createAnchor35: false",
            f"    reference: {audio_rel}/{stem}__target.wav",
            "    stimuli:",
            # NO EXPLICIT hidden_ref. webMUSHRA inserts the reference among the
            # conditions itself -- undocumented, and measured in the browser:
            # listing it here as well produced SEVEN sliders (3 arms + two
            # copies of the reference + 2 anchors) instead of six.
        ]
        for a in arms:
            L.append(f"      {a}: {audio_rel}/{stem}__{a}.wav")
        # The anchor rides in as an ordinary stimulus. randomize: true means
        # its position changes per session and the key is never shown, so it
        # is as blind as the arms are.
        L.append(f"      anchor: {audio_rel}/{stem}__anchor.wav")
    # NO QUESTIONNAIRE. This is an instrument a few lab members take one at a
    # time, not software handed to strangers -- so the machinery a panel needs
    # to identify who submitted what is machinery nobody here needs, and asking
    # for a name would put personal data in a CSV for no analysis anyone wants
    # to run. Sessions are still separable: webMUSHRA mints a uuid per page
    # load and mushra_server.py files each one as its own
    # results/<testId>/session_<uuid>.json with its mtime, which is enough to
    # tell one sitting from the next when there are two of you.
    L += [
        "  - type: finish",
        "    name: Thank you",
        "    content: Your responses have been recorded.",
        "    showResults: true",
        "    writeResults: true",
        "",
    ]

    cfg = out / "configs" / f"{args.id}.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("\n".join(L))
    mb = sum(f.stat().st_size for f in audio_dir.iterdir()) / 1e6
    print(f"  wrote {cfg}  ({mb:.1f} MB of audio)")

    print(f"""
Next, on the machine that will serve it:

  git clone https://github.com/audiolabs/webMUSHRA.git   # once
  cp -r {out}/configs/* ~/webMUSHRA/configs/
  python scripts/mushra_server.py --root ~/webMUSHRA

then tunnel in from wherever you are listening and open the config:

  ssh -L 8000:localhost:8000 <user>@<host>
  # http://localhost:8000/?config={args.id}.yaml

mushra_server.py replaces webMUSHRA's PHP result service -- same POST, same
CSVs, no PHP and no root -- and serves the static files itself. If the box does
have PHP, `cd webMUSHRA && chmod -R a+w results && php -S localhost:8000` is
equivalent.

DO A COMPLETE DUMMY SESSION BEFORE ANYONE ELSE TOUCHES IT, and check three
things -- all three fail silently:
  1. a CSV appears in webMUSHRA/results/{args.id}/ when you finish. Under
     mushra_server.py the terminal prints a line the moment it arrives, so
     silence there means the finish page is not set writeResults: true. Under
     php -S it is the permissions on results/, every time.
  2. FIVE sliders per trial: 3 arms, the hidden reference webMUSHRA inserts
     itself, and the one low anchor this script writes. Six or seven means an
     anchor or a reference is being added twice -- check createAnchor35 is
     false, since webMUSHRA would otherwise add a second, fixed 3.5 kHz one.
  3. CONDITION order differs between two sessions -- the slot the same arm
     lands in should move. That is `randomize: true`. Page order is fixed by
     design; see the note in this script above the mushra pages.

Then back the results up off the box after every session.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
