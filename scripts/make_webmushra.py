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

THE HIDDEN REFERENCE. webMUSHRA's manual documents `reference` as the reference
stimulus and `stimuli` as the rated conditions, and does NOT say whether the
reference is also inserted among them automatically. Standard MUSHRA requires
it, so this script writes it into `stimuli` EXPLICITLY as hidden_ref. VERIFY ON
THE DUMMY RUN: count the sliders. Six (3 arms + 2 anchors + 1 hidden ref) means
webMUSHRA is not auto-inserting and this is correct; seven means it is, and the
hidden_ref line should be dropped from the generated YAML.
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
    """{ir: {arm: path}} from eval_real_ir's '<ir>__<arm>.wav' naming."""
    clips: dict = defaultdict(dict)
    for p in sorted(d.glob("*__*.wav")):
        m = re.match(r"(.+?)__(.+)\.wav$", p.name)
        if m:
            clips[m.group(1)][m.group(2)] = p
    return {k: v for k, v in clips.items() if "target" in v}


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
                   help="which numbered file of each group, e.g. 2,3,4")
    p.add_argument("--groups", default="bright,dark,medium")
    p.add_argument("--match", choices=("lufs", "rms"), default="lufs",
                   help="lufs is BS.1770 and needs pyloudnorm; rms is the "
                        "total-energy convention and needs nothing.")
    p.add_argument("--target-db", type=float, default=-23.0,
                   help="LUFS (or dBFS RMS) every stimulus is matched to")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    out = args.out or Path(f"mushra_{args.id}")
    clips = discover(args.dir)
    if not clips:
        raise SystemExit(f"no <ir>__<arm>.wav under {args.dir}")

    want = [f"{g}_{n}" for g in args.groups.split(",")
            for n in args.pick.split(",")]
    # eval_real_ir stems carry a prefix (emt_140_bright_2); match on the tail so
    # the same --pick works whatever the recordings are called.
    chosen = [s for s in sorted(clips) if any(s.endswith(w) for w in want)]
    missing = [w for w in want if not any(s.endswith(w) for s in chosen)]
    if missing:
        print(f"  not found, skipped: {', '.join(missing)}")
    if not chosen:
        raise SystemExit(f"none of {want} under {args.dir}")

    arms = sorted({a for s in chosen for a in clips[s] if a != "target"})
    print(f"{len(chosen)} trials x {len(arms)} arms + hidden ref + 2 anchors")
    print(f"  arms: {', '.join(arms)}")
    print(f"  trials: {', '.join(chosen)}")

    audio_rel = Path("configs/resources/audio") / args.id
    audio_dir = out / audio_rel
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: measure everything, so the headroom gain can be common.
    loaded, peak_after = {}, 0.0
    for stem in chosen:
        for tag in ["target"] + arms:
            x, sr = sf.read(clips[stem][tag], dtype="float64", always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=1)
            g = 10.0 ** ((args.target_db - loudness(x, sr, args.match)) / 20.0)
            loaded[(stem, tag)] = (x * g, sr)
            peak_after = max(peak_after, float(np.abs(x * g).max()))

    # ONE gain for the whole test if anything would clip. Per-file limiting
    # would undo the match this script exists to apply.
    head = min(1.0, 0.98 / peak_after) if peak_after > 0.98 else 1.0
    if head < 1.0:
        print(f"  peak after matching {peak_after:.3f} -- applying a common "
              f"{20 * np.log10(head):+.2f} dB to every stimulus")

    for (stem, tag), (x, sr) in loaded.items():
        sf.write(audio_dir / f"{stem}__{tag}.wav", (x * head).astype(np.float32),
                 sr, subtype="PCM_24")
    print(f"  wrote {len(loaded)} stimuli to {audio_dir}")

    # --- the config -------------------------------------------------------
    title = args.title or f"EMT-140 plate resynthesis -- {args.id}"
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
        "      You will hear a reference recording of a real EMT-140 plate",
        "      reverb and several versions of it. Rate each version by how",
        "      different it sounds from the reference, on one scale, whatever",
        "      the cause of the difference. Use headphones in a quiet room.",
        "      One of the versions IS the reference and one is a lowpassed",
        "      anchor; neither is labelled.",
        "  - type: generic",
        "    id: instructions",
        "    name: Before you start",
        "    content: >",
        "      Switching between versions keeps the playback position, so",
        "      compare the same instant rather than remembering it. Every",
        "      slider must be moved before you can continue. There is no time",
        "      limit, but the whole test should take about 20 minutes.",
        # A nested page array beginning with `random` is webMUSHRA's idiom for
        # shuffling a SUBSET -- the manual documents `random` as the first
        # element of a pages array, and putting it at the top level would
        # shuffle the welcome and finish pages too. VERIFY on the dummy run
        # that trial order differs between two sessions; if it does not,
        # flatten this and accept fixed page order. With two listeners and
        # nine trials the per-page `randomize` below matters far more.
        "  - - random",
    ]
    for stem in chosen:
        L += [
            "    - type: mushra",
            f"      id: {stem}",
            f"      name: {stem.replace('_', ' ')}",
            "      showWaveform: true",
            "      enableLooping: true",
            # strict forces every slider to be moved, so an untouched slider
            # cannot be silently exported as a genuine 0 -- which is the flaw
            # in make_mushra.py's sliders starting at 0.
            "      strict: true",
            "      randomize: true",
            "      createAnchor35: true",
            "      createAnchor70: true",
            f"      reference: {audio_rel}/{stem}__target.wav",
            "      stimuli:",
            # Explicit hidden reference -- see the docstring. Drop this line if
            # the dummy run shows seven sliders rather than six.
            f"        hidden_ref: {audio_rel}/{stem}__target.wav",
        ]
        for a in arms:
            L.append(f"        {a}: {audio_rel}/{stem}__{a}.wav")
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
  2. six sliders per trial, not seven. Seven means webMUSHRA inserts the
     hidden reference itself and the hidden_ref lines should come out.
  3. trial order differs between two sessions. If it does not, the nested
     `random` is not doing anything -- harmless here, but know which it is.

Then back the results up off the box after every session.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
