"""Assert that space.py, gen.sh and jobs_emt7.txt describe the same plate.

    python -m src.emt.check

THE FAILURE THIS CATCHES HAS NO ERROR MESSAGE. --fmax and --fixed-mode-grid
appear in dataset generation and again in every training job. If they disagree,
the targets are rendered on one plate and the model's attempts on another;
nothing raises, nothing warns, and the loss simply parks at a floor with no
visible cause. --duration is the same: eps_ladder.sh never passes one, so a job
line that omits it silently uses train_encoder's 0.25 default against a 1.0 s
dataset.

Run it after editing any of the three. It imports nothing heavy, so it works in
either venv.
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SPACE = os.environ.get("SPACE", "emt10")
JOBS = os.path.join(ROOT, "scripts", f"jobs_{SPACE}.txt")
GEN = os.path.join(HERE, "gen.sh")


def _space() -> dict:
    ns: dict = {}
    exec(open(os.path.join(HERE, "space.py")).read(), ns)
    return ns


def main() -> int:
    ns = _space()
    n = ns["NUMERICS"][SPACE]
    want = {
        "--fmax": float(n["fmax"]),
        "--fixed-mode-grid": "%d,%d" % tuple(n["grid"]),
        "--duration": float(n["duration"]),
    }
    print(f"space.py [{SPACE}]:  " + "  ".join(f"{k} {v}" for k, v in want.items()))

    bad = 0
    jobs = [l for l in open(JOBS).read().splitlines()
            if l.strip() and not l.lstrip().startswith("#")]
    print(f"\n{JOBS}: {len(jobs)} job line(s)")
    for i, line in enumerate(jobs):
        arm = re.search(r'ARMS="([^"]*)"', line)
        out = re.search(r"OUT=(\S+)", line)
        for flag, exp in want.items():
            m = re.search(re.escape(flag) + r"[= ]+([^\s\"]+)", line)
            got = m.group(1) if m else None
            same = got is not None and (
                float(got) == exp if isinstance(exp, float) else got == exp)
            if not same:
                bad += 1
                print(f"  MISMATCH {arm.group(1) if arm else i}: "
                      f"{flag} is {got!r}, space.py says {exp!r}")
        if arm and out:
            print(f"  ok  {arm.group(1):<16} -> {out.group(1)}")

    # gen.sh must READ the values rather than repeat them; a literal there is
    # the same trap one level down.
    gen = open(GEN).read()
    print(f"\n{GEN}:")
    if "space.py" in gen and "NUMERICS" in gen and 'SPACE' in gen:
        print(f"  ok  reads space.py's NUMERICS[SPACE] rather than hardcoding")
    else:
        bad += 1
        print("  MISMATCH: does not read space.py -- values could drift")
    for flag in ("--fmax", "--fixed-mode-grid", "--duration"):
        if re.search(re.escape(flag) + r"\s+[0-9]", gen):
            bad += 1
            print(f"  MISMATCH: {flag} has a literal value; it must come from "
                  f"space.py")

    # The arms have to be an eps-matched trio or the comparison is confounded.
    arms = [re.search(r'ARMS="([^"]*)"', l).group(1) for l in jobs
            if re.search(r'ARMS="([^"]*)"', l)]
    trio = {a for a in arms if a != "L1_STFT"} | {"L1_STFT"}
    eps = {re.search(r"(?:eps|hyb)(\S+)$", a).group(1)
           for a in trio if re.search(r"(?:eps|hyb)(\S+)$", a)}
    print(f"\narms: {sorted(trio)}")
    if len(eps) > 1:
        bad += 1
        print(f"  MISMATCH: the compressed arms use different eps {sorted(eps)}. "
              f"A hybrid/log pair differing in eps AND in the linear term is "
              f"not single-variable -- that was gamma_ppre's flaw (hyb1e2 "
              f"against eps1e7, five decades apart).")
    else:
        print(f"  ok  eps-matched at {eps.pop() if eps else 'n/a'}")

    print("\n" + ("OK" if not bad else f"{bad} MISMATCH(ES) -- fix before generating"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
