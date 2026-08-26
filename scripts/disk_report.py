"""What is taking the disk, and what is safe to delete because nothing reads it.

    python scripts/disk_report.py
    python scripts/disk_report.py --min-mb 200 --emit       # print rm commands

REPORTS ONLY. It never deletes; --emit prints the commands for you to read and
run yourself.

THE RISK IT EXISTS TO REMOVE. Half the checkpoints here are branch points --
every `--resume` in scripts/ and every `RESUME=` in a jobs file names one, and
deleting the wrong one silently costs a campaign that has already run. `du`
sorts by size and knows nothing about that. This greps every script and jobs
file for referenced paths first, then classifies.

WHAT IT CLASSIFIES

  LOAD-BEARING   a checkpoint or directory some script resumes from, or a
                 dataset a jobs file names. Never suggested for deletion, even
                 when large.
  NUMBERS KEPT   a finished run whose scalars.csv or history.json exists, so
                 the event files and intermediate checkpoints behind it can go
                 without losing a single number. This is where most of the
                 space is.
  REGENERABLE    datasets and listening bundles. Deletable, with the cost of
                 regenerating stated rather than implied.
  ORPHAN         nothing in the repo references it and it holds no numbers.

The distinction that matters is NUMBERS KEPT: results/diffsynth run dirs are
~300 MB of TensorBoard events and Lightning checkpoints wrapped around a
few hundred KB of scalars.csv, and monitor_diffsynth reads the CSV. Deleting
the event files costs the TensorBoard UI and nothing else. What it does cost is
the ability to resume, so a run still in flight is never listed.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOTS = ["results", "data", "external/diffsynth/data", "logs",
         os.path.expanduser("~/gpu_queue_logs")]
# Anything a script names as a checkpoint to resume, a dataset to read, or an
# output another job branches from.
REF_PATTERNS = [
    r"--resume\s+(\S+)", r"RESUME=(\S+)", r"CKPT=(\S+)",
    r"TRAIN=(\S+)", r"VAL=(\S+)", r"ID_DIR=(\S+)", r"OOD_DIR=(\S+)",
    r"--data-dir\s+(\S+)", r"--val-data-dir\s+(\S+)", r"--wav-dir\s+(\S+)",
    r"data\.id_dir=(\S+)", r"data\.ood_dir=(\S+)",
    # THE DEFAULTS, and missing them is the dangerous direction. eps_ladder.sh
    # carries TRAIN=${TRAIN:-data/train-p99} and VAL=${VAL:-data/val-p99}, and
    # every jobs file that does not override them depends on those values. The
    # $-filter below drops any capture containing a variable, which silently
    # excluded the largest dataset in the repo from protection.
    r"\$\{[A-Za-z_][A-Za-z0-9_]*:-([^}]+)\}",
]


def du(path: str) -> int:
    try:
        out = subprocess.run(["du", "-sb", path], capture_output=True,
                             text=True, timeout=300)
        return int(out.stdout.split()[0]) if out.stdout.strip() else 0
    except Exception:
        return 0


def referenced(root: str) -> set[str]:
    """Every path any script or jobs file names, normalised.

    Includes the parent of a checkpoint path: `RESUME=results/ddsp/x/enc.pt`
    makes results/ddsp/x load-bearing, not just that one file.
    """
    out: set[str] = set()
    files = (glob.glob(os.path.join(root, "scripts", "*")) +
             glob.glob(os.path.join(root, "src", "**", "*.sh"), recursive=True) +
             glob.glob(os.path.join(root, "src", "**", "*.py"), recursive=True))
    for f in files:
        if os.path.isdir(f) or os.path.basename(f) == "disk_report.py":
            continue        # its own docstring quotes example paths
        try:
            text = open(f, errors="ignore").read()
        except Exception:
            continue
        for pat in REF_PATTERNS:
            for m in re.findall(pat, text):
                p = m.strip().strip('"\'').rstrip(",")
                if not p or p.startswith("-") or "{" in p or "$" in p:
                    continue
                # A ${VAR:-default} capture is only a path if it looks like one;
                # the same pattern also catches numbers, flags and loss names.
                if "/" not in p and not p.startswith(("data", "results")):
                    continue
                p = os.path.normpath(p)
                out.add(p)
                # RESUME= names a run directory; --resume names a file in one.
                while p not in (".", "/", ""):
                    p = os.path.dirname(p)
                    if p:
                        out.add(p)
    return out


def is_live(d: str, max_age_min: float) -> bool:
    """Written to recently -- a run in flight, never suggested for deletion."""
    newest = 0.0
    for dp, _, fs in os.walk(d):
        for f in fs:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(dp, f)))
            except OSError:
                pass
        if newest and time.time() - newest < max_age_min * 60:
            return True
    return bool(newest) and (time.time() - newest) < max_age_min * 60


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default=".")
    p.add_argument("--min-mb", type=float, default=100.0,
                   help="Ignore anything smaller; the tail is never the problem.")
    p.add_argument("--live-min", type=float, default=60.0, metavar="MIN",
                   help="A directory written to within this many minutes is "
                        "treated as a run in flight and never suggested for "
                        "deletion, whatever else is true of it.")
    p.add_argument("--emit", action="store_true",
                   help="Print rm commands for the safe categories. Still does "
                        "not run them.")
    args = p.parse_args()

    root = os.path.abspath(args.root)
    os.chdir(root)
    refs = referenced(root)
    print(f"{len(refs)} path(s) referenced by scripts/ and src/\n")

    df = subprocess.run(["df", "-h", "."], capture_output=True, text=True)
    print(df.stdout.strip().splitlines()[-1], "\n")

    rows = []
    for r in ROOTS:
        if not os.path.isdir(r):
            continue
        # One level down, except results/ which nests by family.
        depth2 = r == "results"
        cands = []
        for e in sorted(os.listdir(r)):
            q = os.path.join(r, e)
            if not os.path.isdir(q):
                continue
            if depth2 and any(os.path.isdir(os.path.join(q, x))
                              for x in os.listdir(q)[:3]) and not glob.glob(
                                  os.path.join(q, "*.pt")):
                cands += [os.path.join(q, x) for x in sorted(os.listdir(q))
                          if os.path.isdir(os.path.join(q, x))]
            else:
                cands.append(q)
        for q in cands:
            n = du(q)
            if n < args.min_mb * 1e6:
                continue
            rel = os.path.relpath(q, root) if q.startswith(root) else q
            ck = glob.glob(os.path.join(q, "**", "*.ckpt"), recursive=True) + \
                glob.glob(os.path.join(q, "**", "*.pt"), recursive=True)
            ev = glob.glob(os.path.join(q, "**", "events.out.tfevents.*"),
                           recursive=True)
            nums = (glob.glob(os.path.join(q, "**", "scalars.csv"), recursive=True)
                    + glob.glob(os.path.join(q, "**", "history.json"),
                                recursive=True))
            live = is_live(q, args.live_min)
            ref = any(rel == p or rel.startswith(p + os.sep) or p.startswith(rel + os.sep)
                      for p in refs)
            if live:
                cat, act = "LIVE", "in flight -- leave alone"
            elif ref:
                cat, act = "LOAD-BEARING", "a script resumes from or reads this"
            elif nums and (ev or len(ck) > 2):
                cat = "NUMBERS KEPT"
                act = (f"drop {len(ev)} event file(s) and intermediate "
                       f"checkpoints; {len(nums)} numbers file(s) stay")
            elif r.endswith("data") or "listen" in rel or rel.endswith(".tar.gz"):
                cat, act = "REGENERABLE", "regenerate from its seed/script"
            elif not nums and not ref:
                cat, act = "ORPHAN", "nothing references it and it holds no numbers"
            else:
                cat, act = "KEEP", "holds numbers, nothing obvious to strip"
            rows.append((n, cat, rel, act, ev, ck, nums))

    order = {"ORPHAN": 0, "REGENERABLE": 1, "NUMBERS KEPT": 2, "KEEP": 3,
             "LOAD-BEARING": 4, "LIVE": 5}
    rows.sort(key=lambda t: (order[t[1]], -t[0]))
    print(f"{'size':>9}  {'class':<14}path")
    tot = {}
    for n, cat, rel, act, ev, ck, nums in rows:
        tot[cat] = tot.get(cat, 0) + n
        print(f"{n / 1e9:>8.2f}G  {cat:<14}{rel}")
        print(f"{'':>9}  {'':<14}  {act}")
    print(f"\n{'total':>9}  by class")
    for cat in sorted(tot, key=lambda c: order[c]):
        print(f"{tot[cat] / 1e9:>8.2f}G  {cat}")
    free = "  ".join(f"{cat} {tot[cat] / 1e9:.1f}G"
                     for cat in ("ORPHAN", "REGENERABLE") if cat in tot)
    strip = tot.get("NUMBERS KEPT", 0)
    print(f"\nfreeable now: {free or 'nothing'}")
    print(f"freeable by stripping events/checkpoints from finished runs: "
          f"up to {strip / 1e9:.1f}G, keeping every number")

    if not args.emit:
        print("\n--emit prints the commands. Read them before running; this "
              "script deletes nothing.")
        return

    print("\n# --- ORPHAN and REGENERABLE: whole directories ---")
    for n, cat, rel, *_ in rows:
        if cat in ("ORPHAN", "REGENERABLE"):
            print(f"rm -rf {rel!r}".replace("'", ""))
    print("\n# --- NUMBERS KEPT: events and intermediate checkpoints only ---")
    print("# scalars.csv / history.json stay, so monitor_* still works.")
    print("# This makes the run UNRESUMABLE -- only for runs that are done.")
    for n, cat, rel, act, ev, ck, nums in rows:
        if cat != "NUMBERS KEPT":
            continue
        print(f"find {rel} -name 'events.out.tfevents.*' -delete")
        keep = ("latest.ckpt", "last.ckpt", "encoder_last.pt", "ep0049.ckpt")
        drop = [c for c in ck if os.path.basename(c) not in keep
                and not os.path.basename(c).startswith("epoch_")]
        if drop:
            print(f"#   {len(drop)} checkpoint(s), keeping {', '.join(keep)}:")
            for c in drop:
                print(f"rm -f {c}")


if __name__ == "__main__":
    main()
