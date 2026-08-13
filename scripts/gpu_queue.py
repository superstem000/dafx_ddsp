"""Run a queue of jobs, one per GPU, starting each as a card actually frees.

The pattern this exists for: a sweep with more arms than GPUs ends with most
cards idle while the last arm runs for hours. Queueing the next experiment by
hand means either watching for that or wasting the capacity.

Free is defined by memory, and only after N consecutive polls. A single poll is
not enough -- a card goes briefly idle between arms of a running sweep, and
claiming it there would put two jobs on one GPU and slow both. The default of
three polls at 60s means a card has to be quiet for three minutes.

Jobs come from a file, one shell command per line, with {gpu} where the device
index goes. Blank lines and # comments are skipped. Each job is assumed to want
exactly one GPU.

    cat > jobs.txt <<'EOF'
    HEAD_BOUND=stclamp OUT=results/ddsp/eps_ladder_stclamp ARMS="L1_STFT"        scripts/eps_ladder.sh "{gpu}"
    HEAD_BOUND=stclamp OUT=results/ddsp/eps_ladder_stclamp ARMS="L1_STFT_eps1"   scripts/eps_ladder.sh "{gpu}"
    EOF
    python scripts/gpu_queue.py --jobs jobs.txt

Ctrl-C stops the queue and terminates running jobs. Jobs already finished stay
finished; re-running with the same file starts everything again, so trim the
file or use the sweep scripts' own skip logic if that matters.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def gpu_memory() -> dict[int, int]:
    """Used MiB per GPU index, via nvidia-smi."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    used = {}
    for line in out.strip().splitlines():
        i, m = (x.strip() for x in line.split(","))
        used[int(i)] = int(m)
    return used


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--jobs", type=Path, required=True,
                   help="One shell command per line, {gpu} for the device index")
    p.add_argument("--gpus", default=None,
                   help="Space-separated indices to use; default is every GPU")
    p.add_argument("--poll", type=int, default=60, help="Seconds between polls")
    p.add_argument("--stable", type=int, default=3,
                   help="Consecutive free polls before a GPU is claimed. Guards "
                        "against grabbing a card another sweep is between arms on")
    p.add_argument("--free-mib", type=int, default=1000,
                   help="A GPU counts as free below this many MiB in use")
    p.add_argument("--logdir", type=Path, default=Path("~/gpu_queue_logs"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    jobs = [l.strip() for l in args.jobs.read_text().splitlines()]
    jobs = [l for l in jobs if l and not l.startswith("#")]
    if not jobs:
        print("no jobs"); raise SystemExit(1)

    try:
        all_gpus = sorted(gpu_memory())
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"cannot read nvidia-smi: {e}"); raise SystemExit(1)
    gpus = [int(g) for g in args.gpus.split()] if args.gpus else all_gpus

    logdir = args.logdir.expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    print(f"[{stamp()}] {len(jobs)} jobs, gpus {gpus}, "
          f"free<{args.free_mib}MiB for {args.stable} polls of {args.poll}s")
    for i, j in enumerate(jobs):
        print(f"    {i}: {j}")
    if args.dry_run:
        return

    pending = list(enumerate(jobs))
    running: dict[int, tuple[int, subprocess.Popen]] = {}   # gpu -> (idx, proc)
    free_streak: dict[int, int] = defaultdict(int)
    stopping = False

    def stop(_s, _f):
        nonlocal stopping
        stopping = True
        print(f"\n[{stamp()}] interrupted -- terminating {len(running)} running job(s)")
        for _g, (_i, pr) in running.items():
            pr.terminate()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while (pending or running) and not stopping:
        for g, (idx, pr) in list(running.items()):
            if pr.poll() is not None:
                print(f"[{stamp()}] job {idx} on gpu {g} exited rc={pr.returncode}")
                del running[g]
                free_streak[g] = 0          # let it settle before reuse

        if pending:
            try:
                used = gpu_memory()
            except subprocess.CalledProcessError:
                used = {}
            for g in gpus:
                if g in running or g not in used:
                    continue
                free_streak[g] = free_streak[g] + 1 if used[g] < args.free_mib else 0

            for g in gpus:
                if not pending or g in running:
                    continue
                if free_streak[g] < args.stable:
                    continue
                idx, cmd = pending.pop(0)
                real = cmd.replace("{gpu}", str(g))
                log = logdir / f"job{idx}_gpu{g}.log"
                print(f"[{stamp()}] job {idx} -> gpu {g}  ({log})")
                print(f"           {real}")
                with log.open("w") as fh:
                    fh.write(f"# {real}\n\n"); fh.flush()
                    pr = subprocess.Popen(real, shell=True, stdout=fh,
                                          stderr=subprocess.STDOUT,
                                          preexec_fn=os.setsid)
                running[g] = (idx, pr)
                free_streak[g] = 0

        if pending or running:
            time.sleep(args.poll)

    for g, (idx, pr) in running.items():
        pr.wait()
        print(f"[{stamp()}] job {idx} on gpu {g} exited rc={pr.returncode}")
    print(f"[{stamp()}] queue {'stopped' if stopping else 'complete'}; "
          f"{len(pending)} job(s) never started")


if __name__ == "__main__":
    main()
