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

STAGES. A line of the form `# STAGE <label>` starts a new stage, and stages are
barriers: nothing in stage N+1 starts until every job in stage N has exited,
and if any of them failed the remaining stages are abandoned rather than
launched against outputs that were never written. That is what lets a dependent
pipeline be one file and one command -- a 50-epoch base, five branches that
resume from its checkpoint, then eight resumes from theirs. A file with no
STAGE lines is a single stage, which is exactly the previous behaviour.

Ctrl-C stops the queue and terminates running jobs. Jobs already finished stay
finished; re-running with the same file starts everything again, so trim the
file or use the sweep scripts' own skip logic if that matters.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
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


def parse_stages(path: Path) -> list[tuple[str, list[str]]]:
    """[(label, [command, ...]), ...] -- one entry per '# STAGE' section."""
    stages: list[tuple[str, list[str]]] = []
    label, cur = "all", []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        # STAGE must be followed by whitespace, a colon, or end of line. The
        # looser `\b[:\s]*` this replaces also matched ordinary prose -- a
        # comment reading "# stage. ~1.7h on one card" was read as a marker
        # under re.I and renamed the stage it was describing. Mid-stage, the
        # same line would have split one stage into two and dropped the barrier
        # between them, which is exactly the failure the stages exist to prevent.
        m = re.match(r"^#\s*STAGE(?:[:\s]+(.*))?$", line, re.I)
        if m:
            if cur:
                stages.append((label, cur))
                cur = []
            label = (m.group(1) or "").strip() or f"stage{len(stages)}"
            continue
        if line and not line.startswith("#"):
            cur.append(line)
    if cur:
        stages.append((label, cur))
    return stages


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--jobs", type=Path, required=True,
                   help="One shell command per line, {gpu} for the device index; "
                        "'# STAGE <label>' introduces a dependency barrier")
    p.add_argument("--gpus", default=None,
                   help="Space-separated indices to use; default is every GPU")
    p.add_argument("--poll", type=int, default=60, help="Seconds between polls")
    p.add_argument("--stable", type=int, default=3,
                   help="Consecutive free polls before a GPU is claimed. Guards "
                        "against grabbing a card another sweep is between arms on")
    p.add_argument("--free-mib", type=int, default=1000,
                   help="A GPU counts as free below this many MiB in use")
    p.add_argument("--logdir", type=Path, default=Path("~/gpu_queue_logs"),
                   help="Logs go in <logdir>/<jobs-file-stem>/")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    stages = parse_stages(args.jobs)
    if not stages:
        print("no jobs"); raise SystemExit(1)
    n_jobs = sum(len(js) for _, js in stages)

    try:
        all_gpus = sorted(gpu_memory())
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        # --dry-run is for checking the jobs file parses into the stages you
        # meant, which is worth being able to do anywhere -- including on a
        # machine with no GPU.
        if not args.dry_run:
            print(f"cannot read nvidia-smi: {e}"); raise SystemExit(1)
        print(f"(no nvidia-smi: {e} -- dry run continues)")
        all_gpus = []
    gpus = [int(g) for g in args.gpus.split()] if args.gpus else all_gpus

    # Per-jobs-file subdirectory. Job indices restart at 0 for every queue, so
    # a flat log directory means a second queue's job0 overwrites the first
    # one's -- silently, and for jobs that may still be running. Keying on the
    # file name makes two different queues incapable of colliding, while
    # re-running the same file still overwrites its own logs, which is what you
    # want.
    logdir = args.logdir.expanduser() / args.jobs.stem
    logdir.mkdir(parents=True, exist_ok=True)

    print(f"[{stamp()}] {n_jobs} job(s) in {len(stages)} stage(s), gpus {gpus}, "
          f"free<{args.free_mib}MiB for {args.stable} polls of {args.poll}s")
    i = 0
    for label, js in stages:
        print(f"  --- stage '{label}'")
        for j in js:
            print(f"    {i}: {j}")
            i += 1
    if args.dry_run:
        return

    state = {"stopping": False}
    running: dict[int, tuple[int, subprocess.Popen]] = {}   # gpu -> (idx, proc)
    failed: list[tuple[int, int]] = []

    def terminate(pr: subprocess.Popen) -> None:
        """Signal the job's whole process group, not just its shell.

        Jobs start with preexec_fn=os.setsid, so each gets its own process
        group. pr.terminate() then reaches only the shell wrapper -- the python
        training process underneath survives, holds its GPU, and keeps writing
        results after the queue is gone. That happened: killing the queue left
        three trainings running for another two hours, invisible to every
        pattern anyone thought to pkill.
        """
        try:
            os.killpg(os.getpgid(pr.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pr.terminate()

    def stop(_s, _f):
        state["stopping"] = True
        print(f"\n[{stamp()}] interrupted -- terminating {len(running)} running job(s)")
        for _g, (_i, pr) in running.items():
            terminate(pr)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def run_stage(label: str, js: list[str], base: int) -> None:
        """Schedule this stage's jobs, then wait for every one of them."""
        print(f"\n[{stamp()}] === stage '{label}' -- {len(js)} job(s)")
        pending = list(enumerate(js, start=base))
        free_streak: dict[int, int] = defaultdict(int)

        while (pending or running) and not state["stopping"]:
            for g, (idx, pr) in list(running.items()):
                if pr.poll() is not None:
                    print(f"[{stamp()}] job {idx} on gpu {g} exited rc={pr.returncode}")
                    if pr.returncode:
                        failed.append((idx, pr.returncode))
                    del running[g]
                    free_streak[g] = 0      # let it settle before reuse

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

        # The barrier: this stage is not finished until nothing of it is left
        # running, whether we got here normally or via Ctrl-C.
        for g, (idx, pr) in list(running.items()):
            pr.wait()
            print(f"[{stamp()}] job {idx} on gpu {g} exited rc={pr.returncode}")
            if pr.returncode:
                failed.append((idx, pr.returncode))
            del running[g]

    base = 0
    for si, (label, js) in enumerate(stages):
        run_stage(label, js, base)
        base += len(js)
        if state["stopping"]:
            break
        bad = [(j, rc) for j, rc in failed if base - len(js) <= j < base]
        if bad and si + 1 < len(stages):
            # Later stages resume from checkpoints these jobs were supposed to
            # write, so continuing would launch them against files that do not
            # exist -- a cascade of fast confusing failures instead of one clear
            # one.
            print(f"\n[{stamp()}] stage '{label}' had {len(bad)} failure(s): "
                  + ", ".join(f"job {j} (rc={rc})" for j, rc in bad))
            print(f"[{stamp()}] NOT starting the remaining {len(stages) - si - 1} "
                  f"stage(s) -- they depend on this one.")
            print(f"[{stamp()}] logs: {logdir}")
            break

    print(f"\n[{stamp()}] queue {'stopped' if state['stopping'] else 'finished'}; "
          f"{len(failed)} job(s) failed")


if __name__ == "__main__":
    main()
