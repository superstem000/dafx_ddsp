"""Extract a uniform random sample of wavs from an NSynth tar on stdin.

Separate file rather than a heredoc inside get_nsynth.sh on purpose: the
archive arrives on stdin, and

    curl ... | python3 - <<'PY'

does not work -- the heredoc becomes stdin, so python reads its program from
there and the pipe is silently discarded, leaving tarfile with an exhausted
stream. Keeping the program in a file leaves stdin free for the data.

    curl -sL <url> | python3 scripts/nsynth_sample.py nsynth-train 25000

Sampling is by sequential selection: take the next wav with probability
(want - kept) / (total - seen). That yields exactly `want` files, each equally
likely. The alternatives are both worse -- Bernoulli sampling gives a random
count, and truncating to the first N gives one instrument family, because
NSynth filenames lead with the family (bass_synthetic_..., brass_acoustic_...)
and tar order follows them.
"""

import os
import random
import sys
import tarfile

TOTALS = {"nsynth-train": 289205, "nsynth-valid": 12678, "nsynth-test": 4096}


def main() -> None:
    argv = sys.argv[1:]
    only = out = None
    cap = 0
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--only":
            only = argv[i + 1]; i += 2
        elif argv[i] == "--max":
            cap = int(argv[i + 1]); i += 2
        elif argv[i] == "--out":
            out = argv[i + 1]; i += 2
        else:
            rest.append(argv[i]); i += 1
    name = rest[0]
    want = int(rest[1])
    full = len(rest) > 2 and rest[2] == "1"
    random.seed(0)

    total = TOTALS.get(name, 289205)
    out_dir = os.path.join(out or name, "audio")
    os.makedirs(out_dir, exist_ok=True)

    # 'r|gz' is the streaming reader: forward-only, no seeking, so it works on a
    # pipe. The default seekable reader tries to rewind and fails here.
    tar = tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz")

    kept = seen = 0
    for m in tar:
        if not (m.isfile() and m.name.endswith(".wav")):
            continue
        seen += 1
        if only is not None:
            # ALL of one class, not a sample of it. The sampling arithmetic
            # below needs the eligible population up front -- it takes the next
            # file with probability (want - kept) / (total - seen) -- and the
            # per-class count is not knowable from a forward-only stream. A
            # class is small enough to take whole anyway: keyboard_acoustic is
            # 8068 of 289205, about 1 GB. --max prunes afterwards for the few
            # classes that are not (bass_synthetic is ~57k, ~7 GB).
            if not os.path.basename(m.name).startswith(only):
                continue
        elif not full:
            if kept >= want:
                continue
            remaining = max(total - seen + 1, 1)
            if random.random() > (want - kept) / remaining:
                continue
        f = tar.extractfile(m)
        if f is None:
            continue
        with open(os.path.join(out_dir, os.path.basename(m.name)), "wb") as g:
            g.write(f.read())
        kept += 1
        if kept % 500 == 0:
            pct = 100.0 * seen / total
            tgt = only if only is not None else want
            print(f"  kept {kept}/{tgt}   streamed {seen}/{total} (~{pct:.1f}%)",
                  flush=True)

    print(f"  done: kept {kept} of {seen} wav entries seen", flush=True)
    if only is not None:
        if not kept:
            print(f"  ERROR: nothing matched '{only}'. NSynth basenames are "
                  f"<family>_<source>_<instr>-<pitch>-<velocity>.wav, so the "
                  f"prefix wants both halves, e.g. reed_acoustic",
                  file=sys.stderr)
            raise SystemExit(1)
        if cap and kept > cap:
            # Prune after writing rather than reservoir-sampling during it: a
            # reservoir would have to hold the file BYTES for an unknown
            # population, ~128 KB each, which is gigabytes of RAM. Disk is the
            # cheaper place to buffer.
            names = sorted(os.listdir(out_dir))
            drop = random.sample(names, len(names) - cap)
            for d in drop:
                os.remove(os.path.join(out_dir, d))
            print(f"  pruned to --max {cap} (removed {len(drop)})", flush=True)
        return
    if kept < want and not full:
        # Only reachable if the archive held fewer entries than TOTALS claims.
        print(f"  NOTE: archive was shorter than expected ({seen} < {total})",
              flush=True)


if __name__ == "__main__":
    main()
