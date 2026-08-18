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
    name = sys.argv[1]
    want = int(sys.argv[2])
    full = len(sys.argv) > 3 and sys.argv[3] == "1"
    random.seed(0)

    total = TOTALS.get(name, 289205)
    out_dir = os.path.join(name, "audio")
    os.makedirs(out_dir, exist_ok=True)

    # 'r|gz' is the streaming reader: forward-only, no seeking, so it works on a
    # pipe. The default seekable reader tries to rewind and fails here.
    tar = tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz")

    kept = seen = 0
    for m in tar:
        if not (m.isfile() and m.name.endswith(".wav")):
            continue
        seen += 1
        if not full:
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
            print(f"  kept {kept}/{want}   streamed {seen}/{total} (~{pct:.1f}%)",
                  flush=True)

    print(f"  done: kept {kept} of {seen} wav entries seen", flush=True)
    if kept < want and not full:
        # Only reachable if the archive held fewer entries than TOTALS claims.
        print(f"  NOTE: archive was shorter than expected ({seen} < {total})",
              flush=True)


if __name__ == "__main__":
    main()
