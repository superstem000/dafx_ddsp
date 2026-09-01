"""Merge generated dataset shards into one directory, renumbered and verified.

    python scripts/ds_merge_shards.py --out external/diffsynth/data/diffsynth_5-6/harmor_2oscfree_var \
        --shards external/diffsynth/data/varlen/s*/harmor_2oscfree_var

WHY RENUMBERING IS THE DANGEROUS PART. gen_dataset.py numbers from 00000 on
every invocation, so five shards all contain 00000.wav. WaveParamDataset globs
audio/*.wav and param/*.pt INDEPENDENTLY and pairs them by sorted index --
nothing ties a clip to its own parameters. A merge that renumbers the two
halves inconsistently therefore trains every clip against another clip's
targets, and it does not raise anywhere: the loss still falls, just to a worse
floor, which reads as a hard task rather than as a broken dataset.

So this copies each pair together under one new index, then verifies the whole
result the way the loader will see it -- globbing both directories separately,
sorting, and comparing stems. data.py's WaveParamDataset now asserts the same
thing on every load, which is the real guard; this check is here so the failure
is caught at merge time by the tool that could have caused it.

Zero-padded to a width that fits the total, so lexical order and numeric order
agree. That matters: the loader sorts strings, so 10000 sorting before 9999
would silently reorder one half relative to the other if the two halves ever
had different widths.

--move hardlinks instead of copying, for when the shards are large and about to
be deleted anyway. Falls back to copying across filesystems.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys


def shard_pairs(d: str) -> list[tuple[str, str]]:
    """(audio, param) for one shard, paired by stem rather than by position."""
    a = sorted(glob.glob(os.path.join(d, "audio", "*.wav")))
    p = {os.path.splitext(os.path.basename(x))[0]: x
         for x in glob.glob(os.path.join(d, "param", "*.pt"))}
    out = []
    for f in a:
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem not in p:
            raise SystemExit(f"{d}: {os.path.basename(f)} has no param/{stem}.pt")
        out.append((f, p[stem]))
    extra = len(p) - len(out)
    if extra:
        raise SystemExit(f"{d}: {extra} param file(s) with no matching audio")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shards", nargs="+", required=True, metavar="DIR",
                   help="Each must contain audio/ and param/.")
    p.add_argument("--out", required=True, metavar="DIR")
    p.add_argument("--move", action="store_true",
                   help="Hardlink rather than copy; falls back to copy across "
                        "filesystems.")
    args = p.parse_args()

    pairs = []
    for d in args.shards:
        got = shard_pairs(d)
        print(f"{d}: {len(got)} pair(s)")
        pairs += got
    if not pairs:
        raise SystemExit("no pairs found")

    ad = os.path.join(args.out, "audio")
    pd = os.path.join(args.out, "param")
    for d in (ad, pd):
        os.makedirs(d, exist_ok=True)
        if glob.glob(os.path.join(d, "*")):
            raise SystemExit(f"{d} is not empty; refusing to merge into it "
                             f"(numbering would collide with what is there)")

    width = max(5, len(str(len(pairs) - 1)))
    for i, (a, q) in enumerate(pairs):
        name = f"{i:0{width}d}"
        for src, dst in ((a, os.path.join(ad, name + ".wav")),
                         (q, os.path.join(pd, name + ".pt"))):
            if args.move:
                try:
                    os.link(src, dst)
                    continue
                except OSError:
                    pass
            shutil.copy2(src, dst)

    # Verified the way the LOADER sees it: two independent globs, sorted,
    # stems compared. Not by trusting the loop above.
    a = [os.path.splitext(os.path.basename(x))[0]
         for x in sorted(glob.glob(os.path.join(ad, "*.wav")))]
    b = [os.path.splitext(os.path.basename(x))[0]
         for x in sorted(glob.glob(os.path.join(pd, "*.pt")))]
    if a != b:
        raise SystemExit(f"MERGE IS BROKEN: {len(a)} audio vs {len(b)} param, "
                         f"stems differ. Do not train on {args.out}.")
    print(f"\n{args.out}: {len(a)} clip(s), audio and param stems verified "
          f"identical under the loader's own sort.")
    print(f"Train with ID_DIR={args.out}")


if __name__ == "__main__":
    sys.exit(main())
