#!/usr/bin/env bash
# Fetch NSynth, the out-of-domain half of Masuda & Saito's setup.
#
#   scripts/get_nsynth.sh                # sample 25000 of the train split
#   SAMPLE=40000 scripts/get_nsynth.sh   # a bigger pool
#   FULL=1 scripts/get_nsynth.sh         # every file; needs ~60 GB free
#
# Why sample rather than extract everything. nsynth-train is 289205 files of 4 s
# at 16 kHz, ~128 KB each, so a full extraction is ~37 GB on top of a 22 GB
# archive. The paper (sec 4.3.2) uses 20000 sounds "randomly selected from the
# full dataset", and IdOodDataModule then narrows the pool to the in-domain size
# anyway:
#
#     indices = np.random.choice(len(ood_dat), len(id_dat), replace=False)
#
# so the only hard requirement is a pool of at least 20000. 25000 gives that
# with headroom, costs ~3 GB, and -- because the sample is drawn uniformly over
# the whole archive rather than by truncation -- is the same population the
# paper describes.
#
# Truncating instead would NOT be equivalent, which is why this does the extra
# work: NSynth filenames begin with the instrument family (bass_synthetic_...,
# brass_acoustic_...) and tar order follows them, so the first 25000 entries are
# essentially one family. That would make the out-of-domain set a single
# instrument class and quietly change what "out of domain" means.
#
# Sampling happens during the stream, so the archive is never written to disk.
# The cost is that an interruption means starting over -- there is no resuming a
# pipe -- so run it under tmux.
#
# NSynth is needed for EVERY arm, not just Real: setup() builds the OOD dataset
# unconditionally and both P-loss and Synth validate against it, so with nothing
# on disk np.random.choice(0, 20000) raises before the first batch.
#
# Files land at <dest>/nsynth-<split>/audio/*.wav, which is both the ood_dir the
# experiment configs already name and the layout WaveParamDataset globs.
set -euo pipefail

# Resolved before the cd below, because $0 is relative and dirname would then
# resolve against $DEST instead of the repo.
HERE="$(cd "$(dirname "$0")" && pwd)"

SPLIT=${SPLIT:-train}
SAMPLE=${SAMPLE:-25000}
FULL=${FULL:-0}
DEST=${DEST:-"$(cd "$(dirname "$0")/.." && pwd)/external/diffsynth/data"}

NAME="nsynth-${SPLIT}"
URL=${URL:-"http://download.magenta.tensorflow.org/datasets/nsynth/${NAME}.jsonwav.tar.gz"}

mkdir -p "$DEST"
cd "$DEST"

if [[ -d "$NAME/audio" ]] && (( $(ls "$NAME/audio" 2>/dev/null | wc -l) > 0 )); then
  echo "$DEST/$NAME/audio already has $(ls "$NAME/audio" | wc -l) wavs -- nothing to do"
  echo "(delete it to re-fetch, or set SAMPLE higher and re-run into a fresh DEST)"
  exit 0
fi

echo "checking $URL"
if ! curl -sIL --max-time 30 "$URL" | grep -qiE '^HTTP/[0-9.]+ 200'; then
  echo
  echo "ERROR: that URL did not return 200."
  echo "The canonical listing is https://magenta.tensorflow.org/datasets/nsynth"
  echo "Re-run with URL=<...> if it has moved. Do not guess -- a 200 on the"
  echo "wrong file is worse than a 404."
  curl -sIL --max-time 30 "$URL" | head -5
  exit 1
fi

FREE_GB=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc '0-9')
if [[ "$FULL" == "1" ]]; then
  NEED_GB=60
  echo "FULL extraction: need ~${NEED_GB} GB, have ${FREE_GB} GB"
else
  NEED_GB=$(( SAMPLE * 140 / 1000000 + 2 ))   # ~128 KB per wav, plus slack
  echo "sampling $SAMPLE files: need ~${NEED_GB} GB, have ${FREE_GB} GB"
fi
if (( FREE_GB < NEED_GB )); then
  echo "ERROR: not enough free space. Lower SAMPLE, or set DEST to a bigger volume."
  exit 1
fi

echo
echo "streaming (no resume -- run under tmux; ~22 GB over the wire either way)"
# The sampler is a separate file, not a heredoc: the archive arrives on stdin,
# and `curl ... | python3 - <<'PY'` makes the heredoc stdin, so python reads its
# program from there and silently discards the pipe. That failed instantly with
# an exhausted stream and, under `tmux new -d`, took the session down with it
# before anything could be read.
curl -sL "$URL" | python3 "$HERE/nsynth_sample.py" "$NAME" "$SAMPLE" "$FULL"

N=$(ls "$NAME/audio" | wc -l)
echo
echo "$DEST/$NAME/audio contains $N wav files"
if (( N < 20000 )); then
  echo "WARNING: fewer than the 20000 the in-domain set needs."
  echo "IdOodDataModule will raise in np.random.choice. Re-run with a larger"
  echo "SAMPLE, or use the train split if you used a smaller one."
  exit 1
fi

echo
echo "done. This is already the ood_dir the experiment configs name (data/$NAME),"
echo "so no config edit is needed."
