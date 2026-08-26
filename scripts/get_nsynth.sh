#!/usr/bin/env bash
# Fetch NSynth, the out-of-domain half of Masuda & Saito's setup.
#
#   scripts/get_nsynth.sh                # sample 25000 of the train split
#   SAMPLE=40000 scripts/get_nsynth.sh   # a bigger pool
#   FULL=1 scripts/get_nsynth.sh         # every file; needs ~60 GB free
#   CLASS=reed_acoustic scripts/get_nsynth.sh          # one instrument class
#   CLASS=bass_synthetic MAX=12000 scripts/get_nsynth.sh
#   CLASS="reed_acoustic string_acoustic brass_acoustic" MAX=8500 \
#       scripts/get_nsynth.sh                           # several, ONE stream
#
# CLASS may name SEVERAL classes, and one stream then fills all of them. The
# archive is 22 GB and forward-only, so a per-class fetch pays that stream every
# time -- eleven classes one at a time is hours of re-reading the same bytes.
# Disk is what bounds a batch, which is what MAX is for: 8500 files is ~1.1 GB
# per class. With more than one class, OUT is ignored and each lands in its own
# data/nsynth-<class> directory.
#
# CLASS takes EVERY file of one <family>_<source> and ignores SAMPLE. That is
# what a class-specific training set is, and the per-class counts are small
# enough to take whole -- keyboard_acoustic is 8068 files, ~1 GB. MAX prunes
# afterwards for the few that are not; bass_synthetic is ~57k, ~7 GB. The
# stream is the same 22 GB either way: the archive is ordered by family, but
# there is no seeking in a pipe, so the whole thing still goes past.
#
# Output lands in data/nsynth-<class with underscores removed>, e.g.
# nsynth-reedacoustic, unless OUT names it. The archive itself is still the
# split named by SPLIT.
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
CLASS=${CLASS:-}
MAX=${MAX:-0}
DEST=${DEST:-"$(cd "$(dirname "$0")/.." && pwd)/external/diffsynth/data"}

NAME="nsynth-${SPLIT}"
URL=${URL:-"http://download.magenta.tensorflow.org/datasets/nsynth/${NAME}.jsonwav.tar.gz"}
# The archive to stream and the directory to write are the same thing only when
# no class is selected. NAME stays the archive; OUT is where files land.
# Plain word-splitting on the default IFS. An earlier version substituted
# spaces for newlines first, which broke it: `read -a` consumes ONE line, so a
# multi-class CLASS silently collapsed to its first element.
read -r -a CLASS_ARR <<< "$CLASS"
if (( ${#CLASS_ARR[@]} > 1 )); then
  OUT=""                       # per-class directories, named by nsynth_sample
else
  OUT=${OUT:-$([[ -n "$CLASS" ]] && echo "nsynth-${CLASS//_/}" || echo "$NAME")}
fi

mkdir -p "$DEST"
cd "$DEST"

# Skip classes already on disk, and drop out entirely if none is left to fetch
# -- a sweep re-runs this per batch and must not pay 22 GB for nothing.
if (( ${#CLASS_ARR[@]} > 1 )); then
  TODO=()
  for c in "${CLASS_ARR[@]}"; do
    d="nsynth-${c//_/}"
    if [[ -d "$d/audio" ]] && (( $(ls "$d/audio" 2>/dev/null | wc -l) > 0 )); then
      echo "$DEST/$d/audio already has $(ls "$d/audio" | wc -l) wavs -- skipping $c"
    else
      TODO+=("$c")
    fi
  done
  if (( ${#TODO[@]} == 0 )); then
    echo "every requested class is already on disk -- nothing to do"
    exit 0
  fi
  CLASS_ARR=("${TODO[@]}")
elif [[ -n "$OUT" && -d "$OUT/audio" ]] && (( $(ls "$OUT/audio" 2>/dev/null | wc -l) > 0 )); then
  echo "$DEST/$OUT/audio already has $(ls "$OUT/audio" | wc -l) wavs -- nothing to do"
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
if [[ -n "$CLASS" ]]; then
  # Unknown until the stream ends, so budget from the largest class that is not
  # capped. 15000 files is ~2 GB and covers every family_source except
  # bass_synthetic and keyboard_electronic, which want MAX.
  NEED_GB=$(( ( MAX > 0 ? MAX : 15000 ) * 140 * ${#CLASS_ARR[@]} / 1000000 + 2 ))
  echo "classes ${CLASS_ARR[*]} (${#CLASS_ARR[@]}): need ~${NEED_GB} GB, have ${FREE_GB} GB"
elif [[ "$FULL" == "1" ]]; then
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
# `cond && cmd` statements, not `if`, are a set -e trap: when cond is false the
# AND-list's status is 1 and errexit takes the script down. With OUT empty on
# the multi-class path that is exactly what happened.
EXTRA=()
for c in "${CLASS_ARR[@]}"; do EXTRA+=(--only "$c"); done
if [[ -n "$CLASS" && -n "$OUT" ]]; then EXTRA+=(--out "$OUT"); fi
if (( MAX > 0 )); then EXTRA+=(--max "$MAX"); fi
curl -sL "$URL" | python3 "$HERE/nsynth_sample.py" "$NAME" "$SAMPLE" "$FULL" "${EXTRA[@]}"

if (( ${#CLASS_ARR[@]} > 1 )); then
  echo
  for c in "${CLASS_ARR[@]}"; do
    d="nsynth-${c//_/}"
    echo "  $DEST/$d/audio  $(ls "$d/audio" 2>/dev/null | wc -l) wav files"
  done
  exit 0
fi
N=$(ls "$OUT/audio" | wc -l)
echo
echo "$DEST/$OUT/audio contains $N wav files"
if [[ -n "$CLASS" ]]; then
  echo
  echo "A class set is an ood_dir for a class-specific run, not a drop-in for"
  echo "nsynth-train. IdOodDataModule draws len(id_dat) indices from it without"
  echo "replacement (data.py:92), so the ID_DIR paired with it must hold NO MORE"
  echo "files than $N -- and using the SAME id_dir across classes is what makes"
  echo "two class runs comparable, since it fixes the training-set size."
  exit 0
fi
if (( N < 20000 )); then
  echo "WARNING: fewer than the 20000 the in-domain set needs."
  echo "IdOodDataModule will raise in np.random.choice. Re-run with a larger"
  echo "SAMPLE, or use the train split if you used a smaller one."
  exit 1
fi

echo
echo "done. This is already the ood_dir the experiment configs name (data/$NAME),"
echo "so no config edit is needed."
