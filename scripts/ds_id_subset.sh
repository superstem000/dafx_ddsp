#!/usr/bin/env bash
# A symlinked in-domain set of N files, to set a class run's training size.
#
#   scripts/ds_id_subset.sh 3000 external/diffsynth/data/h2of_kb /tmp/id3000
#
# WHY THIS EXISTS. data.py:92 draws len(id_dat) indices from the ood set
# without replacement, so the ID directory's FILE COUNT -- nothing else about
# it -- decides how many out-of-domain clips a run trains on, and a class with
# fewer files than the ID set raises before the first batch. With h2of_kb's
# 8064 files only 11 of NSynth's 25 family_source classes can be trained on at
# all. Sizing the ID set per class removes that limit.
#
# Symlinks, not copies: 8064 wavs is a gigabyte and this may be built a dozen
# times in one sweep. WaveParamDataset globs <dir>/audio/*.wav and
# <dir>/param/*.pt separately and pairs them BY POSITION in sorted order, so
# both subsets must contain the same basenames -- which is why this walks the
# audio files and derives each param path rather than taking the first N of
# each glob independently. A mismatch there would silently pair every clip with
# another clip's parameters.
#
# Deterministic: the first N in sorted order, not a random draw, so two runs of
# the same size get the same set.
set -euo pipefail

N=${1:?usage: ds_id_subset.sh N SRC DST}
SRC=${2:?usage: ds_id_subset.sh N SRC DST}
DST=${3:?usage: ds_id_subset.sh N SRC DST}

SRC="$(cd "$SRC" && pwd)"
[[ -d "$SRC/audio" ]] || { echo "ERROR: $SRC/audio missing"; exit 1; }
[[ -d "$SRC/param" ]] || { echo "ERROR: $SRC/param missing"; exit 1; }

AVAIL=$(ls "$SRC/audio" | wc -l)
if (( N > AVAIL )); then
  echo "ERROR: asked for $N but $SRC/audio has $AVAIL"
  exit 1
fi

rm -rf "$DST"
mkdir -p "$DST/audio" "$DST/param"
i=0
while IFS= read -r f; do
  b="$(basename "$f" .wav)"
  p="$SRC/param/$b.pt"
  [[ -f "$p" ]] || { echo "ERROR: no param for $b"; exit 1; }
  ln -s "$f" "$DST/audio/$b.wav"
  ln -s "$p" "$DST/param/$b.pt"
  (( ++i >= N )) && break
done < <(find "$SRC/audio" -name '*.wav' | sort)

echo "$DST: $(ls "$DST/audio" | wc -l) audio, $(ls "$DST/param" | wc -l) param"
