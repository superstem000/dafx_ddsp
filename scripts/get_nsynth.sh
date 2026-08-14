#!/usr/bin/env bash
# Fetch NSynth, the out-of-domain half of Masuda & Saito's setup.
#
#   scripts/get_nsynth.sh              # the train split, which is what is needed
#   SPLIT=valid scripts/get_nsynth.sh  # small, for testing the plumbing
#
# Why the train split and not the small ones. The paper samples 20000 sounds
# (sec 4.3.2) and IdOodDataModule enforces that by
#
#     indices = np.random.choice(len(ood_dat), len(id_dat), replace=False)
#
# so the OOD pool must hold at least as many files as the in-domain set, which
# is 20000. valid (12678) and test (4096) together are 16774 -- short even
# combined. Only the train split (289205) is large enough, and sampling 20000
# from all of it is what the paper describes.
#
# This is needed for EVERY arm, not just Real. setup() builds the OOD dataset
# unconditionally and both P-loss and Synth validate against it, so with no
# NSynth on disk np.random.choice(0, 20000) raises before the first batch.
#
# The archive extracts to nsynth-<split>/audio/*.wav, and WaveParamDataset globs
# <ood_dir>/audio/*.wav, so unpacking into external/diffsynth/data/ lands
# exactly on the ood_dir the experiment configs already name. No path edits.
#
# NSynth is 4 s mono at 16 kHz, which is what the in-domain set is too, so no
# resampling or trimming is involved. librosa.load returns floats in [-1,1], so
# the /32768 rescale some loaders need does not apply here.
set -euo pipefail

SPLIT=${SPLIT:-train}
DEST=${DEST:-"$(cd "$(dirname "$0")/.." && pwd)/external/diffsynth/data"}
KEEP_TAR=${KEEP_TAR:-0}

NAME="nsynth-${SPLIT}"
TAR="${NAME}.jsonwav.tar.gz"
URL="http://download.magenta.tensorflow.org/datasets/nsynth/${TAR}"

mkdir -p "$DEST"
cd "$DEST"

if [[ -d "$NAME/audio" ]]; then
  echo "$DEST/$NAME/audio already exists ($(ls "$NAME/audio" | wc -l) wavs) -- nothing to do"
  exit 0
fi

# Check the URL before spending hours on it. Magenta has moved this host before;
# if this fails, the canonical listing is https://magenta.tensorflow.org/datasets/nsynth
# and mirrors exist on Hugging Face. Do not guess a replacement URL -- a 200 on
# the wrong file is worse than a 404.
echo "checking $URL"
if ! curl -sIL --max-time 30 "$URL" | grep -qiE '^HTTP/[0-9.]+ 200'; then
  echo
  echo "ERROR: that URL did not return 200."
  echo "Check https://magenta.tensorflow.org/datasets/nsynth for the current"
  echo "location and re-run with URL=<...> if it has moved."
  curl -sIL --max-time 30 "$URL" | head -5
  exit 1
fi

SIZE_B=$(curl -sIL --max-time 30 "$URL" | awk 'BEGIN{IGNORECASE=1}/^content-length:/{l=$2}END{print l+0}' | tr -d '\r')
NEED_GB=$(( (SIZE_B * 5 / 2) / 1073741824 + 1 ))   # archive + extracted, roughly
FREE_GB=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc '0-9')
echo "archive $(( SIZE_B / 1073741824 )) GB; need ~${NEED_GB} GB free during unpack; have ${FREE_GB} GB"
if (( FREE_GB < NEED_GB )); then
  echo "ERROR: not enough free space. Free some, or set DEST to a bigger volume."
  exit 1
fi

# -c so an interrupted download resumes instead of restarting. This is a
# multi-GB transfer and it will get interrupted.
echo
echo "downloading (resumable -- safe to ctrl-C and re-run)"
wget -c -O "$TAR" "$URL"

echo
echo "extracting"
tar -xzf "$TAR"

N=$(ls "$NAME/audio" | wc -l)
echo "  $NAME/audio contains $N wav files"
if (( N < 20000 )); then
  echo "WARNING: fewer than the 20000 the in-domain set needs. IdOodDataModule"
  echo "will raise in np.random.choice. Use the train split."
fi

if [[ "$KEEP_TAR" != "1" ]]; then
  rm -f "$TAR"
  echo "  removed $TAR (KEEP_TAR=1 to keep it)"
fi

echo
echo "done: $DEST/$NAME"
echo "This is already the ood_dir the experiment configs name (data/$NAME),"
echo "so no config edit is needed."
