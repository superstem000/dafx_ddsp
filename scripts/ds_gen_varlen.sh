#!/usr/bin/env bash
# Generate the variable-note-length diffsynth dataset.
#
#   scripts/ds_gen_varlen.sh                    # 20000 clips, NOTE_OFF sampled
#   N_PER=320 scripts/ds_gen_varlen.sh          # smoke test
#   DEVICE=cpu scripts/ds_gen_varlen.sh         # no GPU
#   LEVELS="0.15 0.30 0.45 0.60 0.75" N_PER=4000 scripts/ds_gen_varlen.sh
#                                               # pinned sweep, five shards
#
# WHAT THIS CHANGES, and it is two things, only one of which is the variable.
#
#   noise_mode: mul, in configs/synth/dataset/h2of_var.yaml. THE ENABLING
#   CHANGE, not the variable. h2of.yaml adds the envelope noise outside the
#   A+D+S sum, so a clip that has finished its release still sits at
#   randn*noise_mag -- median -26 dB below its own peak -- and the window is
#   fully occupied no matter how short the note is. Measured on 400 clips of
#   the existing set, occupancy spans 0.958 to 0.998. Multiplicative noise
#   keeps the jitter during the note and lets silence be silence. Setting
#   NOISE_A to 0 would have done the first half and also removed the jitter,
#   confounding note length with a change in low-level structure -- which is
#   the exact thing the loss comparison is about.
#
#   NOTE_OFF, per shard. THE VARIABLE. h2of pins it at 0.75, so every note in
#   the published set releases at the same instant, 3.0 s into a 4.0 s window.
#
# WHY FIVE FIXED LEVELS RATHER THAN SAMPLING IT. ADSREnvelope.forward computes
# `attack = attack * note_off`, so draws near zero collapse the entire envelope
# onto t=0 and a uniform draw would spend a noticeable share of the dataset on
# degenerate clips. Fixed levels also make note length a DISCRETE covariate
# with five values, which is what lets the arm gap be read level by level
# rather than as one pooled number -- with 4000 clips per level that is a
# better design than the continuous version, not a compromise.
#
# THE MERGE IS THE DANGEROUS STEP. gen_dataset.py numbers from 00000 every
# invocation, so all five shards contain 00000.wav, and WaveParamDataset pairs
# audio with param BY SORTED INDEX across two independent globs. A merge that
# renumbers the halves inconsistently trains every clip against another clip's
# targets and raises nothing -- the loss simply parks at a worse floor.
# ds_merge_shards.py copies each pair together and then re-verifies by globbing
# the result the way the loader will; data.py asserts the same on every load.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DS="$ROOT/external/diffsynth"
cd "$DS"

CONF=${CONF:-configs/synth/dataset/h2of_var.yaml}
NAME=$(python3 -c "from omegaconf import OmegaConf; print(OmegaConf.load('$CONF').name)")
# EMPTY IS THE DEFAULT AND THE RECOMMENDED PATH: h2of_var.yaml samples NOTE_OFF
# like every other parameter, so one generation produces continuous note
# lengths and there is no merge to get wrong. Set LEVELS to a list to pin it
# instead, which puts NOTE_OFF back into fixed_params per shard and brings the
# renumbering step back with it -- worth it only for a controlled sweep.
LEVELS=${LEVELS:-}
N_PER=${N_PER:-20000}
DEVICE=${DEVICE:-cuda}
SHARD_ROOT=${SHARD_ROOT:-data/varlen}
OUT=${OUT:-data/diffsynth_5-6/$NAME}

echo "config   $CONF  (name $NAME)"
if [[ -z "$LEVELS" ]]; then
  echo "NOTE_OFF sampled uniform [0,1] -- one generation, no shards, no merge"
  echo "total    $N_PER  ->  $OUT"
  df -h . | tail -1
  echo
  # gen_dataset writes to <base>/<conf.name>/, so the base is OUT's parent and
  # OUT's own basename has to BE conf.name -- otherwise the data lands beside
  # where everything downstream will look for it, with no error.
  if [[ "$(basename "$OUT")" != "$NAME" ]]; then
    echo "ERROR: OUT must end in the config's name ($NAME); got $(basename "$OUT")"
    echo "       gen_dataset.py writes to <base>/<conf.name>/, so it cannot"
    echo "       produce $OUT. Either rename OUT or change 'name:' in $CONF."
    exit 1
  fi
  rm -rf "$OUT"
  python -u gen_dataset.py "$(dirname "$OUT")" "$CONF" \
    --data_size "$N_PER" --save_param --device "$DEVICE"
  echo
  echo "Check the occupancy actually varies before training on it:"
  echo "  python scripts/ds_eval_folder.py --dirs $DS/$OUT \\"
  echo "      --arms synth_magx_halfw --n 400 --seed 0 --device cpu"
  echo "act_p10 should now be well below the 0.958 the h2of set reports."
  exit 0
fi

echo "levels   $LEVELS   (NOTE_OFF pinned per shard, merge required)"
echo "per      $N_PER  ->  $(python3 -c "print(len('$LEVELS'.split()) * $N_PER)") total"
echo "shards   $SHARD_ROOT     merged -> $OUT"
df -h . | tail -1
echo

for lv in $LEVELS; do
  tag="s$(python3 -c "print(str($lv).replace('.',''))")"
  dir="$SHARD_ROOT/$tag"
  # "non-empty" is NOT complete: gen_dataset writes as it goes and leaves no
  # marker, so a killed run leaves a directory a non-empty test would skip
  # forever -- and the merge would then be short without saying so. Count.
  have=$( { ls "$dir/$NAME"/audio/*.wav 2>/dev/null || true; } | wc -l )
  if (( have == N_PER )); then
    echo "=== NOTE_OFF $lv -> $dir  complete ($have/$N_PER), skipping"
    continue
  fi
  if (( have > 0 )); then
    echo "=== NOTE_OFF $lv -> $dir  PARTIAL ($have/$N_PER), regenerating"
    rm -rf "$dir"
  else
    echo "=== NOTE_OFF $lv -> $dir"
  fi
  # -u because this is normally run under `| tee`, which block-buffers stdout
  # and makes the whole run's progress appear at once as if it had hung.
  python -u gen_dataset.py "$dir" "$CONF" \
    --data_size "$N_PER" --save_param --device "$DEVICE" \
    --set "fixed_params.NOTE_OFF=$lv"
done

echo
rm -rf "$OUT"
python "$ROOT/scripts/ds_merge_shards.py" --out "$OUT" \
  --shards $(for lv in $LEVELS; do
      echo "$SHARD_ROOT/s$(python3 -c "print(str($lv).replace('.',''))")/$NAME"
    done)

echo
echo "Check the occupancy actually varies before training on it:"
echo "  python scripts/ds_eval_folder.py --dirs $DS/$OUT \\"
echo "      --arms synth_magx_halfw --n 400 --seed 0 --device cpu"
echo "act_p10 should now be well below the 0.958 the h2of set reports."
