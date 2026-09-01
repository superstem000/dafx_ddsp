#!/usr/bin/env bash
# Generate the one-shot diffsynth dataset: notes that start late and end by decaying.
#
#   scripts/ds_gen_varlen.sh                    # 20000 clips
#   N_PER=320 scripts/ds_gen_varlen.sh          # smoke test
#   DEVICE=cpu scripts/ds_gen_varlen.sh         # no GPU
#
# WHAT IT PRODUCES, and why it is not the published set. h2of.yaml models a HELD
# KEY: the envelope always begins rising at t=0, note_off is pinned at 0.75 so
# every note releases at the same instant, and the noise term is added outside
# the A+D+S sum so a finished note never reaches silence -- measured across 400
# clips of the generated set, occupancy of the 4 s window spans 0.958 to 0.998.
#
# A note from a sample library is a ONE-SHOT: it starts late because the sampler
# trimmed near the transient rather than on it, and it ends because it decayed,
# not because a key was released. h2of_var.yaml is that model --
# delay -> attack -> decay -> silence, with delay, attack and decay drawn per
# clip -- and the config explains each of the three changes it makes.
#
# NOT SHARDED. An earlier version pinned note_off per shard and merged, which
# brought in gen_dataset's per-invocation numbering and WaveParamDataset's
# pair-by-sorted-index; that merge was the only step in the pipeline that could
# fail silently. Note length is now a property of the drawn parameters, so this
# is one generation with nothing to merge. scripts/ds_merge_shards.py still
# exists, verified, if a sharded sweep is ever wanted.
#
# OCCUPANCY IS THE ACCEPTANCE TEST, not a nice-to-have: if the generated set
# does not span a real range of it, the dataset has not solved the problem it
# was made for and there is no point spending training time on it. The command
# to check is printed at the end.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DS="$ROOT/external/diffsynth"
cd "$DS"

CONF=${CONF:-configs/synth/dataset/h2of_var.yaml}
NAME=$(python3 -c "from omegaconf import OmegaConf; print(OmegaConf.load('$CONF').name)")
N_PER=${N_PER:-20000}
DEVICE=${DEVICE:-cuda}
OUT=${OUT:-data/diffsynth_5-6/$NAME}

# gen_dataset writes to <base>/<conf.name>/, so OUT's parent is the base and
# OUT's own basename has to BE conf.name -- otherwise the data lands beside
# where everything downstream will look for it, with no error anywhere.
if [[ "$(basename "$OUT")" != "$NAME" ]]; then
  echo "ERROR: OUT must end in the config's name ($NAME); got $(basename "$OUT")"
  echo "       gen_dataset.py writes to <base>/<conf.name>/, so it cannot"
  echo "       produce $OUT. Rename OUT, or change 'name:' in $CONF."
  exit 1
fi

echo "config   $CONF  (name $NAME)"
echo "total    $N_PER  ->  $OUT"
df -h . | tail -1
echo

rm -rf "$OUT"
# -u because this is normally run under `| tee`, which makes stdout a pipe:
# Python then block-buffers and the whole run's progress appears at once when
# the buffer flushes at exit, which reads exactly like a hang.
python -u gen_dataset.py "$(dirname "$OUT")" "$CONF" \
  --data_size "$N_PER" --save_param --device "$DEVICE"

echo
echo "ACCEPTANCE TEST -- run this before training on it:"
echo
echo "  python scripts/ds_eval_folder.py --dirs $DS/$OUT \\"
echo "      --arms synth_magx_halfw --n 400 --seed 0 --device cpu"
echo
echo "act_p10 must be well below the 0.958 the h2of set reports, and act_p90"
echo "should still be near 1.0 -- a spread, not a shift. If occupancy is still"
echo "pinned near 1.0 the dataset has not solved the problem it was made for."
