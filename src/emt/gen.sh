#!/usr/bin/env bash
# Generate the emt7 IR datasets, train and val, under one pin and one ceiling.
#
#   src/emt/gen.sh                    # 24576 train + 991 val, ~4.3 GB
#   N_TRAIN=49152 src/emt/gen.sh      # 8.7 GB -- check df first
#
# THE PIN AND THE CEILING MUST MATCH TRAINING, and that is the whole reason
# this is a script rather than two remembered commands. --fmax and
# --fixed-mode-grid appear here and in scripts/jobs_emt7.txt, and a mismatch
# between them is not an error anywhere: it silently renders the targets on one
# plate and the model's attempts on another, and the loss goes to a floor
# nobody can explain. Both values come from src/emt/space.py.
#
# WHY NOT MORE CLIPS. train_encoder holds the whole training set as one tensor.
# At 1.0 s and 44.1 kHz float32 that is 176 KB per clip: 24576 is 4.3 GB, the
# same as raw7's 98304 at 0.25 s, and 49152 is 8.7 GB. The second does not fit
# beside the diffsynth class sweep.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$(cd "$HERE/../.." && pwd)"

read -r FMAX GRID DUR <<< "$(python3 - <<'PY'
ns = {}
exec(open("src/emt/space.py").read(), ns)
print(ns["FMAX"], "%d,%d" % ns["FIXED_MODE_GRID"], ns["DURATION"])
PY
)"

N_TRAIN=${N_TRAIN:-24576}
N_VAL=${N_VAL:-991}
SEED=${SEED:-0}
DEVICE=${DEVICE:-cuda}
OUT=${OUT:-data}
NUMERICS=${NUMERICS:-"--batched-plate --chunk-elems 50000000 --mode-bucket 1024"}

echo "emt7: fmax=$FMAX  grid=$GRID  duration=${DUR}s"
echo "      train $N_TRAIN -> $OUT/train-emt7   val $N_VAL -> $OUT/val-emt7"
echo "      $(python3 -c "print(f'{$N_TRAIN*$DUR*44100*4/1e9:.1f} GB training tensor')")"
df -h . | tail -1

for split in train val; do
  n=$([[ $split == train ]] && echo "$N_TRAIN" || echo "$N_VAL")
  seed=$([[ $split == train ]] && echo "$SEED" || echo "$((SEED + 1))")
  dir="$OUT/${split}-emt7"
  if [[ -d "$dir" ]] && (( $( { ls "$dir" 2>/dev/null || true; } | wc -l ) > 0 )); then
    echo "$dir already populated -- skipping"
    continue
  fi
  echo
  echo "=== $split: $n IRs, seed $seed"
  PLATE_PARAM_SPACE=emt7 python -m src.data.make_dataset \
    --number "$n" --duration "$DUR" --seed "$seed" --output-dir "$dir" \
    --device "$DEVICE" --fmax "$FMAX" --fixed-mode-grid "$GRID" \
    --render-path training $NUMERICS
done

echo
echo "done. Train with scripts/jobs_emt7.txt, which reads the same two values."
