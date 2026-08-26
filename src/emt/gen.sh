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
# MUST match scripts/jobs_emt7.txt, and not only for correctness -- these two
# numbers are most of the wall clock. The modal sum walks modes in chunks of
# chunk_elems // (B * n_pad): at 50M and a 1.0 s pad that is ~17 modes a pass,
# so 46,854 modes is ~2,700 launches per batch of 32 with the GPU idle between
# them. 400M gives ~141 a pass, ~330 launches. The 0.25 s spaces never felt this
# because a shorter pad bought 4x the chunk from the same budget. 400M is
# 1.6 GB per tensor, three live at once.
NUMERICS=${NUMERICS:-"--batched-plate --compile-plate --chunk-elems 400000000 --mode-bucket 1024"}

echo "emt7: fmax=$FMAX  grid=$GRID  duration=${DUR}s"
echo "      train $N_TRAIN -> $OUT/train-emt7   val $N_VAL -> $OUT/val-emt7"
echo "      $(python3 -c "print(f'{$N_TRAIN*$DUR*44100*4/1e9:.1f} GB training tensor')")"
df -h . | tail -1

for split in train val; do
  n=$([[ $split == train ]] && echo "$N_TRAIN" || echo "$N_VAL")
  seed=$([[ $split == train ]] && echo "$SEED" || echo "$((SEED + 1))")
  dir="$OUT/${split}-emt7"
  # "non-empty" is NOT complete. make_dataset writes one npz per IR as it goes
  # and generation_summary.txt only after the last one, so a killed run leaves a
  # directory that a non-empty test skips forever -- and training then reads a
  # short dataset without complaining. Both conditions, or regenerate.
  if [[ -d "$dir" ]]; then
    have=$( { ls "$dir"/random_IR_[0-9]*.npz 2>/dev/null || true; } | wc -l )
    if [[ -f "$dir/generation_summary.txt" ]] && (( have == n )); then
      echo "$dir complete ($have/$n) -- skipping"
      continue
    fi
    if (( have > 0 )); then
      echo "$dir is PARTIAL ($have/$n npz, summary $( [[ -f "$dir/generation_summary.txt" ]] && echo present || echo missing ))"
      echo "  regenerating from scratch -- rm -rf then continue"
      rm -rf "$dir"
    fi
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
