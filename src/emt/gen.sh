#!/usr/bin/env bash
# Generate the emt14 IR datasets, train and val, under one pin and one ceiling.
#
#   src/emt/gen.sh                    # 24576 train + 991 val, ~4.5 GB at 1.0 s
#   SPACE=emt13 src/emt/gen.sh        # narrower rho/Ly, T60_DC to 8.0, 1.0 s
#   SPACE=emt12 src/emt/gen.sh        # emt13's box at 2.0 s, ~9.0 GB
#   SPACE=emt11 src/emt/gen.sh        # emt14's box exactly -- same data
#   SPACE=emt10 src/emt/gen.sh        # loss_F1 and fp/op searched, seven params
#   SPACE=emt9 src/emt/gen.sh         # rho fixed, six parameters
#   SPACE=emt8 src/emt/gen.sh         # the 22 kHz ceiling, kept runnable
#   SPACE=emt7 src/emt/gen.sh         # the earlier space, kept runnable
#   N_TRAIN=49152 src/emt/gen.sh      # 8.7 GB -- check df first
#
# THE PIN AND THE CEILING MUST MATCH TRAINING, and that is the whole reason
# this is a script rather than two remembered commands. --fmax and
# --fixed-mode-grid appear here and in scripts/jobs_<SPACE>.txt, and a mismatch
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

# SPACE picks which entry of space.py's NUMERICS table to read, so the values
# cannot drift between generation and training now that the file holds two
# spaces. It is also what PLATE_PARAM_SPACE is set to below.
SPACE=${SPACE:-emt14}
read -r FMAX GRID DUR <<< "$(SPACE="$SPACE" python3 - <<'PY'
import os
ns = {}
exec(open("src/emt/space.py").read(), ns)
n = ns["NUMERICS"][os.environ["SPACE"]]
print(n["fmax"], "%d,%d" % n["grid"], n["duration"])
PY
)"

N_TRAIN=${N_TRAIN:-24576}
N_VAL=${N_VAL:-991}
SEED=${SEED:-0}
DEVICE=${DEVICE:-cuda}
OUT=${OUT:-data}
# MUST match scripts/jobs_<SPACE>.txt exactly, and --compile-plate is the whole
# reason this line exists. Measured here, 64 IRs at 1.0 s: eager 68 s, compiled
# 8 s -- 8.5x, matching the 7.3x that 704c1ea measured on quiet3. It is not a
# free flag either way:
#
#   0f260fc  compile was DROPPED for quiet3, because compiled generation and
#            compiled training disagreed by 7.66% of saturation on the log arm
#            (42.6% in the quietest decile). raw7 tolerated it at <=0.074%.
#            emt8 is on raw7's side of that -- T60_DC is searched, so nothing
#            here sits at the float32 cancellation floor -- but it is a claim,
#            and src.ddsp.diag_gt_floor is what checks it. Run the gate.
#   35e4529  chunk_elems and --batch-size are part of the NUMERICS CONTRACT in
#            eager mode: 1e9 vs 2e8 moved log by 8.51%. Compiled, Inductor
#            fuses the chunk kernel and chunk_elems is a plain memory knob
#            again -- which is the only reason changing it here is safe. It is
#            8e8 for emt12 and emt10, 1e9 for emt13 and emt11. The knob is
#            memory, not FLOPs: total work is B*n_pad*Ts whatever it is, and
#            chunk_elems only decides how many launches that is split across.
#            emt12 runs 8e8 because its 2.0 s training tensor is 8.7 GB and
#            12.0 + 8.7 = 20.7 GB does not leave margin on a 23 GB card;
#            9.6 + 8.7 = 18.3 does. emt13 halves that tensor back to 4.3 GB so
#            12.0 + 4.3 = 16.3 fits and 1e9 is the faster split. Re-run
#            src.ddsp.diag_gt_floor after any change to it. THE DEFAULT BELOW
#            SUITS emt14, emt13 and emt11 -- with SPACE=emt12 or emt10 pass
#            NUMERICS with 800000000 to match their jobs files.
#
# So --batch-size is pinned to eps_ladder.sh's BATCH=64 rather than left at
# make_dataset's default of 32. Under compile the two are interchangeable; if
# compile is ever dropped they are not, and a silent 32/64 split between the
# targets and the renders is exactly the confound above.
NUMERICS=${NUMERICS:-"--batched-plate --compile-plate --chunk-elems 1000000000 --mode-bucket 1024 --batch-size 64"}

echo "$SPACE: fmax=$FMAX  grid=$GRID  duration=${DUR}s"
echo "      train $N_TRAIN -> $OUT/train-$SPACE   val $N_VAL -> $OUT/val-$SPACE"
echo "      $(python3 -c "print(f'{$N_TRAIN*$DUR*44100*4/1e9:.1f} GB training tensor')")"
df -h . | tail -1

for split in train val; do
  n=$([[ $split == train ]] && echo "$N_TRAIN" || echo "$N_VAL")
  seed=$([[ $split == train ]] && echo "$SEED" || echo "$((SEED + 1))")
  dir="$OUT/${split}-$SPACE"
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
  # -u because this is normally run under `| tee`, which makes stdout a pipe:
  # Python then block-buffers and the entire run's progress appears at once when
  # the buffer flushes at exit, which reads exactly like a hang.
  PLATE_PARAM_SPACE="$SPACE" python -u -m src.data.make_dataset \
    --number "$n" --duration "$DUR" --seed "$seed" --output-dir "$dir" \
    --device "$DEVICE" --fmax "$FMAX" --fixed-mode-grid "$GRID" \
    --render-path training $NUMERICS
done

echo
echo "done. Train with scripts/jobs_$SPACE.txt, which reads the same values."
