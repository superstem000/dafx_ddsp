#!/usr/bin/env bash
# Do two runs of the same configuration produce the same numbers, on GPU?
#
#   scripts/diffsynth_determinism_check.sh <gpu>
#
# Run this BEFORE discarding any long run to restart under determinism. Setting
# trainer.deterministic=True asks for reproducibility; it does not guarantee it.
# Two things can go wrong:
#
#   - an op has no deterministic CUDA kernel, and torch raises. Likely
#     candidates in this model are the cuDNN GRU backward and the linear
#     interpolation in util.resample_frames, whose backward uses an atomic
#     scatter.
#   - determinism holds within a card but the runs land on different GPU models,
#     which changes kernel selection.
#
# Either way the answer is cheap: two 2-epoch runs on one card, then an exact
# comparison of every logged scalar. If they match bitwise, restarting the real
# runs buys what it is supposed to buy. If they do not, restarting buys nothing
# and the error bar approach (--spread) is the honest fallback.
#
# Both runs go on the SAME gpu on purpose. Cross-card reproducibility is a
# separate question, and if same-card fails there is no point asking it.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: scripts/diffsynth_determinism_check.sh <gpu>"
  exit 2
fi
GPU=$1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DS="$ROOT/external/diffsynth"
WORK="$DS/outputs/determinism"

rm -rf "$WORK"; mkdir -p "$WORK"
cd "$DS"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

run () {
  local tag=$1 rc=0
  CUDA_VISIBLE_DEVICES="$GPU" python train.py \
    experiment=pretrain \
    data.id_dir="$DS/data/diffsynth_5-6/harmor_2oscfree" \
    data.ood_dir="$DS/data/nsynth-train" \
    data.batch_size=16 \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.max_epochs=2 \
    trainer.limit_train_batches=20 \
    trainer.limit_val_batches=5 \
    trainer.num_sanity_val_steps=0 \
    hydra.run.dir="$WORK/$tag" \
    < /dev/null > "$WORK/$tag.log" 2>&1 || rc=$?
  if (( rc != 0 )); then
    echo "ERROR: run $tag exited $rc. Last 25 lines:"
    echo "--------------------------------------------------------------"
    tail -25 "$WORK/$tag.log"
    echo "--------------------------------------------------------------"
    if grep -qi "deterministic" "$WORK/$tag.log"; then
      echo
      echo "That looks like an op without a deterministic implementation."
      echo "Options, in order of preference:"
      echo "  1. keep determinism and avoid the op (may not be possible here)"
      echo "  2. trainer.deterministic=warn -- runs, but reproducibility is then"
      echo "     NOT guaranteed, so it buys nothing over the current setup"
      echo "  3. leave determinism off and quote the measured run-to-run spread"
      echo "     as an error bar (monitor_diffsynth --spread)"
    fi
    exit $rc
  fi
}

echo "=== run 1 of 2 on gpu $GPU"; run A
echo "=== run 2 of 2 on gpu $GPU"; run B

echo
python3 - "$WORK" <<'PY'
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

work = sys.argv[1]


def scalars(tag):
    ea = EventAccumulator(f"{work}/{tag}/tb_logs", size_guidance={"scalars": 0})
    ea.Reload()
    return {t: {e.step: e.value for e in ea.Scalars(t)}
            for t in ea.Tags().get("scalars", [])}


A, B = scalars("A"), scalars("B")
common = sorted(set(A) & set(B))
if not common:
    print("no scalars logged -- check the run logs"); sys.exit(1)

bad = []
for t in common:
    for step in sorted(set(A[t]) & set(B[t])):
        a, b = A[t][step], B[t][step]
        if a != b:
            bad.append((t, step, a, b))

print(f"compared {len(common)} scalar series")
if not bad:
    print("\nIDENTICAL -- every logged value matches bitwise.")
    print("Restarting the real runs under determinism will make the arms")
    print("differ only by their loss, which is what the comparison needs.")
    sys.exit(0)

print(f"\nDIFFERS in {len({t for t, *_ in bad})} series, {len(bad)} points. First few:")
for t, step, a, b in bad[:8]:
    print(f"  {t:<24} step {step:>7}   {a!r}  vs  {b!r}")
print("\nDeterminism did not hold. Restarting buys nothing; report the measured")
print("run-to-run spread instead (monitor_diffsynth --spread 45).")
sys.exit(1)
PY
