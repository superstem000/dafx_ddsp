#!/usr/bin/env bash
# Does a resumed run get the same train/val split as the run it resumed from?
#
#   scripts/diffsynth_split_check.sh
#
# This is the one open risk that could invalidate the whole comparison rather
# than merely bias a number, so it is worth settling before any real training.
#
# IdOodDataModule.create_split draws from the global RNG at setup() time --
# random_split with no generator, plus np.random.choice for the OOD pool -- so
# the split is only reproducible if every run makes the same number of RNG draws
# before setup(). pl.seed_everything(0) fixes the starting state; nothing fixes
# the draw count. PyTorch-Lightning restores RNG state from a checkpoint when
# resuming, and whether it does so before or after the datamodule's setup() hook
# is an implementation detail we should not be relying on.
#
# It matters because the paper's Synth and Real models ARE resumes: pretrain for
# 200 epochs, then continue. If the resumed run draws a different split, the
# pretrain phase's validation files become the resume phase's training files.
# That is leakage across the phase boundary, it inflates Synth and Real relative
# to P-loss -- i.e. in the direction of the paper's own conclusion -- and no
# metric in the run would show it.
#
# Three runs, one epoch each, on CPU:
#
#   A  fresh
#   B  resumed from A
#   C  fresh again
#
# A vs C is the baseline: two identical fresh runs must agree, or the comparison
# below means nothing. A vs B is the actual question.
#
# No GPU and no NSynth needed: ood_dir points at the in-domain set, which still
# exercises np.random.choice and both random_splits.
set -euo pipefail

cd "$(dirname "$0")/../external/diffsynth"

DATA=${DATA:-"$(pwd)/data/diffsynth_5-6/harmor_2oscfree"}
WORK=${WORK:-"$(pwd)/outputs/split_check"}

if [[ ! -d "$DATA/audio" ]]; then
  echo "ERROR: no dataset at $DATA"
  echo "Generate it first -- see scripts/diffsynth_smoke.sh for the command."
  exit 1
fi

rm -rf "$WORK"
mkdir -p "$WORK"

# hydra.run.dir is pinned so the manifests land where this script can find them;
# by default hydra picks outputs/<date>/<time>/ and we would be globbing.
run () {
  local tag=$1; shift
  python train.py \
    experiment=pretrain \
    data.id_dir="$DATA" \
    data.ood_dir="$DATA" \
    data.batch_size=4 \
    data.num_workers=0 \
    trainer.limit_train_batches=2 \
    trainer.limit_val_batches=1 \
    trainer.num_sanity_val_steps=0 \
    trainer.accelerator=cpu \
    trainer.devices=1 \
    hydra.run.dir="$WORK/$tag" \
    "$@" > "$WORK/$tag.log" 2>&1
}

echo "=== A  fresh, 1 epoch"
run A trainer.max_epochs=1

CKPT=$(find "$WORK/A" -name 'last.ckpt' | head -1)
if [[ -z "$CKPT" ]]; then
  echo "ERROR: no checkpoint written by run A -- see $WORK/A.log"
  tail -20 "$WORK/A.log"
  exit 1
fi
echo "  checkpoint: ${CKPT#$WORK/}"

echo "=== B  resumed from A, to epoch 2"
run B trainer.max_epochs=2 trainer.resume_from_checkpoint="$CKPT"

echo "=== C  fresh again, 1 epoch"
run C trainer.max_epochs=1

echo
python - "$WORK" <<'PY'
import json, os, sys

work = sys.argv[1]
def load(tag):
    p = os.path.join(work, tag, "split_manifest.json")
    if not os.path.exists(p):
        print(f"MISSING manifest for run {tag} -- see {tag}.log")
        return None
    return json.load(open(p))

A, B, C = load("A"), load("B"), load("C")
if not all((A, B, C)):
    sys.exit(1)

def cmp(x, y, xn, yn):
    bad = [k for k in sorted(x) if x[k]["sha1"] != y.get(k, {}).get("sha1")]
    print(f"\n{xn} vs {yn}: {'IDENTICAL' if not bad else 'DIFFERS in ' + ', '.join(bad)}")
    for k in bad:
        print(f"    {k}: {xn} {x[k]['sha1'][:12]} n={x[k]['n']}   "
              f"{yn} {y[k]['sha1'][:12]} n={y[k]['n']}")
    return not bad

print("splits recorded:", ", ".join(sorted(A)))
base = cmp(A, C, "A", "C")
same = cmp(A, B, "A", "B")

print()
if not base:
    print("VERDICT: two fresh runs already disagree. The split is not reproducible")
    print("at all, so every arm sees different data and no comparison between them")
    print("is valid. Fix by passing an explicit generator to random_split and a")
    print("seeded RandomState to np.random.choice in diffsynth/data.py.")
    sys.exit(2)
if same:
    print("VERDICT: OK. A resumed run reproduces the split it resumed from, so")
    print("Synth and Real continue on the same partition pretrain used and there")
    print("is no leakage across the phase boundary.")
else:
    print("VERDICT: LEAKAGE. The resumed run drew a different split, so the")
    print("pretrain phase's validation files are in the resume phase's training")
    print("set. Synth and Real would be flattered relative to P-loss, in exactly")
    print("the direction of the paper's conclusion. Fix before running anything:")
    print("seed the split explicitly in diffsynth/data.py -- random_split(...,")
    print("generator=torch.Generator().manual_seed(0)) and np.random.default_rng(0)")
    print("for the OOD choice -- so it no longer depends on RNG history.")
    sys.exit(3)
PY
