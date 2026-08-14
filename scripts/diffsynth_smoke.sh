#!/usr/bin/env bash
# Does the vendored diffsynth run at all? Answered on CPU, in minutes.
#
#   scripts/diffsynth_smoke.sh
#
# The point is to separate two questions that are easy to conflate:
#
#   1. does the environment work -- do hydra, PL 2, nnAudio and the synth
#      import and fit together
#   2. is the training recipe right
#
# Only (1) is answered here, and it is the one blocking everything else. It
# needs no GPU, so it can run while the cards are busy for the next ten hours.
#
# What it does: generates 128 in-domain examples on CPU, then runs one epoch
# with the p-loss schedule on a handful of batches. Anything that is going to
# break -- a removed PL argument, a missing config group, an nnAudio API change
# -- breaks here in about a minute rather than after a dataset generation.
#
# ood_dir points at the in-domain set on purpose. IdOodDataModule asserts the
# two training splits are the same length and subsamples OOD down to the ID
# size, so with no NSynth on disk the assert is what would fail first. Reusing
# the ID audio as stand-in OOD (params=False, so only the audio is read) keeps
# the datamodule honest without needing a 70 GB download to test an import.
set -euo pipefail

cd "$(dirname "$0")/../external/diffsynth"

N=${N:-128}
DEV=${DEV:-cpu}
OUT=${OUT:-data/smoke}

echo "=== python: $(command -v python)"
python - <<'PY'
import sys
print("python", sys.version.split()[0])
for m in ("torch", "pytorch_lightning", "hydra", "omegaconf", "nnAudio",
          "librosa", "soundfile", "torchaudio", "matplotlib"):
    try:
        mod = __import__(m)
        print(f"  {m:20} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"  {m:20} MISSING/BROKEN -- {type(e).__name__}: {e}")
PY

echo
echo "=== 1/2  generating $N in-domain examples on $DEV"
# --save_param is required, not optional. Without it gen_dataset.py saves every
# external parameter of the GENERATOR dag, ADSR controls included, and the model
# synth has no envelopes so those keys do not exist on its side -- training dies
# with KeyError: 'enva_peak'. The list it reads lives in the dataset config.
python gen_dataset.py "$OUT" configs/synth/dataset/h2of.yaml \
  --data_size "$N" --batch_size 16 --device "$DEV" --save_param

DATA_DIR="$OUT/harmor_2oscfree"
echo "  wrote $(ls "$DATA_DIR/audio" | wc -l) wavs, $(ls "$DATA_DIR/param" | wc -l) param files"
python - "$DATA_DIR" <<'PY'
import sys, torch
p = torch.load(f"{sys.argv[1]}/param/00000.pt", weights_only=False)
print("  target keys:", {k: tuple(v.shape) for k, v in p.items()})
PY

echo
echo "=== 2/2  one epoch, p-loss schedule, a few batches"
# limit_*_batches keeps this to seconds; the question is whether it runs, not
# whether it learns. num_workers=0 because a dataloader worker crash on CPU is
# reported as an opaque hang rather than a traceback.
python train.py \
  experiment=ploss \
  data.id_dir="$DATA_DIR" \
  data.ood_dir="$DATA_DIR" \
  data.batch_size=4 \
  data.num_workers=0 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=3 \
  trainer.limit_val_batches=2 \
  trainer.num_sanity_val_steps=0 \
  trainer.accelerator=cpu \
  trainer.devices=1

echo
echo "=== smoke test passed"
echo "Environment is good. Next: generate the real dataset (20000, sec 4.3.1)"
echo "and fetch NSynth for the Real arm."
