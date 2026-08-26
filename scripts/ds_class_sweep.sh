#!/usr/bin/env bash
# Screen every NSynth class: 30 epochs of class-specific training, three arms.
#
#   tmux new -d -s sweep 'scripts/ds_class_sweep.sh 2>&1 | tee -a logs/sweep.log'
#   python scripts/ds_sweep_report.py            # any time, including mid-run
#
# THE QUESTION. Trained jointly on all of NSynth, magx beats both compressed
# arms on three classes -- reed_acoustic, guitar_acoustic, keyboard_acoustic.
# Trained on keyboard_acoustic alone it lost by 1.59. One class is one data
# point and the run cost fourteen hours, so this asks the same question of
# every class that is large enough to ask it of, at 30 epochs each, and reports
# whether ANY of them puts magx first on val_ood/mfccdb.
#
# It is a screen, not a result. 30 epochs is 3030 optimizer steps against the
# joint run's 50,000, and on keyboard the arms were still 1.46 apart at epoch
# 239 and moving. Nothing here converges. What it can do is rank classes and
# rule out the ones that are not close, so a full run is spent on a candidate
# rather than on a guess -- and if nothing is close, that is the answer.
#
# WHY 200-230 AND NOT 0-30. Every arm branches from its own pre_* at epoch 199,
# the same branch point every class run has used. Epochs 0-199 are the shared
# synthetic pretrain; the losses first differ during the crossfade and are
# fully separated from 200 on. Starting from scratch would spend the whole
# budget on the part where the arms are identical by construction.
#
# THE COST MODEL, and why the download is batched. The archive is 22 GB and
# forward-only, so a per-class fetch pays that stream every time: eleven
# classes one at a time is hours of re-reading the same bytes. One stream fills
# BATCH classes at once, bounded by MAX files each (~1.1 GB at 8500). Training
# is ~25 min per class for 30 epochs with the three arms on three GPUs, so with
# BATCH=3 the sweep is roughly four streams and eleven trainings -- about six
# hours, most of it GPU rather than network.
#
# DISK. Peak is BATCH x MAX x 128 KB while a batch is being fetched. Each
# class's dataset is deleted the moment its three arms finish, and each run
# directory is reduced to its scalars.csv -- a few hundred KB -- and then
# removed. Nothing accumulates except the CSVs.
#
# WHICH CLASSES. data.py:92 draws len(id_dat) indices from the ood set without
# replacement, so a class with fewer files than ID_DIR raises before the first
# batch. ID_DIR is h2of_kb at 8064 files, which is also what fixes the training
# set at 6451 clips for every class -- the same count every class run so far has
# used, so these are comparable to each other and to the keyboard numbers. The
# default list is every class estimated to clear 8064 in nsynth-train; a class
# that comes up short after download is logged and skipped rather than crashing
# the sweep.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

# Estimated counts in nsynth-train, from 25000 sampled files scaled by 11.57.
# bass_electronic (~8200) and keyboard_acoustic (8068 measured) are the two
# that sit close to the 8064 floor; both are kept and verified after download.
CLASSES=${CLASSES:-"reed_acoustic string_acoustic brass_acoustic guitar_acoustic \
guitar_electronic mallet_acoustic organ_electronic keyboard_electronic \
keyboard_acoustic bass_synthetic bass_electronic"}
EPOCHS=${EPOCHS:-230}          # absolute: 199 restored + 31 of real audio
BATCH=${BATCH:-3}              # classes per download stream
MAXF=${MAXF:-8500}             # files kept per class; must exceed the ID count
IDDIR=${IDDIR:-external/diffsynth/data/h2of_kb}
POLL=${POLL:-60}
STABLE=${STABLE:-1}            # 1 poll, so a free gpu is claimed after ~1 min
GPUS=${GPUS:-}
OUT=${OUT:-results/class_sweep}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-101}   # 6451 train clips / batch 64

mkdir -p "$OUT"
ID_N=$(ls "$IDDIR/audio" 2>/dev/null | wc -l)
if (( ID_N == 0 )); then
  echo "ERROR: $IDDIR/audio is empty or missing; it fixes the training-set size"
  exit 1
fi
if (( MAXF <= ID_N )); then
  echo "ERROR: MAX=$MAXF must exceed the ID set's $ID_N files (data.py:92 draws"
  echo "len(id_dat) indices from the ood set without replacement)"
  exit 1
fi
read -r -a ALL <<< "$CLASSES"
echo "sweep: ${#ALL[@]} classes, $EPOCHS epochs, id=$ID_N clips, batch=$BATCH, max=$MAXF"
echo "results -> $OUT"

fetch() {   # fetch a batch of classes in one stream
  echo
  echo "=== fetching: $*"
  CLASS="$*" MAX="$MAXF" "$HERE/get_nsynth.sh"
}

for ((i = 0; i < ${#ALL[@]}; i += BATCH)); do
  batch=("${ALL[@]:i:BATCH}")
  fetch "${batch[@]}"

  for cls in "${batch[@]}"; do
    tag="${cls//_/}"
    dir="external/diffsynth/data/nsynth-${tag}"
    n=$(ls "$dir/audio" 2>/dev/null | wc -l)
    echo
    echo "===================================================================="
    echo "=== $cls   $n files   $(date '+%H:%M:%S')"
    if (( n < ID_N )); then
      echo "SKIP $cls: $n files < the ID set's $ID_N; data.py:92 would raise"
      echo -e "$cls\t$n\tSKIPPED_TOO_SMALL" >> "$OUT/skipped.tsv"
      rm -rf "$dir"
      continue
    fi

    jobs="$OUT/jobs_${tag}.txt"
    # FORCE=1 so a re-run of the sweep overwrites its own leftovers rather than
    # failing three jobs in sixty seconds on the clash guard.
    {
      echo "# generated by ds_class_sweep.sh -- $cls, $n files, $EPOCHS epochs"
      for arm in magx hyb log; do
        case $arm in
          magx) extra="model.sw_loss.log_mag_w=0 model.sw_loss.power=1 schedule.sw_w.end_v=0.5"; res=pre_magx_halfw ;;
          hyb)  extra="model.sw_loss.power=1 model.sw_loss.log_eps_v=1e-2"; res=pre_hybridx ;;
          log)  extra="model.sw_loss.power=1 model.sw_loss.log_eps_v=1e-2 model.sw_loss.mag_w=0 schedule.sw_w.end_v=0.5"; res=pre_logx_halfw ;;
        esac
        echo "FORCE=1 ID_DIR=$IDDIR OOD_DIR=$dir RESUME=$res scripts/ds_run.sh sw_${tag}_${arm} resume_real \"{gpu}\" trainer.max_epochs=$EPOCHS trainer.checkpoint_every_n_epochs=1000 model.log_grad=false $extra"
      done
    } > "$jobs"

    q=(python scripts/gpu_queue.py --jobs "$jobs" --poll "$POLL" --stable "$STABLE")
    [[ -n "$GPUS" ]] && q+=(--gpus "$GPUS")
    "${q[@]}" || echo "WARNING: queue reported failures for $cls"

    # The numbers, and only the numbers. scalars.csv is a few hundred KB; the
    # event files and checkpoints behind it are ~300 MB per arm and there is no
    # disk to keep eleven classes of them.
    python scripts/ds_export_scalars.py --only "^sw_${tag}_" \
        --steps-per-epoch "$STEPS_PER_EPOCH" || true
    for arm in magx hyb log; do
      src="results/diffsynth/sw_${tag}_${arm}/scalars.csv"
      [[ -f "$src" ]] && cp "$src" "$OUT/${cls}__${arm}.csv"
    done
    rm -rf results/diffsynth/sw_${tag}_*
    rm -rf "$dir"
    echo "=== $cls done, dataset and run dirs removed   $(date '+%H:%M:%S')"
    df -h . | tail -1
  done
done

echo
echo "sweep complete."
python scripts/ds_sweep_report.py --dir "$OUT" || true
