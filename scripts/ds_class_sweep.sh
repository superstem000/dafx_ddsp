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
# every class large enough to ask it of, at a fixed budget each, and reports
# whether ANY of them puts magx first on val_ood/mfccdb.
#
# It is a screen, not a result. REF_STEPS is 3030 optimizer steps against the
# joint run's 50,000, and on keyboard the arms were still 1.46 apart at epoch
# 239 and moving. Nothing here converges. What it can do is rank classes and
# rule out the ones that are not close, so a full run is spent on a candidate
# rather than on a guess -- and if nothing is close, that is the answer.
#
# WHY IT STARTS AT 200 AND NOT AT 0. Every arm branches from its own pre_* at epoch 199,
# the same branch point every class run has used. Epochs 0-199 are the shared
# synthetic pretrain; the losses first differ during the crossfade and are
# fully separated from 200 on. Starting from scratch would spend the whole
# budget on the part where the arms are identical by construction.
#
# THE COST MODEL, and why the download is batched. The archive is 22 GB and
# forward-only, so a per-class fetch pays that stream every time: nineteen
# classes one at a time is hours of re-reading the same bytes. One stream fills
# BATCH classes at once, bounded by MAX files each (~1.1 GB at 8500). Training
# is ~25-35 min per class -- REF_STEPS is fixed, so only the validation passes
# vary -- with the three arms on three GPUs. At BATCH=3 that is seven streams
# and nineteen trainings, roughly twelve hours, most of it GPU.
#
# DISK. Peak is BATCH x MAX x 128 KB while a batch is being fetched. Each
# class's dataset is deleted the moment its three arms finish, and each run
# directory is reduced to its scalars.csv -- a few hundred KB -- and then
# removed. Nothing accumulates except the CSVs.
#
# WHICH CLASSES, AND WHY THE ID SET IS SIZED PER CLASS. data.py:92 draws
# len(id_dat) indices from the ood set without replacement, so the ID
# directory's file count -- nothing else about it -- decides the training-set
# size, and a class with fewer files than the ID set raises before the first
# batch. Held at h2of_kb's 8064 that admits only 11 of NSynth's 25
# family_source classes. So the ID set is rebuilt per class at
# min(IDCAP, class files) by ds_id_subset.sh, and the floor becomes MINFILES.
#
# STEPS ARE EQUALISED, NOT EPOCHS, and that is what makes small classes usable
# rather than merely runnable. An epoch is a pass over the data, so 30 epochs
# of a 2000-clip class is a quarter of the optimisation 30 epochs of an
# 8000-clip class gets -- the same trap that made the keyboard run 20,200 steps
# against the joint run's 50,000. Instead every class runs REF_STEPS optimizer
# steps: max_epochs is REF_STEPS / steps_per_epoch, so a small class simply
# takes more passes over less data.
#
# AND THE LR SCHEDULE IS STRETCHED TO MATCH, because ExponentialLR steps per
# EPOCH. Equalising steps alone would leave a small class decaying its LR four
# times faster per unit of optimisation. gamma = 0.99^(spe/101) makes the
# per-STEP decay identical across classes, and model.lr is scaled so the phase
# still begins at 1.340e-4 -- where the pretrain ended -- rather than jumping.
# Both rely on model.py's on_train_start override, since torch serialises gamma
# and base_lrs into the scheduler state and Lightning restores them.
#
# The net effect: every class gets the same number of updates on the same
# per-step LR curve, and the only thing that varies is the audio. A class that
# comes up short after download is logged and skipped rather than crashing.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

# Every family_source with enough files to train on. The six left out --
# mallet_synthetic ~1515, reed_synthetic ~555, organ_acoustic ~150,
# bass_acoustic ~127, reed_electronic ~116, vocal_electronic ~116 -- would run
# hundreds of epochs over a few dozen clips, where validation dominates the
# wall clock and the training split is smaller than a batch.
CLASSES=${CLASSES:-"reed_acoustic string_acoustic brass_acoustic guitar_acoustic \
guitar_electronic mallet_acoustic organ_electronic keyboard_electronic \
keyboard_acoustic bass_synthetic bass_electronic vocal_synthetic \
flute_acoustic mallet_electronic guitar_synthetic synth_lead_synthetic \
vocal_acoustic keyboard_synthetic flute_synthetic"}
REF_STEPS=${REF_STEPS:-3030}   # 30 epochs x 101 steps, the keyboard reference
MINFILES=${MINFILES:-2000}     # below this, epochs explode and validation dominates
IDCAP=${IDCAP:-8064}           # never more in-domain than h2of_kb holds
BATCH=${BATCH:-3}              # classes per download stream
MAXF=${MAXF:-8500}             # files kept per class in the stream
IDSRC=${IDSRC:-external/diffsynth/data/h2of_kb}
IDTMP=${IDTMP:-external/diffsynth/data/_sweep_id}
POLL=${POLL:-60}
STABLE=${STABLE:-1}            # 1 poll, so a free gpu is claimed after ~1 min
GPUS=${GPUS:-}
OUT=${OUT:-results/class_sweep}

mkdir -p "$OUT"
SRC_N=$(ls "$IDSRC/audio" 2>/dev/null | wc -l)
if (( SRC_N == 0 )); then
  echo "ERROR: $IDSRC/audio is empty or missing; the ID set fixes training size"
  exit 1
fi
read -r -a ALL <<< "$CLASSES"
echo "sweep: ${#ALL[@]} classes, $REF_STEPS steps each, id source $SRC_N files,"
echo "       min $MINFILES files, id cap $IDCAP, batch $BATCH, max $MAXF"
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
    if (( n < MINFILES )); then
      echo "SKIP $cls: $n files < MINFILES=$MINFILES"
      echo -e "$cls\t$n\tSKIPPED_TOO_SMALL" >> "$OUT/skipped.tsv"
      rm -rf "$dir"
      continue
    fi

    # The ID set is what sizes the training set, so it is rebuilt to fit this
    # class: capped at IDCAP so a big class gets no more than the keyboard
    # reference, and at n so data.py:92 can draw without replacement.
    id_n=$(( n < IDCAP ? n : IDCAP ))
    "$HERE/ds_id_subset.sh" "$id_n" "$IDSRC" "$IDTMP" > /dev/null
    IDDIR="$IDTMP"

    # Same optimizer steps and same per-step LR curve for every class, whatever
    # its size. Equalising epochs instead would hand a small class a fraction
    # of the optimisation and call it the same experiment.
    read -r spe epochs gamma baselr < <(python3 - "$id_n" "$REF_STEPS" <<'SPE'
import sys
id_n, ref_steps = int(sys.argv[1]), int(sys.argv[2])
train = int(id_n * 0.8)
spe = max(1, -(-train // 64))          # batch 64, trailing partial batch kept
REF_SPE, G, BASE = 101, 0.99, 1e-3     # the keyboard reference run
epochs = max(1, round(ref_steps / spe))
gamma = G ** (spe / REF_SPE)           # per-STEP decay matched to the reference
lr200 = BASE * G ** 200                # where the pretrain leaves off
print(spe, epochs, f"{gamma:.6f}", f"{lr200 / gamma ** 200:.6e}")
SPE
)
    max_ep=$(( 200 + epochs ))
    echo "    id $id_n -> $(( id_n * 8 / 10 )) train clips, $spe steps/epoch,"
    echo "    $epochs epochs (max_epochs=$max_ep) = $(( spe * epochs )) steps,"
    echo "    decay $gamma, lr $baselr"

    jobs="$OUT/jobs_${tag}.txt"
    # FORCE=1 so a re-run of the sweep overwrites its own leftovers rather than
    # failing three jobs in sixty seconds on the clash guard.
    {
      echo "# generated by ds_class_sweep.sh -- $cls, $n files, id $id_n,"
      echo "# $spe steps/epoch x $epochs epochs = $(( spe * epochs )) steps,"
      echo "# decay $gamma, lr $baselr  (steps and per-step LR matched across classes)"
      for arm in magx hyb log; do
        case $arm in
          magx) extra="model.sw_loss.log_mag_w=0 model.sw_loss.power=1 schedule.sw_w.end_v=0.5"; res=pre_magx_halfw ;;
          hyb)  extra="model.sw_loss.power=1 model.sw_loss.log_eps_v=1e-2"; res=pre_hybridx ;;
          log)  extra="model.sw_loss.power=1 model.sw_loss.log_eps_v=1e-2 model.sw_loss.mag_w=0 schedule.sw_w.end_v=0.5"; res=pre_logx_halfw ;;
        esac
        echo "FORCE=1 ID_DIR=$IDDIR OOD_DIR=$dir RESUME=$res scripts/ds_run.sh sw_${tag}_${arm} resume_real \"{gpu}\" trainer.max_epochs=$max_ep trainer.checkpoint_every_n_epochs=100000 model.log_grad=false model.lr=$baselr model.decay_rate=$gamma $extra"
      done
    } > "$jobs"

    q=(python scripts/gpu_queue.py --jobs "$jobs" --poll "$POLL" --stable "$STABLE")
    [[ -n "$GPUS" ]] && q+=(--gpus "$GPUS")
    "${q[@]}" || echo "WARNING: queue reported failures for $cls"

    # The numbers, and only the numbers. scalars.csv is a few hundred KB; the
    # event files and checkpoints behind it are ~300 MB per arm and there is no
    # disk to keep eleven classes of them.
    python scripts/ds_export_scalars.py --only "^sw_${tag}_" \
        --steps-per-epoch "$spe" || true
    for arm in magx hyb log; do
      src="results/diffsynth/sw_${tag}_${arm}/scalars.csv"
      [[ -f "$src" ]] && cp "$src" "$OUT/${cls}__${arm}.csv"
    done
    rm -rf results/diffsynth/sw_${tag}_*
    rm -rf "$dir" "$IDTMP"
    echo "=== $cls done, dataset and run dirs removed   $(date '+%H:%M:%S')"
    df -h . | tail -1
  done
done

echo
echo "sweep complete."
python scripts/ds_sweep_report.py --dir "$OUT" || true
