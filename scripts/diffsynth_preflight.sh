#!/usr/bin/env bash
# Everything checkable without a GPU, run once before committing to the arms.
#
#   scripts/diffsynth_preflight.sh
#
# Four gates, cheapest first:
#
#   1. dataset sizes -- 20000 in-domain exactly, because 16000 train at batch 64
#      is what makes 250 steps/epoch and puts the ramp on epochs 50/200
#   2. NSynth audio shape -- WaveParamDataset asserts every file is exactly
#      length*sample_rate samples, so one short file crashes training hours in
#   3. the loss schedule, evaluated at the paper's own boundaries rather than
#      assumed from arithmetic
#   4. one epoch against the real id+ood pair, which is the first time the OOD
#      dataloader, the val_ood metrics and the monitored checkpoint have all run
#      together
#
# Gate 4 is the one that would otherwise fail late: ModelCheckpoint monitors
# val_ood/lsd, and that metric only exists if the OOD validation dataloader
# produced it.
set -euo pipefail

cd "$(dirname "$0")/../external/diffsynth"

ID=${ID:-"$(pwd)/data/diffsynth_5-6/harmor_2oscfree"}
OOD=${OOD:-"$(pwd)/data/nsynth-train"}
WORK=${WORK:-"$(pwd)/outputs/preflight"}

echo "=== 1/4  dataset sizes"
python3 - "$ID" "$OOD" <<'PY'
import glob, sys
idd, ood = sys.argv[1], sys.argv[2]
n_id = len(glob.glob(f"{idd}/audio/*.wav"))
n_pa = len(glob.glob(f"{idd}/param/*.pt"))
n_od = len(glob.glob(f"{ood}/audio/*.wav"))
print(f"  in-domain : {n_id} wav, {n_pa} param")
print(f"  out-domain: {n_od} wav")
ok = True
if n_id != 20000:
    print(f"  FAIL: in-domain is {n_id}, not 20000. 16000 train at batch 64 is"); ok = False
    print("        what puts the ramp on epochs 50/200; any other count shifts it.")
if n_pa != n_id:
    print("  FAIL: param and audio counts differ"); ok = False
if n_od < n_id:
    print(f"  FAIL: OOD pool {n_od} < in-domain {n_id};"
          " np.random.choice(..., replace=False) will raise"); ok = False
print("  OK" if ok else "  --> fix before continuing")
sys.exit(0 if ok else 1)
PY

echo
echo "=== 2/4  NSynth audio shape (WaveParamDataset asserts exactly 64000 samples)"
python3 - "$OOD" <<'PY'
import glob, random, sys
import soundfile as sf
ws = sorted(glob.glob(f"{sys.argv[1]}/audio/*.wav"))
random.seed(0)
bad, lens, srs = [], set(), set()
for w in random.sample(ws, min(400, len(ws))):
    info = sf.info(w)
    lens.add(info.frames); srs.add(info.samplerate)
    if info.frames != 64000 or info.samplerate != 16000:
        bad.append((w.split('/')[-1], info.frames, info.samplerate))
print(f"  sampled {min(400, len(ws))} files: lengths {sorted(lens)} sr {sorted(srs)}")
if bad:
    print(f"  FAIL: {len(bad)} off-spec, e.g. {bad[:3]}")
    print("        WaveParamDataset asserts audio.shape[0] == length*sample_rate,")
    print("        so these crash mid-epoch. Filter them or pad.")
    sys.exit(1)
print("  OK")
PY

echo
echo "=== 3/4  loss schedule at the paper's boundaries"
python3 - <<'PY'
from omegaconf import OmegaConf
from diffsynth.schedules import ParamSchedule

SPE = 250  # 16000 train / batch 64
def show(cfg_path, label, checks):
    sched = ParamSchedule(OmegaConf.load(cfg_path))
    print(f"  {label}")
    ok = True
    for epoch, want in checks.items():
        w = sched.get_parameters(epoch * SPE)
        got = {k: round(w[k], 4) for k in want}
        flag = "" if got == want else f"   <-- expected {want}"
        if got != want:
            ok = False
        print(f"    epoch {epoch:>3} (step {epoch*SPE:>6}): "
              f"param_w={w['param_w']:>5.2f}  sw_w={w['sw_w']:>4.2f}{flag}")
    return ok

# Paper sec 4.1: P-loss is parameter loss only for 400 epochs.
a = show("configs/schedule/param.yaml", "param.yaml  (P-loss)",
         {0: {"param_w": 10.0, "sw_w": 0.0}, 400: {"param_w": 10.0, "sw_w": 0.0}})
# "pre-trained using parameter loss for 50 epochs. For the next 150 epochs, a
#  spectral loss is gradually introduced ... Finally ... 200 epochs using only
#  the spectral loss."
b = show("configs/schedule/switch.yaml", "switch.yaml (Synth / Real)",
         {0:   {"param_w": 10.0, "sw_w": 0.0},
          50:  {"param_w": 10.0, "sw_w": 0.0},
          125: {"param_w": 5.0,  "sw_w": 0.5},
          200: {"param_w": 0.0,  "sw_w": 1.0},
          400: {"param_w": 0.0,  "sw_w": 1.0}})
print("  OK" if (a and b) else "  --> schedule does not match the paper")
raise SystemExit(0 if (a and b) else 1)
PY

echo
echo "=== 4/4  one epoch against the real id + ood pair"
rm -rf "$WORK"; mkdir -p "$WORK"
rc=0
python train.py \
  experiment=pretrain \
  data.id_dir="$ID" \
  data.ood_dir="$OOD" \
  data.batch_size=8 \
  data.num_workers=0 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=3 \
  trainer.limit_val_batches=2 \
  trainer.num_sanity_val_steps=0 \
  trainer.accelerator=cpu \
  trainer.devices=1 \
  hydra.run.dir="$WORK/run" \
  > "$WORK/run.log" 2>&1 || rc=$?
if (( rc != 0 )); then
  echo "  FAIL (exit $rc). Last 30 lines:"
  tail -30 "$WORK/run.log"
  exit $rc
fi

python3 - "$WORK" <<'PY'
import glob, json, os, sys
w = sys.argv[1]
ok = True
ck = glob.glob(f"{w}/run/**/checkpoints/*.ckpt", recursive=True)
print(f"  checkpoints written: {[os.path.basename(c) for c in ck]}")
if not any("last" in os.path.basename(c) for c in ck):
    print("  FAIL: no last.ckpt"); ok = False
# A monitored checkpoint named with the metric proves val_ood/lsd was produced;
# had the OOD dataloader failed, PL would warn and save nothing for it.
if not any("last" not in os.path.basename(c) for c in ck):
    print("  FAIL: no monitored checkpoint -- val_ood/lsd was never logged, so")
    print("        ModelCheckpoint(monitor='val_ood/lsd') had nothing to track.")
    ok = False
m = f"{w}/run/split_manifest.json"
if os.path.exists(m):
    d = json.load(open(m))
    print("  splits: " + "  ".join(f"{k}={v['n']}" for k, v in sorted(d.items())))
    if d.get("ood_train", {}).get("n") != d.get("id_train", {}).get("n"):
        print("  FAIL: id and ood train splits differ in size"); ok = False
else:
    print("  FAIL: no split manifest"); ok = False
log = open(f"{w}/run.log").read()
for key in ("val_ood/lsd", "val_id/lsd"):
    print(f"  {key} logged: {key in log}")
print("  OK" if ok else "  --> fix before continuing")
sys.exit(0 if ok else 1)
PY

echo
echo "=== preflight passed"
echo "Remaining decisions are yours, not the code's:"
echo "  - log_mag_w=0 doubles the surviving term (loss.py divides by"
echo "    len(fft_sizes)*(mag_w+log_mag_w)); run one control at sw_w end_v=0.5"
echo "  - report best-epoch AND final-epoch; the paper does not say which it used"
