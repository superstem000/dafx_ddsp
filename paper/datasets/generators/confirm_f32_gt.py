#!/usr/bin/env python3
"""confirm_sim_gap.py — is high log gt_loss the numpy-vs-torch sim gap?
Renders ONE IR from the same params via numpy ModalPlate and via torch
BatchedModalPlateTorch, both float32, and compares. Run from repo root."""
import numpy as np, torch, glob, pandas as pd
from pathlib import Path

# --- load one param set from the f32 dataset ---
params_csv = sorted(glob.glob("random-IR-50-0.2s/random_IR_params_*.csv"))[0]
pdict = pd.read_csv(params_csv).iloc[0].to_dict()
DUR, SR = 0.25, 44100

# --- numpy target (organizers' sim), float32 ---
from ModalPlate.ModalPlate import ModalPlate
plate = ModalPlate(sample_rate=SR, plate_params=pdict)
ir_np = plate.synthesize_ir_method(duration=DUR, normalize=False).astype(np.float32)

# --- torch candidate (our sim), float32, SAME params ---
from src.plate.SevenParamPlate import BatchedModalPlateTorch
synth = BatchedModalPlateTorch(sample_rate=SR, device="cpu", dtype=torch.float32)
batch = BatchedModalPlateTorch.params_dicts_to_tensor([pdict], dtype=torch.float32)
with torch.no_grad():
    ir_th = synth(batch, DUR, normalize=False)[0].cpu().numpy().astype(np.float32)

# --- align length ---
n = min(len(ir_np), len(ir_th)); ir_np, ir_th = ir_np[:n], ir_th[:n]

ir_np = ir_np / (np.max(np.abs(ir_np)) + 1e-12)
ir_th = ir_th / (np.max(np.abs(ir_th)) + 1e-12)

# --- also render numpy vs numpy (same sim, sanity) and torch vs torch ---
def stft_mag(x):
    w = torch.hann_window(4096)
    S = torch.stft(torch.tensor(x, dtype=torch.float32), 4096, 1024,
                   window=w, return_complex=True)
    return S.abs()

def losses(a, b, tag):
    A, B = stft_mag(a), stft_mag(b)
    lin = (A - B).abs().mean().item()
    log = (torch.log(A + 1e-7) - torch.log(B + 1e-7)).abs().mean().item()
    print(f"  {tag:24s}  linear={lin:.3e}   log={log:.4f}")

print(f"params: {Path(params_csv).name}   len={n}")
print("GT-loss (target vs candidate at TRUE params):")
losses(ir_np, ir_th, "numpy-tgt vs torch-cand")   # the real gt_loss
losses(ir_np, ir_np, "numpy vs numpy (same sim)") # must be ~0
losses(ir_th, ir_th, "torch vs torch (same sim)") # must be ~0

# --- where does the disagreement live? bin the log error by target energy ---
A, B = stft_mag(ir_np), stft_mag(ir_th)
e = A.flatten(); logerr = (torch.log(A+1e-7) - torch.log(B+1e-7)).abs().flatten()
order = torch.argsort(e)
groups = np.array_split(order.numpy(), 10)
print("\nlog-error share by target-energy decile (1=quietest):")
tot = logerr.sum().item()
for i, g in enumerate(groups, 1):
    share = logerr[g].sum().item() / tot
    print(f"  decile {i:2d}  tgt_mag~{e[g].mean():.2e}   log-err share {share*100:4.1f}%")