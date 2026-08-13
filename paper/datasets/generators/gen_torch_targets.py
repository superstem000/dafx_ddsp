#!/usr/bin/env python3
"""gen_torch_targets.py — targets from the SAME torch synth the fitter uses as
candidate, so gt_loss ~ 0 (matched-model / inverse-crime diagnostic). Run from repo root."""
import numpy as np, torch, glob, pandas as pd
from pathlib import Path
import importlib.util as ilu

spec = ilu.spec_from_file_location("fit", str(Path.home()/"dafxchal/src/cmaes/fit_7param_norm_es.py"))
fit = ilu.module_from_spec(spec); spec.loader.exec_module(fit)
from src.plate.SevenParamPlate import BatchedModalPlateTorch

DTYPE = torch.float32
SR    = fit.SAMPLE_RATE
DUR   = 0.25
DEVICE = "cuda"

synth = BatchedModalPlateTorch(sample_rate=SR, device=DEVICE, dtype=DTYPE)
out_dir = Path("random-IR-torch-50-0.25s"); out_dir.mkdir(exist_ok=True)

param_csvs = sorted(glob.glob("random-IR-50-0.2s/random_IR_params_*.csv"))
assert param_csvs, "no param CSVs found — check the source folder name"

for csv in param_csvs:
    p = pd.read_csv(csv).iloc[0]
    phys = np.array([[float(p[k]) for k in fit.PARAM_KEYS]], dtype=np.float64)
    plate14 = fit.physical_to_plate14_tensor(phys, DEVICE).to(dtype=DTYPE)
    with torch.no_grad():
        ir = synth(plate14, DUR, normalize=False).squeeze(0).cpu().numpy()
    idx = csv.split("params_")[1].split(".csv")[0]
    np.savez(out_dir / f"random_IR_{idx}.npz",
             ir=ir.astype(np.float32),
             sample_rate=np.int32(SR), duration_s=np.float64(DUR),
             normalization_factor=np.float64(1.0))
    pd.read_csv(csv).to_csv(out_dir / f"random_IR_params_{idx}.csv", index=False)

print(f"wrote {len(param_csvs)} torch-rendered targets to {out_dir}")
