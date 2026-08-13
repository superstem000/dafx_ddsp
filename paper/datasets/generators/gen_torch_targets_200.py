#!/usr/bin/env python3
import numpy as np, torch, glob, pandas as pd
from pathlib import Path
import importlib.util as ilu
spec = ilu.spec_from_file_location("fit","src/cmaes/fit_7param_norm_es.py")
fit = ilu.module_from_spec(spec); spec.loader.exec_module(fit)
from src.plate.SevenParamPlate import BatchedModalPlateTorch

DTYPE, SR, DUR, DEVICE = torch.float32, fit.SAMPLE_RATE, 0.25, "cuda"
synth = BatchedModalPlateTorch(sample_rate=SR, device=DEVICE, dtype=DTYPE)
src_dir = "random-IR-200-0.2s"
out_dir = Path(src_dir)
csvs = sorted(glob.glob(f"{src_dir}/random_IR_params_*.csv"))
assert csvs, f"no param CSVs in {src_dir}"
for csv in csvs:
    p = pd.read_csv(csv).iloc[0]
    phys = np.array([[float(p[k]) for k in fit.PARAM_KEYS]], dtype=np.float64)
    p14 = fit.physical_to_plate14_tensor(phys, DEVICE).to(dtype=DTYPE)
    with torch.no_grad():
        ir = synth(p14, DUR, normalize=False).squeeze(0).cpu().numpy()
    idx = csv.split("params_")[1].split(".csv")[0]
    np.savez(out_dir/f"random_IR_{idx}.npz", ir=ir.astype(np.float32),
             sample_rate=np.int32(SR), duration_s=np.float64(DUR),
             normalization_factor=np.float64(1.0))
print(f"rendered {len(csvs)} torch targets into {src_dir}")
