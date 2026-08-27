"""Per-IR scatter of the parameters emt8 pins. Reads probe_best.csv."""
import csv, statistics as st, sys
path = sys.argv[1] if len(sys.argv) > 1 else "emt7_probe_22000/probe_best.csv"
rows = list(csv.DictReader(open(path)))
keys = ["fp_x", "fp_y", "op_x", "op_y"]
BOX = {"fp_x": (0.02, 0.5), "fp_y": (0.02, 0.5), "op_x": (0.05, 0.95), "op_y": (0.05, 0.95)}
PIN = {"fp_x": 0.22, "fp_y": 0.18, "op_x": 0.48, "op_y": 0.59}
print(f"{path}: {len(rows)} IRs\n")
print(f"  {'ir':<20}" + "".join(f"{k:>9}" for k in keys) + f"{'draw':>7}")
for r in rows:
    print(f"  {r['ir']:<20}" + "".join(f"{float(r[k]):>9.3f}" for k in keys)
          + f"{r['draw']:>7}")
print()
print(f"  {'':<20}" + "".join(f"{k:>9}" for k in keys))
for lbl, fn in (("median", st.median), ("min", min), ("max", max)):
    print(f"  {lbl:<20}" + "".join(f"{fn([float(r[k]) for r in rows]):>9.3f}" for k in keys))
print(f"  {'emt8 pin':<20}" + "".join(f"{PIN[k]:>9.3f}" for k in keys))
print(f"  {'spread / box':<20}" + "".join(
    f"{(max(float(r[k]) for r in rows) - min(float(r[k]) for r in rows))/(BOX[k][1]-BOX[k][0]):>8.0%}"
    for k in keys))
print("\n  spread/box near 0%: the fifteen agree, the pin is a real measurement.")
print("  near 100%: they scatter across the box and the pin is a median of noise.")
print(f"  distinct winning draws: {len({r['draw'] for r in rows})} of {len(rows)}")
