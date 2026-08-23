"""Watch a running sweep as trajectories, not as a single latest row.

A last-row view cannot tell "has not started yet" from "started and stopped",
which on this problem is the whole question -- the log arms sit near the
constant-predictor floor from step one, so the informative thing is whether
anything is moving, not where it currently is. Each eval is a row here and each
arm a column, so a stalled arm shows as a flat column and a cliff between two
rungs shows as a step across the table.

Reads history.json, which train_encoder rewrites every --log-every steps, so it
works mid-run. Arms are whatever directories exist under the root.

    python -m src.ddsp.monitor_sweep
    python -m src.ddsp.monitor_sweep --root results/ddsp/lr_probe
    python -m src.ddsp.monitor_sweep --metrics ratio spread train_sat

    # the gamma ladder, all three directories at once: the shared base, the
    # arms that resume it, and the one that does not. The base's rows stop where
    # the resumed arms' begin, so its column ending at 5000 and theirs starting
    # at 6000 is the handover boundary drawn by the table itself.
    python -m src.ddsp.monitor_sweep \
        --root results/ddsp/gamma_pre results/ddsp/gamma_ppre results/ddsp/gamma_raw \
        --metrics spec_w param g03 nmse ratio
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

METRICS = {
    "train": ("train_loss", "{:.4g}"),
    "val": ("val_loss", "{:.4g}"),
    "nmse": ("val_nmse_6d", "{:.4f}"),
    "nmse_geo": ("val_nmse_6d_geo", "{:.4f}"),
    # Train loss is not comparable across arms -- each loss has its own scale --
    # so these two put every arm on one axis when that is what is wanted.
    "train_sat": ("train_loss / saturation; ~1 = gradient uninformative", "{:.3f}"),
    "ratio": ("val_nmse_6d / const_nmse_6d; ~1 = the constant predictor", "{:.2f}"),
    "spread": ("mean spread_* over the raw seven", "{:.3f}"),
    # Head diagnostics. sat should sit near zero under stclamp by construction;
    # if it does not, the hinge is too weak. zmax is the aggregate over the
    # seven coordinates -- use --zmax for the per-parameter breakdown, which is
    # what says whether one coordinate is running while six behave.
    "sat": ("fraction of the batch with |z| > 2.5", "{:.3f}"),
    # leakytanh only. Nonzero says the floor is carrying that step rather than
    # tanh's own derivative -- inert on a healthy arm, so a column that stays
    # high is a coordinate the floor is holding up and worth saying so about.
    "floor": ("fraction of the batch riding the gradient floor", "{:.3f}"),
    "zmax": ("max |z| over all seven parameters", "{:.2f}"),
    # op_x alone: it was already the worst coordinate under tanh (40% of the
    # 6d NMSE, correlation stuck at 0.84) and it is the one that ran away under
    # stclamp, so the aggregate zmax hides exactly the thing being watched.
    "zmax_op_x": ("|z|max for op_x specifically", "{:.2f}"),
    "hinge": ("weighted hinge penalty, kept out of train_loss", "{:.4g}"),
    "h_abs": ("mean |h| entering the head", "{:.3f}"),
    # The handover. Without these a pretrained ladder is unreadable: an arm that
    # is flat at step 4000 is flat because spec_w is still 0, not because it has
    # stalled, and the two look identical in every other column. DiffMoog's
    # narrow_* runs collapsed AT the switch -- 10x their best and then flat for
    # 4000 steps -- while narrow_ploss, which never switches, did not. That is
    # the phenomenon this ladder exists to reproduce, and it is invisible unless
    # the switch is on the table beside the metric.
    # One fixed audio metric for every arm, at Stevens' exponent on intensity --
    # unlike val, which is each arm's own loss and cannot be read across arms.
    # Watch it against nmse through the crossfade: g03 falling while nmse rises
    # is audio similarity bought with parameter accuracy.
    "g03": ("mel cepstral L1 at gamma 0.3 on intensity, same for every arm", "{:.4f}"),
    "spec_w": ("spectral weight, 0 during the hold and 1 after the crossfade", "{:.2f}"),
    "param": ("parameter L1 on z, the objective during the hold", "{:.4f}"),
}


def _rate(h) -> float:
    """Steps per second over the last ~10 logged rows, resume-safe."""
    if len(h) < 2:
        return h[-1]["step"] / max(h[-1].get("elapsed_s", 0.0), 1e-9) if h else 0.0
    k = max(0, len(h) - 11)
    ds = h[-1]["step"] - h[k]["step"]
    dt = h[-1].get("elapsed_s", 0.0) - h[k].get("elapsed_s", 0.0)
    return ds / dt if dt > 0 else 0.0


def load(root, prefix: str = ""):
    """One root, or several. Arms are prefixed when there is more than one.

    This ladder spans two directories on purpose -- gamma_ppre resumes the shared
    parameter-only base, gamma_raw does not -- and the pair IS the experiment:
    whether the handover is what rescues an I^1.0 loss. Both hold an arm called
    L1_STFT_g1, so without the prefix one silently overwrites the other and the
    comparison the campaign was built for disappears from its own monitor.
    """
    if not isinstance(root, str):
        roots = list(root)
        multi = len(roots) > 1
        out = {}
        for r in roots:
            out.update(load(r, f"{Path(r).name}/" if multi else ""))
        return out
    # eps_ladder.sh records the invocation in sweep_command.txt. Reading --steps
    # from it makes the eta column right per root, which matters here because
    # the shared base runs 5000 and the arms resuming from it run 40000: one
    # --steps on the command line is necessarily wrong for one of them.
    target = 0
    try:
        m = re.search(r"steps=(\d+)", (Path(root) / "sweep_command.txt").read_text())
        target = int(m.group(1)) if m else 0
    except Exception:
        pass
    arms = {}
    for hp in sorted(glob.glob(os.path.join(root, "*", "history.json"))):
        name = prefix + Path(hp).parent.name
        try:
            d = json.load(open(hp))
        except Exception:
            continue
        rows = {}
        for r in d["history"]:
            if "val_nmse_6d" not in r:
                continue
            sp = [r[k] for k in r if k.startswith("spread_")]
            zc = {k[5:]: r[k] for k in r if k.startswith("zmax_")}
            rows.setdefault("_zmax", {})[r["step"]] = zc
            rows[r["step"]] = {
                "train": r["train_loss"],
                "val": r.get("val_loss", float("nan")),
                "nmse": r["val_nmse_6d"],
                "nmse_geo": r.get("val_nmse_6d_geo", float("nan")),
                "train_sat": r["train_loss"] / d["saturation"],
                "ratio": r["val_nmse_6d"] / d["const_nmse_6d"],
                "spread": sum(sp) / len(sp) if sp else float("nan"),
                "sat": r.get("sat_frac", float("nan")),
                "floor": r.get("floor_frac", float("nan")),
                "zmax": max(zc.values()) if zc else float("nan"),
                "zmax_op_x": zc.get("op_x", float("nan")),
                "hinge": r.get("hinge", float("nan")),
                "h_abs": r.get("h_absmean", float("nan")),
                "g03": r.get("val_g03", float("nan")),
                "spec_w": r.get("spec_w", float("nan")),
                "param": r.get("param_loss", float("nan")),
                # Every per-parameter error straight through, so a coordinate
                # can be watched by name without a new entry here for each
                # parameter of each space. --metrics perr_T60_DC just works.
                **{k: v for k, v in r.items() if k.startswith("perr_")},
            }
        h = d["history"]
        arms[name] = {
            "rows": rows,
            "last": h[-1]["step"] if h else 0,
            # Over a trailing window, NOT step/elapsed_s. t0 is set when the
            # process starts, so on a --resume run elapsed_s counts only the
            # steps since the resume while step stays absolute: an arm resumed
            # at 5000 and 1000 steps in reported 6x its true rate, and the eta
            # with it. That read as the ladder running three times faster than
            # the identical ladder had before, which is a wrong number that
            # looks like good news.
            "rate": _rate(h),
            # Lifetime clip% is dominated by warmup transients early on -- three
            # of six logged records reads as 50% and says nothing about whether
            # it is settling. Report the recent window too; that is the one that
            # answers "is the clip setting the step size".
            "clip": 100.0 * sum(bool(q.get("clipped")) for q in h) / max(len(h), 1),
            "clip_recent": 100.0 * sum(bool(q.get("clipped")) for q in h[-20:])
                           / max(len(h[-20:]), 1),
            "target": target,
            "gt": d.get("gt_loss", float("nan")),
            "sat": d.get("saturation", float("nan")),
            "floor": d.get("const_nmse_6d", float("nan")),
        }
    return arms


def short(name: str) -> str:
    pre, _, arm = name.rpartition("/")
    # gamma_ppre -> ppre, gamma_raw -> raw: enough to tell the pair apart in a
    # column header without spending the width on the shared stem.
    pre = (pre.rstrip("/").split("_")[-1] + ":") if pre else ""
    if arm == "L1_STFT":
        return pre + "linear"
    return pre + arm.replace("L1_STFT_", "").replace("L1_STFT", "lin")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # action="extend" so that BOTH "--root a b c" and "--root a --root b --root c"
    # work. With plain nargs="+" the repeated form silently keeps only the last
    # flag, which reads as "watch these three" and means "watch the third" -- and
    # if that one is a stage that has not started yet, the whole monitor reports
    # nothing while the run it was asked about is training fine. default=None
    # rather than a list because extend appends to the default rather than
    # replacing it.
    p.add_argument("--root", nargs="+", action="extend", default=None,
                   help="One or more sweep directories, as repeated flags or one "
                        "flag with several paths. With several, arm names are "
                        "prefixed by directory so the same arm run under two "
                        "conditions stays two columns. Directories that do not "
                        "exist yet are skipped, since a queued stage has not "
                        "created its output directory.")
    # No choices=: perr_<param> is per parameter of whichever space is loaded,
    # so the valid set is not knowable here. Unknown names are rejected below
    # against what the histories actually contain, which is the real check.
    p.add_argument("--metrics", nargs="+", default=["train", "val", "nmse"],
                   help="Any of " + ", ".join(sorted(METRICS))
                        + ", or a perr_<param> key logged by train_encoder.")
    p.add_argument("--steps", type=int, default=40000, help="For the ETA column")
    p.add_argument("--zmax", action="store_true",
                   help="Per-parameter |z|max at the latest eval, per arm")
    p.add_argument("--tail", type=int, default=0,
                   help="Show only the last N eval rows (0 = all)")
    args = p.parse_args()

    roots = args.root or ["results/ddsp/eps_ladder"]
    # A root that does not exist yet is normal while a queue is partway through
    # its stages, so it is named and skipped rather than counted as "no runs".
    missing = [r for r in roots if not Path(r).is_dir()]
    present = [r for r in roots if Path(r).is_dir()]
    if missing:
        print(f"not started yet, skipping: {' '.join(missing)}")
    # Prefixing keyed to what was ASKED for, not to what exists: otherwise arm
    # names change the moment a queued stage creates its directory, and a column
    # you were watching as "L1_STFT" becomes "quiet3_ppre/L1_STFT" mid-run.
    multi = len(roots) > 1
    arms = {}
    for r in present:
        arms.update(load(r, f"{Path(r).name}/" if multi else ""))
    if not arms:
        print(f"no runs under {' '.join(present) or ' '.join(roots)}")
        return

    names = sorted(arms)
    w = max(9, max(len(short(n)) for n in names) + 2)

    print(f"{'arm':<18}{'step':>8}{'st/s':>7}{'clip%':>7}{'clip20':>8}{'eta_h':>7}"
          f"{'gt_loss':>11}{'sat':>11}{'floor':>9}")
    for n in names:
        a = arms[n]
        tgt = a.get("target") or args.steps
        eta = (tgt - a["last"]) / a["rate"] / 3600 if a["rate"] > 0 else float("nan")
        print(f"{short(n):<18}{a['last']:>8}{a['rate']:>7.2f}{a['clip']:>7.1f}"
              f"{a['clip_recent']:>8.1f}{eta:>7.1f}"
              f"{a['gt']:>11.3e}{a['sat']:>11.3e}{a['floor']:>9.4f}")

    if args.zmax:
        keys, any_z = None, False
        for n in names:
            zz = arms[n]["rows"].get("_zmax", {})
            if not zz:
                continue
            last = zz[max(zz)]
            if not last:
                continue
            if keys is None:
                keys = sorted(last)
                print("\n=== per-parameter |z|max at the latest eval")
                print(f"{'arm':<18}" + "".join(f"{k:>10}" for k in keys))
            any_z = True
            print(f"{short(n):<18}" + "".join(f"{last[k]:>10.2f}" for k in keys))
        if not any_z:
            print("\n(no zmax_* in history -- runs predate per-parameter logging)")

    all_steps = sorted({s for a in arms.values() for s in a["rows"]
                        if isinstance(s, int)})
    if args.tail:
        all_steps = all_steps[-args.tail:]

    for m in args.metrics:
        if m in METRICS:
            desc, fmt = METRICS[m]
        elif m.startswith("perr_"):
            desc, fmt = (f"median normalized squared error in {m[5:]}; "
                         f"sqrt(x)*100 = percent of that parameter's range"), "{:.5f}"
        else:
            raise SystemExit(f"unknown metric {m!r}; known: "
                             + ", ".join(sorted(METRICS)) + ", or perr_<param>")
        print(f"\n=== {m}   ({desc})")
        print(f"{'step':>8}" + "".join(f"{short(n):>{w}}" for n in names))
        for s in all_steps:
            cells = []
            for n in names:
                r = arms[n]["rows"].get(s)
                r = r if isinstance(r, dict) and "train" in r else None
                cells.append(f"{fmt.format(r[m]):>{w}}" if r else f"{'-':>{w}}")
            print(f"{s:>8}" + "".join(cells))


if __name__ == "__main__":
    main()
