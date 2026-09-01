"""Every parameter a synth config exposes: what is drawn, what is pinned, what is saved.

    python scripts/ds_param_list.py external/diffsynth/configs/synth/dataset/h2of.yaml
    python scripts/ds_param_list.py external/diffsynth/configs/synth/dataset/h2of_var.yaml \
                                    external/diffsynth/configs/synth/h2of.yaml

WHY A TOOL AND NOT A LIST IN A DOCSTRING. The answer is spread over four
places and every one of them can move independently: each module's param_desc
gives the size, range and scale function; the config's `connections` decide
which of those are external at all, since anything wired to another processor
is computed rather than drawn; `fixed_params` removes more; and `static_params`
decides whether a parameter is one value per clip or a curve. A list written by
hand goes stale the first time any of those changes, silently.

Reads them the same way Synthesizer.__init__ does -- ext_params is
`connections` minus processor outputs minus fixed_params, and ext_param_sizes
is built with dict.update() walking the dag in order, so a key shared by two
processors takes the size of the LAST one to claim it. That ordering
dependence is real and worth seeing rather than inferring: h2of_var wires
START and LENGTH to both envelopes, and envc's size 1 overwrites enva's 2,
which is what makes one window apply to both oscillators.

SCALE tells you what the drawn 0..1 actually becomes. 'sigmoid' is linear
interpolation across the range; 'freq_sigmoid' is perceptual, so a uniform
draw is roughly log-uniform in Hz; 'exp_sigmoid' is exponential. A range alone
does not tell you the distribution.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "external", "diffsynth"))

import hydra                                             # noqa: E402
from omegaconf import OmegaConf                          # noqa: E402


def describe(path: str) -> None:
    conf = OmegaConf.load(path)
    fixed = OmegaConf.to_container(conf.get("fixed_params") or {}, resolve=True)
    static = set(conf.get("static_params") or [])
    saved = list(conf.get("save_params") or [])

    nodes = []
    for name, v in conf.dag.items():
        cfg = OmegaConf.to_container(v.config, resolve=True)
        cfg.pop("_target_", None)
        # sample_rate may interpolate ${data.sample_rate}, absent standalone.
        cfg.setdefault("sample_rate", 16000)
        try:
            mod = hydra.utils.instantiate(v.config, name=name, _convert_="all")
        except Exception:
            mod = hydra.utils.instantiate(
                OmegaConf.create({**cfg, "_target_": v.config._target_}),
                name=name)
        nodes.append((name, mod, OmegaConf.to_container(v.connections)))

    proc_names = [n for n, _m, _c in nodes]
    print(f"\n=== {path}")
    print(f"  name: {conf.get('name')}   processors: {', '.join(proc_names)}")

    sizes: dict[str, int] = {}
    rows = []
    for name, mod, conn in nodes:
        for inp, key in conn.items():
            if inp not in mod.param_desc:
                continue
            d = mod.param_desc[inp]
            if key in proc_names:
                rows.append((key, f"{name}.{inp}", "-", "<- " + key, "computed"))
                continue
            if key in fixed:
                v = fixed[key]
                rows.append((key, f"{name}.{inp}", "-",
                             "pinned" if v is not None else "CONDITIONING",
                             str(v) if v is not None else "supplied at runtime"))
                continue
            sizes[key] = d["size"]        # update(), last writer wins
            rows.append((key, f"{name}.{inp}", d["size"],
                         f"{d['range'][0]:g}..{d['range'][1]:g}", d["type"]))

    print(f"\n  {'KEY':<12}{'feeds':<18}{'size':>5}  {'range':<18}scale")
    seen = set()
    for key, feeds, size, rng, scale in rows:
        n = sizes.get(key, size)
        mark = ""
        if key in sizes and key in seen:
            mark = "  (shares KEY; size from the last writer)"
        seen.add(key)
        kind = "static" if key in static else "per-frame"
        extra = f"  [{kind}]" if key in sizes else ""
        print(f"  {key:<12}{feeds:<18}{str(n):>5}  {rng:<18}{scale}{extra}{mark}")

    print(f"\n  ext_param_size = {sum(sizes.values())}"
          f"   ({len(sizes)} drawn key(s), "
          f"{sum(1 for k in sizes if k in static)} static)")
    if fixed:
        print(f"  fixed: " + ", ".join(
            f"{k}={v}" if v is not None else f"{k}=<conditioning>"
            for k, v in fixed.items()))
    if saved:
        print(f"  save_params ({len(saved)}): " + ", ".join(saved))
    else:
        print(f"  save_params: none -- this is a MODEL synth, not a generator")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("configs", nargs="+", metavar="YAML")
    args = p.parse_args()
    for c in args.configs:
        describe(c)
    print("\n  per-frame keys are predicted/drawn once per frame (250 for a 4 s\n"
          "  clip at hop 256); static keys are sliced to the LAST frame by\n"
          "  fill_params, so they are one value per clip.\n"
          "  'computed' means the input is wired to another processor's output\n"
          "  and is never drawn -- in the generator that is how the envelopes\n"
          "  reach harmor, and it is why the envelope's own controls are the\n"
          "  drawn ones instead.")


if __name__ == "__main__":
    main()
