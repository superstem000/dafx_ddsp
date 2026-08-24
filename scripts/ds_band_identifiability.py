"""Per-band identifiability on diffsynth. See src/analysis/band_identifiability.

    python scripts/ds_band_identifiability.py \
        --conf external/diffsynth/configs/synth/h2of.yaml
    python scripts/ds_band_identifiability.py --conf ... --floor-db 100

The plate's counterpart is src/ddsp/diag_band_identifiability, and the analysis
underneath both is the same module, so the two tables are directly comparable.
That is the entire point of running it here: the plate has 65-71% of its bins
below -120 dB and every compressed arm collapsed to a constant, while diffsynth
has 19.4% and its compressed arms are competitive. Those two facts want a
mechanism connecting them, and the mechanism is whether the bins compression
up-weights vote correctly or at random.

Parameters are already normalized to [0,1] -- fill_params takes [B, frames, P]
in that range and it IS the search space -- so a radius here is a fraction of
range with no per-parameter bounds table and no convention to pick.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "diffsynth"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omegaconf import OmegaConf                                    # noqa: E402
from diffsynth.modelutils import construct_synth_from_conf          # noqa: E402

from src.analysis.band_sensitivity import EPS, stft_mag             # noqa: E402
from src.analysis import band_identifiability as bi                 # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--conf", required=True)
    ap.add_argument("--n", type=int, default=24, help="Targets")
    ap.add_argument("--k", type=int, default=32, help="Candidates per target")
    ap.add_argument("--max-rel", type=float, default=0.30,
                    help="Radii uniform in (0, this] as a fraction of range, "
                         "along random directions. A spread, not a fixed "
                         "radius -- concordance needs candidates at different "
                         "true distances to have anything to rank.")
    ap.add_argument("--pin", nargs="+", default=None, metavar="NAME=V",
                    help="Hold a parameter at V in both target and candidates, "
                         "so it contributes no distance and no difference. The "
                         "operating point is a design variable here as it is in "
                         "ds_param_sensitivity.")
    ap.add_argument("--vary", nargs="+", default=None, metavar="NAME",
                    help="Search ONLY these columns; every other column is held "
                         "at --rest for both target and candidates. This defines "
                         "a TASK, the way PLATE_PARAM_SPACE does on the plate, "
                         "and the task is what the decomposition is a property "
                         "of -- a synthesizer does not have one answer, a chosen "
                         "set of searched parameters does. Combine with --pin to "
                         "put the held columns somewhere other than --rest.")
    ap.add_argument("--rest", type=float, default=0.5, metavar="V",
                    help="Where --vary holds the unsearched columns.")
    ap.add_argument("--list", action="store_true",
                    help="Print the parameter columns and exit.")
    ap.add_argument("--cond", nargs="+", default=None, metavar="NAME=V",
                    help="Value for a fixed parameter the config leaves to be "
                         "supplied at run time -- f0_hz in the _f0 chains, and "
                         "anything else whose attribute is None. These are NOT "
                         "in [0,1]: they are physical (f0_hz in Hz), which is "
                         "why fill_params skips scaling them. Unset ones get a "
                         "default and the run says which. They are held equal "
                         "across target and candidates, so they contribute no "
                         "distance and no difference -- conditioning, not a "
                         "searched parameter.")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--audio-len", type=float, default=4.0)
    ap.add_argument("--n-fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--floor-db", type=float, default=None,
                    help="Set the log measure's floor this far below each "
                         "target's peak instead of at the absolute eps 1e-7.")
    ap.add_argument("--render-batch", type=int, default=8, metavar="K",
                    help="Render this many candidates at a time. Candidates are "
                         "a batch dimension, so all K at once is a K/8 larger "
                         "transient for identical output. Lower it to share a "
                         "card with a training job.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    conf = OmegaConf.merge(OmegaConf.create({"data": {"sample_rate": args.sr}}),
                           OmegaConf.load(args.conf))
    synth = construct_synth_from_conf(conf).to(dev)
    names = list(synth.ext_param_sizes.keys())
    sizes = [synth.ext_param_sizes[k] for k in names]
    label = [n if s == 1 else f"{n}[{j}]"
             for n, s in zip(names, sizes) for j in range(s)]
    P = len(label)
    n_samples = int(args.audio_len * args.sr)

    if args.list:
        print(f"{Path(args.conf).name}   {P} columns:")
        for l in label:
            print(f"  {l}")
        return

    # --vary first so an explicit --pin can override where a held column sits.
    pins = {}
    if args.vary:
        bad = set(args.vary) - set(label)
        if bad:
            raise SystemExit(f"unknown: {', '.join(sorted(bad))}; "
                             f"have: {', '.join(label)}")
        pins = {i: args.rest for i, l in enumerate(label) if l not in args.vary}
    for item in args.pin or []:
        k, v = item.split("=")
        if k not in label:
            raise SystemExit(f"unknown parameter {k!r}; have: {', '.join(label)}")
        pins[label.index(k)] = float(v)

    searched = [l for i, l in enumerate(label) if i not in pins]
    if not searched:
        raise SystemExit("every column is held; nothing is being searched")
    print(f"{Path(args.conf).name}   {P} columns, {len(searched)} searched   "
          f"{args.n} targets   {args.k} candidates each   "
          f"radii (0, {args.max_rel:g}] of range")
    print(f"searching: {', '.join(searched)}")
    if pins:
        print(f"held: {', '.join(f'{label[i]}={v:g}' for i, v in sorted(pins.items()))}")

    # Some chains leave a fixed parameter to be supplied at run time -- the _f0
    # configs take f0_hz from the dataset -- and fill_params reads it out of
    # `conditioning`, which defaults to None and then raises TypeError on
    # subscript. Supply a constant for each so the chain can be measured at all.
    # Constant is the right choice here and not a shortcut: conditioning is not
    # a searched parameter, so holding it equal across target and candidates is
    # exactly what it should contribute -- nothing.
    _DEFAULT_COND = {"f0_hz": 220.0}
    need = [n for n in synth.fixed_param_names if getattr(synth, n) is None]
    cond_v = {n: _DEFAULT_COND.get(n, 1.0) for n in need}
    for item in args.cond or []:
        k, v = item.split("=")
        cond_v[k] = float(v)
    unknown = [n for n in need if n not in _DEFAULT_COND
               and n not in {i.split("=")[0] for i in args.cond or []}]
    if need:
        print(f"conditioning: " + ", ".join(f"{k}={v:g}" for k, v in cond_v.items())
              + (f"   ({', '.join(unknown)} had no default -- set with --cond "
                 f"if 1.0 is wrong for it)" if unknown else ""))

    def render(p: torch.Tensor) -> torch.Tensor:
        out = []
        for i in range(0, p.shape[0], args.render_batch):
            chunk = p[i:i + args.render_batch, None, :].to(dev)
            cond = {k: torch.full((chunk.shape[0], 1, 1), v, device=dev)
                    for k, v in cond_v.items()}
            with torch.no_grad():
                audio, _ = synth(synth.fill_params(chunk, cond), n_samples)
            out.append(audio)
        return torch.cat(out, dim=0)

    g = torch.Generator().manual_seed(args.seed)
    rows, marg, dropped = [], [], 0
    for _ in range(args.n):
        tgt = torch.rand(P, generator=g)
        for i, v in pins.items():
            tgt[i] = v

        d = torch.randn((args.k, P), generator=g)
        d /= d.norm(dim=1, keepdim=True).clamp(min=1e-30)
        r = torch.rand((args.k, 1), generator=g) * args.max_rel
        cand = (tgt[None, :] + d * r).clamp(0.0, 1.0)
        for i, v in pins.items():
            cand[:, i] = v
        # After the clamp and after the pins, so a candidate that hit a bound
        # or whose only movement was in a pinned column is labelled with the
        # distance it actually has rather than the one it was drawn at.
        dist = (cand - tgt[None, :]).norm(dim=1)

        x_ref = render(tgt[None, :])[0]
        x_can = render(cand)
        ok = torch.isfinite(x_can).all(dim=-1)
        if not bool(ok.all()):
            dropped += int((~ok).sum())
            x_can, dist = x_can[ok], dist[ok.cpu()]
        if x_can.shape[0] < 4 or not torch.isfinite(x_ref).all():
            continue

        A_ref = stft_mag(x_ref[None, :], args.n_fft, args.hop, True)[0]
        A_can = stft_mag(x_can, args.n_fft, args.hop, True)
        eps = (EPS if args.floor_db is None
               else float(A_ref.max()) * 10.0 ** (-args.floor_db / 20.0))
        dt = dist.to(A_ref.device)
        rows.append(bi.probe(A_ref, A_can, dt, eps))
        marg.append(bi.marginal(A_ref, A_can, dt, eps))

    if not rows:
        raise SystemExit("no usable targets")
    if dropped:
        print(f"  {dropped} non-finite candidate renders dropped")
    if args.floor_db is not None:
        print(f"  log floor: {args.floor_db:g} dB below each target's peak")
    bi.report(bi.accumulate(rows),
              title=f"{Path(args.conf).stem}   {len(rows)} targets")
    bi.report_marginal(marg)


if __name__ == "__main__":
    main()
