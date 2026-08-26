# emt7 — a plate space that can sound like an EMT-140

Everything specific to this experiment lives here. The renderer's physics
(`src/plate/SevenParamPlate.py`) is untouched upstream code and is *not* copied:
what deviated from the original plate work was never the modal sum, it was the
values fed to it.

```
src/emt/space.py   the parameter space, the ceiling, the mode grid, the duration
src/emt/gen.sh     dataset generation, reading all four from space.py
scripts/jobs_emt7.txt   the three arms, reading the same four again
```

## Why

Rendering a real EMT-140 IR through the `raw7` encoders and listening to it,
three things were wrong, and none of them was the encoder or the loss:

| symptom | cause | where it lived |
|---|---|---|
| no initial crash | drive point `fp_x/fp_y` pinned, and `--fixed-mode-grid 60,185` truncating the high modes on every draw | `raw7` fixed set; a CLI flag |
| far darker | `fmax=10000` hardcoded in the renderer's constructor default and never passed, so everything above 10 kHz is dropped before any parameter has a say. `loss_F1=500` then puts the damping corner two octaves below a plate's | `BatchedModalPlateTorch.__init__`; `raw7` fixed set |
| far longer | `T60_DC=6.0`, `T60_F1=2.0` pinned — every render decays over six seconds whatever the target does | `raw7` fixed set |

## What changed

**Damping enters the search.** `T60_DC` 1.5–6.0 s and `loss_F1` 2000–8000 Hz
were both fixed. `T60_F1` stays pinned at 1.2 s — it is near-degenerate with
`loss_F1`, both shaping the same high-frequency tilt.

**The ceiling is raised**, 10 kHz → 16 kHz, via a new `--fmax` on
`train_encoder` and `make_dataset` and a new argument to
`Raw7Space.configure_plate`. `None` keeps the 10 kHz default, so nothing that
ran before renders differently. Not 20 kHz: memory is flat in `fmax` — the
modal sum is bounded by `chunk_elems` — but time is linear in mode count, and
20 kHz costs another 25% for content at the edge of hearing.

**The mode grid is computed, not inherited.** `137,342` is this box's most
expensive corner (smallest `h`, largest `Ly`, smallest `E`, largest `rho`,
smallest `T0`), so nothing is truncated. `raw7`'s `60,185` would truncate
*every* draw here — even the thickest, smallest, stiffest one needs `DDx=108`.

**Geometry is a real plate.** An EMT-140 is 2.0 × 1.0 m of 0.5 mm steel. `Lx`
is pinned at 1.0, `Ly` searched over 1.5–2.5 so 2.0 is mid-range, `h` over
0.4–0.8 mm. `raw7`'s `h` floor of 1 mm is twice the real thickness, which is
why nothing it renders can clang.

**Duration 1.0 s**, from 0.25. At `T60_F1 = 1.2 s` a quarter second captures
12 dB of the high-frequency decay; one second captures 50, and 15 dB at DC.

**The arms are eps-matched.** `L1_STFT` / `L1_STFT_hyb1e2` / `L1_STFT_eps1e2`.
`gamma_ppre` compared `hyb1e2` against `eps1e7` — five decades of eps apart
*on top of* the linear term, so the pair was never single-variable. `eps1e2`
was added to `_EPS_LADDER` for this.

## The cost, stated

Narrowing `rho` to 7000–8500 and `E` to 1.7e11–2.2e11 is steel rather than
`raw7`'s aluminium-to-gold box. That removes degrees of freedom — `mu = rho*h`
and `D/mu` now vary mostly through `h` — so **this estimation task is easier
than `raw7`'s** and a loss comparison on it is correspondingly weaker evidence.
It is also what makes the campaign affordable: wide `rho`/`E` needs 118,048
modes at the corner against 46,854 here, 2.5× the render cost per IR. "Realistic
plates" is a different condition from "wide box", not a better version of it.

```
                                          corner modes    per-IR cost
raw7 today      h .001-.005, fmax 10k          47,742          1.0x
emt7            h .0004-.0008, fmax 16k        46,854         16.9x   (4x the samples)
emt7, wide rho/E                              118,048         42.5x
```

Whole campaign at n=24576 is ~4.2× the current generation cost. 24576 × 1.0 s
is a 4.3 GB training tensor — the same as `raw7`'s 98304 × 0.25 s, so nothing
about memory changes; 49152 would be 8.7 GB and does not fit beside the
diffsynth class sweep.

## Not comparable to anything earlier

Different space, duration, ceiling and mode grid. `emt7` is internally
comparable — three arms, one base, one dataset — and claims nothing else.

## Still unreachable

`fp_x`/`fp_y` stay pinned, so onset brightness is identical across the whole
dataset. If the strike is still wrong, that is why, and the fix is to swap
`T0` out for `fp_x`: the repo's own sensitivity table ranks `T0` last of the
searched parameters (dnorm 5.7/10.3/17.3 against `T60_DC`'s 14.8/34.1/46.7).
And a modal sum has no noise source, so a real plate's hiss, room and driver
transient are outside the model at any parameter setting.
