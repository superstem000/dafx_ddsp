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

**The ceiling is raised**, 10 kHz → 12 kHz, via a new `--fmax` on
`train_encoder` and `make_dataset` and a new argument to
`Raw7Space.configure_plate`. `None` keeps the 10 kHz default, so nothing that
ran before renders differently. Not 16 kHz — see **The budget line**, which is
what actually set this number.

**The mode grid is computed, not inherited.** `100,220` is this box's most
expensive corner (smallest `h`, largest `Ly`, smallest `E`, largest `rho`,
smallest `T0`), verified as the maximum over all 32 corners, so nothing is
truncated. The cheapest draw needs `75,112`. `raw7`'s `60,185` would truncate
*every* draw here.

**Geometry is close to a real plate.** An EMT-140 is 2.0 × 1.0 m of 0.5 mm
steel. `Lx` is pinned at 1.0, `Ly` searched over 1.5–2.2 so 2.0 sits near the
top, `h` over 0.56–0.8 mm — which does **not** contain the real 0.5. Both
ceilings were pulled in to buy render time. `raw7`'s `h` floor of 1 mm is twice
the real thickness, which is why nothing it renders can clang; 0.56 mm is 1.12×.

**10000 steps, not 40000.** Not a cut. 40000 came from `raw7`, which trained on
98,304 clips — `40000 × 64 / 98304 = 26` epochs. At emt7's 24,576 clips the same
40000 steps is **104** epochs, 4× the passes `raw7` ever made, inherited by
accident. 10000 restores 26 exactly. Largest single saving in the campaign and
it gives up nothing.

## The budget line

Every geometry number above was set by a 24 h/arm budget, not by the physics,
and it is worth being explicit about which knob is which. The renderer
materialises the whole `DDx × DDy` grid and masks afterwards, so cost is grid
cells × samples, and

```
cells  ~  Ly * fmax * sqrt(rho/E) / h
```

`Ly`, `fmax` and the `h` floor are therefore **one knob, not three**. At 1.0 s
and 10000 steps, 24 h buys `Ly*fmax/h <= 4.7e7`. Every point on that line costs
the same 23.8 h:

| fmax | h floor | grid | arm |
|---|---|---|---|
| 16,000 | 0.75 mm | 100,220 | 23.8 h |
| 13,000 | 0.61 mm | 100,220 | 23.8 h |
| **12,000** | **0.56 mm** | **100,220** | **23.8 h** |
| 11,000 | 0.50 mm | 100,221 | 23.9 h |

12000 / 0.56 mm was chosen because the two fix different symptoms. `fmax` sets
where the spectrum stops, and 10 kHz was the "far darker" — 12 kHz keeps 1.2× of
that fix. `h` sets how densely modes pack *below* the ceiling, and mode density
is what an initial crash is made of. A 16 kHz ceiling would have bought content
above 13 kHz, where a plate has least energy, by spending the very mode density
the transient needs. Duration is not on this line: it was held at 1.0 s.

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
It is also what makes the campaign affordable: wide `rho`/`E` needs 55,650
cells at the corner against 22,000 here, 2.53× the render cost per IR. "Realistic
plates" is a different condition from "wide box", not a better version of it.

```
                                          grid cells    per-IR cost
raw7 today      h .001-.005, fmax 10k          11,100          1.0x
emt7            h .00056-.0008, fmax 12k       22,000          7.9x   (4x the samples)
emt7, wide rho/E                               55,650         20.1x
```

24576 × 1.0 s is a 4.3 GB training tensor — the same as `raw7`'s 98304 × 0.25 s,
so nothing about memory changes; 49152 would be 8.7 GB and does not fit beside
the diffsynth class sweep.

## Numerics, and the one gate to run first

`--compile-plate` is 8.5× here (64 IRs at 1.0 s: 68 s eager, 8 s compiled),
matching the 7.3× `704c1ea` measured. It is not a free flag. `0f260fc` dropped
it for `quiet3` because compiled generation and compiled training disagreed by
**7.66% of saturation on the log arm**, 42.6% of it in the quietest decile;
`raw7` tolerated it at ≤0.074% because its quiet bins never sat at the float32
cancellation floor. emt7 should be on `raw7`'s side — `T60_DC` is searched, so
nothing here is that far down — but that is a claim, and `35e4529` is what
happens when this family of assumption goes unchecked. **Run
`src.ddsp.diag_gt_floor` at the settings training will use before spending 24 h.**

`chunk_elems` and `--batch-size` are plain memory knobs *only because* compile
fuses the chunk kernel; eager they are part of the numerics contract (`35e4529`:
1e9 vs 2e8 moved log 8.51%). `gen.sh` therefore pins `--batch-size 64` to match
`eps_ladder.sh` rather than leaving `make_dataset`'s default of 32.

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
