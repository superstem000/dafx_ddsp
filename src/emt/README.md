# emt8 — a plate space set by measurement

Two spaces live here. **`emt8` is current**; `emt7` is kept for the record and
stays runnable via `SPACE=emt7`.

```
src/emt/space.py        both spaces, and the NUMERICS table gen.sh/check.py read
src/emt/gen.sh          dataset generation      SPACE=emt8 by default
src/emt/probe.py        the encoder-free measurement every emt8 bound came from
src/emt/why_dark.py     decomposes a render/target gap four ways
src/emt/band.py         brick-wall lowpass to the renderer's ceiling
src/emt/check.py        asserts space.py, gen.sh and the jobs file agree
scripts/jobs_emt8.txt   the three arms
```

## Why emt8 exists

emt7 trained, and its encoders produced near-constants on real audio — six of
seven parameters identical to five figures across fifteen audibly different
EMT-140 IRs. Every prediction was also trapped inside emt7's box. So none of
those numbers said anything about the model, the space, or the losses.

`probe.py` removes the encoder: 2048 random draws from a deliberately wide box,
rendered, scored against all fifteen targets. Every emt8 bound comes from it.

## What it found

**The `rho`/`E` degeneracy, which is the finding that matters most.** The modal
sum sees only `mu = rho*h`, `D/mu` and `T0/mu`. Four searched parameters mapping
to three observables leaves an exact one-parameter family — these render
bit-identically:

```
h .00060  rho 13697  E 1.043e12  ->  mu 8.2180  D/mu 2.51100  T0/mu 988.923
h .00107  rho  7702  E 1.855e11  ->  mu 8.2180  D/mu 2.51100  T0/mu 988.923
h .00350  rho  2348  E 5.256e09  ->  mu 8.2180  D/mu 2.51100  T0/mu 988.923
```

emt7 searched all three and claimed every parameter was individually
identifiable. It is the best explanation of emt7's railing: a tanh-bounded head
free to slide along a flat direction parks at a corner. emt8 fixes `rho` and `E`
and widens `h` to carry the `D/mu` range they gave up.

**The ceiling was the largest single lever and never flattened.** Best-of-2048
against saturation, full band, same seed: `12k 1.781 · 16k 1.665 · 20k 1.544 ·
22k 1.324`. emt7's 12 kHz was chosen on a "0.36% of energy above the ceiling"
argument, and energy share badly understates what those bands are worth. 22 kHz
covers 99.9% of the mel axis, so there is nothing left to chase.

**Four hypotheses tested and killed** — `fmax` doesn't matter (wrong); the bass
excess is the drive point (pinning `fp` to a corner moved 62 Hz by 0.1 dB); the
bass excess is the simply-supported boundary condition (reachable to 0.1 dB, and
jointly at ~20% cost); the collapse is the 46% dead input axis (band-limiting the
input *tightened* the collapse).

## The searched set

| | bounds | probe winner at 22 kHz [90% CI] |
|---|---|---|
| `h` | 0.0006 – 0.004 | 1.07 mm [0.64, 1.31] |
| `Ly` | 1.3 – 2.8 | 1.92 [1.66, 2.50] |
| `T0` | 30 – 5e4, log | 8127 [458, 2.76e4] |
| `T60_DC` | 0.8 – 6.0, log | 2.67 [2.39, 2.92] |
| `T60_ratio` | 0.05 – 0.95, log | 0.665 [0.562, 0.791] |
| `loss_F1` | 3000 – 50000, log | 2.60e4 [2.13e4, 3.42e4] |

Fixed: `rho` 7700, `E` 1.85e11 (degenerate with `h`), `fp_x` 0.22, `fp_y` 0.18,
`op_x` 0.48, `op_y` 0.59 (one plate, one drive point, one pickup — the probe
picked the *same single draw* for bright_1, dark_1 and medium_1, and likewise
for `_2` and `_3`), `Lx` 1.0, `nu` 0.30.

`fmax 22000`, pin `(125, 351)` = 43,875 cells, duration 1.0 s, 24576/991,
2500-step shared base then 10000 steps × 3 arms at batch 64, `--eval-every 250`.
**≈11.8 h/arm**, three in parallel.

## What emt8 still cannot do

At every one of the four probe ceilings the winners put `loss_F1` at ~1.15× the
ceiling and pushed `T60_ratio` up (0.23 → 0.23 → 0.56 → 0.67). Both mean the
same thing: as little frequency-dependent damping as the model allows.
`sig = alpha + beta*omega^2` cannot produce the target's shallow high-frequency
decay, which falls 1.86× from 2 to 8 kHz against the law's ~16×. **Expect both to
sit high in the results, and expect that to be the residual after everything else
is right.** No bound fixes it; a different damping law would.

A modal sum also has no noise source, so a real plate's hiss, room and driver
transient are outside the model at any parameter setting.

## The test of why emt7 collapsed

Narrower than it looks. Fifteen recordings of one plate means `h` and `Ly`
*should* be near-constant across them — that is the right answer, not a collapse.
What must develop spread is **`loss_F1`** (bright vs dark is a spectral
difference) and **`T60_DC`** (the damper is a decay difference). emt7 had spread
on `T60_DC` and none on `loss_F1`, whose optimum sat 3× outside its ceiling.

---

# emt7 — the earlier space, kept for the record

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
