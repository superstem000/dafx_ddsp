"""PLATE_EMT7 -- the parameter space, and every value that deviates from raw7.

Pure data. Nothing here imports from src.cmaes or src.gd, so
fit_7param_norm_es can register this space without a circular import; the
registry entry there is one line and this file is the definition.

WHY A NEW SPACE. Rendering an EMT-140 IR through the raw7 encoders and
listening to it, three things were wrong and none of them was the encoder:

  NO INITIAL TRANSIENT. The strike's brightness is set by the drive point,
  fp_x/fp_y, which raw7 pins -- and by how many high modes exist at all, which
  --fixed-mode-grid 60,185 was truncating on every draw.
  FAR DARKER. The renderer's own ceiling is fmax=10000, hardcoded as a
  constructor default and never passed by Raw7Space, so everything above 10 kHz
  is dropped before any parameter has a say. loss_F1=500 then puts the
  high-frequency damping corner two octaves below where a plate's is.
  FAR LONGER. raw7 pins T60_DC=6.0 and T60_F1=2.0. Every render decays over six
  seconds whatever the target does; an EMT-140 with the damper closed is well
  under one.

So the deviations are: damping enters the search, the ceiling is raised, the
mode grid is computed from this space's own corner instead of inherited, and
the geometry is narrowed to a real plate rather than a wide sampling box.

THE PHYSICAL PLATE. An EMT-140 is 2.0 x 1.0 m of 0.5 mm steel under tension.
Lx is pinned at 1.0 and Ly searched over 1.5-2.2, so 2.0 sits near the top
rather than in the middle; h is searched over 0.56-0.8 mm, which does NOT
contain the real 0.5. Both ceilings were pulled in to buy render time -- see
THE BUDGET LINE. raw7's h floor of 1 mm is twice the real thickness, which is
why nothing it renders can clang; 0.56 mm is 1.12x, which is the closest this
campaign can afford.

WHAT NARROWING COSTS, stated because it is not free. rho 7000-8500 and E
1.7e11-2.2e11 are steel rather than raw7's aluminium-to-gold box. That removes
degrees of freedom: mu = rho*h and D/mu now vary mostly through h, so the
estimation task is EASIER than raw7's, and a loss comparison on it is
correspondingly weaker evidence. It is also what makes the campaign affordable
-- wide rho/E needs (159, 350) = 55,650 cells at the corner against 22,000
here, 2.53x the render cost per IR. Treat "realistic plates" as a different
condition from "wide box", not as a strictly better version of it.

MODE COUNTS, from this box's most expensive corner (smallest h, largest Ly,
smallest E, largest rho, smallest T0):

    Ly 2.2, h 0.00056, E 1.7e11, rho 8500, T0 1   ->  DDx,DDy = 100, 220

which is 22,000 grid cells. FIXED_MODE_GRID below is that corner, verified as
the maximum over all 32 corners of the box, so no draw is ever truncated. The
cheapest draw needs (75, 112). Inheriting raw7's (60, 185) would truncate every
single draw here.

THE BUDGET LINE, which is what set fmax and the h floor and is the only reason
they are not larger. The renderer materialises the whole DDx x DDy grid and
masks afterwards, so cost is grid cells x samples, and

    cells  ~  Ly * fmax * sqrt(rho/E) / h

Ly, fmax and h are therefore ONE knob, not three, and at 1.0 s and 10000 steps
a 24 h arm buys Ly*fmax/h <= 4.7e7. Points on that line, all costing 23.8 h:

    fmax 16000, h 0.75 mm      fmax 13000, h 0.61 mm
    fmax 12000, h 0.56 mm      fmax 11000, h 0.50 mm   <- the real thickness

12000 / 0.56 mm was chosen because fmax and h fix different symptoms. fmax sets
where the spectrum stops, and 10000 was the "far darker"; 12000 keeps 1.2x of
that fix. h sets how densely modes pack BELOW the ceiling, and mode density is
what an initial crash is made of, so spending the budget on a 16 kHz ceiling
would buy content above 13 kHz -- where a plate has least energy -- with the
very mode density the transient needs. Raising either later is one constant and
a re-pin, at a linear cost in training hours.

WHAT IS STILL UNREACHABLE, so it is not rediscovered as a surprise. fp_x/fp_y
stay pinned, so onset brightness is identical across the whole dataset; if the
strike is still wrong that is the reason, and the fix is to swap T0 out for
fp_x -- the repo's own sensitivity table already ranks T0 last of the searched
parameters (dnorm 5.7/10.3/17.3 against T60_DC's 14.8/34.1/46.7). And a modal
sum has no noise source at all, so a real plate's hiss, room and driver
transient are outside the model at any parameter setting.
"""

# The renderer's frequency ceiling. BatchedModalPlateTorch defaults it to
# 10000.0 and Raw7Space never passes one, so every plate result in this repo
# is bandlimited to 10 kHz -- against 48 kHz recordings with content to 20.
# 12000 rather than 16000: see THE BUDGET LINE. It trades against the h floor
# one-for-one, and h is the one that controls mode density.
FMAX = 12000.0

# Computed from the corner above, not inherited. Passed identically to dataset
# generation and to training: a pin that differs between them means the targets
# and the renders are different plates.
FIXED_MODE_GRID = (100, 220)

# 1.0 s, not raw7's 0.25 s. At T60_F1 = 1.2 s a quarter second captures 12 dB
# of the high-frequency decay; one second captures 50, and 15 dB at DC. Two
# seconds would capture more of the DC tail but costs 8x the training tensor
# (98304 x 2 s is 34.7 GB) and takes the encoder from 21 STFT frames to 172,
# which changes the architecture as well as the data.
DURATION = 1.0

# ---------------------------------------------------------------------------
# emt8. Everything below is set by src/emt/probe.py, which removed the encoder
# and scored 2048 random draws against the fifteen real IRs directly. Read that
# file's header for why the encoder's own answers could not be used: emt7's
# arms emitted near-constants, and every prediction was trapped inside a box the
# probe then showed to be wrong in four of seven dimensions.
#
# WHY rho AND E ARE FIXED, and this is the finding that matters most. The modal
# sum sees only three combinations of them: mu = rho*h, D/mu = E*h^2/(12(1-nu^2)rho)
# and T0/mu. Four searched parameters mapping to three observables leaves an
# EXACT one-parameter degeneracy -- these three render bit-identically:
#
#   h .00060  rho 13697  E 1.043e12  ->  mu 8.2180  D/mu 2.51100  T0/mu 988.923
#   h .00107  rho  7702  E 1.855e11  ->  mu 8.2180  D/mu 2.51100  T0/mu 988.923
#   h .00350  rho  2348  E 5.256e09  ->  mu 8.2180  D/mu 2.51100  T0/mu 988.923
#
# emt7 searched all three and its docstring claimed every parameter was
# individually identifiable. That was false, and it is the best explanation of
# emt7's railing: a tanh-bounded head free to slide along a flat direction parks
# at a corner. Fixing rho and E removes redundancy, not freedom -- (h, T0) ->
# (D/mu, T0/mu) is then a bijection -- and it makes the parameter NMSE, which is
# the thesis's own metric, measure something identifiable for the first time.
# h's ceiling is widened to 0.004 to carry the D/mu range rho and E gave up
# (44x against the 9x of emt7's range); the FLOOR sets render cost, so that is
# free.
#
# WHY THE DRIVE POINT AND PICKUP ARE FIXED, and why their VALUES are a
# convention rather than a measurement. Two separate facts, and only the first
# justifies fixing them.
#
# They are EXACTLY EXCHANGEABLE. The renderer forms
#   in_weight  = sin(fp_x*pi*m) * sin(fp_y*pi*n)
#   out_weight = sin(op_x*pi*m) * sin(op_y*pi*n)
#   P = 4 * out_weight * in_weight * ...
# a product, so swapping (fp_x, fp_y) with (op_x, op_y) renders bit-identically.
# Searching all four therefore carries a hard two-fold ambiguity no data
# resolves -- the same class of error as searching rho, E and h together, and it
# would corrupt the parameter NMSE the thesis depends on.
#
# And the objective is FLAT in them. src/emt/scatter.py over the 22 kHz probe:
# the fifteen per-IR winners come from only five distinct draws, whose fp_x
# ranges 0.036-0.486 and op_x 0.064-0.944 -- 94% and 98% of the probe box -- all
# scoring in the top band. So these values are NOT determined by the data. They
# are the top-32-by-mean medians, which is a stable statistic (bootstrap CI
# about a third of the box) computed over an objective that barely varies. Being
# wrong about them costs little, which is exactly what flatness means; but do
# not read them as an estimate of where an EMT-140 is actually driven.
#
# The caveat on that: flat under loss_mfcc is not the same as inaudible. The
# drive point sets onset brightness and the pickup combs the modal amplitudes,
# and why_dark's ONSET table is where a listening difference would show up.
#
# WHY fmax IS 22000. Best-of-2048 against saturation, full band, same seed:
#
#   fmax 12000 -> 1.781    fmax 20000 -> 1.544
#   fmax 16000 -> 1.665    fmax 22000 -> 1.324
#
# Monotone and never flattening; 22 kHz covers 99.9% of the mel axis, so there
# is no further ceiling to chase. emt7's 12000 was chosen when a "0.36% of
# energy above the ceiling" argument said it did not matter. That argument was
# wrong: energy share badly understates what those bands are worth.
#
# WHAT THIS SPACE STILL CANNOT DO, so it is not rediscovered as a surprise. At
# every one of the four ceilings the winners put loss_F1 at ~1.15x fmax and
# pushed T60_ratio up (0.23, 0.23, 0.56, 0.67) -- both meaning "as little
# frequency-dependent damping as the model allows". sig = alpha + beta*omega^2
# cannot produce the target's shallow high-frequency decay, which falls 1.86x
# from 2 to 8 kHz against the law's ~16x. Expect both parameters to sit high in
# the results, and expect that to be the residual after everything else is right.
FMAX_EMT8 = 22000.0

# Corner of the emt8 box (max Ly, min h, min T0), verified as the maximum over
# all eight corners. The cheapest draw needs only (48, 63), so most draws pay
# 14.5x their own requirement -- that is the price of a pin, and the pin is what
# makes targets and training synthesis agree bit-for-bit.
FIXED_MODE_GRID_EMT8 = (125, 351)

DURATION_EMT8 = 1.0

PLATE_EMT8 = dict(
    keys=["h", "Ly", "T0", "T60_DC", "T60_ratio", "loss_F1"],
    bounds={
        # Probe winners ~1.07 mm [0.64, 1.31]. The ceiling is 0.004 rather than
        # ~0.002 because h now carries the D/mu range rho and E used to; the
        # floor is what sets render cost, so the ceiling is free.
        "h": (0.0006, 0.004),
        # 1.92 [1.66, 2.50].
        "Ly": (1.3, 2.8),
        # 8127 [458, 2.76e4]. T0 moved with the ceiling across the four probes
        # (5.34e4, 2.24e4, 4857, 8127), so it is entangled with fmax and gets a
        # generous range rather than a tight one -- but not the 1e5 first
        # proposed, which is 3.6x above the interval and only dilutes the
        # log-uniform sampling density.
        "T0": (30.0, 5e4),
        # 2.67 [2.39, 2.92]; wide because a real EMT-140 damper is.
        "T60_DC": (0.8, 6.0),
        # NOT an absolute T60_F1. beta = 3ln10/dOmSq * (1/T60_F1 - 1/T60_DC), so
        # T60_F1 > T60_DC makes beta negative, sig negative at high omega, and
        # r = exp(-sig*k) > 1: the mode GROWS until it overflows to inf and
        # poisons every metric to nan. An absolute bound cannot express the
        # constraint; quiet7 carries it as a ratio for the same reason, and
        # emt7's pin at 1.2 was safe only by accident of its T60_DC floor.
        # Probe winners 0.665 [0.562, 0.791].
        "T60_ratio": (0.05, 0.95),
        # 2.60e4 [2.13e4, 3.42e4] -- above the 22 kHz ceiling. See the note on
        # the damping law above.
        "loss_F1": (3000.0, 50000.0),
    },
    log_keys={"T0", "T60_DC", "T60_ratio", "loss_F1"},
    fixed={
        "Lx": 1.0,
        "nu": 0.30,
        # Degenerate with h -- see the block above. 7700 / 1.85e11 is rho/E
        # 4.2e-8, a sound speed of 4900 m/s, and both sat within a few percent
        # across all four probe ceilings with the tightest intervals of anything
        # measured.
        "rho": 7700.0,
        "E": 1.85e11,
        # One plate, one drive point, one pickup. Probe medians at 22 kHz.
        "fp_x": 0.22,
        "fp_y": 0.18,
        "op_x": 0.48,
        "op_y": 0.59,
    },
    products={"T60_F1": ("T60_ratio", "T60_DC")},
    composite=False,
)

# ===========================================================================
# emt9: emt8's box, below the Nyquist cliff.
#
# THE SAME SEARCH SPACE. keys, bounds, log_keys, fixed and products are emt8's,
# reused by reference below rather than copied, so the two cannot drift and a
# result under either is a result about the same six coordinates. ONLY THE
# CEILING CHANGES.
#
# WHY. emt8's arms ended ABOVE their own initialisation on their own objective
# -- the parameter-only base reached L1_STFT 0.3945 on val with spec_w 0
# throughout, and the arm trained to minimise L1_STFT finished at 0.6562. That
# is not a bad minimum, it is a surface gradient descent cannot walk, and
# src/emt/slice.py measured it directly: no encoder, no dataset, no training,
# just the loss along one coordinate through a known truth at one Adam step per
# sample. Local minima within +-0.05 of z, and the fraction of steps that move
# AWAY from the truth while pointing at it:
#
#   fmax     Nyq%    h: linear      Ly: linear        h: eps1e2
#   22,000  99.8%    61  /  22%     44  /  18%        70  /  34%
#   21,000  95.2%     2  /   0%     13  /   3%        69  /  35%
#   20,000  90.7%     1  /   0%      9  /   2%        84  /  35%
#   18,000  81.6%     1  /   0%     10  /   3%        77  /  38%
#   12,000  54.4%     1  /   0%     10  /   3%        69  /  35%
#
# Two separate readings, and both matter.
#
# THE CLIFF IS AT NYQUIST, NOT AT BANDWIDTH. 21 kHz is clean and 22 kHz is
# destroyed, so the constraint is proximity to sr/2 rather than how much band
# the plate covers. It is NOT the number of modes crossing the om <= max_omega
# gate -- that is 255 at 18k and 283 at 22k, an 11% difference against 1 minimum
# versus 61, and it rises smoothly across the whole ladder while the damage is a
# step. At 22 kHz the top modes sit at om/sr = 3.134 against pi = 3.1416, about
# 2.005 samples per cycle. The mechanism there is not established; the threshold
# is, and 20 kHz leaves 2 kHz of margin below a cliff bracketed to within 1 kHz.
#
# Ly's 9-13 minima at every ceiling from 10k to 21k is its own roughness floor,
# not a ceiling effect -- 20 kHz is marginally CLEANER than 12 kHz on both
# coordinates.
#
# THE COMPRESSION LADDER IS IN THE SURFACE, and this is the campaign's actual
# result. At every clean ceiling the log arm has 69-84 local minima where linear
# has one. That is "log compression is bad for parameter estimation" as a
# measured property of the objective rather than an outcome correlation, and it
# predicts every arm ever run here: a clean basin (1 minimum, 0%) and the arm
# improves its own loss 2-14x while holding parameters (quiet7 linear 0.95 ->
# 0.068 at ratio 0.05, emt7 linear 0.199 -> 0.105 at ratio 0.14); a rough
# surface and it stalls or worsens its own loss and destroys them (emt7 hyb1e2
# 0.430 -> 0.544 at ratio 2.40, quiet7 hyb1e4 1.053 -> 3.693 at ratio 3.28).
# Ten arm-runs, three campaigns, no exceptions. emt8 is the one campaign whose
# LINEAR control was also on a rough surface, which is why it inverted.
#
# WHAT IT COSTS. probe best-of-2048 against the fifteen real IRs reads 12k
# 1.781, 16k 1.665, 20k 1.544, 22k 1.324, so 20 kHz keeps 52% of the 12k->22k
# reachability gain. It is also 9% CHEAPER than emt8: 39,746 modes against
# 43,875. chunk_elems 8e8 needs no change -- launches go 156 -> 140, which is
# the safe direction.
FMAX_EMT9 = 20000.0

# Corner of the box (max Ly, min h, min T0), maximum over all eight corners,
# recomputed at 20 kHz from
#   DDx = floor(Lx/pi * sqrt((-T0 + sqrt(T0^2 + 4*max_omega^2*rho*h*D))/(2D)))
# The same computation returns emt8's (125, 351) at 22 kHz, which is what
# checks it.
FIXED_MODE_GRID_EMT9 = (119, 334)

DURATION_EMT9 = 1.0

# By reference, not by copy: emt9 IS emt8's space at a different ceiling, and
# an edit to the bounds above has to reach both or the claim is false.
PLATE_EMT9 = PLATE_EMT8

# What gen.sh and check.py read, so the four numbers cannot drift between
# dataset generation and training. Keyed by PLATE_PARAM_SPACE.
NUMERICS = {
    "emt7": dict(fmax=FMAX, grid=FIXED_MODE_GRID, duration=DURATION),
    "emt8": dict(fmax=FMAX_EMT8, grid=FIXED_MODE_GRID_EMT8, duration=DURATION_EMT8),
    "emt9": dict(fmax=FMAX_EMT9, grid=FIXED_MODE_GRID_EMT9, duration=DURATION_EMT9),
}


PLATE_EMT7 = dict(
    keys=["Ly", "h", "T0", "rho", "E", "T60_DC", "loss_F1"],
    bounds={
        # The real plate is 2.0 m; the ceiling is 2.2 rather than 2.5 because
        # cells scale linearly in Ly. See THE BUDGET LINE.
        "Ly": (1.5, 2.2),
        # The real plate is 0.5 mm and this floor is 0.56: cells scale as 1/h,
        # so the last 0.06 mm costs 12% of every training hour. raw7's floor was
        # 1 mm, twice the real thickness.
        "h": (0.00056, 0.0008),
        # raw7 went to 0.01 N/m, which is not a tensioned plate.
        "T0": (1.0, 500.0),
        # Steel is 7850.
        "rho": (7000.0, 8500.0),
        # Steel is ~2.0e11.
        "E": (1.7e11, 2.2e11),
        # The damper range. FIXED at 6.0 in raw7, which is the "far longer".
        "T60_DC": (1.5, 6.0),
        # The high-frequency corner. FIXED at 500 in raw7, two octaves below a
        # plate's, which is the "far darker".
        "loss_F1": (2000.0, 8000.0),
    },
    # T0 spans 2.7 decades; the rest are within one and stay linear.
    log_keys={"T0"},
    fixed={
        "Lx": 1.0,
        # Steel, not raw7's 0.25. Enters only through 1/(1-nu^2), ~3%.
        "nu": 0.30,
        # Near-degenerate with loss_F1 -- both shape the same high-frequency
        # tilt -- so it is pinned and loss_F1 carries it. At T60_DC 4.0,
        # T60_F1 1.2 and loss_F1 5000 the decay runs 4.0 s at 100 Hz, 3.7 at
        # 1 k, 1.2 at 5 k and 0.39 at 10 k.
        "T60_F1": 1.2,
        # Drive point: arbitrary, and pinned as in raw7. See the docstring --
        # this is what fixes onset brightness across the whole dataset.
        "fp_x": 0.335,
        "fp_y": 0.467,
        # Pickup: a measurement nuisance, unknown on a real IR and not worth
        # spending two of seven search dimensions on. raw7 searched these.
        "op_x": 0.6,
        "op_y": 0.7,
    },
    # No derived column: T60_F1 is a constant here, not a ratio of T60_DC as in
    # quiet7.
    products={},
    # No mu / D_mu reduction. Geometry and damping are mixed, every parameter is
    # individually identifiable, and the raw NMSE over `keys` is the whole
    # story -- so the 6d/5d composite reports are skipped rather than faked,
    # exactly as for quiet3 and quiet7.
    composite=False,
)
