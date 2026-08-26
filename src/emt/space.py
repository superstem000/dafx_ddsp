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
Lx is pinned at 1.0 and Ly searched over 1.5-2.5 so 2.0 sits mid-range; h is
searched over 0.4-0.8 mm around the real 0.5. raw7's h floor of 1 mm is twice
the real thickness, which is why nothing it renders can clang.

WHAT NARROWING COSTS, stated because it is not free. rho 7000-8500 and E
1.7e11-2.2e11 are steel rather than raw7's aluminium-to-gold box. That removes
degrees of freedom: mu = rho*h and D/mu now vary mostly through h, so the
estimation task is EASIER than raw7's, and a loss comparison on it is
correspondingly weaker evidence. It is also what makes the campaign affordable
-- wide rho/E needs 118,048 modes at the corner against 46,854 here, 2.5x the
render cost per IR. Treat "realistic plates" as a different condition from
"wide box", not as a strictly better version of it.

MODE COUNTS, at fmax 16000, from this box's most expensive corner (smallest h,
largest Ly, smallest E, largest rho, smallest T0):

    Ly 2.5, h 0.0004, E 1.7e11, rho 8500, T0 1    ->  DDx,DDy = 137, 342

which is 46,854 modes -- almost exactly what raw7 already hits at ITS worst
corner (47,742 at fmax 10000). The extra ceiling is paid for by the narrower
geometry. FIXED_MODE_GRID below is that corner, so no draw is ever truncated;
inheriting raw7's (60, 185) would truncate every single draw in this space,
including the thickest, smallest, stiffest one, which needs DDx=108.

fmax 16000 rather than 20000: memory is flat in fmax -- chunk_elems bounds the
modal sum regardless -- but time is linear in mode count, and 20 kHz costs
another 25% for content at the edge of hearing. Raising it later is one
constructor argument and a re-pin.

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
FMAX = 16000.0

# Computed from the corner above, not inherited. Passed identically to dataset
# generation and to training: a pin that differs between them means the targets
# and the renders are different plates.
FIXED_MODE_GRID = (137, 342)

# 1.0 s, not raw7's 0.25 s. At T60_F1 = 1.2 s a quarter second captures 12 dB
# of the high-frequency decay; one second captures 50, and 15 dB at DC. Two
# seconds would capture more of the DC tail but costs 8x the training tensor
# (98304 x 2 s is 34.7 GB) and takes the encoder from 21 STFT frames to 172,
# which changes the architecture as well as the data.
DURATION = 1.0

PLATE_EMT7 = dict(
    keys=["Ly", "h", "T0", "rho", "E", "T60_DC", "loss_F1"],
    bounds={
        # 2.0 m mid-range: the real plate.
        "Ly": (1.5, 2.5),
        # 0.5 mm mid-range. raw7's floor was 1 mm, twice the real thickness.
        "h": (0.0004, 0.0008),
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
