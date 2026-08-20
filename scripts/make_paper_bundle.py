"""Assemble paper/ from the runs that actually produced the paper's numbers.

Copies, never moves: the working tree stays exactly as it is, and this can be
re-run whenever a sweep finishes to refresh the bundle. Anything already in
paper/ for a given entry is replaced, so a stale partial result cannot survive
a rerun and be mistaken for a final one.

The manifest below is the point of the file. results/ holds many more runs than
the paper uses -- earlier parameterizations, abandoned variants, smoke tests --
and which one is "the" result is a judgement that belongs written down rather
than remembered. Each entry records where it came from and what produced it, and
the generated README carries that through to whoever reads the bundle.

Sources that do not exist are reported and skipped rather than failing the run,
so the bundle can be built while a sweep is still going.

    python scripts/make_paper_bundle.py
    python scripts/make_paper_bundle.py --only plate_ddsp_eps_ladder
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

# --------------------------------------------------------------------------
# Entries. `sources` are copied into <slug>/results, `scripts` into
# <slug>/scripts. `note` is prose that lands in the README next to the entry --
# state what the run is and what is known to be wrong with it.
# --------------------------------------------------------------------------
MANIFEST = [
    dict(
        section="plate",
        slug="01_cmaes_full",
        title="CMA-ES, 20 restarts, all losses (the 'CMA-ES full' row)",
        sources=["results/standard_sweep"],
        scripts=["scripts/standard_sweep/run_standard_sweep.sh"],
        note=(
            "src/analysis/compare_methods.py line 228 reads "
            "results/standard_sweep/l1_stft for the row it labels 'CMA-ES "
            "full L1_STFT', so this is that run and cmaes_norm_es* is not. "
            "50 IRs from data/random-IR-100-1.0s at float32, 14 losses, two "
            "stages each -- sweep_run.log records DSET_ROOT, N_SAMPLES=50 and "
            "DTYPE per run along with wall clock, so the invocation is "
            "recovered rather than assumed. Everything else comes from "
            "fit_7param_norm_es defaults: n_trials 400, budget 25000, sigma0 "
            "0.6, tolfun 1e-5, lhs_seed 42, popsize 30-60, early stop at "
            "0.01. All 14 losses are kept, not just L1_STFT: the cross-loss "
            "comparison at a full restart budget is the counterpart to the "
            "one-restart ladder.\n\n"
            "    WARNING -- the bundled run_standard_sweep.sh will NOT "
            "reproduce this run. It reads --n_trials 1 and defaults N_SAMPLES "
            "to 100; the run used 20 and 50. The script was edited after the "
            "fact and sweep_run.log records only DSET_ROOT, N_SAMPLES and "
            "DTYPE, so neither artifact catches it. The data does: "
            "n_restarts_run is len(terminal Optuna trials) and it caps at "
            "exactly 20 for every one of the 14 losses, with LSD, Mel, MSS and "
            "SC+LogMag sitting at 20 on all 50 IRs -- a ceiling that can only "
            "be --n_trials 20. Set --n_trials 20 and N_SAMPLES=50 to rerun."
        ),
    ),
    dict(
        section="plate",
        slug="02_cmaes_ladder_1restart",
        title="CMA-ES compression ladder, one restart per IR",
        sources=["results/ladder_1restart"],
        scripts=["scripts/ladder_1restart.sh"],
        note=(
            "200 IRs, linear / c2 / log / pow plus mss and smoothmss, each in "
            "two stages, --n_trials 1 against the standard sweep's 20. One "
            "restart per IR is the point: it removes the restart budget as a "
            "confound between losses. compare_methods scores these at stage 2 "
            "using stage2/mu_refined_summary.csv's refined_* columns, so the "
            "figure's medians (l1_stft 1.00e-3) do not match stage1 "
            "summary.csv's nmse (1.59e-2) -- read the right stage."
        ),
    ),
    dict(
        section="plate",
        slug="03_gradient_descent_l1stft",
        title="Per-IR gradient descent, L1-STFT",
        sources=["results/gd/graddescent_l1_stft", "results/gd/raw7_budget"],
        scripts=["scripts/gd.sh"],
        note=(
            "KNOWN INCOMPLETE. gd.sh defaults to NUM=10, so this covers ten IRs "
            "against the CMA-ES runs' hundred, and the smoke/ and torch50/ "
            "directories in results/gd are throwaway. This is the run that "
            "needs redoing at the CMA-ES sample count before it can carry a "
            "claim -- and it is also the run that would separate 'the loss is "
            "bad for gradients' from 'our encoder is bad', since it is "
            "gradient descent without an encoder."
        ),
    ),
    dict(
        section="plate",
        slug="04_ddsp_eps_ladder",
        title="DDSP encoder, log(x + eps) ladder",
        sources=[
            "results/ddsp/eps_ladder",
            "results/ddsp/sweep120k_L1_STFT",
            "results/ddsp/sweep120k_L1_STFT_c2",
            "results/ddsp/sweep120k_L1_STFT_log",
            "results/ddsp/sweep120k_MSS",
            "results/ddsp/l1_stft_tgtnorm",
        ],
        scripts=["scripts/eps_ladder.sh", "scripts/lr_probe.sh"],
        note=(
            "eps_ladder is the single-variable sweep; sweep120k_* is the "
            "earlier four-loss sweep it supersedes, kept because the ladder "
            "runs at 40k steps and sweep120k at 120k. l1_stft_tgtnorm is the "
            "250k linear run and is where the converged linear number comes "
            "from -- the ladder's linear arm reaches 0.0061 at 40k against "
            "that run's 0.0002 at 250k. Note the ladder uses BatchNorm and "
            "sweep120k GroupNorm, so their numbers are not interchangeable."
        ),
    ),
    dict(
        section="plate",
        slug="05_ddsp_lr_probe",
        title="DDSP encoder, learning-rate grid",
        sources=[
            "results/ddsp/lr_probe",
            "results/ddsp/lr_L1_STFT_c2_1e-4",
            "results/ddsp/lr_L1_STFT_c2_3e-5",
            "results/ddsp/lr_L1_STFT_c2_1e-5",
            "results/ddsp/lr_L1_STFT_log_1e-4",
            "results/ddsp/lr_L1_STFT_log_3e-5",
            "results/ddsp/lr_L1_STFT_log_1e-5",
            "results/ddsp/lr_MSS_1e-4",
            "results/ddsp/lr_MSS_3e-5",
            "results/ddsp/lr_MSS_1e-5",
        ],
        scripts=["scripts/lr_probe.sh", "scripts/jobs_lr_probe.txt"],
        note=(
            "The stated grid answering 'you did not tune the learning rate'. "
            "The lr_* directories are the older cells under pre-ladder names "
            "(c2 is eps1, log is eps1e7); lr_probe holds the rest, including "
            "the L1_STFT control the older probes never covered."
        ),
    ),
    dict(
        section="diffmoog",
        slug="06_ddsp_eps_ladder",
        title="DiffMoog encoder, log(x + eps) ladder on the fixed-pitch task",
        sources=["results/diffmoog"],
        # The recreation of Masuda & Saito's design in DiffMoog, plus the
        # linear swap. Its header is where the 16000-train / 250-steps-per-epoch
        # alignment is explained -- DiffMoog's ramp is epoch-indexed and theirs
        # is step-indexed, so the dataset size is what makes the two land on
        # each other.
        scripts=["scripts/jobs_diffmoog_dsrecreation.txt"],
        extra_globs=[
            "external/diffmoog/configs/loss_study/q_*.yaml",
            # The resolved config and the git commit + argv each run actually
            # used. configs/ is the input; config_dump/ is what happened.
            "external/diffmoog/experiments/current/q_*/config_dump/*",
            # create_data.py writes commit_and_args.txt beside each split; it
            # is the only record of how the dataset was generated, since the
            # training run's own commit_and_args only says "-d fixed10k".
            "external/diffmoog/data/fixed10k/*/commit_and_args.txt",
            "external/diffmoog/data/fixed10k/*/params_dataset.csv",
        ],
        note=(
            "The q_* runs are the ladder: ploss, linear at two rates, the eps "
            "ladder, and mss. Scalars only -- experiments/ is 21 GB of "
            "Lightning checkpoints and none of it is needed for a figure. "
            "SAW_FIXED_FILTER is amp plus filter_freq with pitch pinned, which "
            "is why this task can be learned at all.\n\n"
            "    Why q_* and not the synth_/filter_/narrow_/sawfilt_ families: "
            "those set parameters_loss_weight 1 with "
            "spectrogram_loss_warmup_epochs 25 and loss_switch_epochs 75, i.e. "
            "they train on parameter loss and then hand the model over to a "
            "spectral one. That handover destroys the model. At ~40 steps per "
            "epoch the schedule puts the switch between steps 1000 and 4000, "
            "and narrow_spectrogram_magnitude tracks it exactly: minimum 0.055 "
            "at step 559 during the p-loss phase, rising through 0.205 / 0.467 "
            "/ 0.552 as the ramp proceeds, then FLAT at 0.568 for the final "
            "4000 steps -- 10x its best, and frozen. "
            "narrow_spectrogram_logmag_eps_l1 is worse: 0.054 to 0.962, 18x, "
            "also flat from step 5000.\n\n"
            "    narrow_ploss is the control that isolates it. Parameter loss "
            "throughout, never switching, best-to-final 1.4x and no collapse. "
            "So it is not the p-loss stage that hurts, it is the handover. The "
            "q_* and fx_* families never switch (parameters_loss_weight 0, "
            "warmup 0) and degrade 1.1-2.4x with no flatline -- ordinary late "
            "drift, peaking around step 2000-4500.\n\n"
            "    All 26 runs' histories are included, so the switch families "
            "are here as evidence rather than as an assertion. It also bears "
            "on ds_switch.yaml, which uses the same handover at warmup 50 / "
            "switch 150 and should be expected to reproduce the collapse."
        ),
    ),
    dict(
        section="plate",
        slug="07_ddsp_controls",
        title="DDSP encoder, the three knobs that do not rescue a compressed loss",
        sources=[
            "results/ddsp/eps_ladder_adam1e8",
            "results/ddsp/head_probe",
        ],
        scripts=[
            "scripts/eps_ladder.sh",
            # The launchers for the two runs this section is actually built
            # from. eps_ladder.sh alone is the ladder's own recipe, not these.
            "scripts/head_probe.sh",
            "scripts/jobs_head_probe.txt",
            "scripts/jobs_ladder_adam1e8.txt",
        ],
        extra_globs=[],
        note=(
            "The answers to 'you did not tune X'. 05 covers the learning rate; "
            "these cover the optimiser epsilon and the output-head bound, and "
            "neither moves the result.\n\n"
            "    eps_ladder_adam1e8 is the whole ladder rerun at adam_eps=1e-8 "
            "against the default 1e-16. It matters most where it could matter "
            "at all: the gentlest rung improves from 0.1473 to 0.0789, and the "
            "linear arm is unchanged at 0.0061. 0.0789 is still 1.7x the "
            "constant-predictor floor of 0.0473, so the rung that benefits "
            "most from fixing the optimiser still fails to beat predicting the "
            "mean.\n\n"
            "    head_probe swaps the bounded output head three ways -- tanh, "
            "softcap, normtanh -- at 12k steps. Linear lands at 0.0156-0.0170 "
            "and eps 1e-1 at 0.1673-0.2281 whichever is used, so saturation of "
            "the head is not what separates them. Recorded because a saturating "
            "head WAS a real failure earlier in this project (see "
            "train_encoder.py's leakytanh, kept in the source as a measured "
            "failure), and ruling it out here is what makes that episode "
            "irrelevant to the ladder.\n\n"
            "    jobs_ladder_adam1e8.txt carries the argument for why 1e-16 is "
            "a bug rather than a setting, and it is not written down anywhere "
            "else: once a coordinate's surface goes flat both m and v are ~0, "
            "so m/(sqrt(v) + 1e-16) is noise divided by noise and the "
            "coordinate takes full-lr steps in a rounding-determined "
            "direction. head_probe.sh and jobs_head_probe.txt are the same six "
            "cells -- {tanh, normtanh, softcap} x {L1_STFT, L1_STFT_eps1e1} at "
            "12k steps -- run as a fixed claim on GPUs and as a queue "
            "respectively; either reproduces the runs here."
        ),
    ),
    dict(
        section="diffsynth",
        slug="08_reproduction",
        title="Masuda & Saito reproduced, and the loss swapped inside their schedule",
        sources=["results/diffsynth"],
        scripts=[
            "scripts/ds_run.sh",
            "scripts/jobs_diffsynth.txt",
            "scripts/gpu_queue.py",
            "scripts/ds_export_scalars.py",
            # The gates. Run in this order before any arm; each answers one
            # question and stops before the next becomes expensive.
            "scripts/diffsynth_smoke.sh",
            "scripts/diffsynth_preflight.sh",
            "scripts/diffsynth_split_check.sh",
            "scripts/diffsynth_determinism_check.sh",
        ],
        extra_globs=[],
        note=(
            "RUN scripts/ds_export_scalars.py FIRST. results/diffsynth is 16 GB "
            "of Lightning checkpoints and TensorBoard event files; SKIP drops "
            "both, which would leave the hydra configs and no numbers. The "
            "exporter writes scalars.csv into each run directory and that is "
            "what lands here.\n\n"
            "    All four diffsynth sections copy the same results/diffsynth "
            "tree, so the split between 08-11 is editorial rather than "
            "physical: which runs answer which question. The run names carry "
            "it -- pre_/synth_/real_ on the published schedule, *x for the "
            "magnitude family, hold_/cos_ for the schedule study, spec_ for "
            "from scratch.\n\n"
            "    The published design: 50 epochs of parameter loss, a ramp to "
            "spectral over epochs 50-200, spectral only to 400, branching to "
            "synth (harmor) and real (NSynth). Epochs 0-50 are identical for "
            "every arm, so pre_base is trained once and every variant resumes "
            "from it -- both more faithful to the paper and free of the ~1% "
            "Param spread that five independent pretrainings showed, "
            "trainer.deterministic being unavailable here (see "
            "configs/trainer/default.yaml).\n\n"
            "    Four gates establish that the reproduction is faithful, and "
            "they are what to run first. diffsynth_smoke.sh answers 'does the "
            "vendored code import and fit at all' on CPU in minutes, so an "
            "environment fault is never mistaken for a recipe fault. "
            "diffsynth_preflight.sh checks everything else that needs no GPU: "
            "20000 in-domain sounds exactly, because 16000 train at batch 64 "
            "is what makes 250 steps/epoch and puts the ramp on epochs 50 and "
            "200; that every NSynth clip is exactly length*sample_rate samples, "
            "since WaveParamDataset asserts it and one short file crashes "
            "training hours in; and the loss schedule evaluated at its own "
            "boundaries rather than trusted.\n\n"
            "    diffsynth_split_check.sh is the one that could invalidate the "
            "comparison rather than merely bias a number, and it is why "
            "split_manifest.py exists. IdOodDataModule.create_split draws from "
            "the global RNG at setup() time -- random_split with no generator, "
            "plus np.random.choice for the OOD pool -- so the split is "
            "reproducible only if every run makes the same number of draws "
            "before setup(). Every arm here is a resume from pre_base, and "
            "Lightning restores RNG state from a checkpoint; if it does so "
            "before setup(), the resumed run draws a DIFFERENT split and "
            "pretrain's validation files become the resume's training files. "
            "That is leakage across the phase boundary, biased towards the "
            "paper's own result, and invisible in every metric. The script "
            "runs a pretrain and a resume and diffs their manifests; the "
            "manifests in each run directory here are the standing record.\n\n"
            "    diffsynth_determinism_check.sh runs the same configuration "
            "twice on GPU. trainer.deterministic=True asks for reproducibility "
            "and does not guarantee it -- the cuDNN GRU backward and the linear "
            "interpolation in util.resample_frames are the likely offenders -- "
            "so this is what settles whether to pay for it, and it is why "
            "epochs 0-50 are trained once and shared rather than repeated per "
            "arm.\n\n"
            "    Seven upstream defects were fixed to get this running at all; "
            "they are in the vendored external/diffsynth under code/, each "
            "with the reason in a comment. The one that changes numbers rather "
            "than merely running: save_last=True does NOT give a latest "
            "checkpoint under save_top_k=1 with a monitor, so the first "
            "attempt's 50-epoch base was silently epoch 37."
        ),
    ),
    dict(
        section="diffsynth",
        slug="09_magnitude",
        title="The same losses on magnitude rather than power",
        sources=["results/diffsynth"],
        scripts=[
            "scripts/jobs_diffsynth_magx.txt",
            "scripts/jobs_diffsynth_magx2.txt",
            "scripts/ds_scale_balance.py",
        ],
        extra_globs=[],
        note=(
            "The *x arms: magx, magx_halfw, hybridx, logx_halfw, each with "
            "synth and real branches, all resuming from the same pre_base at "
            "epoch 50 on the published schedule.\n\n"
            "    Why. diffsynth's spectral loss is L1 on POWER (amp = re^2 + "
            "im^2), so for a partial of amplitude a it is |a^2 - t^2| and the "
            "derivative 2a vanishes as a -> 0: silence is a stationary point "
            "with exactly zero gradient. Measured -- from random init the "
            "linear-only arms froze at 10.0138 and 5.0069, unchanged to four "
            "decimals from epoch 4 and exactly 2x apart, one trajectory stuck "
            "and scaled by the weight. Arms carrying a log term escape, since "
            "d log(p+eps)/dp is 1e4 at the origin rather than 0. On magnitude "
            "the loss is |a - t|, derivative +-1, and there is no such point. "
            "That is also what the plate's linear loss has always been "
            "(losses.py, torch.abs of the stft), so 'linear' meant two "
            "different things across the two systems and only one had the "
            "trap.\n\n"
            "    The eps moves with the domain: log_eps 1e-4 on power is 1e-2 "
            "on magnitude, so the knee stays at the same signal level and "
            "`power` changes the domain and nothing else. Getting this wrong "
            "would put the log four decades deeper, near the rungs that "
            "collapse on the plate.\n\n"
            "    ds_scale_balance.py is the supporting measurement: with a "
            "fixed eps against six unnormalised FFT sizes, the knee sits 30 dB "
            "apart between the 64- and 2048-point terms, and the linear half "
            "carries 40-52% of its weight at n_fft=2048 against an equal-weight "
            "16.7% while the log half is near-flat. So the two halves are not "
            "equally multi-scale, which is a confound in any mag-vs-log "
            "comparison and is measured here rather than assumed."
        ),
    ),
    dict(
        section="diffsynth",
        slug="10_schedule",
        title="Where the spectral phase settles, once the learning rate stops being the cap",
        sources=["results/diffsynth"],
        scripts=[
            "scripts/jobs_diffsynth_hold.txt",
            "scripts/jobs_diffsynth_cos.txt",
            "scripts/jobs_diffsynth600.txt",
        ],
        extra_globs=[],
        note=(
            "hold_*, cos_* and the *600 arms. The published schedule is "
            "ExponentialLR at gamma 0.99, which is a convergent series: total "
            "parameter travel is capped at lr0/(-ln gamma) = 0.0995 however "
            "long it runs, and 98.2% of that is spent by epoch 400. So 'flat "
            "at 400' cannot be distinguished from 'the learning rate ran out', "
            "and the paper's own 400 epochs are not evidence of a floor.\n\n"
            "    Three attempts, in order, and the first two are recorded "
            "because they failed usefully. The *600 arms stretch gamma to "
            "0.99332217 so the final rate at 600 matches the old one at 400, "
            "which buys 1.5x the budget -- but leaving the phase boundaries at "
            "50/200 ran the whole ramp at roughly twice the step size, and "
            "pre600_hybrid reached epoch 200 at Param 0.0906 against "
            "pre_hybrid's 0.0699. Since param_w is 0 after epoch 200 there is "
            "no supervision left to recover with, so it measured the ramp "
            "rather than the ceiling.\n\n"
            "    hold_* branches the published epoch-200 checkpoints onto a "
            "CONSTANT rate at the value the schedule had there (1.3398e-4), "
            "which removes the cap: 400 epochs at that rate is 4x the entire "
            "post-200 budget of the original run. Every arm then read flat, "
            "which at a constant rate is a real plateau. cos_* anneals from "
            "epoch 300 to zero at 400 so the endpoint is unambiguous, and "
            "lands on the paper's own epoch count.\n\n"
            "    Two mechanisms this needed, both in the vendored source: "
            "ExponentialLR.state_dict carries gamma and load_state_dict does "
            "__dict__.update, so a resumed run silently takes the schedule "
            "from the checkpoint and ignores the config -- hence ConfigLambdaLR, "
            "whose lambda comes from the config and which tolerates a "
            "state_dict written by a different scheduler class. And "
            "trainer.checkpoint_every_n_epochs, because branching an anneal "
            "from a chosen epoch is impossible when only best-by-lsd and latest "
            "are kept."
        ),
    ),
    dict(
        section="diffsynth",
        slug="11_from_scratch_and_metrics",
        title="Spectral loss from random init, and what the evaluation metrics measure",
        sources=["results/diffsynth"],
        scripts=[
            "scripts/jobs_diffsynth_spec.txt",
            "scripts/ds_param_breakdown.py",
            "scripts/ds_param_baseline.py",
            "scripts/ds_mfcc_check.py",
            "src/ddsp/monitor_diffsynth.py",
        ],
        extra_globs=["results/diffsynth/param_baseline.json"],
        note=(
            "spec_* uses configs/schedule/spec.yaml -- sw_w 1.0 and nothing "
            "else, so the spectral loss is the whole objective from step one. "
            "It ships with the upstream repo and had never been used. The point "
            "was to put harmor in the condition diffmoog and the plate are in, "
            "since both of those have no parameter supervision at all and both "
            "say linear wins.\n\n"
            "    Result: from random init NOTHING works on harmor. spec_hybrid "
            "and spec_log_halfw end at 1.14 and 1.16 of the constant-predictor "
            "error -- worse than guessing the dataset mean -- and the two "
            "linear-only arms froze at initialisation, which is the "
            "observation that led to 09. So the from-scratch condition cannot "
            "carry the linear-vs-log claim here, and the published schedule is "
            "where the evidence sits. Kept because a null result that "
            "redirected the work is worth more written down than remembered.\n\n"
            "    THE METRICS. param_loss sums an L1 over six parameter groups "
            "and divides by the count, so val_id/param is a six-way mean of "
            "quantities on different scales -- roughly half osc_mix and q by "
            "magnitude alone, with f0_hz contributing under 1% whatever any arm "
            "does. ds_param_baseline computes the error of a constant "
            "predictor per group and ds_param_breakdown reports each group as a "
            "fraction of it, which is comparable across groups and is the same "
            "normalisation the plate work quotes against. It reverses at least "
            "one reading: amplitudes has the smallest baseline of the six, so "
            "its small raw L1 was hiding that it is the LEAST recovered "
            "group.\n\n"
            "    ds_mfcc_check exists because Mfcc reaches torch.stft through "
            "MelSpec.forward with five positional arguments, so `window` keeps "
            "its None default and the transform runs with a RECTANGULAR window "
            "-- while compute_lsd, two functions away, passes a Hann window "
            "explicitly. It recomputes the distance as logged, with a window, "
            "and with library conventions. On the synth branch all three give "
            "the same ordering, so the defect changes the spread and not the "
            "ranking; the gaps roughly double once the leakage is removed.\n\n"
            "    And the reason MFCC is reported at all: LSD applies "
            "10*log10 to every bin independently, which is the same per-bin log "
            "the training loss's log term applies, differing by the constant "
            "4.343 and the eps. So it grades a log-trained arm with a rescaled "
            "version of its own objective. MFCC's log lands after mel-band "
            "summation, so it stays compressive without being the objective "
            "under test."
        ),
    ),
    dict(
        section="diffsynth",
        slug="12_masking_metric",
        title="A dB floor that is not a free parameter: masking as the evaluation threshold",
        sources=["masking", "results/eval"],
        scripts=[
            "scripts/ds_masking.py",
            "scripts/ds_mfcc_check.py",
            "scripts/ds_floor_audio.py",
            "scripts/ds_compare_audio.py",
        ],
        extra_globs=[],
        note=(
            "THE PROBLEM. MFCC logs mel-band totals, so it keeps scoring bins "
            "far below anything audible, and how far down it counts decides "
            "which loss wins. Clamping everything more than N dB below a "
            "clip's peak makes that explicit -- and on the real branch the "
            "arm ordering FLIPS across the ladder. So N is doing the arguing, "
            "and 'why N' has no answer from inside the metric.\n\n"
            "    THE REPLACEMENT. ds_masking.py computes, per bin and per "
            "frame, the level below which the target signal renders content "
            "inaudible: MPEG-1 psychoacoustic model 1, following Painter & "
            "Spanias (2000) section V. Tonal maskers at local maxima 7 dB "
            "above their neighbourhood, one noise masker per critical band "
            "from what is left, decimation against the absolute threshold and "
            "within 0.5 Bark, two-slope spreading (Schroeder et al. 1979) with "
            "the level-dependent upward slope, masking index 14.5 + z_m for "
            "tonal and 5.5 for noise, power-summed with the Terhardt ATH. That "
            "floor has NO free parameter to argue about.\n\n"
            "    ONE ISO PARAMETER HAD TO BE RESCALED, and it is the only "
            "judgement call in the model. The tonal neighbourhood is +/-2 bins "
            "at ISO's 86.13 Hz per bin (512-point at 44.1 kHz); ours are "
            "15.625 Hz, so the same 172.3 Hz is +/-11 of our bins. "
            "--tonal-window switches between the frequency-scaled reading "
            "(default) and the literal bin count, so the choice is a flag "
            "rather than an assumption.\n\n"
            "    FOUR SELF-TESTS, and what each rules out. A 1 kHz sine at "
            "80 dB SPL gives one tonal masker at 8.51 Bark and SMR 21.2 dB -- "
            "printed APART from the canonical ~24 dB tone-masks-noise figure, "
            "because reaching it via 14.5 + z_m at 1 kHz is arithmetic and "
            "would otherwise read as agreement. White noise gives zero tonal "
            "maskers. A 20-partial tone gives a composite threshold at most "
            "3.53 dB above the best single masker against the superposition "
            "bound 10*log10(21) = 13.22 dB -- this check was WRONG on first "
            "writing, reporting 80.44 dB, because it subtracted a global "
            "threshold carrying the ATH from a spread masker decayed to "
            "-200 dB. The ATH minimum is -4.98 dB SPL at 3325 Hz, which is "
            "Terhardt's fit rather than the idealised '0 dB at 4 kHz'.\n\n"
            "    That bug mattered beyond the test. At PN = 90.302 a bin 70 dB "
            "below peak sits near 20 dB SPL, under the ATH outright below a "
            "few hundred Hz where Terhardt is +30 dB, so a threshold carrying "
            "the ATH would let 'too quiet to hear' masquerade as 'masked by "
            "the signal'. Every floor is therefore reported twice: "
            "frac_energy_masked against the full threshold, and "
            "frac_energy_smasked against the maskers alone. Over the 2000-clip "
            "held-out split at a 70 dB floor those are 0.8701 and 0.7722, so "
            "masking carries 89% of the verdict and the claim is about masking "
            "rather than about quietness. Family medians at that floor run "
            "0.83 (flute) to 0.94 (string).\n\n"
            "    AS A METRIC. --dump-thresholds writes the thresholds once and "
            "ds_mfcc_check --mask uses them as the MFCC floor, giving a `mask` "
            "column and a mask+N cross against each dB rung (both clamp the "
            "same dB quantity, so the cross takes the higher of the two). The "
            "physics is NOT reimplemented for the metric: the 0.5 Bark "
            "decimation is a sequential sweep that does not vectorise cleanly, "
            "and a second batched implementation would be a second thing that "
            "can be subtly wrong. The numpy path the self-tests validate "
            "writes the thresholds; the metric loads them. The cache is keyed "
            "by a content hash of the target audio, so a split that moved "
            "raises rather than pairing a clip with the wrong threshold, and "
            "it carries the SPL offset so the consumer never re-derives the "
            "window normalisation.\n\n"
            "    The 484 MB cache is NOT copied here -- regenerate it with "
            "scripts/ds_masking.py --dump-thresholds masking/T_ood.npz, which "
            "also reproduces masking.csv.\n\n"
            "    THE CLAMP MOVED PRE-MEL, and this restates an earlier number. "
            "A masking threshold is defined per FFT bin and a 6 kHz mel band "
            "spans several critical bands, so the mask has no choice; the dB "
            "columns moved with it or the table would be two conventions side "
            "by side. It is not cosmetic. Post-mel a mel band holding one "
            "strong partial sits above the floor and shelters every quiet bin "
            "in it, so the clamp barely bites; pre-mel those bins are raised "
            "individually. At 70 dB post-mel put hybridx marginally ahead "
            "(8.0404 vs magx 8.0631, a dead tie against 2*se 0.4274); pre-mel "
            "magx leads by 0.49 (7.0168 vs 7.5084, 2*se 0.4264). At 80 dB both "
            "conventions agree, and every rung below 70 agrees. So the "
            "crossover is real but its LOCATION is convention-dependent -- it "
            "must not be quoted as sitting at 70 dB. db70post is retained as "
            "the old convention so nothing here is a silent restatement.\n\n"
            "    THE RESULT. Under the mask, magx_halfw beats hybridx by "
            "0.4082, which is 3.9 se and 4.1% of the metric's range against "
            "the unrelated-pairs saturation of 9.9442. Statistically clear, "
            "perceptually modest: the claim it supports is not that the linear "
            "loss is much better, but that the log term's advantage lives in "
            "the 70-80 dB shell and does not survive at the threshold of "
            "audibility. gam03, trained on the gamma=0.3 compression the "
            "metric ladder calls perceptually calibrated, does not beat magx "
            "(0.1149 against 2*se 0.2078) -- which is the answer to 'why not "
            "just train on the metric'.\n\n"
            "    THE SATURATION ROW IS WHAT MAKES THOSE READABLE, and it "
            "disqualifies two columns. Scoring each target against an "
            "unrelated real clip gives each column's range. `mask` has the "
            "LOWEST best-arm fraction of any column at 30.4% -- a trained "
            "model is 3.3x closer to the target than unrelated audio, where "
            "under the conventional log and dB columns it is only ~2.1x -- "
            "while separating arms better than the gamma ladder (17.0% vs "
            "10.9%). Meanwhile db20 and mask+20 sit at 78.7% and 84.7%: a "
            "trained model is barely distinguishable from randomly paired "
            "audio there, so whatever those columns rank is not synthesis "
            "quality, and a large-looking gap in them means nothing. Only "
            "floors at 60 dB and above, and the mask itself, carry usable "
            "signal.\n\n"
            "    ds_floor_audio.py and ds_compare_audio.py are the listening "
            "checks behind all of this: the first gates a clip into keep<N> "
            "and drop<N> so what a floor discards can be played at its own "
            "level (they sum back to the original sample-for-sample), the "
            "second resynthesises the SAME clips through several arms via "
            "load_arm's reproduced split. An argument about audibility settled "
            "by listening rather than by asserting that librosa's default is "
            "reasonable."
        ),
    ),
]


# Copied once, shared by every entry, rather than traced per result: an import
# trace would be more precise and would also be one more thing that can be
# subtly wrong.
ANALYSIS = [
    # THE generator for the paper's table and ECDF figure. Reads
    # standard_sweep/l1_stft as "CMA-ES full", every ladder_1restart/* as
    # "CMA-ES 1rst", and the encoder from --ddsp-ckpt. Scores CMA-ES at stage 2
    # via the refined_* columns when mu_refined_summary.csv exists, which is
    # why a stage-1 median read straight off summary.csv does not reproduce the
    # figure's numbers.
    "src/analysis/compare_methods.py",
    # Produces the cross-loss comparison tables; lives under results/ rather
    # than src/, so nothing else in the manifest would pick it up.
    "results/cmaes/compare_all_losses.py",
    # Per-IR NMSE for the CMA-ES full run, every 1-restart ladder arm and the
    # 250k DDSP run -- the data behind the ECDF figure, and the only per-IR
    # record that survives for the full CMA-ES run.
    "docs/figures/nmse_per_ir.csv",
    "docs/figures/nmse_ecdf.pdf",
    # The diffsynth reader. Everything in sections 08-11 is read through this:
    # milestone trajectories rather than last-N-epochs, because adjacent epochs
    # differ by noise, and the per-group Param table normalised against the
    # constant predictor.
    "src/ddsp/monitor_diffsynth.py",
]

# The original numpy plate and its dataset generator. random-IR-100-1.0s and
# random-IR-200-0.2s -- the CMA-ES and gradient-descent sets -- came from
# DatasetGen.py, not from src/data/make_dataset.py, and nothing else in the
# manifest reaches it.
GENERATORS = [
    ("ModalPlate", "datasets/generators/ModalPlate"),
    # Re-render an existing dataset's IRs through the torch plate, so target
    # and model share a code path. Referenced by train_encoder's docstring as
    # what was done for the fitting datasets.
    ("gen_torch_targets.py", "datasets/generators/gen_torch_targets.py"),
    ("gen_torch_targets_200.py", "datasets/generators/gen_torch_targets_200.py"),
    # The diagnostic that found the float32 target/synthesis disagreement in
    # the first place.
    ("confirm_f32_gt.py", "datasets/generators/confirm_f32_gt.py"),
    ("src/data/make_dataset.py", "datasets/generators/make_dataset.py"),
    # The diffsynth halves. Without these the synth_ and real_ arms are not
    # reproducible at all -- the in-domain half is rendered from harmor and the
    # out-of-domain half is a sample of NSynth, and neither is shipped as audio.
    ("external/diffsynth/gen_dataset.py", "datasets/generators/diffsynth_gen_dataset.py"),
    ("scripts/get_nsynth.sh", "datasets/generators/get_nsynth.sh"),
    # Separate from get_nsynth.sh on purpose: the archive arrives on stdin, so a
    # heredoc would take stdin away from the pipe and leave tarfile with an
    # exhausted stream.
    ("scripts/nsynth_sample.py", "datasets/generators/nsynth_sample.py"),
]

CODE = [
    ("src", "code/plate_src"),
    # The vendored diffsynth, with the seven upstream fixes needed to run it on
    # modern torch/Lightning and the four changes that are ours: sw_loss.power,
    # sw_loss.log_eps_v, ConfigLambdaLR and the per-group Param logging. Each
    # carries its reason in a comment at the site.
    ("external/diffsynth/diffsynth", "code/diffsynth_src"),
    ("external/diffsynth/configs", "code/diffsynth_configs"),
    ("external/diffsynth/train.py", "code/diffsynth_train.py"),
    ("external/diffsynth/split_manifest.py", "code/diffsynth_split_manifest.py"),
    # The AudioLogger train.py imports, ours as modified: upstream logged audio
    # and figures every validation, which dominated wall clock once the arms
    # were running in parallel.
    ("external/diffsynth/plot.py", "code/diffsynth_plot.py"),
    ("external/diffmoog/src", "code/diffmoog_src"),
    ("external/diffmoog/configs/loss_study", "code/diffmoog_configs"),
    ("external/diffmoog/tools", "code/diffmoog_tools"),
    # y(mu) = y(mu_ref)*(mu_ref/mu) is exact, not approximate, and that identity
    # is what lets the mu stage cost one multiply instead of a resynthesis. It
    # would break silently if a change made the mode grid depend on mu other
    # than through T0/mu and D/mu, so it is checked rather than asserted. The
    # only executable proof of a structural claim in the plate pipeline.
    ("tests", "code/plate_tests"),
    # The environments. Three, because the two vendored repos are pinned
    # against each other's era rather than against ours; diffsynth's is
    # deliberately NOT a reconstruction of the 2021 stack, since PL 1.4/1.5
    # pairs with torch wheels that have no kernel image for a current card.
    ("requirements.txt", "code/requirements/plate.txt"),
    ("external/diffsynth/requirements-modern.txt", "code/requirements/diffsynth.txt"),
    ("external/diffmoog/requirements.txt", "code/requirements/diffmoog.txt"),
    # The bundle's own builder and its checker, so paper/ can be rebuilt and
    # audited by its own criteria rather than by ours. check_paper_bundle.py
    # separates FUNDAMENTAL (a number cannot be recomputed) from the rest.
    ("scripts/make_paper_bundle.py", "code/bundle/make_paper_bundle.py"),
    ("scripts/check_paper_bundle.py", "code/bundle/check_paper_bundle.py"),
]

DATASETS = [
    "data/train_100000_params.csv.gz",
    "data/val_1000_params.csv",
    "docs/DATASETS.md",
]

# Per-IR parameters for the sets the CMA-ES and gradient-descent runs used.
# Those directories are 80 MB and 75 MB of rendered audio; the parameters are
# what make them regenerable, and they are a few hundred KB.
DATASET_GLOBS = [
    ("data/random-IR-100-1.0s/random_IR_params_*.csv", "random-IR-100-1.0s"),
    ("data/random-IR-100-1.0s/generation_summary.txt", "random-IR-100-1.0s"),
    ("random-IR-200-0.2s/random_IR_params_*.csv", "random-IR-200-0.2s"),
    ("random-IR-200-0.2s/generation_summary.txt", "random-IR-200-0.2s"),
]


SKIP = ["*.pt", "*.ckpt", "*.db", "__pycache__", "events.out.tfevents.*",
        # The masking threshold cache, 484 MB for the 2000-clip ood split.
        # Regenerable, and gitignored for the same reason -- ds_masking.py
        # --dump-thresholds rebuilds it alongside masking.csv. Named rather
        # than "*.npz", which would also drop the rendered plate IRs under
        # results/postprocessing.
        "T_*.npz",
        # monitor_diffsynth's accelerator, keyed on file size and mtime and
        # carrying only the tags it happens to plot. scalars.csv from
        # ds_export_scalars.py is the record; this is not.
        ".monitor_cache.json"]


GENERATORS_MD = """# Which script made which dataset, and which ones have a floor

Three generators produced the data in this paper, and they are not
interchangeable. `docs/DATASETS.md` covers the flags; this covers which script.

## `generators/ModalPlate/DatasetGen.py`

The original numpy plate. Produced the per-IR fitting sets:

| set | IRs | duration | generated |
|---|---|---|---|
| `random-IR-100-1.0s` | 100 | 1.0 s | 2026-04-24 |
| `random-IR-200-0.2s` | 200 | 0.25 s | 2026-07-28 |

These are what the CMA-ES sweeps and the gradient-descent runs fit. Each ships a
`generation_summary.txt` recording the parameter ranges, which are identical
between them: Lx 1.0 and nu 0.25 fixed, Ly [1.1, 4.0], h [0.001, 0.005],
T0 [0.01, 1000], rho [2430, 21230], E [6.7e10, 2.2e11].

## `generators/gen_torch_targets*.py` -- and the one that matters most

Re-render an existing set's IRs through the *torch* plate, so a target and the
model that fits it share a code path exactly. `gen_torch_targets.py` says what
that buys in its own docstring: "targets from the SAME torch synth the fitter
uses as candidate, so gt_loss ~ 0 (matched-model / inverse-crime diagnostic)".

**`gen_torch_targets_200.py` writes back into `random-IR-200-0.2s` in place**
(`out_dir = Path(src_dir)`). So that set's `.npz` files are torch-rendered while
its `generation_summary.txt` still records DatasetGen and 2026-07-28 -- the
summary describes the *parameters*, not the current rendering. Do not read it as
provenance for the audio.

That single fact separates the two CMA-ES results:

| run | IRs | dataset | rendering | `gt_loss` median |
|---|---|---|---|---|
| `standard_sweep/l1_stft` ("CMA-ES full") | 50 | `data/random-IR-100-1.0s` | numpy | **1.37e-05** |
| `ladder_1restart/*` (1 restart) | 200 | `random-IR-200-0.2s` | torch | **exactly 0** |
| `on_separate_50ir/phase1` | 50 | numpy | numpy | 1.33e-05 |

### Why this decides which comparison is usable

On the numpy targets the floor is not small, and it is not the same size for
every loss. Median across the 50 IRs of `standard_sweep`:

| loss | `gt_loss` | `best_loss` | floor as a fraction of what was achieved |
|---|---|---|---|
| L2 | 3.0e-12 | 8.6e-08 | 0.00 |
| ESR | 2.5e-11 | 9.8e-08 | 0.00 |
| L1 | 1.28e-06 | 1.35e-06 | 0.95 |
| L1_STFT | 1.37e-05 | 1.39e-05 | 0.98 |
| Mel | 0.188 | 0.522 | 0.36 |
| MSS | 0.818 | 1.073 | 0.76 |
| SC+LogMag | 0.409 | 0.474 | 0.86 |
| LSD | 13.7 | 10.05 | **1.36** |

LSD's optimizer found a loss *below* the value at the true parameters: on those
targets the ground truth is not the minimum, so "LSD did badly" there is partly
a statement about the targets. MSS and SC+LogMag are three quarters floor. The
compressed and perceptual losses take the mismatch hardest, which is the same
asymmetry `docs/DATASETS.md` measures for the encoder -- so a cross-loss
comparison run on numpy targets is confounded in exactly the direction the paper
is arguing about.

`ladder_1restart` has `gt_loss` **exactly 0.0 for all six arms**, so its
cross-loss comparison carries no floor at all. That makes it the trustworthy one
-- and it also forecloses an obvious objection: the ladder shows log losing to
linear with no target/synthesis disagreement anywhere in the picture, so the
result cannot be attributed to one.

The cost is that a zero floor is a matched-model result, the "inverse crime" the
generator names. It is the right control for comparing losses and the wrong
setting for claiming an absolute accuracy, so quote the ladder for the former
and say plainly which targets it used.

## `generators/make_dataset.py`

The encoder datasets. Its `--render-path` is the flag that separates the two
generations of them:

- `direct` -- the historical path, builds plate14 straight from the CSV. Leaves
  `T0` quantised by its *range* rather than its value, a ~6e-5 quantum on a
  range of (0.01, 1000), which is ~1e-4 on the mode frequencies. Invisible to a
  linear loss and **19.8% of saturation to log(x + 1e-7)**.
- `training` -- renders through the float32 `z` the encoder emits, so targets
  and training synthesis agree bit-for-bit.

`--fixed-mode-grid` is the second axis. Without it `n_modes` follows the batch
maximum, so an IR renders differently depending on which batch it lands in:
6.1% of saturation for log against ~0 for linear.

| set | grid pinned | `gt_loss` observed |
|---|---|---|
| `train-100000-0.25s`, `val-1000-0.25s` | no | **1.2490e-05** |
| `*-v3` | (107, 403) | 0.0 |
| `train-p99`, `val-p99` | (86, 282) | 0.0 |

The 250k linear run (`l1_stft_tgtnorm`) is on the first row; the 120k sweep and
the eps ladder are on the last. That is why their numbers are not on the same
footing, and why `diag_gt_floor` has to read `0.0000e+00` on the SHUFFLED row
before a sweep is attributable at all.

Audio is not shipped. `datasets/` carries the parameter CSVs; the commands in
`docs/DATASETS.md` regenerate the audio, and every flag in them is load-bearing.

## `generators/diffsynth_gen_dataset.py` and `generators/get_nsynth.sh`

The two halves of the diffsynth data (sections 08-11), which have no parameter
CSV to ship and so are regenerated rather than copied.

**In-domain**, `diffsynth_gen_dataset.py`: renders from the harmor synth against
a hydra synth config, sampling each static and time-varying parameter and
writing the audio alongside the parameters that produced it. This is the half
with a true theta, and it is the only half on which `val_id/param` means
anything -- the paper's own headline metric is defined only here.

**Out-of-domain**, `get_nsynth.sh` plus `nsynth_sample.py`: NSynth, sampled
rather than extracted whole. `nsynth-train` is 289205 files of 4 s at 16 kHz,
so a full extraction is ~37 GB on top of a 22 GB archive; the paper (sec 4.3.2)
uses 20000 sounds "randomly selected from the full dataset", and
`IdOodDataModule` then narrows whatever pool it is given down to the in-domain
size anyway, so sampling 25000 during extraction reaches the same place for a
fraction of the disk. `nsynth_sample.py` is a separate file rather than a
heredoc because the archive arrives on stdin and a heredoc would take stdin
away from the pipe, leaving `tarfile` with an exhausted stream and no error.

There is no `gt_loss` for the real half, and that is the point of having both:
NSynth has no true parameters, so the model is necessarily misspecified there
and the only question the weighting decides is where the bias lands. The
in-domain half is what makes the bias measurable at all.
"""


def prune_split_manifests(root: Path) -> int:
    """Drop the file lists SplitManifest used to write for every split.

    Runs made before split_manifest.py was narrowed to the valid splits carry
    all six -- 16000 id_train names among them -- which is 1.2 MB per run
    against 1.6 KB for the hash-only form, and 124 copies of it made the
    diffsynth sections 165 MB. Nothing reads anything but id_valid.

    Rewritten on the COPY, never on the source: results/ is the record and this
    is the bundle. Returns bytes saved.
    """
    saved = 0
    for f in root.rglob("split_manifest.json"):
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        before = f.stat().st_size
        touched = False
        for k, v in rec.items():
            if isinstance(v, dict) and "files" in v and not k.endswith("_valid"):
                del v["files"]
                touched = True
        if not touched:
            continue
        f.write_text(json.dumps(rec, indent=2))
        saved += before - f.stat().st_size
    return saved


def copy(src: Path, dst: Path, skip=None) -> tuple[int, int]:
    """Copy a file or tree. Returns (files, bytes)."""
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        # Checkpoints are the reason experiments/ is 21 GB and results/ddsp is
        # hundreds of MB. Nothing in the paper reads them.
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*(skip or SKIP)))
        prune_split_manifests(dst)
        files = [p for p in dst.rglob("*") if p.is_file()]
        return len(files), sum(p.stat().st_size for p in files)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return 1, dst.stat().st_size


def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024 or unit == "GB":
            return f"{x:.1f}{unit}"
        x /= 1024.0
    return f"{x:.1f}GB"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=Path("paper"))
    p.add_argument("--only", nargs="+", default=None, help="Slugs to rebuild")
    p.add_argument("--with-figures", action="store_true",
                   help="Include per-IR diagnostic PNGs. They are 1212 files "
                        "and ~96 MB in the 1-restart ladder alone, they "
                        "regenerate from the CSVs beside them, and no figure "
                        "in the paper is one of them.")
    args = p.parse_args()
    skip = list(SKIP) + ([] if args.with_figures else ["*_diagnostic.png"])

    root = Path(".").resolve()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip() or "unknown"

    lines = [
        "# Paper bundle",
        "",
        f"Generated by `scripts/make_paper_bundle.py` at commit `{commit}`.",
        "",
        "Every path under `results/` here is a copy; the working tree is",
        "unchanged. Checkpoints (`*.pt`, `*.ckpt`), Optuna databases and",
        "TensorBoard event files are excluded -- they are large and nothing in",
        "the paper reads them. For DiffMoog the scalars were extracted first",
        "with `external/diffmoog/tools/extract_scalars.py`.",
        "",
        "Datasets are not copied as audio. `datasets/` holds the parameter CSVs",
        "and the regeneration commands; `docs/DATASETS.md` explains why the",
        "flags in them are load-bearing rather than defaults.",
        "",
    ]

    missing = []
    for e in MANIFEST:
        if args.only and e["slug"] not in args.only:
            continue
        base = args.out / e["section"] / e["slug"]
        nf = nb = 0
        got, absent = [], []
        for s in e["sources"]:
            src = root / s
            if not src.exists():
                absent.append(s)
                continue
            f, b = copy(src, base / "results" / Path(s).name, skip)
            nf, nb = nf + f, nb + b
            got.append(s)
        for s in e.get("scripts", []):
            src = root / s
            if src.exists():
                f, b = copy(src, base / "scripts" / Path(s).name)
                nf, nb = nf + f, nb + b
                got.append(s)
            else:
                absent.append(s)
        for pat in e.get("extra_globs", []):
            hits = sorted(root.glob(pat))
            # config_dump/config.yaml and commit_and_args.txt carry the same
            # basename in every one of the nine runs, and params_dataset.csv in
            # both splits. Flattening on basename kept one of each and dropped
            # the rest without a word. Widen the name by whole path components
            # until every hit in this glob is distinct, so the naming is uniform
            # rather than "whichever one landed first keeps the short name".
            depth = 1
            while depth < 5:
                names = {"_".join(h.parts[-depth:]) for h in hits}
                if len(names) == len(hits):
                    break
                depth += 1
            for h in hits:
                f, b = copy(h, base / "scripts" / "_".join(h.parts[-depth:]))
                nf, nb = nf + f, nb + b
            got.append(f"{pat} ({len(hits)} files)")

        missing.extend(absent)
        print(f"{e['slug']:<32} {nf:>5} files  {human(nb):>9}"
              + (f"   MISSING: {', '.join(absent)}" if absent else ""))

        lines += [
            f"## `{e['section']}/{e['slug']}` -- {e['title']}",
            "",
            e["note"],
            "",
            "| copied from |",
            "|---|",
            *[f"| `{g}` |" for g in got],
        ]
        if absent:
            lines += ["", "**Not found at build time:** "
                      + ", ".join(f"`{a}`" for a in absent)]
        lines += [""]

    # Shared code and datasets
    for src, dst in CODE:
        s = root / src
        if s.exists():
            f, b = copy(s, args.out / dst)
            print(f"{dst:<32} {f:>5} files  {human(b):>9}")
        else:
            missing.append(src)
    for d in DATASETS:
        s = root / d
        if s.exists():
            copy(s, args.out / "datasets" / Path(d).name)
        else:
            missing.append(d)
    for pat, sub in DATASET_GLOBS:
        hits = sorted(root.glob(pat))
        for h in hits:
            copy(h, args.out / "datasets" / sub / h.name)
        if not hits:
            missing.append(pat)
        else:
            print(f"{'datasets/' + sub:<32} {len(hits):>5} files")
    for src, dst in GENERATORS:
        sp = root / src
        if sp.exists():
            f, b = copy(sp, args.out / dst)
            print(f"{dst:<32} {f:>5} files  {human(b):>9}")
        else:
            missing.append(src)
    (args.out / "datasets").mkdir(parents=True, exist_ok=True)
    (args.out / "datasets" / "GENERATORS.md").write_text(GENERATORS_MD)
    for a in ANALYSIS:
        s = root / a
        if s.exists():
            copy(s, args.out / "analysis" / Path(a).name)
        else:
            missing.append(a)

    lines += [
        "## `code/`",
        "",
        "Copied wholesale rather than traced per result. An import trace would",
        "be more precise and would also be one more thing that can be quietly",
        "wrong; the trees here are small.",
        "",
        "| tree | from |",
        "|---|---|",
        *[f"| `{dst}` | `{src}` |" for src, dst in CODE],
        "",
    ]

    (args.out / "README.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}/README.md")
    if missing:
        print("\nmissing at build time (entries were skipped, not failed):")
        for m in sorted(set(missing)):
            print(f"  {m}")


if __name__ == "__main__":
    main()
