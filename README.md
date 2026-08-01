# DAFx Challenge TODO-list

If you are currently working on something, claim it as yours so that nobody does redundant work.

Also do feel free to write more items. 

## Essentials

- [ ] Put together a pipeline to estimate "mu" after an initial CMA-ES run
  - [x] Only thing "μ" affects (if "μ" is regarded independent from D/μ or T₀/μ) is amplitude. It's just a scaling variable with one optimum, which can probably be optimized with a ternary search.
    - [B] Tried ternary search, results: drops mean NMSE from 4.6e-3 to ~1e-8.
    - [x] Ask organizers if the ternary search strategy is acceptable -> Michele says yes
- [x] ~~Implement "early-stopping" to save time~~ -> now takes 1/4 time. `results/cmaes_norm_es` compared to `results/cmaes/l1_stft`
- [x] Submit report (mid-may)
- [ ] Submit entry (end of May)
  - [x] format? -> 2 to 4 pages, "loosely"
  - [x] prepare overleaf

## Less Essential

- [ ] Do some configuration optimization
  - [x] ~~e.g. μ could be excluded from current runs, might make optimization a bit faster~~ -> no easy way to do this in the seven-param setting
  - [x] ~~try the *five* param config D/μ, T₀/μ, Ly, xo, yo ? <- I think one of us has the answer to this~~ -> worse, see `results/cmaes/incremental_ablation/3_5_norm`
- [ ] Do some more hyperparameter optimization
  - [J] loss: STFT size, hope size, segment length and location, log-less losses, plain FFT loss,
    - tried with no avail. Anyone else up to try?
  - [x] dtype: ~~float64? float32? float16? bfloat16? Are the latter two even viable? Perhaps float64 as a stage-two after float16?~~ -> float16 immediately leads to NaN. float64 is better but time taken is 3x
  - [ ] Optuna: pruning speed/rate?
  - [ ] CMA-ES hyperparam? (read the below with page 29 of [https://arxiv.org/pdf/1604.00772](https://arxiv.org/pdf/1604.00772) )
     - [x] population size?
        - for a constant budget (= population x gens), runtime is (very) roughly proportional to sqrt population.
        - Sometimes having too many particles hinder search (e.g. IRs 3 and 12). It might be more economic to have moderate (~50) number of particles and do many restarts.
        - See `results/diagnostics/success_rate`
     - [x] sigma-0 (tutorial by the inventor thinks we should be using about 30% of the whole search space or else this is going to take too long, but my script is using sigma0 of 1.0)? ~~**He also says *search spaces for variables should not differ by orders of magnitude, but the code I wrote with Codex is sort of doing that*, in contrast to the one Brian wrote in `cmaes_customloss/fit_cmaes_optuna_hyperband.py`. Sorry, feel free to fix that part and how it goes.**~~ no discernable difference, but this seems to be the "more-principled" option, so we'll use this one. 
  - [ ] Can "monotonicity" be used to justify or guide many of the choices above?
  
# ICASSP TODO-list

- [ ] Justify "monotonicity". 
  - [ ] unsure if you can claim this is a scheme "derived" from Fitness-Distance Correlation (Jones & Forrest). It looks more original to me tbh
- [ ] So how the hell are we going to extend this to be "scientifically meaningful"?
- [ ] bayesian search as a baseline?
- [ ] neural network (IR -> param) as a baseline?
  - [ ] Task?
  - [ ] in conjunction with neural networks or neural proxies (of DSP modules)?
