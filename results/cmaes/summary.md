# Loss Tierlist (CMA-ES):

* S: l1_stft
* A: ESR(i.e. waveform L2, essentially; used in ESRGAN), complexstft
* A-: GroupDelay
* B: rest
* F: Modal Density (zero does not always correspond to the correct configuration), OnsetDisp (is always zero)

We'd still like to know:
* When (and why) does the S- and A-tiers fail? Are the failures random, or do they correlate with each other? -> retry l1 stft and see which fail this time
* Should I (still) try a combination of these? -> ???
