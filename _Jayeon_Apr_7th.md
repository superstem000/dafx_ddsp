# Usage

Compare: for comparing the implementation of this plate with others' to see there is no error.
`python compare_plate_v5.py`
`python compare_plate_v2.py`

Sweeps: they sweep. Outputs to `sweep`
`python sweep_rerereparam_2x.py --loss sc --alpha 0 --device cuda --seed 2`
`python sweep_rerereparam_3x.py --loss sc --alpha 0 --device cuda --seed 2` <- this won't be run because there is way too much to run

fit: try to use Adam optimizer to fit the params. Outputs to a folder you give as an argument. Just run `fit_plate_rerereparam.sh`. GPU preferred

## Careful,

Currently `diff_plate_rerereparam.py` does not use `velCalc=True`. I have a one-liner there, commented out, which makes it do that. Before you run the plate, please confirm you are using it with the right setting.

For the `compare_plate_v5.py` we're using `False`. For the fitting I'm using `True`