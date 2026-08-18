"""Record which files landed in which split, so a resume can be checked.

IdOodDataModule.create_split calls torch.utils.data.random_split with no
explicit generator, and the OOD pool is subsampled with np.random.choice. Both
therefore depend on the global RNG state at the moment setup() runs, which
pl.seed_everything(0) fixes -- but only if the number of RNG draws before
setup() is the same in every run. The upstream comment concedes the fragility:
"should be seeded fine but probably better to split test set in some other way".

That matters here because the paper's Synth and Real models are a resume from
the pretrain checkpoint. PyTorch-Lightning restores RNG state from a checkpoint
when resuming. If it does so before the datamodule's setup() hook, the resumed
run draws a DIFFERENT split, and pretrain's validation files become resume's
training files -- leakage across the phase boundary, biased in the direction of
the paper's own result, and invisible in any metric.

This writes the split membership to split_manifest.json in the run directory so
two runs can be compared directly. It is a Callback rather than a patch to the
datamodule so that nothing about the split itself changes.
"""

import hashlib
import json
import os

from pytorch_lightning.callbacks import Callback


def _files(ds):
    """Resolve a (possibly nested) Subset down to the underlying file paths.

    The OOD side is a Subset of a Subset -- np.random.choice narrows the pool to
    the in-domain size, then random_split cuts that into train/valid/test -- so
    unwrapping has to be iterative rather than one deep.
    """
    idx = list(range(len(ds)))
    while hasattr(ds, "indices"):
        idx = [ds.indices[i] for i in idx]
        ds = ds.dataset
    return [os.path.basename(ds.raw_files[i]) for i in idx]


class SplitManifest(Callback):
    def __init__(self, path="split_manifest.json"):
        super().__init__()
        self.path = path

    def on_fit_start(self, trainer, pl_module):
        # on_fit_start rather than the setup hook: setup() must already have run
        # for id_datasets/ood_datasets to exist, and hook ordering between the
        # datamodule and callbacks is a PL implementation detail worth not
        # depending on.
        dm = getattr(trainer, "datamodule", None)
        if dm is None:
            return
        rec = {}
        for domain in ("id", "ood"):
            splits = getattr(dm, f"{domain}_datasets", None)
            if not splits:
                continue
            for name, ds in splits.items():
                names = _files(ds)
                rec[f"{domain}_{name}"] = {
                    "n": len(names),
                    # Hash of the SORTED membership: what matters is which files
                    # are in the split, not the order random_split happened to
                    # emit them in.
                    "sha1": hashlib.sha1("\n".join(sorted(names)).encode()).hexdigest(),
                    "first5": sorted(names)[:5],
                    # The membership itself, not only its hash. A hash can tell
                    # an offline evaluation that it reproduced the wrong split;
                    # it cannot tell it the right one. Reproducing the split
                    # means matching every RNG draw train.py makes before
                    # setup() -- including the estimator's weight init -- which
                    # is a fragile thing for any later script to depend on.
                    # 2000 file names is a few tens of KB.
                    "files": sorted(names),
                }
        with open(self.path, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"split manifest written to {os.path.abspath(self.path)}")
