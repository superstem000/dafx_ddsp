import os

import hydra
import pytorch_lightning as pl
from plot import AudioLogger
from split_manifest import SplitManifest
import warnings
from pytorch_lightning.callbacks import ModelCheckpoint
from diffsynth.model import EstimatorSynth

# version_base="1.1" keeps hydra's pre-1.2 behaviour of chdir-ing into the run
# directory. train.py writes tb_logs and checkpoints to relative paths, so
# without it every run would scatter its outputs into the launch directory.
@hydra.main(config_path="configs/", config_name="config.yaml", version_base="1.1")
def main(cfg):
    pl.seed_everything(0, workers=True)
    warnings.simplefilter('ignore', RuntimeWarning)
    # Hydra chdirs into outputs/<date>/<time>/ before main() runs, which is what
    # keeps each run's tb_logs and checkpoints separate. The side effect is that
    # every relative path in the config -- including one passed on the command
    # line as data.id_dir=data/foo -- resolves against the run directory rather
    # than where the command was typed. The failure mode is quiet: the dataset
    # globs nothing, prints "loaded 0 files", and dies later on a missing
    # param dir. Re-anchor the data paths to the launch directory instead.
    for _k in ('id_dir', 'ood_dir'):
        _v = cfg.data.get(_k, None)
        if _v is not None and not os.path.isabs(_v):
            cfg.data[_k] = os.path.join(hydra.utils.get_original_cwd(), _v)

    model = EstimatorSynth(cfg.model, cfg.synth, cfg.schedule)
    logger = pl.loggers.TensorBoardLogger("tb_logs", "", default_hp_metric=False, version='')
    hparams = {'data': cfg.data.train_type, 'schedule': cfg.schedule.name, 'synth': cfg.synth.name}
    # dummy value
    logger.log_hyperparams(hparams, {'val_id/lsd': 40, 'val_ood/lsd': 40})
    # log audio examples
    checkpoint_callback = ModelCheckpoint(monitor="val_ood/lsd", save_top_k=1, filename="epoch_{epoch:03}_{val_ood/lsd:.2f}", save_last=True, auto_insert_metric_name=False)
    # SplitManifest records which files landed in which split. The train/val
    # split is drawn from the global RNG at setup() time, and Synth and Real are
    # resumes -- so if PL restores RNG state before setup(), a resumed run gets a
    # different split and the pretrain phase's validation data becomes the
    # resume phase's training data. Writing the membership makes that checkable
    # instead of assumed, and costs one small json per run.
    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval='step'), AudioLogger(),
                 SplitManifest(), checkpoint_callback]
    # PL 2 takes the resume path at fit() rather than as a Trainer argument, so
    # it is pulled out of the trainer config here. The config key is kept so the
    # README's `trainer.resume_from_checkpoint=<path>` commands still work.
    trainer_cfg = {k: v for k, v in cfg.trainer.items() if k != 'resume_from_checkpoint'}
    ckpt_path = cfg.trainer.get('resume_from_checkpoint', None)
    trainer = hydra.utils.instantiate(trainer_cfg, callbacks=callbacks, logger=logger)
    datamodule = hydra.utils.instantiate(cfg.data)
    # make model
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()
