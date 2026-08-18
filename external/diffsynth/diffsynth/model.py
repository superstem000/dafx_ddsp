import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import diffsynth.util as util
from diffsynth.spectral import compute_lsd, loudness_loss, Mfcc
import pytorch_lightning as pl
from diffsynth.modelutils import construct_synth_from_conf
from diffsynth.schedules import ParamSchedule
import hydra
from diffsynth.estimator import F0MelEstimator

class ConfigLambdaLR(torch.optim.lr_scheduler.LambdaLR):
    """LambdaLR that can resume from a checkpoint another scheduler wrote.

    The point of this run is to branch the published epoch-200 checkpoints onto
    a different learning-rate schedule, and those checkpoints were written by
    ExponentialLR. LambdaLR.load_state_dict opens with

        lr_lambdas = state_dict.pop("lr_lambdas")

    so restoring one raises KeyError before anything else happens. Supplying the
    missing key -- as None per group, meaning "keep the lambda I was constructed
    with" -- is the whole fix. gamma is dropped because __dict__.update would
    otherwise leave ExponentialLR's decay rate sitting on an object that has no
    use for it, which is the kind of thing that reads as meaningful two months
    later.

    last_epoch and base_lrs are restored as usual, so the branch continues at
    the epoch and base rate it left off at.
    """

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        state_dict.pop('gamma', None)
        state_dict.setdefault('lr_lambdas', [None] * len(self.lr_lambdas))
        super().load_state_dict(state_dict)


class EstimatorSynth(pl.LightningModule):
    """
    audio -> Estimator -> Synth -> audio
    """
    def __init__(self, model_cfg, synth_cfg, sched_cfg):
        super().__init__()
        self.synth = construct_synth_from_conf(synth_cfg)
        self.estimator = hydra.utils.instantiate(model_cfg.estimator, output_dim=self.synth.ext_param_size)
        self.loss_w_sched = ParamSchedule(sched_cfg) # loss weighting
        self.sw_loss = hydra.utils.instantiate(model_cfg.sw_loss) # reconstruction loss
        if model_cfg.perc_model is not None:
            self.perc_model = hydra.utils.instantiate(model_cfg.perc_model)
        else:
            self.perc_model = None
        self.log_grad = model_cfg.log_grad
        self.lr = model_cfg.lr
        self.decay_rate = model_cfg.decay_rate
        # None keeps the published ExponentialLR exactly. See configure_optimizers.
        self.lr_schedule = model_cfg.get('lr_schedule', None)
        self.mfcc = Mfcc(n_fft=1024, hop_length=256, n_mels=40, n_mfcc=20, sample_rate=16000)
        # The same cepstral distance at Stevens' exponent instead of the log.
        # mfcc and lsd both measure at gamma -> 0, which is far past where
        # hearing sits and is also the domain a log-trained arm is optimised
        # for, so ranking arms by them is close to circular. 0.3 is where
        # loudness actually is. Identical in every other respect to self.mfcc,
        # so the pair isolates the compression.
        self.mfcc03 = Mfcc(n_fft=1024, hop_length=256, n_mels=40, n_mfcc=20, sample_rate=16000, gamma=0.3)
        self.save_hyperparameters()

    def param_group_losses(self, synth_output, param_dict):
        """The per-parameter terms param_loss sums over.

        Split out so validation can log them individually. The aggregate hides
        more than it summarises: the six L1s live on different scales, so the
        unweighted mean is roughly half osc_mix and q by magnitude alone while
        f0_hz contributes under 1% of it whatever any arm does, and the groups
        move in opposite directions between arms -- a linear loss recovers
        level and frequency better while a log loss recovers spectral shape
        better, which the mean cancels into a single number showing neither.
        """
        out = {}
        for k, target in param_dict.items():
            output_name = self.synth.dag_summary[k]
            if output_name in self.synth.fixed_param_names:
                continue
            if target.numel() == 0:
                continue
            x = synth_output[output_name]
            if target.shape[1] > 1:
                x = util.resample_frames(x, target.shape[1])
            out[k] = F.l1_loss(x, target)
        return out

    def param_loss(self, synth_output, param_dict):
        # Divided by the number of keys in param_dict, not by the number of
        # terms actually summed -- skipped and empty groups still count in the
        # denominator. Preserved exactly as it was, since every number recorded
        # so far is on that convention.
        losses = self.param_group_losses(synth_output, param_dict)
        loss = sum(losses.values()) if losses else 0.0
        loss = loss / len(param_dict.keys())
        return loss

    def estimate_param(self, conditioning):
        """
        Args:
            conditioning (dict): {'PARAM NAME': Conditioning Tensor, ...}

        Returns:
            torch.Tensor: estimated parameters in Tensor ranged 0~1
        """
        if isinstance(self.estimator, F0MelEstimator):
            return self.estimator(conditioning['audio'], conditioning['f0_hz'])
        return self.estimator(conditioning['audio'])

    def log_param_grad(self, params_dict):
        def save_grad(name):
            def hook(grad):
                # batch, n_frames, feat_size
                grad_v = grad.abs().mean(dim=(0, 1))
                for i, gv in enumerate(grad_v):
                    self.log('train/param_grad/'+name+f'_{i}', gv, on_step=False, on_epoch=True)
            return hook

        if self.log_grad:
            for k, v in params_dict.items():
                if v.requires_grad == True:
                    v.register_hook(save_grad(k))

    def forward(self, conditioning):
        """
        Args:
            conditioning (dict): {'PARAM NAME': Conditioning Tensor, ...}

        Returns:
            torch.Tensor: audio
        """
        audio_length = conditioning['audio'].shape[1]
        est_param = self.estimate_param(conditioning)
        params_dict = self.synth.fill_params(est_param, conditioning)
        if self.log_grad is not None:
            self.log_param_grad(params_dict)

        resyn_audio, outputs = self.synth(params_dict, audio_length)
        return resyn_audio, outputs

    def get_params(self, conditioning):
        """
        Don't render audio
        """
        est_param = self.estimate_param(conditioning)
        params_dict = self.synth.fill_params(est_param, conditioning)
        if self.log_grad is not None:
            self.log_param_grad(params_dict)
        
        synth_params = self.synth.calculate_params(params_dict)
        return synth_params

    def train_losses(self, target, output, loss_w=None, sw_loss=None, perc_model=None):
        sw_loss = self.sw_loss if sw_loss is None else sw_loss
        perc_model = self.perc_model if perc_model is None else perc_model
        # always computes mean across batch dimension
        if loss_w is None:
            loss_w = {'param_w': 1.0, 'sw_w':1.0, 'perc_w':1.0}
        loss_dict = {}
        # parameter L1 loss
        if loss_w['param_w'] > 0.0 and 'params' in target:
            loss_dict['param'] = loss_w['param_w'] * self.param_loss(output, target['params'])
        else:
            loss_dict['param'] = 0.0
        # Audio losses
        target_audio = target['audio']
        resyn_audio = output['output']
        if loss_w['sw_w'] > 0.0 and sw_loss is not None:
            # Reconstruction loss
            # log_mag_w is forwarded only when the schedule carries it.
            # ParamSchedule returns every key in sched_cfg, so a schedule
            # without it yields None here and SpecWaveLoss keeps its
            # constructed weight -- every existing config is unaffected.
            #
            # Validation calls this with loss_w=None, so the reported val
            # spec/wave always use the CONSTRUCTED weight rather than the
            # scheduled one. That is deliberate: it keeps val/spec comparable
            # across epochs while the balance is still moving. The metrics the
            # paper reads -- param, mfcc, lsd -- do not depend on it at all.
            spec_loss, wave_loss = sw_loss(target_audio, resyn_audio,
                                           log_mag_w=loss_w.get('log_mag_w'))
            loss_dict['spec'], loss_dict['wave'] = loss_w['sw_w'] * spec_loss, loss_w['sw_w'] * wave_loss
        else:
            loss_dict['spec'], loss_dict['wave'] = (0, 0)
        if loss_w['perc_w'] > 0.0 and perc_model is not None:
            loss_dict['perc'] = loss_w['perc_w']*perc_model.perceptual_loss(target_audio, resyn_audio)
        else:
            loss_dict['perc'] = 0
        return loss_dict

    def monitor_losses(self, target, output):
        mon_losses = {}
        # Audio losses
        target_audio = target['audio']
        resyn_audio = output['output']
        # losses not used for training
        mon_losses['lsd'] = compute_lsd(target_audio, resyn_audio)
        mon_losses['loud'] = loudness_loss(resyn_audio, target_audio)
        mon_losses['mfcc'] = F.l1_loss(self.mfcc(target_audio), self.mfcc(resyn_audio))
        mon_losses['mfcc03'] = F.l1_loss(self.mfcc03(target_audio), self.mfcc03(resyn_audio))
        return mon_losses

    def training_step(self, batch_dict, batch_idx):
        # get loss weights
        loss_weights = self.loss_w_sched.get_parameters(self.global_step)
        self.log_dict({'lw/'+k: v for k, v in loss_weights.items()}, on_epoch=True, on_step=False)
        if loss_weights['sw_w']+loss_weights['perc_w'] == 0:
            # do not render audio because reconstruction is unnecessary
            synth_params = self.get_params(batch_dict)
            # Parameter loss
            batch_loss = loss_weights['param_w'] * self.param_loss(synth_params, batch_dict['params'])
        else:
            # render audio
            resyn_audio, outputs = self(batch_dict)
            losses = self.train_losses(batch_dict, outputs, loss_weights)
            self.log_dict({'train/'+k: v for k, v in losses.items()}, on_epoch=True, on_step=False)
            batch_loss = sum(losses.values())
        self.log('train/total', batch_loss, prog_bar=True, on_epoch=True, on_step=False)
        return batch_loss

    def validation_step(self, batch_dict, batch_idx, dataloader_idx=0):
        # render audio
        resyn_audio, outputs = self(batch_dict)
        losses = self.train_losses(batch_dict, outputs)
        eval_losses = self.monitor_losses(batch_dict, outputs)
        losses.update(eval_losses)
        prefix = 'val_id/' if dataloader_idx==0 else 'val_ood/'
        losses = {prefix+k: v for k, v in losses.items()}
        self.log_dict(losses, prog_bar=True, on_epoch=True, on_step=False, add_dataloader_idx=False)
        # The breakdown behind val_id/param, as a trajectory rather than only at
        # whatever checkpoints happen to survive. Six extra scalars an epoch.
        # In-domain only: the OOD loader is NSynth, which has no ground-truth
        # parameters, so batch_dict carries no 'params' there.
        if dataloader_idx == 0 and 'params' in batch_dict:
            groups = self.param_group_losses(outputs, batch_dict['params'])
            self.log_dict({'val_id/param_group/'+k: v for k, v in groups.items()},
                          on_epoch=True, on_step=False, add_dataloader_idx=False)
        return losses

    # get_progress_bar_dict was removed in PL 2 (it is now
    # get_metrics on the progress bar callback). It only hid two keys from the
    # progress bar, so dropping it changes display and nothing else.

    def test_step(self, batch_dict, batch_idx, dataloader_idx=0):
        # render audio
        resyn_audio, outputs = self(batch_dict)
        losses = self.train_losses(batch_dict, outputs)
        eval_losses = self.monitor_losses(batch_dict, outputs)
        losses.update(eval_losses)
        prefix = 'val_id/' if dataloader_idx==0 else 'val_ood/'
        losses = {prefix+k: v for k, v in losses.items()}
        self.log_dict(losses, prog_bar=True, on_epoch=True, on_step=False, add_dataloader_idx=False)
        return losses

    @staticmethod
    def make_lr_lambda(cfg):
        """LambdaLR multiplier on lr, as an explicit function of the epoch.

        Two shapes beyond the published ExponentialLR, both expressed relative
        to it so that everything up to `hold_from` is bit-identical to the
        original schedule and a run can branch off a checkpoint without a step
        change in learning rate:

          hold     decay^min(e, hold_from) -- the published decay to hold_from,
                   then constant. A geometric schedule is a convergent series,
                   so it caps total travel at lr0/(-ln gamma) whatever the epoch
                   count; holding removes the cap, which is what "train until it
                   plateaus" requires.

          cosine   hold, then a cosine ramp from anneal_from to anneal_to down
                   to eta_min_factor of the held rate.

        Deliberately LambdaLR rather than ExponentialLR or CosineAnnealingLR.
        Their state_dicts carry gamma / T_max, and load_state_dict does
        __dict__.update, so a resumed run silently takes the schedule from the
        checkpoint and ignores the config -- which makes changing the schedule
        at a resume impossible without editing checkpoints. LambdaLR does not
        store a plain-function lambda, so the config wins.
        """
        kind = cfg['type']
        decay = float(cfg.get('decay_rate', 0.99))
        hold_from = int(cfg.get('hold_from', 200))
        if kind == 'hold':
            return lambda e: decay ** min(e, hold_from)
        if kind == 'cosine':
            t0, t1 = int(cfg['anneal_from']), int(cfg['anneal_to'])
            floor = float(cfg.get('eta_min_factor', 0.0))
            def f(e):
                base = decay ** min(e, hold_from)
                if e <= t0:
                    return base
                t = min((e - t0) / max(t1 - t0, 1), 1.0)
                return base * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t)))
            return f
        raise ValueError(f"unknown lr_schedule type {kind!r}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.estimator.parameters(), self.lr)
        if self.lr_schedule is None:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.decay_rate)
        else:
            scheduler = ConfigLambdaLR(
                optimizer, self.make_lr_lambda(self.lr_schedule))
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler
            }
        }