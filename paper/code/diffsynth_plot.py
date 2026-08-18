import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
try:  # PL >= 2.0
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except ImportError:  # PL 1.x, which this repo was written against
    from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.callbacks import Callback

def plot_spec(y, ax, sr=16000):
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.label_outer()

def plot_recons(x, x_tilde, plot_dir, name=None, epochs=None, sr=16000, num=6, save=True):
    """Plot spectrograms/waveforms of original/reconstructed audio

    Args:
        x (numpy array): [batch, n_samples]
        x_tilde (numpy array): [batch, n_samples]
        sr (int, optional): sample rate. Defaults to 16000.
        dir (str): plot directory.
        name (str, optional): file name.
        epochs (int, optional): no. of epochs.
        num (int, optional): number of spectrograms to plot. Defaults to 6.
    """
    fig, axes = plt.subplots(num, 4, figsize=(15, 30))
    for i in range(num):
        plot_spec(x[i], axes[i, 0], sr)
        plot_spec(x_tilde[i], axes[i, 1], sr)
        axes[i, 2].plot(x[i])
        axes[i, 2].set_ylim(-1,1)
        axes[i, 3].plot(x_tilde[i])
        axes[i, 3].set_ylim(-1,1)
    if save:
        if epochs:
            fig.savefig(os.path.join(plot_dir, 'epoch{:0>3}_recons.png'.format(epochs)))
            plt.close(fig)
        else:
            fig.savefig(os.path.join(plot_dir, name+'.png'))
            plt.close(fig)
    else:
        return fig

def save_to_board(i, name, writer, orig_audio, resyn_audio, plot_num=4, sr=16000):
    orig_audio = orig_audio.detach().cpu()
    resyn_audio = resyn_audio.detach().cpu()
    for j in range(plot_num):
        writer.add_audio('{0}_orig/{1}'.format(name, j), orig_audio[j].unsqueeze(0), i, sample_rate=sr)
        writer.add_audio('{0}_resyn/{1}'.format(name, j), resyn_audio[j].unsqueeze(0), i, sample_rate=sr)
    fig = plot_recons(orig_audio.detach().cpu().numpy(), resyn_audio.detach().cpu().numpy(), '', sr=sr, num=plot_num, save=False)
    writer.add_figure('plot_recon_{0}'.format(name), fig, i)

class AudioLogger(Callback):
    """Log example audio and spectrogram figures, occasionally.

    epoch_frequency exists because batch_frequency alone does not bound this.
    There are 250 train batches per epoch, so `batch_idx % 1000 == 0` is only
    ever true at batch 0 -- i.e. it fires EVERY epoch, three times over (train,
    val_id, val_ood), each writing 8 audio clips and a 15x30in figure. Over 400
    epochs that is gigabytes of event file, and it makes reading the scalars
    back out take minutes.

    Every 25th epoch keeps the qualitative check without that cost. This is
    logging only; nothing about training changes.
    """

    def __init__(self, batch_frequency=1000, epoch_frequency=25):
        super().__init__()
        self.batch_freq = batch_frequency
        self.epoch_freq = epoch_frequency

    @rank_zero_only
    def log_local(self, writer, name, current_epoch, orig_audio, resyn_audio):
        save_to_board(current_epoch, name, writer, orig_audio, resyn_audio)

    def log_audio(self, pl_module, batch, batch_idx, name="train"):
        if self.epoch_freq and pl_module.current_epoch % self.epoch_freq:
            return
        if batch_idx % self.batch_freq == 0:
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            # get audio
            with torch.no_grad():
                resyn_audio, _outputs = pl_module(batch)
            resyn_audio = torch.clamp(resyn_audio.detach().cpu(), -1, 1)
            orig_audio = torch.clamp(batch['audio'].detach().cpu(), -1, 1)
            
            self.log_local(pl_module.logger.experiment, name, pl_module.current_epoch, orig_audio, resyn_audio)

            if is_train:
                pl_module.train()

    # PL 2 dropped dataloader_idx from the train hook and made it optional on
    # the validation hook. Defaults keep this callable under both.
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_audio(pl_module, batch, batch_idx, name="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        data_type='id' if dataloader_idx==0 else 'ood'
        self.log_audio(pl_module, batch, batch_idx, name="val_"+data_type)

def save_to_board_mel(i, writer, orig_mel, recon_mel, plot_num=8):
    orig_mel = orig_mel.detach().cpu()
    recon_mel = recon_mel.detach().cpu()

    fig, axes = plt.subplots(2, plot_num, figsize=(30, 8))
    for j in range(plot_num):
        axes[0, j].imshow(orig_mel[j], aspect=0.25)
        axes[1, j].imshow(recon_mel[j], aspect=0.25)
    fig.tight_layout()
    writer.add_figure('plot_recon', fig, i)

def plot_param_dist(param_stats):
    """
    violin plot of parameter values
    """

    fig, ax = plt.subplots(figsize=(15, 5))
    labels = param_stats.keys()
    parts = ax.violinplot(param_stats.values(), showmeans=True)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_ylim(0, 1)
    return fig