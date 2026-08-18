import torch
import torch.nn as nn
from diffsynth.spectral import multiscale_fft, compute_loudness
from diffsynth.util import log_eps
import torch.nn.functional as F
import functools

def spectrogram_loss(x_audio, target_audio, fft_sizes=[64, 128, 256, 512, 1024, 2048], hop_ls=None, win_ls=None, log_mag_w=0.0, mag_w=1.0, norm=None, power=2, log_eps_v=1e-4, gamma=None):
    """`gamma`, when set, compresses the POWER spectrogram as (p + eps)^gamma
    before the L1, in place of the plain difference the mag_w term otherwise
    takes. It requires power=2, because the exponent is defined on intensity.

    One knob that subsumes `power` for the linear term and extends past both
    ends of it:

        gamma = 1.0   the published loss, L1 on power
        gamma = 0.5   exactly the magnitude loss, since magnitude = power^0.5
        gamma = 0.3   Stevens' law, loudness ~ I^0.3
        gamma -> 0    the log term

    Why it exists. The evaluation metrics -- MFCC, LSD -- all measure at
    gamma -> 0, far past where hearing sits, and an arm trained with a log term
    is optimised for exactly that domain. Having established that, the obvious
    question is why not train at 0.3 too, and this is what answers it with a run
    instead of a paragraph.

    eps is log_eps_v, deliberately the SAME epsilon the log term uses. Two
    reasons. It is needed at all because d(p^gamma)/dp = gamma*p^(gamma-1)
    diverges at p -> 0 for gamma < 1 -- the mirror image of the power-domain
    dead zone, an exploding gradient at silence rather than a vanishing one.
    And sharing it makes gamma=0.3 against log a one-variable comparison: same
    knee at the same signal level, different compression above it. At 1e-4 the
    slope at silence is 0.3*(1e-4)^-0.7 ~ 190, against the log term's 1/1e-4 =
    1e4, so this is the tamer of the two.
    """
    if gamma is not None and power != 2:
        raise ValueError(
            f"gamma={gamma} needs power=2: the exponent is defined on the "
            f"power spectrogram, and power={power} would make it "
            f"p^{gamma / 2:g} instead.")
    x_specs = multiscale_fft(x_audio, fft_sizes, hop_ls, win_ls, power)
    target_specs = multiscale_fft(target_audio, fft_sizes, hop_ls, win_ls, power)
    loss = 0.0
    spec_loss = {}
    log_spec_loss = {}
    for n_fft, x_spec, target_spec in zip(fft_sizes, x_specs, target_specs):
        spec_norm = norm['spec'][n_fft] if norm is not None else 1.0
        log_spec_norm = norm['logspec'][n_fft] if norm is not None else 1.0
        if mag_w > 0:
            if gamma is None:
                xs, ts = x_spec, target_spec
            else:
                xs = (x_spec + log_eps_v) ** gamma
                ts = (target_spec + log_eps_v) ** gamma
            spec_loss[n_fft] = mag_w * torch.mean(torch.abs(xs - ts)) / spec_norm
        if log_mag_w > 0:
            log_spec_loss[n_fft] = log_mag_w * torch.mean(torch.abs(log_eps(x_spec, log_eps_v) - log_eps(target_spec, log_eps_v))) / log_spec_norm
    return {'spec':spec_loss, 'logspec':log_spec_loss}

def waveform_loss(x_audio, target_audio, l1_w=0, l2_w=1.0, linf_w=0, linf_k=1024, norm=None):
    norm = {'l1':1.0, 'l2':1.0} if norm is None else norm
    l1_loss = l1_w * torch.mean(torch.abs(x_audio - target_audio)) / norm['l1'] if l1_w > 0 else 0.0
    # mse loss
    l2_loss = l2_w * torch.mean((x_audio - target_audio)**2) / norm['l2'] if l2_w > 0 else 0.0
    if linf_w > 0:
        # actually gets k elements
        residual = (x_audio - target_audio)**2
        values, _ = torch.topk(residual, linf_k, dim=-1)
        linf_loss = torch.mean(values) / norm['l2']
    else:
        linf_loss = 0.0
    return {'l1':l1_loss, 'l2':l2_loss, 'linf':linf_loss}

class SpecWaveLoss():
    """
    loss for reconstruction with multiscale spectrogram loss and waveform loss
    """
    def __init__(self, fft_sizes=[64, 128, 256, 512, 1024, 2048], hop_lengths=None, win_lengths=None, mag_w=1.0, log_mag_w=1.0, l1_w=0, l2_w=0.0, linf_w=0.0, linf_k=1024, norm=None, power=2, log_eps_v=1e-4, gamma=None):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_lengths = hop_lengths
        self.win_lengths = win_lengths
        self.mag_w = mag_w
        self.log_mag_w = log_mag_w
        self.l1_w=l1_w
        self.l2_w=l2_w
        self.linf_w=linf_w
        self.spec_loss = functools.partial(spectrogram_loss, fft_sizes=fft_sizes, hop_ls=hop_lengths, win_ls=win_lengths, log_mag_w=log_mag_w, mag_w=mag_w, norm=norm, power=power, log_eps_v=log_eps_v, gamma=gamma)
        self.wave_loss = functools.partial(waveform_loss, l1_w=l1_w, l2_w=l2_w, linf_w=linf_w, linf_k=linf_k, norm=norm)
        
    def __call__(self, x_audio, target_audio, log_mag_w=None):
        # log_mag_w overrides the constructed weight for this call, so the
        # linear/log balance can be SCHEDULED during training the way param_w
        # and sw_w already are. None keeps the constructed value, which is what
        # every run before this did, so nothing existing changes.
        #
        # Worth noting that the normalisation below divides by (mag_w + lmw),
        # i.e. the loss is a weighted AVERAGE of the two halves rather than a
        # sum. That is what makes a crossfade safe: ramping lmw from 0 to 1
        # moves from pure linear to a 50/50 average with no jump in scale, so
        # the optimiser sees the balance change and not a step in magnitude.
        #
        # self.spec_loss is a functools.partial that already binds log_mag_w;
        # passing it again as a keyword overrides the bound value.
        lmw = self.log_mag_w if log_mag_w is None else log_mag_w
        if (self.mag_w + lmw) > 0:
            spec_losses = self.spec_loss(x_audio, target_audio, log_mag_w=lmw)
            multi_spec_loss = sum(spec_losses['spec'].values()) + sum(spec_losses['logspec'].values())
            multi_spec_loss /= (len(self.fft_sizes)*(self.mag_w + lmw))
        else: # no spec loss
            multi_spec_loss = torch.tensor([0.0], device=x_audio.device)
        if (self.l1_w + self.l2_w + self.linf_w) > 0:
            wave_losses = self.wave_loss(x_audio, target_audio)
            waveform_loss = wave_losses['l1'] + wave_losses['l2'] + wave_losses['linf']
            waveform_loss /= (self.l1_w + self.l2_w + self.linf_w)
        else: # no waveform loss
            waveform_loss = torch.tensor([0.0], device=x_audio.device)
        return multi_spec_loss, waveform_loss

def calculate_norm(loader, fft_sizes, hop_ls, win_ls):
    """
    calculate stats for scaling losses
    based on jukebox
    doesn't really work
    """
    n, spec_n = 0, 0
    spec_total = {n_fft: 0.0 for n_fft in fft_sizes}
    log_spec_total = {n_fft: 0.0 for n_fft in fft_sizes}
    total, total_sq, l1_total = 0.0, 0.0, 0.0
    print('calculating bandwidth')
    for data_dict in loader:
        x_audio = data_dict['audio']
        total = torch.sum(x_audio)
        total_sq = torch.sum(x_audio**2)
        l1_total = torch.sum(torch.abs(x_audio))
        x_specs = multiscale_fft(x_audio, fft_sizes, hop_ls, win_ls)
        for n_fft, spec in zip(fft_sizes, x_specs): 
            # spec: power spectrogram [batch_size, n_bins, time]
            spec_total[n_fft] += torch.mean(spec)
            # probably not right
            log_spec_total[n_fft] += torch.mean(torch.abs(log_eps(spec)))
        n += x_audio.shape[0] * x_audio.shape[1]
        spec_n += 1
    
    print('done.')
    mean = total / n
    for n_fft in fft_sizes:
        spec_total[n_fft] /= spec_n
        log_spec_total[n_fft] /= spec_n

    return {'l2': total_sq/n - mean**2, 'l1': l1_total/n, 'spec': spec_total, 'logspec': log_spec_total}
