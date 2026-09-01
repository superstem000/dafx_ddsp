import numpy as np
import torch
from diffsynth.processor import Processor

def soft_clamp_min(x, min_v, T=100):
    return torch.sigmoid((min_v-x)*T)*(min_v-x)+x

class ADSREnvelope(Processor):
    def __init__(self, n_frames=250, name='env', min_value=0.0, max_value=1.0, channels=1, noise_mode='add', delay_range=(0.0, 0.25)):
        """noise_mode decides whether a clip can ever fall silent.

        'add' is the published behaviour and the default, so every dataset
        generated before this argument existed is still reproducible: the
        noise is added to the envelope OUTSIDE the A+D+S sum, so once the
        release has finished and A+D+S is exactly 0, the envelope still sits
        at randn*noise_mag. noise_mag is drawn on (0, 0.1), which puts that
        floor between -20 and -60 dB below the clip's own peak with a median
        of -26. Measured consequence: across 400 clips of the h2of set the
        fraction of the window within 60 dB of peak spans 0.958 to 0.998. The
        occupancy of the window is pinned near 1.0 and sus_level, drawn
        uniform on [0,1], never gets to matter.

        'mul' scales the noise BY the envelope instead, so the jitter during
        the note is preserved at the same relative depth while silence stays
        silence. That is the difference between varying note length and not:
        with 'add', shortening a note leaves a -26 dB bed for the rest of the
        window and the clip is still fully occupied.

        Not folded into 'add' as a fix. The two differ in the character of
        the noise during the note as well as at its end, so a dataset that
        changed both at once would confound note length with jitter -- and
        jitter is low-level structure, which is the exact thing the loss
        comparison is about.
        """
        super().__init__(name=name)
        self.n_frames = int(n_frames)
        self.param_names = ['total_level', 'attack', 'decay', 'sus_level', 'release']
        self.min_value = min_value
        self.max_value = max_value
        self.channels = channels
        assert noise_mode in ('add', 'mul'), noise_mode
        self.noise_mode = noise_mode
        self.delay_range = tuple(delay_range)
        self.param_desc = {
                'floor':        {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'}, 
                'peak':         {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'}, 
                'attack':       {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'},
                'decay':        {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'},
                'sus_level':    {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'},
                'release':      {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'},
                'noise_mag':    {'size':self.channels, 'range': (0, 0.1), 'type': 'sigmoid'},
                'note_off':     {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'},
                # ONSET TIME, absent from the published envelope, which always
                # begins rising at t=0. A sampled instrument note does not: the
                # sampler trims near the transient, not exactly on it, so a
                # one-shot carries a short lead-in of silence. Measured on a
                # Juno-6 saw-bass sample, the onset sits about 13% into a 1.31 s
                # file. Unconnected in every config that predates it, and
                # synthesizer.py:43 builds ext_params from CONNECTIONS, so a
                # param_desc entry nothing connects is never sampled and those
                # datasets are bit-identical.
                'delay':        {'size':self.channels, 'range': self.delay_range, 'type': 'sigmoid'},
                }

    def forward(self, floor, peak, attack, decay, sus_level, release, noise_mag=0.0, note_off=0.8, delay=0.0, n_frames=None):
        """generate envelopes from parameters

        Args:
            floor (torch.Tensor): floor level of the signal 0~1, 0=min_value (batch, 1, channels)
            peak (torch.Tensor): peak level of the signal 0~1, 1=max_value (batch, 1, channels)
            attack (torch.Tensor): relative attack point 0~1 (batch, 1, channels)
            decay (torch.Tensor): actual decay point is attack+decay (batch, 1, channels)
            sus_level (torch.Tensor): sustain level 0~1 (batch, 1, channels)
            release (torch.Tensor): release point is attack+decay+release (batch, 1, channels)
            note_off (float or torch.Tensor, optional): note off position. Defaults to 0.8.
            n_frames (int, optional): number of frames. Defaults to None.

        Returns:
            torch.Tensor: envelope signal (batch_size, n_frames, 1)
        """
        torch.clamp(floor, min=0, max=1)
        torch.clamp(peak, min=0, max=1)
        torch.clamp(attack, min=0, max=1)
        torch.clamp(decay, min=0, max=1)
        torch.clamp(sus_level, min=0, max=1)
        torch.clamp(release, min=0, max=1)

        batch_size = attack.shape[0]
        if n_frames is None:
            n_frames = self.n_frames
        # batch, n_frames, 1
        x = torch.linspace(0, 1.0, n_frames)[None, :, None].repeat(batch_size, 1, self.channels)
        x = x.to(attack.device)
        # The whole envelope shifted right, by clamping the pre-onset region to
        # x=0 rather than by rolling: at x=0, A=0 and D and S are already
        # clamped, so the envelope is exactly its value at time zero, which is
        # silence. delay defaults to 0.0 and a config that does not connect it
        # gets identical output.
        x = torch.clamp(x - delay, min=0.0)
        attack = attack * note_off
        A = x / (attack)
        A = torch.clamp(A, max=1.0)
        D = (x - attack) * (sus_level - 1) / (decay+1e-5)
        D = torch.clamp(D, max=0.0)
        D = soft_clamp_min(D, sus_level-1)
        S = (x - note_off) * (-sus_level / (release+1e-5))
        S = torch.clamp(S, max=0.0)
        S = soft_clamp_min(S, -sus_level)
        peak = peak * self.max_value + (1 - peak) * self.min_value
        floor = floor * self.max_value + (1 - floor) * self.min_value
        env = A + D + S
        noise = torch.randn_like(A) * noise_mag
        # 'add' reproduces the published expression exactly, character included.
        env = env * (1.0 + noise) if self.noise_mode == 'mul' else env + noise
        signal = env*(peak - floor) + floor
        return torch.clamp(signal, min=self.min_value, max=self.max_value)