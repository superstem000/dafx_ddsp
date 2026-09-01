import numpy as np
import torch
from diffsynth.processor import Processor

def soft_clamp_min(x, min_v, T=100):
    return torch.sigmoid((min_v-x)*T)*(min_v-x)+x

class ADSREnvelope(Processor):
    def __init__(self, n_frames=250, name='env', min_value=0.0, max_value=1.0, channels=1, length_range=(1.0, 1.0)):
        """A published-shape envelope placed in a WINDOW of the clip.

        length_range bounds the note's duration as a fraction of the clip.
        (1.0, 1.0) is the default and reproduces the published envelope
        exactly: the note fills the clip and `start` has nowhere to move it.
        """
        super().__init__(name=name)
        self.n_frames = int(n_frames)
        self.param_names = ['total_level', 'attack', 'decay', 'sus_level', 'release']
        self.min_value = min_value
        self.max_value = max_value
        self.channels = channels
        self.length_range = tuple(length_range)
        self.param_desc = {
                'floor':        {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'}, 
                'peak':         {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'}, 
                'attack':       {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'},
                'decay':        {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'},
                'sus_level':    {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'},
                'release':      {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'},
                'noise_mag':    {'size':self.channels, 'range': (0, 0.1), 'type': 'sigmoid'},
                'note_off':     {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'},
                # THE NOTE'S WINDOW WITHIN THE CLIP. The published envelope
                # always spans the whole clip: it begins rising at t=0 and
                # note_off is a fixed fraction of the clip rather than of the
                # note, so nothing could place a note anywhere else or make it
                # any shorter. A sampled instrument note does both -- the
                # sampler trims near the transient rather than on it, and the
                # note lasts as long as it lasts.
                #
                # `length` is the note's duration as a fraction of the clip and
                # `start` positions it, scaled by (1 - length) in forward so the
                # note always fits and is never cut by the clip's end.
                #
                # Unconnected in every config that predates them, and
                # synthesizer.py:43 builds ext_params from CONNECTIONS, so
                # param_desc entries nothing connects are never sampled and
                # those datasets are bit-identical.
                'start':        {'size':self.channels, 'range': (0, 1), 'type': 'sigmoid'},
                'length':       {'size':self.channels, 'range': self.length_range, 'type': 'sigmoid'},
                }

    def forward(self, floor, peak, attack, decay, sus_level, release, noise_mag=0.0, note_off=0.8, start=0.0, length=1.0, n_frames=None):
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
        # THE TIME WARP, and it is the only thing here that is not published.
        # x stops being "position in the clip" and becomes "position in the
        # NOTE": 0 at the note's onset, 1 at its end. Every line below is then
        # the published expression unchanged, so the envelope inside the window
        # is exactly what the old generator produced over the whole clip, on a
        # compressed time axis. At start=0, length=1 the two are identical.
        #
        # start is scaled by (1 - length) so the note always fits: a note of
        # half the clip can begin anywhere in the first half, and none is ever
        # cut short by the clip running out.
        #
        # `inside` gates the result to silence outside the window. Without it,
        # clamping x at 1 would HOLD the envelope's final value for the rest of
        # the clip, and that value is not zero whenever note_off + release > 1
        # -- the published set has the same notes, it simply stops rendering at
        # the clip's end. Gating reproduces that boundary exactly rather than
        # smearing it forward.
        u = (x - start * (1.0 - length)) / length
        inside = ((u >= 0.0) & (u <= 1.0)).to(x.dtype)
        x = torch.clamp(u, 0.0, 1.0)
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
        # The published expression, with the noise ADDED as it always was --
        # gating handles the silence, so the -26 dB bed that made every clip
        # fully occupied stays where it belongs, INSIDE the note, and the sound
        # within the window is unchanged.
        env = (A + D + S + torch.randn_like(A)*noise_mag) * inside
        signal = env*(peak - floor) + floor
        return torch.clamp(signal, min=self.min_value, max=self.max_value)