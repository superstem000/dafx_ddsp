import sys, torch
DM = '/home/user/dafx_ddsp/external/diffmoog/src'
DS = '/home/user/dafx_ddsp/external/diffsynth'
sys.path.insert(0, DS); sys.path.insert(0, DM)

from diffsynth.loss import spectrogram_loss          # diffsynth
from model.loss.spectral_loss import SpectralLoss    # diffmoog
from model.loss.spectral_loss_presets import loss_presets
from synth.synth_constants import SynthConstants

torch.manual_seed(0)
a = torch.randn(4, 64000) * 0.1
b = torch.randn(4, 64000) * 0.1
FFT = [64, 128, 256, 512, 1024, 2048]
sc = SynthConstants()

# diffsynth: SpecWaveLoss with mag_w = log_mag_w = 1.0, then /(6*2)
d = spectrogram_loss(a, b, fft_sizes=FFT, log_mag_w=1.0, mag_w=1.0)
ds_total = (sum(d['spec'].values()) + sum(d['logspec'].values())) / (len(FFT) * 2.0)
ds_mag = sum(d['spec'].values()) / (len(FFT) * 2.0)
ds_log = sum(d['logspec'].values()) / (len(FFT) * 2.0)

for name, ref in (('diffsynth_mss', ds_total), ('diffsynth_mss_mag', ds_mag), ('diffsynth_mss_log', ds_log)):
    L = SpectralLoss(loss_presets[name], sc, device='cpu')
    got, _, _ = L.call(a, b, step=10_000)
    rel = abs(got.item() - ref.item()) / max(abs(ref.item()), 1e-12)
    print(f'{name:<20} diffmoog {got.item():.10e}   diffsynth {ref.item():.10e}   rel {rel:.3e}  {"OK" if rel < 1e-5 else "MISMATCH"}')

# the split must reconstitute the whole
L = SpectralLoss(loss_presets['diffsynth_mss'], sc, device='cpu')
w, _, _ = L.call(a, b, step=10_000)
m, _, _ = SpectralLoss(loss_presets['diffsynth_mss_mag'], sc, device='cpu').call(a, b, step=10_000)
g, _, _ = SpectralLoss(loss_presets['diffsynth_mss_log'], sc, device='cpu').call(a, b, step=10_000)
print(f'\nhalves sum to whole: {(m+g).item():.10e} vs {w.item():.10e}  rel {abs((m+g-w).item())/w.item():.3e}')

# gradients must flow to the audio in every arm
for name in ('diffsynth_mss', 'diffsynth_mss_mag', 'diffsynth_mss_log'):
    p = b.clone().requires_grad_(True)
    l, _, _ = SpectralLoss(loss_presets[name], sc, device='cpu').call(a, p, step=10_000)
    l.backward()
    print(f'{name:<20} grad norm {p.grad.norm().item():.4e}  finite {torch.isfinite(p.grad).all().item()}')
