import torch
import soundfile as sf
import tqdm
import argparse, os
import diffsynth.util as util
from diffsynth.modelutils import construct_synth_from_conf
from omegaconf import OmegaConf

def unit_of(value, desc):
    """Natural units -> the 0..1 draw that produces them, inverting param_desc.

    Synthesizer.uniform draws every external parameter as rand(0, 1) and each
    processor scales it by its own SCALE_FNS entry, so a config that wants to
    restrict a parameter to specific REAL values (MULT of exactly 2.0) has to
    say which draws those correspond to. Inverting here rather than storing
    normalised numbers in the yaml keeps the config readable and keeps it
    correct if a range ever moves.
    """
    lo, hi = desc['range']
    if desc['type'] == 'sigmoid':      # x*(high-low) + low
        return (value - lo) / (hi - lo)
    if desc['type'] == 'freq_sigmoid': # linear in MIDI, not in Hz
        m_lo, m_hi = util.hz_to_midi(lo), util.hz_to_midi(hi)
        return (util.hz_to_midi(value) - m_lo) / (m_hi - m_lo)
    raise ValueError("cannot invert scale type '{0}'".format(desc['type']))

def draw_slots(synth, quant_conf, range_conf):
    """[(key, offset, size, kind, payload)] into the ext parameter tensor.

    kind 'set'   payload is a tensor of allowed 0..1 draws; one is picked
                 uniformly per clip (quantize_params)
    kind 'range' payload is (lo, hi) in 0..1; the draw is uniform BETWEEN them
                 (range_params)

    fill_params walks ext_param_sizes in insertion order to slice the tensor,
    so walking it the same way here gives offsets that are correct by
    construction rather than by a hardcoded index.

    A parameter of size > 1 gets an independent draw per channel, which is what
    a per-oscillator quantity like M_OSC would want.

    WHY range_params EXISTS SEPARATELY FROM NARROWING THE PROCESSOR. MULT's
    (1, 8) lives in harmor.py's param_desc, and that range is the model's
    parameter space as much as the data's: every checkpoint's MULT head maps
    its output through it. Narrowing it there would silently reinterpret every
    existing checkpoint. Restricting the DRAW leaves harmor, ext_param_size,
    the heads and the normalised targets all untouched, so the same model
    config and the same commands work against the new data.
    """
    desc_of = {}
    for processor, connections in synth.dag:
        for input_name, key in connections.items():
            if input_name in processor.param_desc:
                desc_of[key] = processor.param_desc[input_name]
    for name, conf in (('quantize_params', quant_conf),
                       ('range_params', range_conf)):
        unknown = [k for k in conf if k not in synth.ext_param_sizes]
        if unknown:
            raise ValueError('{0} names {1}, which is not an external '
                             'parameter of this synth. Available: {2}'.format(
                                 name, unknown,
                                 list(synth.ext_param_sizes.keys())))
    both = sorted(set(quant_conf) & set(range_conf))
    if both:
        raise ValueError('{0} appears in both quantize_params and '
                         'range_params; a parameter is drawn one way or the '
                         'other'.format(both))
    slots, offset = [], 0
    for key, size in synth.ext_param_sizes.items():
        if key in quant_conf:
            units = [unit_of(float(v), desc_of[key]) for v in quant_conf[key]]
            slots.append((key, offset, size, 'set', torch.tensor(units)))
        elif key in range_conf:
            vals = list(range_conf[key])
            if len(vals) != 2:
                raise ValueError('range_params[{0}] must be [low, high], got '
                                 '{1}'.format(key, vals))
            lo, hi = (unit_of(float(v), desc_of[key]) for v in vals)
            if not 0.0 <= lo < hi <= 1.0:
                raise ValueError(
                    'range_params[{0}] = {1} maps to 0..1 draws {2:.4f}..{3:.4f},'
                    ' which is not an increasing interval inside the '
                    "parameter's own range {4}".format(
                        key, vals, lo, hi, desc_of[key]['range']))
            slots.append((key, offset, size, 'range', (lo, hi)))
        offset += size
    return slots

def draw_batch(synth, batch_size, n_samples, device, slots):
    """synth.uniform, with the restricted parameters redrawn.

    A 'set' parameter picks one listed value uniformly. Snapping a uniform draw
    to the nearest listed value would instead weight them by the width of their
    nearest-neighbour cells, which for [1, 2, 4] out of (1, 8) is 7/21/71 per
    cent rather than the intended thirds.

    A 'range' parameter is drawn uniform on the sub-interval, in the SAME 0..1
    space the parameter already uses -- so for a 'sigmoid' scale like MULT the
    result is uniform in the natural units too, and for 'freq_sigmoid' it stays
    uniform in MIDI exactly as the unrestricted draw was.
    """
    if not slots:
        return synth.uniform(batch_size, n_samples, device)
    params = torch.rand(batch_size, 1, synth.ext_param_size).to(device)
    for _key, offset, size, kind, payload in slots:
        if kind == 'set':
            units = payload.to(device)
            pick = torch.randint(len(units), (batch_size, 1, size), device=device)
            params[:, :, offset:offset+size] = units[pick]
        else:
            lo, hi = payload
            params[:, :, offset:offset+size] = (
                torch.rand(batch_size, 1, size, device=device) * (hi - lo) + lo)
    dag_input = synth.fill_params(params)
    return synth(dag_input, n_samples)

def make_dirs(base_dir, synth_name):
    dat_dir = os.path.join(base_dir, synth_name)
    audio_dir = os.path.join(dat_dir, 'audio')
    param_dir = os.path.join(dat_dir, 'param')
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(param_dir, exist_ok=True)
    return audio_dir, param_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir',  type=str,   help='')
    parser.add_argument('synth_conf',   type=str,   help='')
    parser.add_argument('--data_size',  type=int,   default=20000)
    parser.add_argument('--audio_len',  type=float, default=4.0)
    parser.add_argument('--sr',         type=int,   default=16000)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--save_param', action='store_true')
    # 'cuda' was hardcoded below. Made switchable so a small dataset can be
    # generated -- and the whole pipeline smoke-tested -- with no free GPU.
    parser.add_argument('--device',     type=str,   default='cuda')
    args = parser.parse_args()

    conf = OmegaConf.load(args.synth_conf)
    synth = construct_synth_from_conf(conf).to(args.device)

    # Optional, generation-side only: restrict some parameters to a set of real
    # values instead of the uniform draw. Nothing about the model changes --
    # ext_param_size, the heads and the normalised targets are all untouched.
    def _conf(name):
        c = conf.get(name, None)
        return {} if c is None else OmegaConf.to_container(c, resolve=True)

    quant, rng = _conf('quantize_params'), _conf('range_params')
    slots = draw_slots(synth, quant, rng)
    for key, offset, size, kind, payload in slots:
        if kind == 'set':
            print('{0}[{1}:{2}] drawn from {3}  (0..1: {4})'.format(
                key, offset, offset + size, list(quant[key]),
                [round(float(u), 4) for u in payload]))
        else:
            print('{0}[{1}:{2}] drawn uniform on {3}  (0..1: {4:.4f}..{5:.4f} '
                  'of the full range)'.format(
                      key, offset, offset + size, list(rng[key]),
                      payload[0], payload[1]))

    audio_dir, param_dir = make_dirs(args.dataset_dir, conf.name)

    n_samples = int(args.audio_len * args.sr)
    count = 0
    break_flag = False
    skip_count = 0
    if args.save_param:
        save_params = conf.save_params # harmor_q, harmor_cutoff, etc.
    else: # save all external params
        rev_dag_summary = {v: k for k,v in synth.dag_summary.items()} # HARM_Q: harmor_q
        save_params = [rev_dag_summary[k] for k in synth.ext_param_sizes.keys()]
    with torch.no_grad():
        with tqdm.tqdm(total=args.data_size) as pbar:
            while True:
                if break_flag:
                    break
                audio, output = draw_batch(synth, args.batch_size, n_samples, args.device, slots)
                params = {k: output[synth.dag_summary[k]].cpu() for k in save_params}
                for j in range(args.batch_size):
                    if count >= args.data_size:
                        break_flag=True
                        break
                    aud = audio[j]
                    # remove silence
                    if aud.abs().max() < 0.05:
                        skip_count += 1
                        continue
                    p = {k:pv[j] for k, pv in params.items()}
                    param_path = os.path.join(param_dir, '{0:05}.pt'.format(count))
                    torch.save(p, param_path)
                    audio_path = os.path.join(audio_dir, '{0:05}.wav'.format(count))
                    sf.write(audio_path, aud.cpu().numpy(), samplerate=args.sr)
                    count+=1
                    pbar.update(1)
    print('skipped {0} quiet sounds'.format(skip_count))