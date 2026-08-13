"""
This file contains the constants used in the synth module.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SynthConstants:
    wave_type_dict = {"sine": 0,
                      "square": 1,
                      "sawtooth": 2}

    filter_type_dict = {"low_pass": 0,
                        "high_pass": 1}

    sample_rate: int = 16000
    fixed_note_off: bool = True

    semitones_max_offset: int = 36
    middle_c_freq: float = 261.6255653005985
    min_amp: float = 0.01
    max_amp: float = 1

    min_mod_index: float = 0.01
    max_mod_index: float = 0.1
    min_fm_lfo_mod_index: float = 0.0001
    max_fm_lfo_mod_index: float = 0.01
    min_lfo_freq: float = 0.5
    max_lfo_freq: float = 15
    min_filter_freq: float = 100
    max_filter_freq: float = sample_rate / 2
    min_amount_tremolo: float = 0.05
    max_amount_tremolo: float = 1
    min_intensity_filter: float = 0
    max_intensity_filter: float = 1
    filter_adsr_frame_size = 512

    lfo_signal_sampling_rate = 100

    # non-active operation defaults
    non_active_waveform_default = 'sine'
    non_active_freq_default = 0
    non_active_amp_default = 0
    non_active_phase_default = 0
    non_active_mod_index_default = 0
    non_active_fm_lfo_mod_index_default = 0
    non_active_tremolo_amount_default = 0
    non_active_filter_intensity_default = 0

    # When predicting oscillator frequency by regression, the defines are used to normalize the output from the model
    margin: float = 200
    # --------------------------------------
    # -----------Modular Synth--------------
    # --------------------------------------
    # Modular Synth attributes:

    # Seed for random parameters generator
    seed = 2345124

    # Modular synth possible modules from synth_modules.py
    modular_synth_operations = ['osc', 'osc_sine', 'osc_square', 'osc_saw', 'fm', 'lfo', 'mix', 'filter', 'env_adsr',
                                'fm_lfo', 'lfo_sine', 'lfo_non_sine',
                                'fm_sine', 'fm_square', 'fm_saw', 'lowpass_filter',
                                'osc_saw_narrowfreq', 'osc_saw_fixedfreq']

    # defined modules and their parameters
    modular_synth_params = {'osc': ['amp', 'freq', 'waveform', 'active', 'phase'],
                            'osc_sine': ['amp', 'active', 'freq'],  # 'freq' is discrete from piano notes
                            'osc_square': ['amp', 'active', 'freq'],
                            'osc_saw': ['amp', 'active', 'freq'],
                            'osc_sine_no_activeness': ['amp', 'freq'],
                            'osc_square_no_activeness': ['amp', 'freq'],
                            'osc_saw_no_activeness': ['amp', 'freq'],
                            'osc_saw_narrowfreq': ['amp', 'freq'],  # freq restricted to under an octave
                            'osc_saw_fixedfreq': ['amp'],           # no freq head at all
                            'osc_sine_no_activeness_cont_freq': ['amp', 'freq'],  # 'freq' is continuous
                            'osc_square_no_activeness_cont_freq': ['amp', 'freq'],
                            'osc_saw_no_activeness_cont_freq': ['amp', 'freq'],
                            'lfo_sine': ['active', 'freq'],
                            'lfo_non_sine': ['freq', 'waveform'],
                            'lfo': ['freq', 'waveform', 'active'],
                            'surrogate_lfo': ['freq', 'waveform', 'active'],
                            'fm_lfo': ['active', 'fm_active', 'freq_c', 'waveform', 'fm_lfo_mod_index'],
                            'fm': ['freq_c', 'waveform', 'mod_index', 'active'],
                            'fm_sine': ['active', 'fm_active', 'amp_c', 'freq_c', 'mod_index'],
                            'fm_square': ['active', 'fm_active', 'amp_c', 'freq_c', 'mod_index'],
                            'fm_saw': ['active', 'fm_active', 'amp_c', 'freq_c', 'mod_index'],
                            'surrogate_fm_saw': ['active', 'fm_active', 'amp_c', 'freq_c', 'mod_index'],
                            'surrogate_fm_sine': ['active', 'fm_active', 'amp_c', 'freq_c', 'mod_index'],
                            'saw_square_osc': ['saw_amp', 'square_amp', 'freq', 'factor'],
                            'mix': [],
                            'filter': ['filter_freq', 'filter_type'],
                            'lowpass_filter': ['filter_freq'],
                            'lowpass_filter_adsr': ['filter_freq', 'intensity', 'attack_t', 'decay_t',
                                                    'sustain_t', 'sustain_level', 'release_t'],
                            'env_adsr': ['attack_t', 'decay_t', 'sustain_t', 'sustain_level', 'release_t'],
                            'amplitude_shape': ['envelope', 'attack_t', 'decay_t', 'sustain_t', 'sustain_level',
                                                'release_t'],
                            'tremolo': ['amount', 'active', 'fm_active']}

    def __post_init__(self):
        self.wave_type_dic_inv = {v: k for k, v in self.wave_type_dict.items()}
        self.filter_type_dic_inv = {v: k for k, v in self.filter_type_dict.items()}

        # build a list of possible frequencies
        self.semitones_list = [*range(-self.semitones_max_offset, self.semitones_max_offset + 1)]
        self.osc_freq_list = [self.middle_c_freq * (2 ** (1 / 12)) ** x for x in self.semitones_list]
        self.osc_freq_dic = {round(key, 4): value for value, key in enumerate(self.osc_freq_list)}
        self.osc_freq_dic_inv = {v: k for k, v in self.osc_freq_dic.items()}
        self.min_oscillator_freq = self.osc_freq_list[0]
        self.max_oscillator_freq = self.osc_freq_list[-1] + self.margin

        self.all_params_presets = {
            'lfo': {'freq': np.asarray([0.5] + [k + 1 for k in range(int(self.max_lfo_freq))])},
            'fm': {'freq_c': np.asarray(self.osc_freq_list),
                   'mod_index': np.linspace(0, self.max_mod_index, 16)},
            'filter': {'filter_freq': np.asarray([100 * 1.4 ** k for k in range(14)])}
        }

        self.sampling_configurations = self._create_sampling_config()
        self.param_configs = self._create_op_types_dict()

    def _create_sampling_config(self):
        """
        Create a dictionary of sampling configurations for each parameter. Used for data generation.
        """
        sampling_configurations = {
            'uniform_amp': {'type': 'uniform',
                            'values': (self.min_amp, self.max_amp),
                            'non_active_default': self.non_active_amp_default},
            'constant_amp': {'type': 'choice',
                             'values': (1,),
                             'non_active_default': self.non_active_amp_default},
            'osc_freq': {'type': 'choice',
                         'values': self.osc_freq_list,
                         'non_active_default': self.non_active_freq_default},
            # Eleven semitones centred on middle C: 196.0 Hz to 349.2 Hz, a
            # ratio of 1.78, deliberately under an octave.
            #
            # Two things make pitch hard for a bin-wise spectrogram distance.
            # Locally the distance stops responding once harmonics leave their
            # bins; globally, predicting twice the true pitch aligns every even
            # harmonic, so octave errors sit in a deep false basin. Staying
            # inside an octave removes the second outright. The first survives
            # here: at n_fft 2048 and 16 kHz the bins are 7.8 Hz and a semitone
            # near middle C is about 15 Hz, so adjacent notes overlap within the
            # Hann mainlobe and a usable gradient exists.
            #
            # The point is a task where pitch is still estimated and can still
            # be got wrong -- not one where it has been removed.
            # Pitch pinned to a single note. Wrong pitch puts the predicted
            # harmonics in different bins from the target's, and an L1 over
            # disjoint supports is sum(target) + sum(pred) -- monotonically
            # increasing in the model's own output, so the descent direction is
            # silence, and the loss then equals sum(target), a constant. That is
            # the 78.596 / 1.388 / 1.394 plateau reached from every
            # initialisation, and it is why amp collapsed while nothing else
            # could be learned either: a wrong pitch removes the gradient for
            # every other coordinate.
            #
            # Fixing pitch is not sidestepping the hard part. It removes the one
            # coordinate whose failure poisons the others, so that amp and
            # filter_freq -- both smooth in spectral terms -- can be estimated at
            # all, and a comparison between losses can mean something.
            'osc_freq_fixed': {'type': 'choice',
                               'values': (261.6255653005985,),
                               'non_active_default': self.non_active_freq_default},
            'osc_freq_narrow': {'type': 'choice',
                                'values': [self.middle_c_freq * (2 ** (1 / 12)) ** x
                                           for x in range(-5, 6)],
                                'non_active_default': self.non_active_freq_default},
            'osc_cont_freq': {'type': 'uniform',
                              'values': (self.min_oscillator_freq, self.max_oscillator_freq),
                              'non_active_default': self.non_active_freq_default},
            'osc_phase': {'type': 'uniform',
                          'values': (0, 2 * np.pi),
                          'non_active_default': self.non_active_phase_default},
            'waveform': {'type': 'choice',
                         'values': list(self.wave_type_dict),
                         'non_active_default': self.non_active_waveform_default},
            'non_sine_waveform': {'type': 'choice',
                                  'values': [k for k in self.wave_type_dict.keys() if k != 'sine'],
                                  'non_active_default': self.non_active_waveform_default},
            'lfo_freq': {'type': 'uniform',
                         'values': (self.min_lfo_freq, self.max_lfo_freq),
                         'non_active_default': self.non_active_freq_default},
            'fm_freq': {'type': 'freq_c',
                        'non_active_default': self.non_active_freq_default},
            'mod_index': {'type': 'uniform',
                          'values': (self.min_mod_index, self.max_mod_index),
                          'non_active_default': self.non_active_mod_index_default,
                          'activity_signal': 'fm_active'},
            'fm_lfo_mod_index': {'type': 'uniform',
                                 'values': (self.min_fm_lfo_mod_index, self.max_fm_lfo_mod_index),
                                 'non_active_default': self.non_active_fm_lfo_mod_index_default,
                                 'activity_signal': 'fm_active'},
            'filter_freq': {'type': 'uniform',
                            'values': (self.min_filter_freq, self.max_filter_freq)},
            'filter_type': {'type': 'choice',
                            'values': list(self.filter_type_dict)},
            'amount': {'type': 'uniform',
                       'values': (self.min_amount_tremolo, self.max_amount_tremolo),
                       'non_active_default': self.non_active_tremolo_amount_default},
            'intensity': {'type': 'uniform',
                          'values': (self.min_intensity_filter, self.max_intensity_filter),
                          'non_active_default': self.non_active_filter_intensity_default},
            'uniform_mix_factor': {'type': 'unit_uniform',
                                   'values': (0, 1),
                                   'sum': 1},
        }

        return sampling_configurations

    def _create_op_types_dict(self):
        sampling_configurations = self.sampling_configurations
        op_types = {
            'osc': {'amp': sampling_configurations['uniform_amp'],
                    'freq': sampling_configurations['osc_freq'],
                    'phase': sampling_configurations['osc_phase'],
                    'waveform': sampling_configurations['waveform']},
            'osc_sine': {'amp': sampling_configurations['uniform_amp'],
                         'freq': sampling_configurations['osc_freq']},
            'osc_square': {'amp': sampling_configurations['uniform_amp'],
                           'freq': sampling_configurations['osc_freq']},
            'osc_saw': {'amp': sampling_configurations['uniform_amp'],
                        'freq': sampling_configurations['osc_freq']},
            'osc_sine_no_activeness': {'amp': sampling_configurations['uniform_amp'],
                                       'freq': sampling_configurations['osc_freq']},
            'osc_square_no_activeness': {'amp': sampling_configurations['uniform_amp'],
                                         'freq': sampling_configurations['osc_freq']},
            'osc_saw_no_activeness': {'amp': sampling_configurations['uniform_amp'],
                                      'freq': sampling_configurations['osc_freq']},
            'osc_saw_narrowfreq': {'amp': sampling_configurations['uniform_amp'],
                                   'freq': sampling_configurations['osc_freq_narrow']},
            'osc_saw_fixedfreq': {'amp': sampling_configurations['uniform_amp']},
            'osc_sine_no_activeness_cont_freq': {'amp': sampling_configurations['uniform_amp'],
                                                 'freq': sampling_configurations['osc_cont_freq']},
            'osc_square_no_activeness_cont_freq': {'amp': sampling_configurations['uniform_amp'],
                                                   'freq': sampling_configurations['osc_cont_freq']},
            'osc_saw_no_activeness_cont_freq': {'amp': sampling_configurations['uniform_amp'],
                                                'freq': sampling_configurations['osc_cont_freq']},
            'lfo_sine': {'freq': sampling_configurations['lfo_freq']},
            'lfo_non_sine': {'freq': sampling_configurations['lfo_freq'],
                             'waveform': sampling_configurations['non_sine_waveform']},
            'lfo': {'freq': sampling_configurations['lfo_freq'], 'waveform': sampling_configurations['waveform']},
            'surrogate_lfo': {'freq': sampling_configurations['lfo_freq'],
                              'waveform': sampling_configurations['waveform']},
            'fm_lfo': {'freq_c': sampling_configurations['lfo_freq'], 'waveform': sampling_configurations['waveform'],
                       'fm_lfo_mod_index': sampling_configurations['fm_lfo_mod_index']},
            'fm': {'freq_c': sampling_configurations['fm_freq'], 'waveform': sampling_configurations['waveform'],
                   'mod_index': sampling_configurations['mod_index']},
            'fm_sine': {'amp_c': sampling_configurations['uniform_amp'], 'freq_c': sampling_configurations['fm_freq'],
                        'mod_index': sampling_configurations['mod_index']},
            'fm_square': {'amp_c': sampling_configurations['uniform_amp'], 'freq_c': sampling_configurations['fm_freq'],
                          'mod_index': sampling_configurations['mod_index']},
            'fm_saw': {'amp_c': sampling_configurations['uniform_amp'], 'freq_c': sampling_configurations['fm_freq'],
                       'mod_index': sampling_configurations['mod_index']},
            'surrogate_fm_saw': {'amp_c': sampling_configurations['uniform_amp'],
                                 'freq_c': sampling_configurations['fm_freq'],
                                 'mod_index': sampling_configurations['mod_index']},
            'surrogate_fm_sine': {'amp_c': sampling_configurations['uniform_amp'],
                                  'freq_c': sampling_configurations['fm_freq'],
                                  'mod_index': sampling_configurations['mod_index']},
            'saw_square_osc': {'square_amp': sampling_configurations['uniform_amp'],
                               'saw_amp': sampling_configurations['uniform_amp'],
                               'freq': sampling_configurations['fm_freq'],
                               'factor': sampling_configurations['uniform_mix_factor']},
            'oscillator_mix': {'amp': sampling_configurations['uniform_amp'],
                               'freq': sampling_configurations['fm_freq'],
                               'waveform': sampling_configurations['waveform']},
            'mix': {'mix_factor': sampling_configurations['uniform_mix_factor']},
            'filter': {'filter_freq': sampling_configurations['filter_freq'],
                       'filter_type': sampling_configurations['filter_type']},
            'lowpass_filter': {'filter_freq': sampling_configurations['filter_freq']},
            'lowpass_filter_adsr': {'filter_freq': sampling_configurations['filter_freq'],
                                    'intensity': sampling_configurations['intensity']},
            'env_adsr': {'attack_t', 'decay_t', 'sustain_t', 'sustain_level', 'release_t'},
            'amplitude_shape': {'envelope', 'attack_t', 'decay_t', 'sustain_t', 'sustain_level',
                                'release_t'},
            'tremolo': {'amount': sampling_configurations['amount']}
        }

        return op_types


synth_constants = SynthConstants()
