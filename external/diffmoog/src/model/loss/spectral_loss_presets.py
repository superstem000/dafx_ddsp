"""
this config file is used to define the spectral loss presets
each preset is a dictionary with the following args:

fft_sizes: the fft sizes for STFT.
transform: ['SPECTROGRAM', 'MEL_SPECTROGRAM', 'BOTH']
frame_overlap: window overlap for STFT
n_mels: the number of mel bins for MEL_SPECTROGRAM transform
f_min: the minimum frequency for MEL_SPECTROGRAM transform
f_max: the maximum frequency for MEL_SPECTROGRAM transform

weighting factors for preprocessing function (magnitude, log magnitude, cumulative sum time, cumulative sum frequency):
    multi_spectral_loss_norm: ['L1', 'L2']
    multi_spectral_mag_weight: the magnitude weight to be used for the loss
    multi_spectral_logmag_weight: the log magnitude weight to be used for the loss
    multi_spectral_cumsum_time_weight: the cumulative sum time weight to be used for the loss
    multi_spectral_cumsum_freq_weight: the cumulative sum frequency weight to be used for the loss

normalize_loss_by_nfft: whether to normalize the loss by the nfft
"""

MSS_MAGNITUDE_LOSS = {'fft_sizes': (2048, 1024, 512, 256, 128, 64),
                      'transform': 'SPECTROGRAM',
                      'frame_overlap': 0.75,
                      'multi_spectral_loss_norm': 'L1',
                      'multi_spectral_mag_weight': 1,
                      'multi_spectral_logmag_weight': 0,
                      'normalize_loss_by_nfft': False}

MSS_MEL_SPECTROGRAM_MAGNITUDE_LOSS = {'fft_sizes': (2048, 1024, 512, 256, 128, 64),
                                  'transform': 'MEL_SPECTROGRAM',
                                  'frame_overlap': 0.75,
                                  'n_mels': 1024,
                                  'f_min': 30,
                                  'f_max': 4000,
                                  'multi_spectral_loss_norm': 'L1',
                                  'multi_spectral_mag_weight': 1,
                                  'multi_spectral_mag_warmup': 0,
                                  'normalize_loss_by_nfft': False,
                                  'multi_spectral_mag_gradual': False,
                                  }

MSS_CUMSUM_TIME_LOSS = {'fft_sizes': (2048, 1024, 512, 256, 128, 64),
                        'transform': 'SPECTROGRAM',
                        'frame_overlap': 0.75,
                        'multi_spectral_loss_norm': 'L1',
                        'multi_spectral_cumsum_time_weight': 1,
                        'normalize_loss_by_nfft': False}

MSS_CUMSUM_FREQ_LOSS = {'fft_sizes': (2048, 1024, 512, 256, 128, 64),
                        'transform': 'SPECTROGRAM',
                        'frame_overlap': 0.75,
                        'multi_spectral_loss_norm': 'L1',
                        'multi_spectral_cumsum_freq_weight': 1,
                        'normalize_loss_by_nfft': False}

MSS_MEL_SPECTROGRAM_CUMSUM_TIME_LOSS = {'fft_sizes': (2048, 1024, 512, 256, 128, 64),
                        'transform': 'MEL_SPECTROGRAM',
                        'frame_overlap': 0.75,
                        'n_mels': 1024,
                        'f_min': 30,
                        'f_max': 4000,
                        'multi_spectral_loss_norm': 'L1',
                        'multi_spectral_cumsum_time_weight': 1,
                        'normalize_loss_by_nfft': False}

MSS_MEL_SPECTROGRAM_CUMSUM_FREQ_LOSS = {'fft_sizes': (2048, 1024, 512, 256, 128, 64),
                        'transform': 'MEL_SPECTROGRAM',
                        'frame_overlap': 0.75,
                        'n_mels': 1024,
                        'f_min': 30,
                        'f_max': 4000,
                        'multi_spectral_loss_norm': 'L1',
                        'multi_spectral_cumsum_freq_weight': 1,
                        'normalize_loss_by_nfft': False}

MSS_LOG_MAGNITUDE_LOSS = {'fft_sizes': (2048, 1024, 512, 256, 128, 64),
                          'transform': 'SPECTROGRAM',
                          'frame_overlap': 0.75,
                          'multi_spectral_loss_norm': 'L1',
                          'multi_spectral_mag_weight': 0,
                          'multi_spectral_logmag_weight': 1,
                          'normalize_loss_by_nfft': False}

MEL_SPECTROGRAM_MAGNITUDE_LOSS = {'fft_sizes': (1024,),
                                  'transform': 'MEL_SPECTROGRAM',
                                  'frame_overlap': 0.5,
                                  'n_mels': 1024,
                                  'f_min': 30,
                                  'f_max': 4000,
                                  'multi_spectral_loss_norm': 'L1',
                                  'multi_spectral_mag_weight': 1,
                                  'multi_spectral_mag_warmup': 0,
                                  'normalize_loss_by_nfft': False,
                                  'multi_spectral_mag_gradual': False,
                                  }

CUMSUM_TIME_LOSS = {'fft_sizes': (2048, ),
                    'transform': 'SPECTROGRAM',
                    'frame_overlap': 0.75,
                    'n_mels': 1024,
                    'f_min': 30,
                    'f_max': 4000,
                    'multi_spectral_loss_norm': 'L1',
                    'multi_spectral_cumsum_time_weight': 1,
                    'normalize_loss_by_nfft': False}

CUMSUM_FREQ_LOSS = {'fft_sizes': (2048, ),
                    'transform': 'SPECTROGRAM',
                    'frame_overlap': 0.75,
                    'n_mels': 1024,
                    'f_min': 30,
                    'f_max': 4000,
                    'multi_spectral_loss_norm': 'L1',
                    'multi_spectral_cumsum_freq_weight': 1,
                    'normalize_loss_by_nfft': False}

MSS_CUMSUM_TIME_FREQ_LOSS = {'fft_sizes': (2048, 1024, 512, 256, 128, 64),
                            'transform': 'SPECTROGRAM',
                             'frame_overlap': 0.75,
                             'n_mels': 1024,
                            'f_min': 30,
                            'f_max': 4000,
                            'multi_spectral_loss_norm': 'L1',
                            'multi_spectral_cumsum_time_weight': 1,
                            'multi_spectral_cumsum_freq_weight': 1,
                            'normalize_loss_by_nfft': False}




CUMSUM_TIME_LOW_FFT_LOSS = {'fft_sizes': (128, 64),
                            'transform': 'SPECTROGRAM',
                            'multi_spectral_loss_norm': 'L1',
                            'multi_spectral_cumsum_time_weight': 1/2000,
                            'normalize_loss_by_nfft': False}

FM_ONLY_LOSS = {'fft_sizes': (128, 64),
                'transform': 'SPECTROGRAM',
                'multi_spectral_loss_norm': 'L1',
                'multi_spectral_cumsum_time_weight': 1/2000,
                'multi_spectral_mag_weight': 1/200,
                'multi_spectral_mag_warmup': 20000,
                'multi_spectral_mag_gradual': True,
                'normalize_loss_by_nfft': True}


CUMSUM_TIME_FREQ_W_LOGMAG = {'fft_sizes': (2048, 1024, 512, 256, 128, 64),
                             'transform': 'SPECTROGRAM',
                             'multi_spectral_loss_norm': 'L1',
                             'multi_spectral_cumsum_time_weight': 1 / 2000,
                             'multi_spectral_cumsum_freq_weight': 1 / 5000,
                             'multi_spectral_mag_weight': 1/200,
                             'multi_spectral_mag_warmup': 20000,
                             'multi_spectral_mag_gradual': True,
                             'normalize_loss_by_nfft': True}

SPECTROGRAM_MAGNITUDE_LOSS = {'fft_sizes': (2048,),
                  'transform': 'SPECTROGRAM',
                              'frame_overlap': 0.75,
                              'n_mels': 1024,
                  'f_min': 30,
                  'f_max': 4000,
                  'multi_spectral_loss_norm': 'L1',
                  'multi_spectral_mag_weight': 1,
                  'multi_spectral_mag_warmup': 0,
                  'normalize_loss_by_nfft': False,
                  'multi_spectral_mag_gradual': False,
                  }

SPECTROGRAM_LOG_MAGNITUDE_LOSS = {'fft_sizes': (2048,),
                                  'transform': 'SPECTROGRAM',
                                  'frame_overlap': 0.75,
                                  'n_mels': 1024,
                                  'f_min': 30,
                                  'f_max': 4000,
                                  'multi_spectral_loss_norm': 'L2',
                                  'multi_spectral_mag_weight': 1,
                                  'multi_spectral_mag_warmup': 0,
                                  'normalize_loss_by_nfft': False,
                                  'multi_spectral_mag_gradual': False}

MEL_SPECTROGRAM_MAGNITUDE_LOSS = {'fft_sizes': (1024,),
                                  'transform': 'MEL_SPECTROGRAM',
                                  'frame_overlap': 0.75,
                                  'n_mels': 1024,
                                  'f_min': 30,
                                  'f_max': 4000,
                                  'multi_spectral_loss_norm': 'L1',
                                  'multi_spectral_mag_weight': 1,
                                  'multi_spectral_mag_warmup': 0,
                                  'normalize_loss_by_nfft': False,
                                  'multi_spectral_mag_gradual': False,
                                  }

MSS_MEL_SPECTROGRAM_MAGNITUDE_LOSS = {'fft_sizes': (2048, 1024, 512, 256, 128, 64),
                                        'transform': 'MEL_SPECTROGRAM',
                                      'frame_overlap': 0.75,
                                      'n_mels': 1024,
                                        'f_min': 30,
                                        'f_max': 4000,
                                        'multi_spectral_loss_norm': 'L1',
                                        'multi_spectral_mag_weight': 1,
                                        'multi_spectral_mag_warmup': 0,
                                        'normalize_loss_by_nfft': False,
                                        'multi_spectral_mag_gradual': False,
                                        }

MAG_LOGMAG_LOSS = {'fft_sizes': (2048,),
                   'transform': 'SPECTROGRAM',
                   'frame_overlap': 0.75,
                   'n_mels': 1024,
                   'f_min': 30,
                   'f_max': 4000,
                   'multi_spectral_loss_norm': 'L2',
                   'multi_spectral_mag_weight': 0,
                   'multi_spectral_logmag_weight': 1,
                   'multi_spectral_mag_warmup': 0,
                   'normalize_loss_by_nfft': False,
                   'multi_spectral_mag_gradual': False,}

# ---------------------------------------------------------------------------
# Loss study: a 2x2 over resolution x compression, single-factor throughout.
#
# The two shipped presets that look like they would serve do not. Weights are
# read with .get(name, 0), so a missing key is a zero weight, and on that
# reading SPECTROGRAM_LOG_MAGNITUDE_LOSS is mag_weight 1 with no logmag_weight
# under L2 -- L2 on linear magnitude, despite the name -- while MAG_LOGMAG_LOSS
# is mag_weight 0 / logmag_weight 1 under L2, i.e. log only, also despite the
# name. The two appear to have been swapped. Either would silently answer a
# different question than the one being asked, so the study defines its own.
# ---------------------------------------------------------------------------

# The single-factor flip from SPECTROGRAM_MAGNITUDE_LOSS: same resolution,
# same L1 norm, same overlap, mag_weight moved to logmag_weight.
SPECTROGRAM_LOGMAG_L1_LOSS = {'fft_sizes': (2048,),
                              'transform': 'SPECTROGRAM',
                              'frame_overlap': 0.75,
                              'multi_spectral_loss_norm': 'L1',
                              'multi_spectral_mag_weight': 0,
                              'multi_spectral_logmag_weight': 1,
                              'power': 1,
                              'normalize_loss_by_nfft': False}

# The plate's MSS with the spectral-convergence term removed, on DiffMoog's
# window list. loss_mss there is, per resolution, an SC term plus
# mean|log(t + 1e-7) - log(c + 1e-7)| -- so dropping SC leaves the unbounded log
# magnitude alone, which is what this is. Not DDSP's published loss, which also
# carries a linear magnitude term; the name says what it is.
#
# Against spectrogram_logmag_eps_l1 this isolates resolution at fixed
# compression, and against spectrogram_mag_l1 it is the multi-resolution end of
# the compression axis.
MSS_LOGMAG_EPS_LOSS = {'fft_sizes': (2048, 1024, 512, 256, 128, 64),
                       'transform': 'SPECTROGRAM',
                       'frame_overlap': 0.75,
                       'multi_spectral_loss_norm': 'L1',
                       'multi_spectral_mag_weight': 0,
                       'multi_spectral_logmag_eps_weight': 1,
                       'power': 1,
                       'normalize_loss_by_nfft': False}

# DDSP's original multi-scale spectral loss (Engel et al., 2020): linear
# magnitude L1 plus log magnitude L1 at equal weight, summed over six
# resolutions. There is no spectral-convergence term to omit -- SpectralLoss
# implements only mag, logmag, cumsum_time and cumsum_freq -- so this is the
# whole objective, not a reduced form of one.
MSS_DDSP_LOSS = {'fft_sizes': (2048, 1024, 512, 256, 128, 64),
                 'transform': 'SPECTROGRAM',
                 'frame_overlap': 0.75,
                 'multi_spectral_loss_norm': 'L1',
                 'multi_spectral_mag_weight': 1,
                 'multi_spectral_logmag_weight': 1,
                 'power': 1,
                 'normalize_loss_by_nfft': False}

# Linear magnitude at power 1, i.e. |X| rather than |X|^2. Their
# SPECTROGRAM_MAGNITUDE_LOSS uses the Spectrogram default power=2.0, so an L1 on
# it spans a squared dynamic range; from a random init the gradient is dominated
# by a few loud bins and the cheapest descent is to go silent, which is exactly
# what the cold-start linear arms did -- amplitude pinned at 0.003 with std
# 0.001, spectral loss frozen at 78.596 from the first epoch. At power 1 this is
# the plate's L1_STFT: |X|, L1, single resolution, hop n_fft/4.
SPECTROGRAM_MAG_L1_LOSS = {'fft_sizes': (2048,),
                           'transform': 'SPECTROGRAM',
                           'frame_overlap': 0.75,
                           'multi_spectral_loss_norm': 'L1',
                           'multi_spectral_mag_weight': 1,
                           'power': 1,
                           'normalize_loss_by_nfft': False}

# The unbounded end of the ladder: log(x + 1e-7) rather than log(x + 1).
# Identical to SPECTROGRAM_LOGMAG_L1_LOSS in every other respect, so the pair
# isolates where the compression knee sits and nothing else. Note x here is a
# power spectrogram (the Spectrogram transform is built with power=2.0), so the
# knee falls at a different point relative to the magnitude distribution than
# the plate's log(|X| + 1e-7) does -- a parameter of the experiment, not an
# accident, and worth stating wherever the two are compared.
SPECTROGRAM_LOGMAG_EPS_L1_LOSS = {'fft_sizes': (2048,),
                                  'transform': 'SPECTROGRAM',
                                  'frame_overlap': 0.75,
                                  'multi_spectral_loss_norm': 'L1',
                                  'multi_spectral_mag_weight': 0,
                                  'multi_spectral_logmag_eps_weight': 1,
                                  'power': 1,
                                  'normalize_loss_by_nfft': False}

# The compression ladder as one variable: log(|X| + eps), eps from 1e-7 to 1.
# Identical in every other respect to SPECTROGRAM_MAG_L1_LOSS, so a step along
# the ladder moves the knee and nothing else. eps = 1 is log1p, i.e. DiffMoog's
# own 'logmag' and the plate's C2; eps = 1e-7 is the unbounded log the standard
# multi-scale spectral loss uses.
def _logmag_eps_preset(eps):
    return {'fft_sizes': (2048,),
            'transform': 'SPECTROGRAM',
            'frame_overlap': 0.75,
            'multi_spectral_loss_norm': 'L1',
            'multi_spectral_mag_weight': 0,
            'multi_spectral_logmag_eps_weight': 1,
            'logmag_eps_value': eps,
            'power': 1,
            'normalize_loss_by_nfft': False}


EPS_LADDER = {f'logmag_eps_{tag}': _logmag_eps_preset(v) for tag, v in
              (('1', 1.0), ('1e1', 1e-1), ('1e3', 1e-3), ('1e5', 1e-5), ('1e7', 1e-7))}

loss_presets = {**EPS_LADDER,
                'mss_cumsum_time': MSS_CUMSUM_TIME_LOSS,
                'mss_logmag_eps': MSS_LOGMAG_EPS_LOSS,
                'spectrogram_mag_l1': SPECTROGRAM_MAG_L1_LOSS,
                'spectrogram_logmag_l1': SPECTROGRAM_LOGMAG_L1_LOSS,
                'spectrogram_logmag_eps_l1': SPECTROGRAM_LOGMAG_EPS_L1_LOSS,
                'mss_ddsp': MSS_DDSP_LOSS,
                'mss_cumsum_freq': MSS_CUMSUM_FREQ_LOSS,
                'mss_cumsum_time_freq': MSS_CUMSUM_TIME_FREQ_LOSS,
                'cumsum_time': CUMSUM_TIME_LOSS,
                'cumsum_freq': CUMSUM_FREQ_LOSS,
                'cumsum_time_low_fft': CUMSUM_TIME_LOW_FFT_LOSS,
                'lfo_only': CUMSUM_TIME_LOSS,
                'fm_only': FM_ONLY_LOSS,
                'cumsum_time_freq_mag': CUMSUM_TIME_FREQ_W_LOGMAG,
                'mag_logmag': MAG_LOGMAG_LOSS,
                'spectrogram_magnitude': SPECTROGRAM_MAGNITUDE_LOSS,
                'spectrogram_log_magnitude': SPECTROGRAM_LOG_MAGNITUDE_LOSS,
                'mel_spectrogram_magnitude': MEL_SPECTROGRAM_MAGNITUDE_LOSS,
                'mss_magnitude': MSS_MAGNITUDE_LOSS,
                'mss_log_magnitude': MSS_LOG_MAGNITUDE_LOSS,
                'mss_mel_spectrogram_magnitude': MSS_MEL_SPECTROGRAM_MAGNITUDE_LOSS,
                'mss_mel_spectrogram_cumsum_time': MSS_MEL_SPECTROGRAM_CUMSUM_TIME_LOSS,
                'mss_mel_spectrogram_cumsum_freq': MSS_MEL_SPECTROGRAM_CUMSUM_FREQ_LOSS}

