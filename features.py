import json
import librosa
import numpy as np

from scipy.interpolate import interp1d
from scipy import signal

import matplotlib as mpl

# for headless mode we don't want to
# change the backend
try:
    mpl.use('TkAgg')
except ImportError as e:
    pass
import matplotlib.pyplot as plt

# from util import get_melspec


with open('params.json', 'r') as f:
    param_default = json.load(f)['audio']


# Calculate the audio features
# audio_paths can be dictionary of paths with keys in track_id, or directly a path
# Can also directly feed in audio array
def get_features(audio_paths,
                 track_id=None,
                 param=param_default,
                 source_sr=None,
                 pass_zero=True,
                 offset=0.0):

    feature_list = []

    # If input type is already audio array just pass it on
    if type(audio_paths) == np.ndarray:
        y = audio_paths

        # Potentially resample:
        if source_sr is not None and param['SAMPLING_RATE'] != source_sr:
            y = librosa.resample(y, orig_sr=source_sr, target_sr=param['SAMPLING_RATE'])
            if param['SAMPLING_RATE'] > source_sr:
                print('Warning: Tried to increase sampling rate.')

    # Otherwise load audio
    else:
        # Set correct path to file
        if type(audio_paths) == str:
            audio_path = audio_paths
        elif type(audio_paths) == dict and track_id is not None:
            audio_path = audio_paths[track_id]
        else:
            raise Exception('Incompatible parameters given to get_features.')

        # Potentially load offset and duration
        duration = None
        if param['single_slice_audio']:
            duration = param['sample_sec']
        # if 'offset' in param.keys():
        #     offset = param['offset']

        # Load track
        # TODO: Quick hack to make compatible with GPU, fix properly
        try:
            y, _ = librosa.load(audio_path, sr=param['SAMPLING_RATE'], duration=duration, offset=offset)
        except:
            try:
                audio_path = audio_path.replace('/Users/maxfrenzel/Dropbox (Personal)/AI_DJ_Selection/',
                                                '/home/max/Data/AI_DJ/')
                y, _ = librosa.load(audio_path, sr=param['SAMPLING_RATE'], duration=duration, offset=offset)
            except:
                try:
                    audio_path = audio_path.replace('/Volumes/G-DRIVE ev RaW/Datasets/',
                                                    '/media/max/G-DRIVE ev RaW/Datasets/')
                    y, _ = librosa.load(audio_path, sr=param['SAMPLING_RATE'], duration=duration, offset=offset)
                except:
                    # TODO: This is another quick hack and not ideal for many reasons (e.g. fixed feature size)...
                    if pass_zero:
                        print(f'Cannot load audio file {audio_path}. Passing empty features instead')
                        return np.zeros((128, 10000))
                    else:
                        print(f'Cannot load audio file {audio_path}.')
                        raise

    # Calculate spectrum
    y_stft_full = get_spectrum(y, param['N_FFT'], param['HOP_LENGTH'])

    # If HPSS, split spectrum here and perform feature extraction on both parts
    if 'USE_HPSS' in param.keys() and param['USE_HPSS']:
        y_h, y_p = librosa.decompose.hpss(y_stft_full)
        y_list = [y_h, y_p]
    else:
        y_list = [y_stft_full]

    for y_stft in y_list:
        # Get melspectrogram of track, as well as deltas
        melspec = get_melspec(y_stft, n_mels=param['MELSPEC_BANDS'])
        if param['USE_DELTA']:
            delta = librosa.feature.delta(melspec)
        if param['USE_DELTADELTA']:
            delta_delta = librosa.feature.delta(melspec, order=2)

        # Get MFCC
        if param['USE_MFCC'] or param['USE_MFCC_DELTA'] or param['USE_MFCC_DELTADELTA']:
            mfcc = get_mfcc(melspec, n_mfcc=param['N_MFCC'])
            if param['USE_MFCC_DELTA']:
                mfcc_delta = librosa.feature.delta(mfcc)
            if param['USE_MFCC_DELTADELTA']:
                mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)

        # Get Fluctogram
        if param['USE_FLUCT']:
            fluct, spec_contrac, spec_flat = get_fluctogram(y_stft,
                                                            sr=param['SAMPLING_RATE'],
                                                            n_fft=param['N_FFT'])

            if 'MASK_FLUCT' in param.keys() and param['MASK_FLUCT']:
                fluct = mask_fluctogram(fluct, spec_contrac, spec_flat, param)

        # Concatenate
        if param['USE_SPEC']:
            # Convert melspec from -80 to 0dB to range [0,1]
            spec = (melspec + 80.0) / 80.0
            feature_list.append(spec)
        if param['USE_DELTA']:
            feature_list.append(delta)
        if param['USE_DELTADELTA']:
            feature_list.append(delta_delta)
        if param['USE_MFCC']:
            feature_list.append(mfcc)
        if param['USE_MFCC_DELTA']:
            feature_list.append(mfcc_delta)
        if param['USE_MFCC_DELTADELTA']:
            feature_list.append(mfcc_delta_delta)
        if param['USE_FLUCT']:
            feature_list.append(fluct)
        if 'USE_SC' in param.keys() and param['USE_SC']:
            feature_list.append(spec_contrac)
        if 'USE_SF' in param.keys() and param['USE_SF']:
            feature_list.append(spec_flat)

    features = np.concatenate(feature_list)

    return features


def get_spectrum(y, n_fft, hop_length, sr=None):

    # If path to audio file, load track
    if type(y) == str:
        y, _ = librosa.load(y, sr=sr)

    y_stft = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length)

    return y_stft

def get_melspec(spec, n_mels):

    # Power spectrum
    powerspec = np.abs(spec)**2

    melspec = librosa.feature.melspectrogram(S=powerspec, n_mels=n_mels)

    S = librosa.power_to_db(melspec, np.max)

    return S

def get_mfcc(melspec,
             n_mfcc):

    mfcc = librosa.feature.mfcc(S=melspec, n_mfcc=n_mfcc)

    return mfcc

def get_fluctogram(spec, sr, n_fft,
                   n_bands=11,
                   bandwith = 240,
                   bands_offset = 30,
                   note_min='A#3',
                   note_max='E8',
                   octaves=4.5,
                   bins_per_octave=120,
                   n_shifts=5,
                   apply_mask=False):

    bin_shift = np.arange(-n_shifts, n_shifts+1)

    # Take magnitude spectrogram
    spec = np.abs(spec)

    # Map the frequency axis of the spectrum to a logarithmic scale that relates to pitch
    target_freqs = librosa.cqt_frequencies(octaves * bins_per_octave, fmin=librosa.note_to_hz(note_min),
                                           bins_per_octave=bins_per_octave)
    y_stft_log = stft_interp(spec, librosa.core.fft_frequencies(sr=sr, n_fft=n_fft), target_freqs)

    # Find correct range of spectrum that's of interest for calculating the fluctogram
    f_start = librosa.note_to_hz(note_min)
    f_end = librosa.note_to_hz(note_max)
    f_start_idx = np.argmin(np.abs(target_freqs - f_start))
    f_end_idx = np.argmin(np.abs(target_freqs - f_end))

    spec_log = y_stft_log[f_start_idx:f_end_idx + 1, :]

    # Initialise empty fluctogram array
    fluctogram = np.zeros((n_bands, spec_log.shape[1]))

    # Get window function as a matrix
    def _get_triangle_window(shape):
        win = np.bartlett(shape[0])
        return np.tile(win, (shape[1], 1)).T

    win = _get_triangle_window((bandwith, spec_log.shape[1]))

    # Extract the subbands
    for cur_band_idx in np.arange(n_bands):

        cur_band_start = cur_band_idx * bands_offset
        cur_band_end = cur_band_start + bandwith

        # Assign the subbands
        cur_band = spec_log[cur_band_start:cur_band_end, :].copy()

        # Weight the subbands with the triangular window
        cur_band *= win

        for cur_frame in np.arange(spec_log.shape[1] - 1):
            cur_frame_spec = cur_band[:, cur_frame]
            next_frame_spec = cur_band[:, cur_frame + 1]

            # Cross-correlate both frames
            xc = np.correlate(cur_frame_spec, next_frame_spec, 'same')

            # Normalize according to Pearson at lag 0 (center bin)
            center_bin = int(np.floor(len(xc) / 2))
            xc /= xc[center_bin]

            # Bins of interest: get +- 5 bins around center
            # With standard setting this is quivalent to +- half a semiton
            # --> Reveals sub-semitone fluctuations
            boi = bin_shift + center_bin
            xc_boi = xc[boi.tolist()]

            # Take maximum idx and center it
            fluctogram[cur_band_idx, cur_frame] = np.argmax(xc_boi) + np.min(bin_shift)

    # Calculate reliability indicators
    spec_contract = bandwise_contraction(spec, target_freqs)
    spec_flat = bandwise_flatness(spec, target_freqs)

    return fluctogram, spec_contract, spec_flat

def mask_fluctogram(fluct, spec_contrac, spec_flat, param):

    if param['SPECTRAL_CONTRACTION']:
        fluct *= np.float32(spec_contrac > param['SC_THRESHOLD'])
    if param['SPECTRAL_FLATNESS']:
        fluct *= np.float32(spec_flat < param['SF_THRESHOLD'])

    return fluct

def visualize_fluct(fluct):

    n_bands = fluct.shape[0]
    for cur_band in np.arange(n_bands):
        plt.plot(fluct[cur_band, :] + (cur_band + 1) * 3, 'k')

def stft_interp(spec, source_freqs, target_freqs):
    """Compute an interpolated version of the spectrogram. Uses scipy.interp1d to map
       to the new frequency axis.
    """

    set_interp = interp1d(source_freqs, spec, kind='linear', axis=0)
    spec_interp = set_interp(target_freqs)

    return spec_interp

def spectral_contraction(X_mag):
    """Spectral Contraction measure.

       As suggested in _[1].

    Parameters
    ----------
    X_mag : ndarray
        Magnitude spectrum of a time frame.

    Returns
    -------
    spectral_contraction :

    References
    ----------
    .. [1] Bernhard Lehner, Gerhard Widmer, Reinhard Sonnleitner
           "ON THE REDUCTION OF FALSE POSITIVES IN SINGING VOICE DETECTION",
           ICASSP 2014

    """
    window = signal.windows.chebwin(X_mag.shape[0], 200)
    if X_mag.ndim > 1:
        window = np.tile(window, (X_mag.shape[1], 1)).T

    spectral_contraction = np.sum(np.square(X_mag) * window, axis=0) / (np.sum(np.square(X_mag), axis=0) + np.finfo(float).eps)

    return spectral_contraction

# def bandwise_contraction(X_log, freq_ax_log, f_start=164, f_end=10548, n_bands=17, bandwith=240, bands_offset=30):
def bandwise_contraction(X_log, target_freqs,
                         n_bands=11,
                         bandwith=240,
                         bands_offset=30,
                         note_min='A#3',
                         note_max='E8'):
    f_start = librosa.note_to_hz(note_min)
    f_end = librosa.note_to_hz(note_max)

    # Get indices for frequency range
    f_start_idx = np.argmin(np.abs(target_freqs - f_start))
    f_end_idx = np.argmin(np.abs(target_freqs - f_end))
    X_log = X_log[f_start_idx:f_end_idx + 1, :]

    bw_contraction = np.zeros((n_bands, X_log.shape[1]))

    # extract the subbands
    for cur_band_idx in np.arange(n_bands):
        cur_band_start = cur_band_idx * bands_offset
        cur_band_end = cur_band_start + bandwith

        # assign the subbands
        cur_band = X_log[cur_band_start:cur_band_end, :].copy()

        # Call standard spectral contraction for each band
        bw_contraction[cur_band_idx, :] = spectral_contraction(cur_band)

    return bw_contraction


def spectral_flatness(X_mag):
    """Spectral Flatness measure.

    We use the log-spec version e use the log-spec version as suggested by _[1].

    Parameters
    ----------
    X_mag : ndarray
        Magnitude spectrum of a time frame.

    Returns
    -------
    spectral_flatness :

    References
    ----------
    .. [1] Alexander Lerch, "Audio Content Analysis"

    """
    # geometric mean
    spectral_flatness = np.exp(np.mean(np.log(X_mag + np.finfo(float).eps), axis=0))

    # divided by arithmetic mean
    spectral_flatness /= np.mean(X_mag, axis=0) + np.finfo(float).eps

    return spectral_flatness


def bandwise_flatness(X_log, target_freqs,
                      n_bands=11,
                      bandwith=240,
                      bands_offset=30,
                      note_min='A#3',
                      note_max='E8'):
    f_start = librosa.note_to_hz(note_min)
    f_end = librosa.note_to_hz(note_max)

    f_start_idx = np.argmin(np.abs(target_freqs - f_start))
    f_end_idx = np.argmin(np.abs(target_freqs - f_end))
    X_log = X_log[f_start_idx:f_end_idx + 1, :]

    bw_flatness = np.zeros((n_bands, X_log.shape[1]))

    # extract the subbands
    for cur_band_idx in np.arange(n_bands):
        cur_band_start = cur_band_idx * bands_offset
        cur_band_end = cur_band_start + bandwith

        # assign the subbands
        cur_band = X_log[cur_band_start:cur_band_end, :].copy()

        # Call standard spectral flatness for each band
        bw_flatness[cur_band_idx, :] = spectral_flatness(cur_band)

    return bw_flatness
