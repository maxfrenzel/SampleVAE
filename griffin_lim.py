import librosa
import numpy as np
import json

# with open('audio_params.json', 'r') as f:
#     param = json.load(f)
#
# N_FFT = param['N_FFT']
# HOP_LENGTH = param['HOP_LENGTH']
# SAMPLING_RATE = param['SAMPLING_RATE']
# MELSPEC_BANDS = param['MELSPEC_BANDS']

_inv_mel_basis = None


def _mel_to_linear(mel_spectrogram, N_FFT, SAMPLING_RATE, MELSPEC_BANDS):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(N_FFT, SAMPLING_RATE, MELSPEC_BANDS))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -80.0) + 80.0


def inv_magphase(mag, phase_angle):
    phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
    return mag * phase


def _build_mel_basis(N_FFT, SAMPLING_RATE, MELSPEC_BANDS):
    n_fft = N_FFT
    return librosa.filters.mel(SAMPLING_RATE, n_fft, n_mels=MELSPEC_BANDS)


def griffin_lim(melspec, num_iters=10, phase_angle=0.0, n_fft=1024, hop=256, sr=22050, mspec_bands=128):
    """Iterative algorithm for phase retrival from a melspectrogram.

    Args:
    mag: Magnitude spectrogram.
    phase_angle: Initial condition for phase.
    n_fft: Size of the FFT.
    hop: Stride of FFT. Defaults to n_fft/2.
    num_iters: Griffin-Lim iterations to perform.

    Returns:
    audio: 1-D array of float32 sound samples.
    """
    mag = _mel_to_linear(_db_to_amp(melspec), n_fft, sr, mspec_bands)

    fft_config = dict(n_fft=n_fft, win_length=n_fft, hop_length=hop, center=True)
    ifft_config = dict(win_length=n_fft, hop_length=hop, center=True)
    complex_specgram = inv_magphase(mag, phase_angle)
    for i in range(num_iters):
        audio = librosa.istft(complex_specgram, **ifft_config)
        if i != num_iters - 1:
            complex_specgram = librosa.stft(audio, **fft_config)
            _, phase = librosa.magphase(complex_specgram)
            phase_angle = np.angle(phase)
            complex_specgram = inv_magphase(mag, phase_angle)
    return audio