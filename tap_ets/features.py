"""Log mel-spectrogram targets and their normalizer."""
import os
import pickle

import numpy as np
import torch
from librosa.filters import mel as librosa_mel_fn

NORMALIZER_FILES = ('normalizer.pkl', 'normalizer.npz')

_mel_basis = {}
_hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """[B, N] waveform -> [B, num_mels, T] log mel, matching the HiFi-GAN front end."""
    key = (str(y.device), sampling_rate, n_fft, num_mels, fmin, fmax)
    if key not in _mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
    window_key = (str(y.device), win_size)
    if window_key not in _hann_window:
        _hann_window[window_key] = torch.hann_window(win_size).to(y.device)

    pad = int((n_fft - hop_size) / 2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode='reflect').squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=_hann_window[window_key],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    return torch.log(torch.clamp(torch.matmul(_mel_basis[key], spec), min=1e-5))


class FeatureNormalizer:
    """Per-dimension standardization."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
        self.std = np.broadcast_to(np.asarray(std, dtype=np.float32).reshape(1, -1), self.mean.shape).copy()

    def normalize(self, features):
        return (features - self.mean) / self.std

    def inverse(self, features):
        return features * self.std + self.mean

    def save(self, path):
        np.savez(path, mean=self.mean.reshape(-1), std=self.std.reshape(-1))

    @classmethod
    def load(cls, path):
        """Load normalizer.npz, or the normalizer.pkl shipped with the released models."""
        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                legacy = _LegacyUnpickler(f).load()
            legacy = legacy[0] if isinstance(legacy, tuple) else legacy  # (feature, emg)
            return cls(legacy.feature_means, legacy.feature_stddevs)
        stats = np.load(path)
        return cls(stats['mean'], stats['std'])


class _LegacyNormalizer:
    """Holds feature_means / feature_stddevs of a pickled normalizer."""


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _LegacyNormalizer if name == 'FeatureNormalizer' else super().find_class(module, name)


def find_normalizer(*directories):
    """First normalizer file found in `directories`."""
    for directory in directories:
        for file_name in NORMALIZER_FILES:
            path = os.path.join(directory, file_name)
            if os.path.exists(path):
                return path
    raise FileNotFoundError(f'no normalizer in {directories}')
