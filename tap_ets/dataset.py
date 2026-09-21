"""EMG preprocessing and datasets."""
import os
import random
from glob import glob

import numpy as np
import scipy.signal
import torch

from tap_ets.phonemes import PAD_ID, merge_frame_phonemes, read_frame_phonemes

EMG_SAMPLE_RATE = 1000  # Gaddy et al. dataset


def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)


def notch(signal, freq, fs):
    b, a = scipy.signal.iirnotch(freq, 30, fs)
    return scipy.signal.filtfilt(b, a, signal)


def notch_harmonics(signal, freq, fs):
    for harmonic in range(1, 8):
        signal = notch(signal, freq * harmonic, fs)
    return signal


def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal)) / old_freq
    sample_times = np.arange(0, times[-1], 1 / new_freq)
    return np.interp(sample_times, times, signal)


def apply_to_all(function, signal_array, *args, **kwargs):
    return np.stack([function(signal_array[:, i], *args, **kwargs) for i in range(signal_array.shape[1])], 1)


def emg_downsample_factor(frame_rate):
    """EMG samples per acoustic frame; the conv front end downsamples by 8."""
    return round(100 / frame_rate) * 8


def emg_sample_rate(frame_rate):
    # truncated, not rounded: the released models were evaluated at int(86.13 * 8) = 689 Hz
    return int(frame_rate * emg_downsample_factor(frame_rate))


def load_emg(path, frame_rate):
    """Filter and resample one utterance; neighbours are concatenated first to avoid edge artifacts."""
    base_dir, file_name = os.path.split(path)
    index = int(file_name.split('_')[0])
    raw = np.load(path)
    neighbours = []
    for offset in (-1, 1):
        neighbour = os.path.join(base_dir, f'{index + offset}_emg.npy')
        neighbours.append(np.load(neighbour) if os.path.exists(neighbour) else np.zeros([0, raw.shape[1]]))
    before, after = neighbours

    x = np.concatenate([before, raw, after], 0)
    x = apply_to_all(notch_harmonics, x, 60, EMG_SAMPLE_RATE)
    x = apply_to_all(remove_drift, x, EMG_SAMPLE_RATE)
    x = x[before.shape[0]:x.shape[0] - after.shape[0], :]
    x = apply_to_all(subsample, x, emg_sample_rate(frame_rate), EMG_SAMPLE_RATE)
    x = x / 20
    x = 50 * np.tanh(x / 50.)
    return x.astype(np.float32)


class EMGDataset(torch.utils.data.Dataset):
    """(EMG, target feature, frame-wise phoneme) examples.

    Train uses voiced, non-parallel and silent recordings, dev/test only silent ones.
    Silent EMG takes its targets from the paired voiced recording.
    """

    def __init__(self, data_path, feature_dir, data_split, frame_rate, target_sec=None, normalizer=None):
        self.dataset_dir = os.path.join(data_path, 'silent_speech_dataset', data_split)
        self.feature_dir = os.path.join(feature_dir, data_split)
        pattern = '*/*/*.flac' if data_split == 'train' else 'silent_parallel_data/*/*.flac'
        self.paths = sorted(glob(os.path.join(self.dataset_dir, pattern)))
        self.frame_rate = frame_rate
        self.downsample = emg_downsample_factor(frame_rate)
        self.target_sec = target_sec
        self.normalizer = normalizer
        if target_sec is not None:
            self.target_speech_len = int(frame_rate * target_sec)
            self.target_emg_len = self.target_speech_len * self.downsample
        random.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

    def resolve(self, path):
        data_type, session, file_name = os.path.relpath(path, self.dataset_dir).split(os.sep)
        idx = file_name.split('_')[0]
        silent = data_type == 'silent_parallel_data'
        target_type = 'voiced_parallel_data' if silent else data_type
        emg_path = os.path.join(self.dataset_dir, data_type, session, f'{idx}_emg.npy')
        feat_path = os.path.join(self.feature_dir, target_type, session, f'{idx}_feat.npy')
        tg_path = os.path.join(self.dataset_dir, target_type, session, f'{idx}_tg.TextGrid')
        return emg_path, feat_path, tg_path, silent

    def __getitem__(self, i):
        emg_path, feat_path, tg_path, silent = self.resolve(self.paths[i])
        speech_feature = np.load(feat_path).astype(np.float32)
        emg = load_emg(emg_path, self.frame_rate)
        if not silent and speech_feature.shape[0] > len(emg) // self.downsample:
            speech_feature = speech_feature[:len(emg) // self.downsample]
        phonemes = read_frame_phonemes(tg_path, self.frame_rate, max_len=speech_feature.shape[0])
        if self.target_sec is not None:
            speech_feature, emg, phonemes = self.crop(speech_feature, emg, phonemes)
        if self.normalizer is not None:
            speech_feature = self.normalizer.normalize(speech_feature)
        return (torch.from_numpy(speech_feature).float(), torch.from_numpy(emg).float(),
                torch.from_numpy(phonemes), silent)

    def crop(self, speech_feature, emg, phonemes):
        """Random `target_sec` window, keeping EMG and features aligned."""
        speech_len, emg_len = len(speech_feature), len(emg)
        ratio = speech_len / emg_len
        diff = speech_len - self.target_speech_len
        speech_start = 0 if diff <= 0 else random.randint(0, diff)
        emg_start = int(speech_start / ratio)
        max_speech_crop = min(speech_len - speech_start, self.target_speech_len)
        max_emg_crop = min(emg_len - emg_start, self.target_emg_len)
        speech_crop_len = min(max_speech_crop, int(max_emg_crop * ratio))
        emg_crop_len = min(max_emg_crop, int(speech_crop_len / ratio))
        return (speech_feature[speech_start:speech_start + speech_crop_len],
                emg[emg_start:emg_start + emg_crop_len],
                phonemes[speech_start:speech_start + speech_crop_len])

    def collate(self, batch):
        batch.sort(key=lambda item: len(item[0]), reverse=True)
        speech_features = [item[0] for item in batch]
        emg_signals = [item[1] for item in batch]
        phonemes = [item[2] for item in batch]
        silents = [item[3] for item in batch]

        target_lengths = [len(f) for f in speech_features]
        est_lengths = [len(e) // self.downsample for e in emg_signals]
        max_len = max(target_lengths)
        max_emg_len = max(est_lengths) * self.downsample

        def pad(sequences, length, dtype=torch.float32):
            return torch.stack([torch.cat([s[:length], torch.zeros(length - len(s[:length]), *s.shape[1:], dtype=dtype)])
                                for s in sequences])

        return {
            'speech_features': pad(speech_features, max_len),
            'emg': pad(emg_signals, max_emg_len),
            'phonemes': pad(phonemes, max_len, dtype=torch.int64),
            'target_lengths': torch.tensor(target_lengths),
            'est_lengths': torch.tensor(est_lengths),
            'silents': torch.tensor(silents),
        }


class RefinerDataset(torch.utils.data.Dataset):
    """Frame-wise / merged phoneme pairs from LibriSpeech MFA TextGrids."""

    def __init__(self, mfa_path, data_split, frame_rate, target_sec=None):
        self.paths = sorted(glob(os.path.join(mfa_path, f'{data_split}-clean', '*.TextGrid')))
        self.frame_rate = frame_rate
        self.target_sec = target_sec
        random.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        frame_ids = torch.from_numpy(read_frame_phonemes(self.paths[i], self.frame_rate))
        if self.target_sec is not None:
            max_len = int(self.target_sec * self.frame_rate)
            start = random.randint(0, max(len(frame_ids) - max_len, 0))
            frame_ids = frame_ids[start:start + max_len]
        return frame_ids, merge_frame_phonemes(frame_ids)

    def collate(self, batch):
        batch.sort(key=lambda item: len(item[0]), reverse=True)
        frame_ids = [item[0] for item in batch]
        merged_ids = [item[1] for item in batch]
        frame_lengths = [len(f) for f in frame_ids]
        merged_lengths = [len(m) for m in merged_ids]

        def pad(sequences, length):
            return torch.stack([torch.cat([s, torch.full((length - len(s),), PAD_ID, dtype=torch.int64)])
                                for s in sequences])

        return {
            'frame_ids': pad(frame_ids, max(frame_lengths)),
            'merged_ids': pad(merged_ids, max(merged_lengths)),
            'frame_lengths': torch.tensor(frame_lengths),
            'merged_lengths': torch.tensor(merged_lengths),
        }
