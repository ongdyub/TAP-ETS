"""Target mel-spectrograms for the voiced recordings, plus normalizer.npz fitted on the train split."""
import os
import sys

import hydra
import librosa
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tap_ets.features import FeatureNormalizer, mel_spectrogram  # noqa: E402
from tap_ets.training import feature_dir  # noqa: E402


class RunningStatistics:
    """Per-dimension mean and std over a stream of [T, D] arrays."""

    def __init__(self, dim):
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        n = x.shape[0]
        if n == 0:
            return
        batch_mean, batch_m2 = x.mean(0), ((x - x.mean(0)) ** 2).sum(0)
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.m2 += batch_m2 + delta ** 2 * self.count * n / total
        self.count = total

    @property
    def std(self):
        return np.sqrt(self.m2 / max(self.count, 1))


@hydra.main(version_base=None, config_path='../configs', config_name='speech2feat')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    feature = cfg.feature
    assert feature.group == 'mspec', f'Unsupported feature group: {feature.group}'
    data_dir = os.path.join(cfg.data_path, 'silent_speech_dataset')
    output_dir = feature_dir(cfg.data_path, feature)
    print(f'writing features to {output_dir}')

    stats = RunningStatistics(feature.dim)
    for root, _, files in tqdm(sorted(os.walk(data_dir))):
        if 'silent_parallel' in root:  # silent recordings have no speech target
            continue
        for file_name in sorted(files):
            if not file_name.endswith('.flac'):
                continue
            audio, sr = librosa.load(os.path.join(root, file_name), sr=None)
            if sr != feature.sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=feature.sr)
            mel = mel_spectrogram(torch.tensor(audio, dtype=torch.float32).unsqueeze(0), feature.nfft, feature.dim,
                                  feature.sr, feature.hop, feature.nfft, feature.fmin, feature.fmax, center=False)
            mel = mel[0].T.numpy()  # [T, dim]
            if os.path.relpath(root, data_dir).startswith('train'):
                stats.update(mel)
            save_dir = root.replace(data_dir, output_dir)
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, file_name.replace('audio_clean.flac', 'feat.npy')), mel)

    print(f'normalizer fitted on {stats.count} training frames')
    FeatureNormalizer(stats.mean, stats.std).save(os.path.join(output_dir, 'normalizer.npz'))


if __name__ == '__main__':
    main()
