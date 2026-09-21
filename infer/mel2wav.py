"""Vocode the mel-spectrograms of emg2feat.py; writes .wav next to them."""
import os
import sys
from glob import glob

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.io.wavfile import write
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tap_ets.released import VOCODER_CHECKPOINT  # noqa: E402
from tap_ets.vocoder import HiFiGANVocoder  # noqa: E402


@hydra.main(version_base=None, config_path='../configs', config_name='mel2wav')
def main(cfg: DictConfig):
    synth_dir = os.path.join(cfg.exp_path, cfg.exp_name, 'synth', cfg.split)
    vocoder = HiFiGANVocoder(VOCODER_CHECKPOINT)
    mel_paths = sorted(glob(os.path.join(synth_dir, '*_mel.npy')))
    print(f'{len(mel_paths)} mel-spectrograms in {synth_dir}')
    for mel_path in tqdm(mel_paths):
        audio = vocoder(np.load(mel_path))
        audio = (audio / np.max(np.abs(audio))).astype(np.float32)
        write(mel_path.replace('_mel.npy', '.wav'), vocoder.sampling_rate, audio)


if __name__ == '__main__':
    main()
