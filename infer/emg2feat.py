"""EMG -> mel-spectrogram, conditioned on the refined phoneme sequence (Lev then NN)."""
import json
import os
import sys
from glob import glob

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tap_ets.dataset import load_emg  # noqa: E402
from tap_ets.features import FeatureNormalizer, find_normalizer  # noqa: E402
from tap_ets.phonemes import SIL_ID, SPECIAL_IDS  # noqa: E402
from tap_ets.refinement import refine_frame_ids  # noqa: E402
from tap_ets.released import GUIDE_JSON, load_refiner, load_tap_ets  # noqa: E402
from tap_ets.training import feature_dir  # noqa: E402


@hydra.main(version_base=None, config_path='../configs', config_name='emg2feat')
def main(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_dir = os.path.join(cfg.exp_path, cfg.exp_name)

    model, path = load_tap_ets(cfg.ckpt, map_location=device)
    model = model.to(device).eval()
    print(f'TAP-ETS weights: {path}')
    refiner, path = load_refiner(cfg.refiner_ckpt, map_location=device)
    refiner = refiner.to(device).eval()
    print(f'refinement weights: {path}')

    feature = model.hparams.feature_config
    normalizer = None
    if feature.normalize:
        normalizer_path = find_normalizer(exp_dir, feature_dir(cfg.data_path, feature))
        print(f'feature normalizer: {normalizer_path}')
        normalizer = FeatureNormalizer.load(normalizer_path)

    with open(GUIDE_JSON) as f:
        guide = json.load(f)
    print(f'guide sequences: {GUIDE_JSON} ({len(guide)} utterances)')

    emg_paths = sorted(glob(os.path.join(cfg.data_path, 'silent_speech_dataset', cfg.split,
                                         'silent_parallel_data', '*', '*_emg.npy')))
    output_dir = os.path.join(exp_dir, 'synth', cfg.split)
    os.makedirs(output_dir, exist_ok=True)
    print(f'{len(emg_paths)} utterances -> {output_dir}')

    with torch.no_grad():
        for emg_path in tqdm(emg_paths):
            session = os.path.basename(os.path.dirname(emg_path))
            idx = os.path.basename(emg_path).split('_')[0]
            key = f'{session}_{idx}'

            emg = torch.from_numpy(load_emg(emg_path, feature.frame_rate)).unsqueeze(0).to(device)
            h, phoneme_logits = model.encode(emg)
            corrected = [i for i in guide[key] if i not in SPECIAL_IDS and i != SIL_ID]
            if not corrected:
                raise ValueError(f'empty guide sequence for {key}')

            frame_ids = model.predict_phoneme_ids(phoneme_logits)[0]
            frame_ids = torch.tensor(refine_frame_ids(frame_ids.tolist(), corrected), device=device)
            frame_ids = refiner.refine(torch.tensor(corrected), frame_ids)

            mel = model.decode(h, frame_ids.unsqueeze(0))[0].cpu().numpy()
            if normalizer is not None:
                mel = normalizer.inverse(mel)
            np.save(os.path.join(output_dir, f'{key}_mel.npy'), mel.astype(np.float32))
            np.save(os.path.join(output_dir, f'{key}_phonemes.npy'), frame_ids.cpu().numpy())


if __name__ == '__main__':
    main()
