"""HiFi-GAN wrapper."""
import json
import os

import numpy as np
import torch

from hifi_gan.env import AttrDict
from hifi_gan.models import Generator


class HiFiGANVocoder:
    """`config.json` must sit next to the checkpoint."""

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        with open(os.path.join(os.path.dirname(checkpoint_path), 'config.json')) as f:
            hparams = AttrDict(json.load(f))
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = hparams.sampling_rate
        self.generator = Generator(hparams)
        state = torch.load(checkpoint_path, map_location='cpu')
        self.generator.load_state_dict(state['generator'] if 'generator' in state else state)
        self.generator.eval()
        self.generator.remove_weight_norm()
        self.generator.to(self.device)

    @torch.no_grad()
    def __call__(self, mel) -> np.ndarray:
        """[T, num_mels] log mel -> waveform."""
        mel = torch.as_tensor(np.asarray(mel), dtype=torch.float32)
        return self.generator(mel.T.unsqueeze(0).to(self.device)).squeeze().cpu().numpy()
