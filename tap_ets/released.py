"""Released weights.

The two .pt files are not in the repository; download them to these paths, or point
`ckpt` / `refiner_ckpt` at your own copies.

    data/models/tap_ets.pt       TAP-ETS backbone
    data/models/tap_refine.pt    refinement model

https://github.com/ongdyub/TAP-ETS/releases/tag/v1.0
"""
import os

import torch
from omegaconf import OmegaConf

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASED = 'released'  # config value selecting the paths below

TAP_ETS_WEIGHTS = os.path.join(REPO_DIR, 'data', 'models', 'tap_ets.pt')
REFINER_WEIGHTS = os.path.join(REPO_DIR, 'data', 'models', 'tap_refine.pt')
VOCODER_CHECKPOINT = os.path.join(REPO_DIR, 'data', 'hifigan_finetuned', 'checkpoint')
GUIDE_JSON = os.path.join(REPO_DIR, 'data', 'MONA_LISA', 'guide.json')

# The .pt files are bare state_dicts; these are the hyper-parameters of the training runs.
ARCHITECTURE = {
    'emg_enc_config': {
        'model_size': 768, 'num_layers': 6, 'dropout': 0.2,
        'use_channel': [0, 1, 2, 3, 4, 5, 6, 7], 'channel_dropout': 0.0,
    },
    'optimizer_config': {
        'target': 'torch.optim.AdamW', 'lr_warmup': 500,
        'params': {'lr': 0.001, 'weight_decay': 1e-07},
    },
    'feature_config': {
        'group': 'mspec', 'target': 'mspec', 'sub_option': 'sr', 'sr': 22050,
        'hop': 256, 'nfft': 1024, 'dim': 80, 'frame_rate': 86.13, 'normalize': True,
    },
}


def _config(name):
    return OmegaConf.create(ARCHITECTURE[name])


def require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found - see the download step in README.md')
    return path


def load_tap_ets(path=RELEASED, map_location='cpu'):
    """Load the backbone from a released .pt or a Lightning .ckpt."""
    from tap_ets.model import LEGACY_PREFIXES, LEGACY_TOKENS, TAPETS
    from tap_ets.utils import rename_state_dict_keys

    path = require(TAP_ETS_WEIGHTS if path == RELEASED else path)
    if path.endswith('.ckpt'):
        return TAPETS.load_from_checkpoint(path, map_location=map_location), path
    model = TAPETS(_config('emg_enc_config'), _config('optimizer_config'), _config('feature_config'))
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(rename_state_dict_keys(state.get('state_dict', state), LEGACY_PREFIXES, LEGACY_TOKENS))
    return model, path


def load_refiner(path=RELEASED, map_location='cpu'):
    """Load the refinement model from a released .pt or a Lightning .ckpt."""
    from tap_ets.refiner import LEGACY_PREFIXES, FrameWisePhonemeRefiner
    from tap_ets.utils import rename_state_dict_keys

    path = require(REFINER_WEIGHTS if path == RELEASED else path)
    if path.endswith('.ckpt'):
        return FrameWisePhonemeRefiner.load_from_checkpoint(path, map_location=map_location), path
    model = FrameWisePhonemeRefiner(_config('emg_enc_config'), _config('optimizer_config'))
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(rename_state_dict_keys(state.get('state_dict', state), LEGACY_PREFIXES))
    return model, path
