"""Seeding, checkpoint lookup and state-dict helpers."""
import os
import random
from typing import Dict, Optional

import numpy as np
import torch
from pytorch_lightning import seed_everything


def set_global_seed(seed: int):
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed, workers=True)


def enable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def rename_state_dict_keys(state_dict: Dict[str, torch.Tensor], prefix_map: Dict[str, str],
                           token_map: Optional[Dict[str, str]] = None) -> Dict[str, torch.Tensor]:
    """Replace the first matching key prefix, then map dotted components."""
    renamed = {}
    for key, value in state_dict.items():
        for old, new in prefix_map.items():
            if key.startswith(old):
                key = new + key[len(old):]
                break
        if token_map:
            key = '.'.join(token_map.get(part, part) for part in key.split('.'))
        renamed[key] = value
    return renamed
