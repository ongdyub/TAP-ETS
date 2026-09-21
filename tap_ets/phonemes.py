"""Phoneme inventory and frame-wise phoneme utilities."""
import string
from typing import Iterable, List, Sequence

import numpy as np
import torch
from textgrids import TextGrid

PHONEME_INVENTORY = [
    '<pad>', '<bos>', '<eos>', '<unk>',
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'axr', 'ay', 'b', 'ch', 'd',
    'dh', 'dx', 'eh', 'el', 'em', 'en', 'er', 'ey', 'f', 'g',
    'hh', 'hv', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'nx',
    'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh',
    'uw', 'v', 'w', 'y', 'z', 'zh', 'sil', '<blk>',
]
NUM_PHONEMES = len(PHONEME_INVENTORY)
PAD_ID = PHONEME_INVENTORY.index('<pad>')
UNK_ID = PHONEME_INVENTORY.index('<unk>')
SIL = 'sil'
SIL_ID = PHONEME_INVENTORY.index(SIL)
MASK_ID = PHONEME_INVENTORY.index('<blk>')  # [MASK] of the refinement model
SPECIAL_IDS = [PHONEME_INVENTORY.index(t) for t in ('<pad>', '<bos>', '<eos>', '<unk>', '<blk>')]


def normalize_phoneme(token: str) -> str:
    """ARPAbet/MFA label -> inventory entry."""
    token = token.strip().lower()
    if token in ('', 'sp', 'spn'):
        return SIL
    if token[-1] in string.digits:
        token = token[:-1]
    return token


def phoneme_to_id(token: str) -> int:
    token = normalize_phoneme(token)
    return PHONEME_INVENTORY.index(token) if token in PHONEME_INVENTORY else UNK_ID


def phoneme_string_to_ids(text: str) -> List[int]:
    """'HH EH1 N ...' -> ids."""
    return [phoneme_to_id(t) for t in text.split()]


def ids_to_phonemes(ids: Iterable[int]) -> List[str]:
    return [PHONEME_INVENTORY[int(i)] for i in ids]


def read_frame_phonemes(textgrid_path: str, frame_rate: float, max_len: int = None) -> np.ndarray:
    """Read the `phones` tier of an MFA TextGrid as frame-wise ids."""
    intervals = TextGrid(textgrid_path)['phones']
    phone_ids = np.full(int(intervals[-1].xmax * frame_rate) + 1, -1, dtype=np.int64)
    phone_ids[-1] = SIL_ID
    for interval in intervals:
        start, end = int(interval.xmin * frame_rate), int(interval.xmax * frame_rate)
        phone_ids[start:end] = PHONEME_INVENTORY.index(normalize_phoneme(interval.text))
    assert (phone_ids >= 0).all(), f'missing aligned phones in {textgrid_path}'
    if max_len is not None:
        phone_ids = phone_ids[:max_len]
        assert phone_ids.shape[0] == max_len, f'{textgrid_path} is shorter than {max_len} frames'
    return phone_ids


def merge_frame_phonemes(frame_ids: Sequence[int]) -> torch.Tensor:
    """Collapse repeats and drop silence; silence still separates repeated phonemes."""
    if torch.is_tensor(frame_ids):
        frame_ids = frame_ids.tolist()
    merged, prev = [], None
    for p in frame_ids:
        p = int(p)
        if p == prev:
            continue
        prev = p
        if p == SIL_ID:
            prev = None
            continue
        if p in SPECIAL_IDS:
            continue
        merged.append(p)
    return torch.tensor(merged or [UNK_ID], dtype=torch.long)
