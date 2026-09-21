import torch

from tap_ets.phonemes import (NUM_PHONEMES, PAD_ID, PHONEME_INVENTORY, SIL_ID, UNK_ID, merge_frame_phonemes,
                              normalize_phoneme, phoneme_string_to_ids)


def test_inventory_size_matches_released_checkpoints():
    assert NUM_PHONEMES == 53
    assert PHONEME_INVENTORY[PAD_ID] == '<pad>'


def test_normalize_phoneme():
    assert normalize_phoneme('AH0') == 'ah'
    assert normalize_phoneme('') == 'sil'
    assert normalize_phoneme('spn') == 'sil'


def test_phoneme_string_to_ids():
    ids = phoneme_string_to_ids('HH EH1 N D ER0 XYZ')
    assert ids[:5] == [PHONEME_INVENTORY.index(t) for t in ['hh', 'eh', 'n', 'd', 'er']]
    assert ids[-1] == UNK_ID


def test_merge_frame_phonemes():
    s, ay = PHONEME_INVENTORY.index('s'), PHONEME_INVENTORY.index('ay')
    merged = merge_frame_phonemes(torch.tensor([SIL_ID, s, s, ay, SIL_ID, ay, ay, SIL_ID]))
    assert merged.tolist() == [s, ay, ay]  # silence separates repeated phonemes
    assert merge_frame_phonemes([SIL_ID, SIL_ID]).tolist() == [UNK_ID]
