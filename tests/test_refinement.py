"""Levenshtein refinement."""
import random

import pytest

from tap_ets.phonemes import PHONEME_INVENTORY, SIL_ID, ids_to_phonemes
from tap_ets.refinement import (expand_segments, levenshtein_align, levenshtein_refine, redistribute_island,
                                refine_frame_ids, run_length_segments)

PHONES = [t for t in PHONEME_INVENTORY if t not in ('<pad>', '<bos>', '<eos>', '<unk>', '<blk>', 'sil')]


def test_run_length_roundtrip():
    tokens = ['sil', 'sil', 's', 's', 's', 'ay', 'sil']
    segments = run_length_segments(tokens)
    assert segments == [('sil', 2), ('s', 3), ('ay', 1), ('sil', 1)]
    assert expand_segments(segments) == tokens


def test_alignment_operations():
    alignment = levenshtein_align(['s', 'ay', 'l', 'n', 'd'], ['s', 'ay', 'l', 'ah', 'n', 't'])
    assert [info['op'] for info in alignment.per_source] == ['keep', 'keep', 'keep', 'keep', 'sub']
    assert alignment.insertions[3] == ['ah']  # inserted before 'n'


def test_island_duration_is_preserved():
    segments = [('s', 3), ('ay', 5), ('l', 2), ('n', 4), ('d', 1)]
    refined = redistribute_island(segments, ['s', 'ay', 'l', 'ah', 'n', 't'])
    assert [token for token, _ in refined] == ['s', 'ay', 'l', 'ah', 'n', 't']
    assert sum(length for _, length in refined) == sum(length for _, length in segments)


def test_empty_island_is_absorbed_into_silence():
    segments = [('sil', 2), ('s', 3), ('sil', 4), ('t', 2), ('sil', 1)]
    refined = levenshtein_refine(segments, ['t'])
    assert refined == [('sil', 9), ('t', 2), ('sil', 1)]


def test_refine_frame_ids_keeps_length_and_silence_structure():
    random.seed(0)
    for _ in range(200):
        frame_tokens = [random.choice(['sil', 'sil'] + PHONES[:6]) for _ in range(random.randint(1, 80))]
        frame_ids = [PHONEME_INVENTORY.index(t) for t in frame_tokens]
        corrected = [PHONEME_INVENTORY.index(random.choice(PHONES[:6])) for _ in range(random.randint(1, 15))]
        refined = refine_frame_ids(frame_ids, corrected)
        assert len(refined) == len(frame_ids)
        assert SIL_ID not in corrected
        assert set(ids_to_phonemes(refined)) <= set(ids_to_phonemes(corrected)) | {'sil'}


def test_all_silence_input_is_returned_unchanged():
    assert refine_frame_ids([SIL_ID, SIL_ID], [PHONEME_INVENTORY.index('s')]) == [SIL_ID, SIL_ID]


def test_expand_segments_rejects_negative_durations():
    with pytest.raises(ValueError):
        expand_segments([('s', -1)])
