"""Levenshtein refinement (Sec. 3.2.1).

The frame-wise sequence is run-length encoded, the corrected tokens are aligned to its
non-silence segments, and each such region ("island") keeps its total duration.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from tap_ets.phonemes import PHONEME_INVENTORY, SIL, ids_to_phonemes

Segment = Tuple[str, int]


def run_length_segments(tokens: Sequence[str]) -> List[Segment]:
    """['sil', 'sil', 's'] -> [('sil', 2), ('s', 1)]."""
    segments: List[Segment] = []
    for token in tokens:
        if segments and segments[-1][0] == token:
            segments[-1] = (token, segments[-1][1] + 1)
        else:
            segments.append((token, 1))
    return segments


def expand_segments(segments: Sequence[Segment]) -> List[str]:
    """Inverse of run_length_segments."""
    tokens: List[str] = []
    for token, length in segments:
        if length < 0:
            raise ValueError(f'Negative duration {length} for token {token}')
        tokens.extend([token] * int(length))
    return tokens


@dataclass
class TokenAlignment:
    """per_source[i] = {'source', 'target' (None if deleted), 'op'};
    insertions[i] = tokens inserted before source[i], insertions[-1] is the tail."""
    per_source: List[Dict[str, Optional[str]]]
    insertions: List[List[str]]


def levenshtein_align(source: List[str], target: List[str]) -> TokenAlignment:
    n, m = len(source), len(target)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0], back[i][0] = i, 'D'
    for j in range(1, m + 1):
        cost[0][j], back[0][j] = j, 'I'
    priority = {'S': 0, 'D': 1, 'I': 2}  # tie-break order
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = [
                (cost[i - 1][j - 1] + (0 if source[i - 1] == target[j - 1] else 1), 'S'),
                (cost[i - 1][j] + 1, 'D'),
                (cost[i][j - 1] + 1, 'I'),
            ]
            cost[i][j], back[i][j] = min(candidates, key=lambda c: (c[0], priority[c[1]]))

    steps = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == 'S':
            steps.append((source[i - 1], target[j - 1]))
            i, j = i - 1, j - 1
        elif move == 'D':
            steps.append((source[i - 1], None))
            i -= 1
        else:
            steps.append((None, target[j - 1]))
            j -= 1
    steps.reverse()

    per_source: List[Dict[str, Optional[str]]] = []
    insertions: List[List[str]] = [[] for _ in range(n + 1)]
    for src, tgt in steps:
        if src is None:
            insertions[len(per_source)].append(tgt)
        elif tgt is None:
            per_source.append({'source': src, 'target': None, 'op': 'del'})
        else:
            per_source.append({'source': src, 'target': tgt, 'op': 'keep' if src == tgt else 'sub'})
    return TokenAlignment(per_source=per_source, insertions=insertions)


def _split_evenly(total: int, parts: int) -> List[int]:
    if parts <= 0:
        return []
    base, remainder = divmod(total, parts)
    return [base + (1 if i < remainder else 0) for i in range(parts)]


def redistribute_island(segments: List[Segment], corrected_tokens: List[str],
                        prefer_left_for_del: bool = True) -> List[Segment]:
    """Spread one island's duration over `corrected_tokens`.

    Kept and substituted tokens keep their duration, a deleted token gives its duration to
    the nearest survivor, and inserted tokens share the duration of their cluster evenly.
    """
    source_tokens = [token for token, _ in segments]
    source_lengths = [int(length) for _, length in segments]
    if any(length < 0 for length in source_lengths):
        raise ValueError('Negative frame length in segments.')
    if not corrected_tokens:
        return []

    alignment = levenshtein_align(source_tokens, corrected_tokens)
    n = len(source_tokens)
    out_tokens: List[str] = []
    out_lengths: List[int] = []
    survivor: List[bool] = []
    deletions: List[Tuple[int, int]] = []          # (output position, duration)
    insertion_start = [0] * (n + 1)
    for i in range(n):
        insertion_start[i] = len(out_tokens)
        for token in alignment.insertions[i]:
            out_tokens.append(token)
            out_lengths.append(0)
            survivor.append(False)
        info = alignment.per_source[i]
        if info['op'] == 'del':
            deletions.append((len(out_tokens), source_lengths[i]))
            continue
        out_tokens.append(info['target'])
        out_lengths.append(source_lengths[i])
        survivor.append(True)
    insertion_start[n] = len(out_tokens)
    for token in alignment.insertions[n]:
        out_tokens.append(token)
        out_lengths.append(0)
        survivor.append(False)

    if not out_tokens:
        return []
    survivors = [k for k, is_survivor in enumerate(survivor) if is_survivor]
    if not survivors:
        return list(zip(out_tokens, _split_evenly(sum(source_lengths), len(out_tokens))))

    def nearest_left(position):
        return next((k for k in reversed(survivors) if k < position), None)

    def nearest_right(position):
        return next((k for k in survivors if k >= position), None)

    for position, duration in deletions:
        left, right = nearest_left(position), nearest_right(position)
        if left is None and right is None:
            continue
        if left is None:
            out_lengths[right] += duration
        elif right is None:
            out_lengths[left] += duration
        else:
            out_lengths[left if prefer_left_for_del else right] += duration

    for i in range(n + 1):
        count = len(alignment.insertions[i])
        if count == 0:
            continue
        start = insertion_start[i]
        inserted = list(range(start, start + count))
        left, right = nearest_left(start), nearest_right(start + count)
        cluster, total = [], 0
        if left is not None:
            cluster.append(left)
            total += out_lengths[left]
        cluster.extend(inserted)
        if right is not None and right != left:
            cluster.append(right)
            total += out_lengths[right]
        if len(cluster) == count:  # no surviving neighbour
            continue
        for index, length in zip(cluster, _split_evenly(total, len(cluster))):
            out_lengths[index] = length

    return [(token, int(length)) for token, length in zip(out_tokens, out_lengths)]


def levenshtein_refine(segments: List[Segment], corrected_tokens: List[str], sil_token: str = SIL,
                       prefer_left_for_del: bool = True, absorb_side: str = 'left') -> List[Segment]:
    """Align `corrected_tokens` (no silence) to the islands of `segments`, keeping the sil segments.

    An island left without tokens gives its duration to a neighbouring sil, and adjacent sils merge.
    """
    if absorb_side not in ('left', 'right'):
        raise ValueError("absorb_side must be 'left' or 'right'.")

    islands: List[Tuple[int, int]] = []   # inclusive index ranges of non-sil runs
    ns_tokens, ns_lengths, ns_island = [], [], []
    start = None
    for index, (token, length) in enumerate(segments):
        if token == sil_token:
            if start is not None:
                islands.append((start, index - 1))
                start = None
            continue
        if start is None:
            start = index
        ns_tokens.append(token)
        ns_lengths.append(int(length))
        ns_island.append(len(islands))
    if start is not None:
        islands.append((start, len(segments) - 1))
    if not ns_tokens:
        return list(segments)

    # align all non-sil tokens at once, then attribute the corrected tokens to islands
    alignment = levenshtein_align(ns_tokens, corrected_tokens)
    corrected_by_island: Dict[int, List[str]] = {}
    for i, island in enumerate(ns_island):
        corrected_by_island.setdefault(island, []).extend(alignment.insertions[i])
        info = alignment.per_source[i]
        if info['op'] != 'del':
            corrected_by_island.setdefault(island, []).append(info['target'])
    if alignment.insertions[len(ns_tokens)]:
        corrected_by_island.setdefault(ns_island[-1], []).extend(alignment.insertions[len(ns_tokens)])

    segments_by_island: Dict[int, List[Segment]] = {}
    for token, length, island in zip(ns_tokens, ns_lengths, ns_island):
        segments_by_island.setdefault(island, []).append((token, length))

    PENDING = '_PENDING_SIL_'
    out: List[Segment] = []
    position = 0
    for island, (first, last) in enumerate(islands):
        while position < first:                       # sil before the island
            out.append((segments[position][0], int(segments[position][1])))
            position += 1
        corrected = corrected_by_island.get(island, [])
        if not corrected:                             # empty island: absorb into a sil
            duration = sum(length for _, length in segments_by_island[island])
            left_sil = next((k for k in range(len(out) - 1, -1, -1) if out[k][0] == sil_token), None)
            if absorb_side == 'left' and left_sil is not None:
                out[left_sil] = (sil_token, out[left_sil][1] + duration)
            else:
                out.append((PENDING, duration))
        else:
            out.extend(redistribute_island(segments_by_island[island], corrected, prefer_left_for_del))
        position = last + 1
    out.extend((token, int(length)) for token, length in segments[position:])

    merged: List[Segment] = []
    pending = 0
    for token, length in out:
        if token == PENDING:
            pending += length
            continue
        if token == sil_token and pending:
            length, pending = length + pending, 0
        if merged and token == sil_token and merged[-1][0] == sil_token:
            merged[-1] = (sil_token, merged[-1][1] + length)
        else:
            merged.append((token, length))
    if pending:
        for k in range(len(merged) - 1, -1, -1):
            if merged[k][0] == sil_token:
                merged[k] = (sil_token, merged[k][1] + pending)
                break
    return merged


def refine_frame_ids(frame_ids: Sequence[int], corrected_ids: Sequence[int]) -> List[int]:
    """Project corrected ids onto the frame grid, keeping the length."""
    segments = run_length_segments(ids_to_phonemes(frame_ids))
    refined = levenshtein_refine(segments, ids_to_phonemes(corrected_ids))
    frame_tokens = expand_segments(refined)
    if len(frame_tokens) != len(frame_ids):
        raise RuntimeError(f'Refinement changed the sequence length from {len(frame_ids)} to {len(frame_tokens)}.')
    return [PHONEME_INVENTORY.index(token) for token in frame_tokens]
