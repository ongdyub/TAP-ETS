"""Training objective, Eq. (3)."""
import numpy as np
import torch
import torch.nn.functional as F

from tap_ets.dtw import align_from_distances
from tap_ets.phonemes import NUM_PHONEMES, SIL_ID


def _count_phonemes(predicted, target, correct_total, confusion):
    """Frame-wise accuracy, skipping frames that are silence on both sides."""
    for p, t in zip(predicted.tolist(), target.tolist()):
        if p == SIL_ID and t == SIL_ID:
            continue
        correct_total[1] += 1
        correct_total[0] += int(p == t)
        confusion[p, t] += 1


def aligned_loss(mel_pred, phoneme_logits, mel_target, phoneme_target, silent, target_lengths, pred_lengths,
                 mel_weight=0.5, phoneme_weight=0.5, evaluate_phonemes=False):
    """Sum of mel distance and phoneme cross-entropy over the alignment path.

    The path is the identity for voiced EMG and DTW on the joint cost for silent EMG.
    Returns (loss, mel loss, phoneme loss, accuracy, confusion), normalized by target frames.
    """
    confusion = np.zeros((NUM_PHONEMES, NUM_PHONEMES)) if evaluate_phonemes else None
    correct_total = [0, 0]
    losses, mel_losses, phoneme_losses = [], [], []
    total_frames = 0
    for pred, y, logits, y_ph, is_silent, target_len, pred_len in zip(
            mel_pred, mel_target, phoneme_logits, phoneme_target, silent, target_lengths, pred_lengths):
        valid_pred = pred_len if is_silent else target_len
        pred, logits = pred[:valid_pred], logits[:valid_pred]
        y, y_ph = y[:target_len], y_ph[:target_len].to(pred.device)

        if is_silent:
            dists = torch.cdist(pred.unsqueeze(0), y.unsqueeze(0)).squeeze(0)     # [T', T]
            log_probs = F.log_softmax(logits, -1)[:, y_ph]
            costs = mel_weight * dists - phoneme_weight * log_probs
            path = align_from_distances(costs.T.detach().cpu().numpy())           # target -> prediction
            frames = range(len(path))
            loss = costs[path, frames].sum()
            mel_loss = dists[path, frames].sum()
            phoneme_loss = -log_probs[path, frames].sum()
            if evaluate_phonemes:
                path_t = align_from_distances(costs.detach().cpu().numpy())       # prediction -> target
                _count_phonemes(logits.argmax(-1), y_ph[path_t], correct_total, confusion)
        else:
            assert y.size(0) == pred.size(0)
            mel_loss = F.pairwise_distance(y, pred).sum()
            phoneme_loss = F.cross_entropy(logits, y_ph, reduction='sum')
            loss = mel_weight * mel_loss + phoneme_weight * phoneme_loss
            if evaluate_phonemes:
                _count_phonemes(logits.argmax(-1), y_ph, correct_total, confusion)

        losses.append(loss)
        mel_losses.append(mel_loss)
        phoneme_losses.append(phoneme_loss)
        total_frames += y.size(0)

    accuracy = correct_total[0] / max(correct_total[1], 1e-5) if evaluate_phonemes else None
    return (sum(losses) / total_frames, sum(mel_losses) / total_frames, sum(phoneme_losses) / total_frames,
            accuracy, confusion)
