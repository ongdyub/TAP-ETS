"""Transformer blocks. Relative-position attention follows fairseq PR #2225."""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def padding_mask_from_lengths(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """[B, max_len], True where padded."""
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)


def padding_mask_from_ids(ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """[B, T], True where padded."""
    return ids == pad_id


class ResBlock(nn.Module):
    """Residual 1-D conv block of the EMG front end."""

    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)
        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.residual_path is not None:
            residual = self.res_norm(self.residual_path(residual))
        return F.relu(x + residual)


class LearnedRelativePositionalEmbedding(nn.Module):
    """Relative-position logits within a window.

    `unmasked` learns 2 * max_relative_pos - 1 offsets (both directions) instead of
    max_relative_pos; offsets outside the window are suppressed.
    """

    def __init__(self, max_relative_pos: int, num_heads: int, embedding_dim: int, unmasked: bool = True):
        super().__init__()
        self.max_relative_pos = max_relative_pos
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.unmasked = unmasked
        num_embeddings = 2 * max_relative_pos - 1 if unmasked else max_relative_pos
        self.embeddings = nn.Parameter(torch.zeros(num_heads, num_embeddings, embedding_dim, 1))
        nn.init.normal_(self.embeddings, mean=0.0, std=embedding_dim ** -0.5)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """[L, B * H, d] queries -> [B * H, L, L] logits."""
        length = query.shape[0]
        embeddings = self._embeddings_for_length(length)[..., 0]
        logits = self._positional_logits(query, embeddings)
        return self._relative_to_absolute(logits)

    def _embeddings_for_length(self, length):
        pad_length = max(length - self.max_relative_pos, 0)
        start = max(self.max_relative_pos - length, 0)
        if self.unmasked:
            padded = F.pad(self.embeddings, (0, 0, 0, 0, pad_length, pad_length))
            return padded.narrow(-3, start, 2 * length - 1)
        padded = F.pad(self.embeddings, (0, 0, 0, 0, pad_length, 0))
        return padded.narrow(-3, start, length)

    def _positional_logits(self, query, embeddings):
        query = query.view(query.shape[0], -1, self.num_heads, self.embedding_dim)
        logits = torch.einsum('lbhd,hmd->lbhm', query, embeddings)
        logits = logits.contiguous().view(logits.shape[0], -1, logits.shape[-1])
        length = query.size(0)
        if length > self.max_relative_pos:
            pad_length = length - self.max_relative_pos
            logits[:, :, :pad_length] -= 1e8
            if self.unmasked:
                logits[:, :, -pad_length:] -= 1e8
        return logits

    def _relative_to_absolute(self, x):
        length, bsz_heads, _ = x.shape
        if self.unmasked:
            x = F.pad(x, (0, 1)).transpose(0, 1).contiguous().view(bsz_heads, length * 2 * length)
            x = F.pad(x, (0, length - 1)).view(bsz_heads, length + 1, 2 * length - 1)
            return x[:, :length, length - 1:]
        x = F.pad(x, (1, 0)).transpose(0, 1).contiguous().view(bsz_heads, length + 1, length)
        return x[:, 1:, :]


class MultiHeadAttention(nn.Module):
    """Self-attention over time-major inputs."""

    def __init__(self, d_model, n_head, dropout=0.1, relative_positional=True,
                 relative_positional_distance=100, relative_unmasked=True):
        super().__init__()
        assert d_model % n_head == 0, 'd_model must be divisible by n_head'
        self.n_head = n_head
        self.d_qkv = d_model // n_head
        self.w_q = nn.Parameter(torch.Tensor(n_head, d_model, self.d_qkv))
        self.w_k = nn.Parameter(torch.Tensor(n_head, d_model, self.d_qkv))
        self.w_v = nn.Parameter(torch.Tensor(n_head, d_model, self.d_qkv))
        self.w_o = nn.Parameter(torch.Tensor(n_head, self.d_qkv, d_model))
        for weight in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.xavier_normal_(weight)
        self.dropout = nn.Dropout(dropout)
        self.relative_positional = (
            LearnedRelativePositionalEmbedding(relative_positional_distance, n_head, self.d_qkv, relative_unmasked)
            if relative_positional else None
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [T, B, D], key_padding_mask: [B, T]."""
        T, B, _ = x.shape
        q = torch.einsum('tbd,hdf->bhtf', x, self.w_q)
        k = torch.einsum('tbd,hdf->bhtf', x, self.w_k)
        v = torch.einsum('tbd,hdf->bhtf', x, self.w_v)
        logits = torch.einsum('bhtf,bhsf->bhts', q, k) / math.sqrt(self.d_qkv)
        if self.relative_positional is not None:
            q_pos = q.permute(2, 0, 1, 3).contiguous().view(T, B * self.n_head, self.d_qkv)
            logits = logits + self.relative_positional(q_pos).view(B, self.n_head, T, T)
        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask.bool()[:, None, None, :], float('-inf'))
        probs = self.dropout(F.softmax(logits, dim=-1))
        out = torch.einsum('bhts,bhsf->bhtf', probs, v)
        return torch.einsum('bhtf,hfd->tbd', out, self.w_o)


class MultiHeadCrossAttention(nn.Module):
    """Cross-attention from x_q to x_kv. Relative-position logits only apply when Tq == Tk."""

    def __init__(self, d_model, n_head, dropout=0.1, relative_positional=False, relative_positional_distance=100):
        super().__init__()
        assert d_model % n_head == 0, 'd_model must be divisible by n_head'
        self.n_head = n_head
        self.d_qkv = d_model // n_head
        self.w_q = nn.Parameter(torch.Tensor(n_head, d_model, self.d_qkv))
        self.w_k = nn.Parameter(torch.Tensor(n_head, d_model, self.d_qkv))
        self.w_v = nn.Parameter(torch.Tensor(n_head, d_model, self.d_qkv))
        self.w_o = nn.Parameter(torch.Tensor(n_head, self.d_qkv, d_model))
        for weight in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.xavier_normal_(weight)
        self.dropout = nn.Dropout(dropout)
        self.relative_positional = (
            LearnedRelativePositionalEmbedding(relative_positional_distance, n_head, self.d_qkv, unmasked=True)
            if relative_positional else None
        )

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x_q: [Tq, B, D], x_kv: [Tk, B, D], key_padding_mask: [B, Tk]."""
        Tq, B, _ = x_q.shape
        Tk = x_kv.shape[0]
        q = torch.einsum('tbd,hdf->bhtf', x_q, self.w_q)
        k = torch.einsum('tbd,hdf->bhtf', x_kv, self.w_k)
        v = torch.einsum('tbd,hdf->bhtf', x_kv, self.w_v)
        logits = torch.einsum('bhtf,bhsf->bhts', q, k) / math.sqrt(self.d_qkv)
        if self.relative_positional is not None and Tq == Tk:
            q_pos = q.permute(2, 0, 1, 3).contiguous().view(Tq, B * self.n_head, self.d_qkv)
            logits = logits + self.relative_positional(q_pos).view(B, self.n_head, Tq, Tk)
        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask.bool()[:, None, None, :], float('-inf'))
        probs = self.dropout(F.softmax(logits, dim=-1))
        out = torch.einsum('bhts,bhsf->bhtf', probs, v)
        return torch.einsum('bhtf,hfd->tbd', out, self.w_o)


class TransformerEncoderLayer(nn.Module):
    """Encoder layer. `normalized_residual` normalizes the residual stream itself
    (x = norm(x); x = x + f(x)) instead of the pre-norm x = x + f(norm(x))."""

    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1,
                 relative_positional=True, relative_positional_distance=100, normalized_residual=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, relative_positional=relative_positional,
                                            relative_positional_distance=relative_positional_distance)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.normalized_residual = normalized_residual

    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = src
        if self.normalized_residual:
            x = self.norm1(x)
            x = x + self.dropout1(self.self_attn(x, key_padding_mask=src_key_padding_mask))
            x = self.norm2(x)
            x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))
            return x
        x = x + self.dropout1(self.self_attn(self.norm1(x), key_padding_mask=src_key_padding_mask))
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x))))))
        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer. Self-attention is not causally masked: the refinement model
    sees the whole corrupted sequence at once."""

    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1,
                 relative_positional=True, relative_positional_distance=100):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, relative_positional=relative_positional,
                                            relative_positional_distance=relative_positional_distance,
                                            relative_unmasked=False)
        self.cross_attn = MultiHeadCrossAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = tgt
        x = x + self.dropout1(self.self_attn(self.norm1(x), key_padding_mask=tgt_key_padding_mask))
        x = x + self.dropout2(self.cross_attn(self.norm2(x), memory, key_padding_mask=memory_key_padding_mask))
        x = x + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(self.norm3(x))))))
        return x


class TransformerEncoder(nn.Module):
    """Batch-first [B, T, D]."""

    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1,
                 relative_positional=True, relative_positional_distance=100, final_norm=True,
                 normalized_residual=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, relative_positional,
                                    relative_positional_distance, normalized_residual)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model) if final_norm else None

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x.transpose(0, 1)
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=key_padding_mask)
        if self.final_norm is not None:
            h = self.final_norm(h)
        return h.transpose(0, 1)


class TransformerDecoder(nn.Module):
    """Batch-first [B, T, D] attending to memory [B, S, D]."""

    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1,
                 relative_positional=True, relative_positional_distance=100):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, relative_positional,
                                    relative_positional_distance)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = tgt.transpose(0, 1)
        mem = memory.transpose(0, 1)
        for layer in self.layers:
            x = layer(x, mem, tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask)
        return self.final_norm(x).transpose(0, 1)
