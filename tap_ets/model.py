"""TAP-ETS (Sec. 3.1).

    h     = EMGEncoder(X)
    Z_ph  = PhonemeHead(h)
    Y_hat = MelHead(Decoder(Q=h, K,V=PhonemeEncoder(phi)))

phi is the ground-truth frame-wise phoneme sequence during training, and the encoder
prediction or a refined sequence at inference.
"""
import random

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tap_ets.layers import (MultiHeadCrossAttention, ResBlock, TransformerEncoder, padding_mask_from_ids,
                            padding_mask_from_lengths)
from tap_ets.loss import aligned_loss
from tap_ets.phonemes import NUM_PHONEMES, PAD_ID, SPECIAL_IDS
from tap_ets.utils import rename_state_dict_keys

RELATIVE_POSITIONAL_DISTANCE = 100


class EMGEncoder(nn.Module):
    """Three stride-2 conv blocks (8x downsampling) + Transformer."""

    def __init__(self, num_channels, d_model, num_layers, dropout):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ResBlock(num_channels, d_model, 2),
            ResBlock(d_model, d_model, 2),
            ResBlock(d_model, d_model, 2),
        )
        self.proj = nn.Linear(d_model, d_model)
        self.transformer = TransformerEncoder(d_model, nhead=8, num_layers=num_layers, dim_feedforward=4 * d_model,
                                              dropout=dropout, relative_positional_distance=RELATIVE_POSITIONAL_DISTANCE,
                                              final_norm=False, normalized_residual=True)

    def forward(self, emg, lengths=None):
        """[B, T, C] -> [B, T // 8, D]."""
        x = self.proj(self.conv_blocks(emg.transpose(1, 2)).transpose(1, 2))
        mask = padding_mask_from_lengths(lengths, x.shape[1]) if lengths is not None else None
        return self.transformer(x, mask)


class PhonemeEncoder(nn.Module):
    """Phoneme embedding + 3-layer Transformer."""

    def __init__(self, num_phonemes, d_model, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_phonemes, d_model, padding_idx=PAD_ID)
        self.transformer = TransformerEncoder(d_model, nhead=8, num_layers=3, dim_feedforward=4 * d_model,
                                              dropout=dropout, relative_positional_distance=RELATIVE_POSITIONAL_DISTANCE,
                                              final_norm=False, normalized_residual=True)

    def forward(self, phoneme_ids, key_padding_mask=None):
        return self.transformer(self.embedding(phoneme_ids), key_padding_mask)


class CrossAttentionDecoderLayer(nn.Module):
    """Q = EMG, K/V = phonemes; the EMG path stays residual."""

    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadCrossAttention(d_model, n_head=n_head, dropout=dropout, relative_positional=True,
                                                  relative_positional_distance=RELATIVE_POSITIONAL_DISTANCE)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(0.5))  # learned scale of the phoneme context
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, h, phoneme_ctx, phoneme_padding_mask=None):
        context = self.cross_attn(self.norm_q(h).transpose(0, 1), self.norm_kv(phoneme_ctx).transpose(0, 1),
                                  key_padding_mask=phoneme_padding_mask).transpose(0, 1)
        h = h + self.gate * self.dropout(context)
        return h + self.ff(self.norm_ff(h))


class CrossAttentionDecoder(nn.Module):
    def __init__(self, d_model, n_head, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([CrossAttentionDecoderLayer(d_model, n_head, dropout) for _ in range(num_layers)])

    def forward(self, h, phoneme_ctx, phoneme_padding_mask=None):
        for layer in self.layers:
            h = layer(h, phoneme_ctx, phoneme_padding_mask)
        return h


# key names of the released weights
LEGACY_PREFIXES = {
    'conv_blocks.': 'emg_encoder.conv_blocks.',
    'w_raw_in.': 'emg_encoder.proj.',
    'transformer.': 'emg_encoder.transformer.',
    'w_aux.': 'phoneme_head.',
    'ph_embedding.': 'phoneme_encoder.embedding.',
    'ph_transformer.': 'phoneme_encoder.transformer.',
    'emg_ph_fusion.': 'decoder.',
    'w_out.': 'mel_head.',
}
LEGACY_TOKENS = {'ln_ca_q': 'norm_q', 'ln_ca_mem': 'norm_kv', 'cross_gate': 'gate', 'ln_ff': 'norm_ff'}


class TAPETS(pl.LightningModule):
    def __init__(self, emg_enc_config: DictConfig, optimizer_config: DictConfig, feature_config: DictConfig,
                 num_phonemes: int = NUM_PHONEMES, phoneme_loss_weight: float = 0.5,
                 mel_loss_weight: float = 0.5, batch_size=None):
        super().__init__()
        self.save_hyperparameters('emg_enc_config', 'optimizer_config', 'feature_config', 'num_phonemes',
                                  'phoneme_loss_weight', 'mel_loss_weight', 'batch_size')
        d_model = emg_enc_config.model_size
        dropout = emg_enc_config.dropout
        self.emg_channels = list(emg_enc_config.use_channel)

        self.emg_encoder = EMGEncoder(len(self.emg_channels), d_model, emg_enc_config.num_layers, dropout)
        self.phoneme_head = nn.Linear(d_model, num_phonemes)
        self.phoneme_encoder = PhonemeEncoder(num_phonemes, d_model, dropout)
        self.decoder = CrossAttentionDecoder(d_model, n_head=8, num_layers=1, dropout=dropout)
        self.mel_head = nn.Linear(d_model, feature_config.dim)

        self.optimizer_config = optimizer_config
        self.phoneme_loss_weight = phoneme_loss_weight
        self.mel_loss_weight = mel_loss_weight
        self.batch_size = batch_size
        self.num_phonemes = num_phonemes

    def encode(self, emg, emg_lengths=None):
        """-> h [B, T', D], Z_ph [B, T', V]."""
        h = self.emg_encoder(emg[:, :, self.emg_channels], emg_lengths)
        return h, self.phoneme_head(h)

    def decode(self, h, phoneme_ids):
        """Mel from h, conditioned on phoneme_ids [B, T']."""
        mask = padding_mask_from_ids(phoneme_ids, PAD_ID)
        phoneme_ctx = self.phoneme_encoder(phoneme_ids, mask)
        return self.mel_head(self.decoder(h, phoneme_ctx, mask))

    def forward(self, emg, phoneme_ids=None, emg_lengths=None):
        """Without phoneme_ids the encoder prediction is used as conditioning."""
        h, phoneme_logits = self.encode(emg, emg_lengths)
        if phoneme_ids is None:
            phoneme_ids = phoneme_logits.argmax(-1)
        return self.decode(h, phoneme_ids), phoneme_logits

    @staticmethod
    def predict_phoneme_ids(phoneme_logits):
        """argmax Z_ph over real phonemes and silence."""
        logits = phoneme_logits.clone()
        logits[..., SPECIAL_IDS] = float('-inf')
        return logits.argmax(-1)

    def configure_optimizers(self):
        optimizer_class = hydra.utils.get_class(self.optimizer_config.target)
        optimizer = optimizer_class(self.parameters(), **self.optimizer_config.params)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.0,
                                      threshold_mode='abs')
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'epoch', 'frequency': 1}}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Linear LR warmup."""
        warmup_steps = self.optimizer_config.lr_warmup
        if self.trainer.global_step < warmup_steps:
            scale = float(self.trainer.global_step + 1) / float(max(1, warmup_steps))
            for group in optimizer.param_groups:
                group['lr'] = self.optimizer_config.params.lr * scale
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    @staticmethod
    def _random_shift(emg):
        """Augmentation: shift raw EMG left by up to 7 samples."""
        shift = random.randrange(8)
        if shift > 0:
            emg[:, :-shift, :] = emg[:, shift:, :].clone()
            emg[:, -shift:, :] = 0
        return emg

    def _compute_loss(self, batch, mel_pred, phoneme_logits, evaluate_phonemes):
        return aligned_loss(mel_pred, phoneme_logits, batch['speech_features'], batch['phonemes'], batch['silents'],
                            batch['target_lengths'], batch['est_lengths'], mel_weight=self.mel_loss_weight,
                            phoneme_weight=self.phoneme_loss_weight, evaluate_phonemes=evaluate_phonemes)

    def training_step(self, batch, batch_idx):
        emg = self._random_shift(batch['emg'])
        mel_pred, phoneme_logits = self(emg, phoneme_ids=batch['phonemes'], emg_lengths=batch['est_lengths'])
        loss, mel_loss, phoneme_loss, _, _ = self._compute_loss(batch, mel_pred, phoneme_logits, False)
        self.log('train_loss', loss, batch_size=self.batch_size)
        self.log('train_loss_mel', mel_loss, batch_size=self.batch_size)
        self.log('train_loss_ph', phoneme_loss, batch_size=self.batch_size)
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True,
                 batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        mel_pred, phoneme_logits = self(batch['emg'], emg_lengths=batch['est_lengths'])
        loss, mel_loss, phoneme_loss, accuracy, _ = self._compute_loss(batch, mel_pred, phoneme_logits, True)
        log_kwargs = dict(on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log('val_loss', loss, prog_bar=True, **log_kwargs)
        self.log('val_loss_mel', mel_loss, **log_kwargs)
        self.log('val_loss_ph', phoneme_loss, **log_kwargs)
        self.log('val_phone_accuracy', accuracy, prog_bar=True, **log_kwargs)
        return loss

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint['state_dict']
        if any(key.startswith(tuple(LEGACY_PREFIXES)) for key in state_dict):
            checkpoint['state_dict'] = rename_state_dict_keys(state_dict, LEGACY_PREFIXES, LEGACY_TOKENS)
