"""Masking-based refinement model (Sec. 3.2.2).

Encoder memory is the merged phoneme sequence, decoder input the corrupted frame-wise
sequence, target the clean one.
"""
import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tap_ets.layers import TransformerDecoder, TransformerEncoder, padding_mask_from_lengths
from tap_ets.phonemes import MASK_ID, NUM_PHONEMES, PAD_ID, PHONEME_INVENTORY
from tap_ets.utils import rename_state_dict_keys

RELATIVE_POSITIONAL_DISTANCE = 100
LEGACY_PREFIXES = {'ph_embed.': 'embedding.', 'dec_out.': 'classifier.'}  # released weights


class FrameWisePhonemeRefiner(pl.LightningModule):
    def __init__(self, model_config: DictConfig, optimizer_config: DictConfig, p_corrupt: float = 1.0,
                 p_mask: float = 0.2, p_replace: float = 0.5, alpha_clean: float = 0.1):
        super().__init__()
        self.save_hyperparameters('model_config', 'optimizer_config', 'p_corrupt', 'p_mask',
                                  'p_replace', 'alpha_clean')
        if p_mask < 0 or p_replace < 0 or p_mask + p_replace > 1:
            raise ValueError('p_mask and p_replace must be non-negative and sum to at most 1.')
        d_model, num_layers, dropout = model_config.model_size, model_config.num_layers, model_config.dropout
        self.optimizer_config = optimizer_config
        self.p_corrupt, self.p_mask, self.p_replace, self.alpha_clean = p_corrupt, p_mask, p_replace, alpha_clean

        self.vocab_size = NUM_PHONEMES
        self.mask_id = MASK_ID
        self.embedding = nn.Embedding(NUM_PHONEMES, d_model, padding_idx=PAD_ID)
        self.encoder = TransformerEncoder(d_model, nhead=8, num_layers=num_layers, dim_feedforward=4 * d_model,
                                          dropout=dropout, relative_positional_distance=RELATIVE_POSITIONAL_DISTANCE)
        self.decoder = TransformerDecoder(d_model, nhead=8, num_layers=num_layers, dim_feedforward=4 * d_model,
                                          dropout=dropout, relative_positional_distance=RELATIVE_POSITIONAL_DISTANCE)
        self.classifier = nn.Linear(d_model, NUM_PHONEMES)

        excluded = {'<pad>', '<bos>', '<eos>', '<blk>'}
        replacement_ids = [i for i, token in enumerate(PHONEME_INVENTORY) if token not in excluded]
        self.register_buffer('replacement_ids', torch.tensor(replacement_ids, dtype=torch.long), persistent=False)
        self.token_loss = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction='none')

    def forward(self, merged_ids, merged_lengths, frame_ids, frame_lengths):
        """-> logits [B, T, V]."""
        memory_mask = padding_mask_from_lengths(merged_lengths, merged_ids.size(1))
        frame_mask = padding_mask_from_lengths(frame_lengths, frame_ids.size(1))
        memory = self.encoder(self.embedding(merged_ids), memory_mask)
        decoded = self.decoder(self.embedding(frame_ids), memory, frame_mask, memory_mask)
        return self.classifier(decoded)

    @torch.no_grad()
    def refine(self, merged_ids: torch.Tensor, frame_ids: torch.Tensor) -> torch.Tensor:
        """One sequence: merged [L] + frames [T] -> refined ids [T]."""
        device = self.embedding.weight.device
        merged_ids = merged_ids.to(device).long().view(1, -1)
        frame_ids = frame_ids.to(device).long().view(1, -1)
        logits = self(merged_ids, torch.tensor([merged_ids.size(1)], device=device),
                      frame_ids, torch.tensor([frame_ids.size(1)], device=device))
        return logits.argmax(-1).squeeze(0)

    @torch.no_grad()
    def corrupt(self, clean_ids, lengths):
        """Select a frame with prob. p_corrupt, then [MASK] it with prob. p_mask or
        replace it with a random phoneme with prob. p_replace."""
        valid = ~padding_mask_from_lengths(lengths, clean_ids.size(1))
        corrupted = clean_ids.masked_fill(~valid, PAD_ID)
        selected = valid & (torch.rand_like(clean_ids, dtype=torch.float) < self.p_corrupt)
        u = torch.rand_like(clean_ids, dtype=torch.float)
        do_mask = (u < self.p_mask) & selected
        do_replace = (u >= self.p_mask) & (u < self.p_mask + self.p_replace) & selected
        corrupted = corrupted.masked_fill(do_mask, self.mask_id)
        random_ids = self.replacement_ids[torch.randint(0, len(self.replacement_ids), clean_ids.shape,
                                                        device=clean_ids.device)]
        corrupted = torch.where(do_replace, random_ids, corrupted)
        return corrupted, do_mask | do_replace

    def loss(self, logits, targets, lengths, corrupt_mask):
        """Eq. (4)."""
        B, T, V = logits.shape
        valid = ~padding_mask_from_lengths(lengths, T) & (targets != PAD_ID)
        token_loss = self.token_loss(logits.reshape(B * T, V), targets.reshape(B * T)).reshape(B, T)
        corrupt, clean = corrupt_mask & valid, ~corrupt_mask & valid
        loss_corrupt = (token_loss * corrupt).sum() / corrupt.sum().clamp_min(1)
        loss_clean = (token_loss * clean).sum() / clean.sum().clamp_min(1)
        return loss_corrupt + self.alpha_clean * loss_clean, loss_corrupt, loss_clean

    @torch.no_grad()
    def accuracy(self, logits, targets, lengths, corrupt_mask):
        pred = logits.argmax(-1)
        valid = ~padding_mask_from_lengths(lengths, targets.size(1)) & (targets != PAD_ID)
        corrupt = corrupt_mask & valid
        acc_all = ((pred == targets) & valid).sum() / valid.sum().clamp_min(1)
        acc_corrupt = ((pred == targets) & corrupt).sum() / corrupt.sum().clamp_min(1)
        return acc_all, acc_corrupt

    def _step(self, batch):
        frame, merged = batch['frame_ids'], batch['merged_ids']
        frame_lengths, merged_lengths = batch['frame_lengths'], batch['merged_lengths']
        corrupted, corrupt_mask = self.corrupt(frame, frame_lengths)
        logits = self(merged, merged_lengths, corrupted, frame_lengths)
        loss, loss_corrupt, loss_clean = self.loss(logits, frame, frame_lengths, corrupt_mask)
        acc_all, acc_corrupt = self.accuracy(logits, frame, frame_lengths, corrupt_mask)
        return loss, loss_corrupt, loss_clean, acc_all, acc_corrupt

    def training_step(self, batch, batch_idx):
        loss, loss_corrupt, loss_clean, acc_all, acc_corrupt = self._step(batch)
        B = batch['frame_ids'].size(0)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=B)
        self.log('train_loss_corrupt', loss_corrupt, on_step=True, on_epoch=True, batch_size=B)
        self.log('train_loss_clean', loss_clean, on_step=True, on_epoch=True, batch_size=B)
        self.log('train_acc_all', acc_all, prog_bar=True, on_step=True, on_epoch=True, batch_size=B)
        self.log('train_acc_corrupt', acc_corrupt, prog_bar=True, on_step=True, on_epoch=True, batch_size=B)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _, acc_all, acc_corrupt = self._step(batch)
        B = batch['frame_ids'].size(0)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=B)
        self.log('val_acc_all', acc_all, prog_bar=True, on_step=False, on_epoch=True, batch_size=B)
        self.log('val_acc_corrupt', acc_corrupt, prog_bar=True, on_step=False, on_epoch=True, batch_size=B)

    def configure_optimizers(self):
        optimizer_class = hydra.utils.get_class(self.optimizer_config.target)
        optimizer = optimizer_class(self.parameters(), **self.optimizer_config.params)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.0,
                                      threshold_mode='abs')
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'epoch', 'frequency': 1}}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Linear learning-rate warmup over ``optimizer_config.lr_warmup`` steps."""
        warmup_steps = self.optimizer_config.lr_warmup
        if self.trainer.global_step < warmup_steps:
            scale = float(self.trainer.global_step + 1) / float(max(1, warmup_steps))
            for group in optimizer.param_groups:
                group['lr'] = self.optimizer_config.params.lr * scale
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint['state_dict']
        if any(key.startswith(tuple(LEGACY_PREFIXES)) for key in state_dict):
            checkpoint['state_dict'] = rename_state_dict_keys(state_dict, LEGACY_PREFIXES)
