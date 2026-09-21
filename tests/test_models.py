"""Shape checks on CPU with small configs."""
import torch
from omegaconf import OmegaConf

from tap_ets.layers import padding_mask_from_ids, padding_mask_from_lengths
from tap_ets.model import TAPETS
from tap_ets.phonemes import NUM_PHONEMES, PAD_ID
from tap_ets.refiner import FrameWisePhonemeRefiner

OPTIMIZER = OmegaConf.create({'target': 'torch.optim.AdamW', 'lr_warmup': 10, 'params': {'lr': 1e-3}})


def test_padding_masks():
    assert padding_mask_from_lengths(torch.tensor([1, 3]), 3).tolist() == [[False, True, True], [False, False, False]]
    assert padding_mask_from_ids(torch.tensor([[PAD_ID, 5]])).tolist() == [[True, False]]


def test_tap_ets_forward_shapes():
    model_cfg = OmegaConf.create({'model_size': 32, 'num_layers': 1, 'dropout': 0.0, 'use_channel': [0, 1, 2]})
    feature_cfg = OmegaConf.create({'dim': 8, 'frame_rate': 86.13, 'normalize': True})
    model = TAPETS(model_cfg, OPTIMIZER, feature_cfg).eval()
    emg = torch.randn(2, 64, 3)
    mel, logits = model(emg, emg_lengths=torch.tensor([64, 40]))
    assert mel.shape == (2, 8, 8) and logits.shape == (2, 8, NUM_PHONEMES)
    phoneme_ids = torch.randint(4, 52, (2, 8))
    mel_cond, _ = model(emg, phoneme_ids=phoneme_ids)
    h, _ = model.encode(emg)
    assert torch.allclose(mel_cond, model.decode(h, phoneme_ids), atol=1e-6)
    assert not (model.predict_phoneme_ids(logits) < 4).any()


def test_refiner_shapes_and_corruption():
    model_cfg = OmegaConf.create({'model_size': 32, 'num_layers': 1, 'dropout': 0.0})
    refiner = FrameWisePhonemeRefiner(model_cfg, OPTIMIZER, p_mask=0.2, p_replace=0.5).eval()
    frame_ids = torch.randint(4, 52, (2, 20))
    lengths = torch.tensor([20, 12])
    corrupted, corrupt_mask = refiner.corrupt(frame_ids, lengths)
    assert corrupted.shape == frame_ids.shape and not corrupt_mask[1, 12:].any()
    assert ((corrupted == refiner.mask_id) == (corrupt_mask & (corrupted == refiner.mask_id))).all()
    logits = refiner(torch.randint(4, 52, (2, 6)), torch.tensor([6, 4]), corrupted, lengths)
    assert logits.shape == (2, 20, NUM_PHONEMES)
    refined = refiner.refine(torch.tensor([5, 6, 7]), frame_ids[0])
    assert refined.shape == (20,)
