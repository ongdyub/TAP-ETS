"""Train TAP-ETS: python main.py exp_name=tap_ets [seed=97]"""
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from tap_ets.dataset import EMGDataset
from tap_ets.features import FeatureNormalizer, find_normalizer
from tap_ets.model import TAPETS
from tap_ets.training import build_trainer, feature_dir
from tap_ets.utils import enable_tf32, seed_worker, set_global_seed

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg: DictConfig):
    enable_tf32()
    set_global_seed(cfg.seed)
    exp_dir = os.path.join(cfg.exp_path, f'{cfg.exp_name}_{cfg.seed}')
    os.makedirs(exp_dir, exist_ok=True)
    log.info(OmegaConf.to_yaml(cfg))
    log.info(f'experiment directory: {exp_dir}')

    feat_dir = feature_dir(cfg.data_path, cfg.feature)
    normalizer = None
    if cfg.feature.normalize:
        normalizer_path = find_normalizer(feat_dir)
        log.info(f'feature normalizer: {normalizer_path}')
        normalizer = FeatureNormalizer.load(normalizer_path)

    model = TAPETS(cfg.emg_enc, cfg.optimizer, cfg.feature, phoneme_loss_weight=cfg.phoneme_loss_weight,
                   mel_loss_weight=cfg.mel_loss_weight, batch_size=cfg.batch_size)

    checkpoint = ModelCheckpoint(dirpath=exp_dir, filename='epoch={epoch:03d}-val_phone_accuracy={val_phone_accuracy:.4f}',
                                 auto_insert_metric_name=False, monitor='val_phone_accuracy', mode='max',
                                 save_top_k=3, save_last=True)
    trainer = build_trainer(cfg, exp_dir, [checkpoint])
    if trainer.is_global_zero:
        OmegaConf.save(cfg, os.path.join(exp_dir, 'config.yaml'))
        if normalizer is not None:  # for inference
            normalizer.save(os.path.join(exp_dir, 'normalizer.npz'))

    trainset = EMGDataset(cfg.data_path, feat_dir, 'train', cfg.feature.frame_rate, target_sec=cfg.target_sec,
                          normalizer=normalizer)
    train_loader = DataLoader(trainset, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                              collate_fn=trainset.collate, worker_init_fn=seed_worker)
    validset = EMGDataset(cfg.data_path, feat_dir, 'dev', cfg.feature.frame_rate, normalizer=normalizer)
    valid_loader = DataLoader(validset, batch_size=1, num_workers=1, collate_fn=validset.collate)

    trainer.fit(model, train_loader, valid_loader, ckpt_path=cfg.resume)


if __name__ == '__main__':
    main()
