"""Train the refinement model on LibriSpeech MFA alignments: python main_refine.py exp_name=refiner"""
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from tap_ets.dataset import RefinerDataset
from tap_ets.refiner import FrameWisePhonemeRefiner
from tap_ets.training import build_trainer
from tap_ets.utils import enable_tf32, seed_worker, set_global_seed

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='configs', config_name='refine')
def main(cfg: DictConfig):
    enable_tf32()
    set_global_seed(cfg.seed)
    exp_dir = os.path.join(cfg.exp_path, f'{cfg.exp_name}_{cfg.seed}')
    os.makedirs(exp_dir, exist_ok=True)
    log.info(OmegaConf.to_yaml(cfg))
    log.info(f'experiment directory: {exp_dir}')

    model = FrameWisePhonemeRefiner(cfg.refiner, cfg.optimizer, p_corrupt=cfg.p_corrupt, p_mask=cfg.p_mask,
                                    p_replace=cfg.p_replace, alpha_clean=cfg.alpha_clean)
    checkpoint = ModelCheckpoint(dirpath=exp_dir, filename='epoch={epoch:03d}-val_acc_corrupt={val_acc_corrupt:.4f}',
                                 auto_insert_metric_name=False, monitor='val_acc_corrupt', mode='max',
                                 save_top_k=3, save_last=True)
    trainer = build_trainer(cfg, exp_dir, [checkpoint])
    if trainer.is_global_zero:
        OmegaConf.save(cfg, os.path.join(exp_dir, 'config.yaml'))

    trainset = RefinerDataset(cfg.mfa_path, 'train', cfg.feature.frame_rate, target_sec=cfg.target_sec)
    train_loader = DataLoader(trainset, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                              collate_fn=trainset.collate, worker_init_fn=seed_worker)
    validset = RefinerDataset(cfg.mfa_path, 'dev', cfg.feature.frame_rate, target_sec=cfg.target_sec)
    valid_loader = DataLoader(validset, batch_size=32, num_workers=1, collate_fn=validset.collate)

    trainer.fit(model, train_loader, valid_loader, ckpt_path=cfg.resume)


if __name__ == '__main__':
    main()
