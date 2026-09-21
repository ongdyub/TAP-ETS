"""Shared by the training entry points."""
import os

import pytorch_lightning as pl
from omegaconf import ListConfig
from pytorch_lightning.loggers import CSVLogger


def feature_dir(data_path, feature):
    return os.path.join(data_path, 'preprocessed', 'target_feature', feature.target,
                        f'{feature.sub_option}{feature[feature.sub_option]}')


def build_trainer(cfg, exp_dir, callbacks):
    devices = cfg.trainer.devices
    multi_gpu = isinstance(devices, (list, ListConfig)) and len(devices) > 1 or (isinstance(devices, int) and devices > 1)
    return pl.Trainer(logger=CSVLogger(save_dir=exp_dir, name='logs'), callbacks=callbacks,
                      strategy='ddp_find_unused_parameters_true' if multi_gpu else 'auto',
                      deterministic=True, num_sanity_val_steps=0, **cfg.trainer)
