from typing import Optional
import pytorch_lightning as pl
import torch
from yacs.config import CfgNode

from shamer.datasets.wlasl.wlasl import WLASLDataset



class SHAMERDataModule(pl.LightningDataModule):

    def __init__(
        self,
        cfg: CfgNode,
        dataset_cfg: CfgNode,
    ) -> None:
        super(SHAMERDataModule, self).__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg

    def setup(self, stage: Optional[str] = None) -> None:

        train_cfg = self.dataset_cfg.clone()
        train_cfg.defrost()
        train_cfg.MODE = "train"
        self.train_dataset = WLASLDataset(train_cfg)

        val_cfg = self.dataset_cfg.clone()
        val_cfg.defrost()
        val_cfg.MODE = "val"
        self.val_dataset = WLASLDataset(val_cfg)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.GENERAL.NUM_WORKERS,
            prefetch_factor=self.cfg.GENERAL.PREFETCH_FACTOR,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.GENERAL.NUM_WORKERS,
            prefetch_factor=self.cfg.GENERAL.PREFETCH_FACTOR,
            pin_memory=True,
        )
