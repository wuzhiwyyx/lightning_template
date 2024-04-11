'''
 # @ Author: Zhi Wu
 # @ Create Time: 2023-04-01 01:07:51
 # @ Modified by: Zhi Wu
 # @ Modified time: 2023-04-03 12:30:40
 # @ Description: Base lightning model for all models.
 '''

from copy import deepcopy
from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim

# from .mynet import MyNet
from .tools import Registry


class PLModule(pl.LightningModule):

    def __init__(self, optim={'name': 'Adam', 'lr': 1e-3, 'weight_decay': 0.01}, 
                    sched={'name': 'StepLR', 'step_size': 4, 'gamma': 0.5}, **config):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.build_model(**config)
        self.optim = optim
        self.sched = sched
        self.metrics = {}

    def build_model(self, name, **kwargs):
        model = Registry.instantiate(name, **kwargs)
        return model

    def sum_losses(self, losses):
        assert isinstance(losses, dict)
        if not 'loss' in losses:
            losses['loss'] = sum(losses.values())
        return losses

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        data, target = batch
        loss, pred, correct = self.forward(data, target)
        return loss, pred, correct

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        data, target = batch
        outputs = self.forward(data, target)

        losses = self.sum_losses(outputs[0])
        return losses
    
    def validation_step(self, batch, batch_idx):
        data, target = batch
        outputs = self.forward(data, target)

        self.sum_losses(outputs[0])
        return outputs

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        data, target = batch
        outputs = self.forward(data)
        return outputs

    def configure_optimizers(self):
        opt = deepcopy(self.optim)
        optimizer = eval(f'optim.{opt.pop("name")}')(self.parameters(), **opt)

        sch = deepcopy(self.sched)
        scheduler = eval(f'optim.lr_scheduler.{sch.pop("name")}')(optimizer, **sch)
        
        lr_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
        return lr_dict


# class PLData(pl.LightningDataModule):
#     def __init__(self, train=None, val=None, test=None, predict=None) -> None:
#         super().__init__()
#         self.train_cfg = train
#         self.val_cfg = val
#         self.test_cfg = test
#         self.predict_cfg = predict

#     def build_dataset_and_loader(self, config):
#         CLASS = Registry.get_class(config.pop('name'))
#         dataset, loader = CLASS.build_dataset(**config)
#         return dataset, loader
        
#     def train_dataloader(self) -> Any:
#         if not self.train_cfg is None:
#             cfg = deepcopy(self.train_cfg)
#             dataset, loader = self.build_dataset_and_loader(cfg)
#             return loader
        
#     def val_dataloader(self) -> Any:
#         if not self.val_cfg is None:
#             cfg = deepcopy(self.val_cfg)
#             dataset, loader = self.build_dataset_and_loader(cfg)
#             return loader
        
#     def test_dataloader(self) -> Any:
#         if not self.test_cfg is None:
#             cfg = deepcopy(self.test_cfg)
#             dataset, loader = self.build_dataset_and_loader(cfg)
#             return loader
        
#     def predict_dataloader(self) -> Any:
#         if not self.predict_cfg is None:
#             cfg = deepcopy(self.predict_cfg)
#             dataset, loader = self.build_dataset_and_loader(cfg)
#             return loader