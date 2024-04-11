'''
 # @ Author: Zhi Wu
 # @ Create Time: 2023-04-01 01:07:51
 # @ Modified by: Zhi Wu
 # @ Modified time: 2023-04-03 12:30:40
 # @ Description: Base lightning model for all models.
 '''

from copy import deepcopy
from typing import Any

import lightning.pytorch as pl
from torch import optim

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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        data, target = batch
        loss, pred, correct = self.forward(data, target)
        return loss, pred, correct

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        # training_step defined the train loop.
        data, target = batch
        outputs = self.forward(data, target)

        losses = self.sum_losses(outputs[0])
        return losses
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
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
