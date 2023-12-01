'''
 # @ Author: Zhi Wu
 # @ Create Time: 2023-04-01 01:07:51
 # @ Modified by: Zhi Wu
 # @ Modified time: 2023-04-03 12:30:40
 # @ Description: Base lightning model for all models.
 '''

from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim
from .mynet import MyNet
from .tools import ConfigDict


class PLModule(pl.LightningModule):

    def __init__(self, learning_rate=1e-3, optim='adam', sched='steplr', **config):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = self.build_model(**config)
        self.optim = optim
        self.sched = sched
        self.metrics = {}

    def build_model(self, name, **kwargs):
        models = {
            'mynet' : MyNet
        }
        model = models[name.lower()](**ConfigDict(kwargs).to_dict())
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
        optimizers = {
            'sgd': optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9),
            'adam': optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0),
            'nadam': optim.NAdam(self.parameters(), lr=self.learning_rate, weight_decay=0.01),
            'adadelta': optim.Adadelta(self.parameters(), lr=self.learning_rate),
            'rmsprop': optim.RMSprop(self.parameters(), lr=self.learning_rate)
        }
        optimizer = optimizers[self.optim]

        schedulers = {
            'steplr': optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma = 0.5),
            'cos': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
        }
        scheduler = schedulers[self.sched]
        
        lr_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
        return lr_dict
