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

# from .utils import build_model


class PLModule(pl.LightningModule):

    def __init__(self, learning_rate=1e-3, batch_size=12, exper=None, out=None,
                        sync_dist=True, txtlogger=None, save_pred=False, vis=False, 
                        optim='adam', sched='steplr', **config):
        super().__init__()
        self.save_hyperparameters(ignore=['txtlogger', 'save_pred', 'vis', 'config'])
        self.learning_rate = learning_rate
        self.bs = batch_size
        self.sync = sync_dist
        self.txtlogger = txtlogger
        self.save_pred = save_pred
        self.vis = vis
        self.exper = exper
        self.model = self.build_model(**config)
        self.optim = optim
        self.sched = sched
        self.out = out

    def build_model(self, name, **kwargs):
        models = {
            'mynet' : MyNet
        }
        model = models[name.lower()](**ConfigDict(kwargs).to_dict())
        return model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        data, target = batch
        loss, pred, correct = self.forward(data, target)
        return loss, pred, correct

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        data, target = batch
        loss, pred, correct = self.forward(data, target)
            
        # Logging to TensorBoard by default
        self.log_to_all({'loss': loss})
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, target = batch
        loss, pred, correct = self.forward(data, target)
            
        # Logging to TensorBoard by default
        self.log_to_all({'loss': loss})
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        data, target = batch
        loss, pred, correct = self.forward(data, target)
        return loss

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

    def prefix_keys(self, x: dict, prefix: str, add_total_loss=True) -> dict:
        # assign prefix according to the stage
        if self.trainer.training:
            prefix = 'train'
        elif self.trainer.validating:
            prefix = 'val'
        elif self.trainer.sanity_checking:
            prefix = 'check'
        else:
            prefix = 'others'

        # add total loss
        if add_total_loss and not 'loss' in x.keys():
            x['loss'] = sum(x.values())

        out = dict([(prefix + '/' + k, v) for k, v in x.items()])
        return out

    def log_to_txt(self, losses):
        if self.txtlogger is None:
            return
        epoch = self.current_epoch
        max_epochs = self.trainer.max_epochs

        total_loss = sum(losses.values()).item()
        msg = ', '.join([f'loss: {total_loss:.04f}'] + 
                        [f'{k}: {v.item():.04f}' for k, v in losses.items()])

        e = f'{epoch + 1}/{max_epochs}'
        s = f'{self.cur_step + 1}/{self.nsteps}' \
                if hasattr(self, 'cur_step') else self.global_step
        msg = f'Epoch [{e}], Step [{s}] : {msg}'
        self.txtlogger.info(msg)

    def log_to_all(self, losses, add_total_loss=True):
        msg = self.prefix_keys(losses, add_total_loss)
        self.log_dict(msg, batch_size=self.bs, sync_dist=self.sync, prog_bar=True)
        self.log_to_txt(msg)
