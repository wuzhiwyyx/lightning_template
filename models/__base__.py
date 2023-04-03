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


class CustomPLBaseModel(pl.LightningModule):

    def __init__(self, learning_rate=1e-3, batch_size=12, exper=None, 
                        sync_dist=True, txtlogger=None, save_pred=False, vis=False):
        super().__init__()
        self.save_hyperparameters(ignore=['txtlogger', 'save_pred', 'vis'])
        self.learning_rate = learning_rate
        self.bs = batch_size
        self.sync = sync_dist
        self.txtlogger = txtlogger
        self.save_pred = save_pred
        self.vis = vis
        self.exper = exper

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