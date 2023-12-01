'''
 # @ Author: Zhi Wu
 # @ Create Time: 2023-04-01 01:07:51
 # @ Modified by: Zhi Wu
 # @ Modified time: 2023-04-03 12:30:40
 # @ Description: TxtLogger implementation to log infos into txt file.
 '''

from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
import logging
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint


class TxtLogger(Callback):
    def __init__(self, name, batch_size, sync=True, summary_depth=3,
                        on_bar=[]) -> None:
        super().__init__()
        self.name = name
        self.sync = sync
        self.bs = batch_size
        self.summary_depth = summary_depth
        self.txt_logger = logging.getLogger(name)
        self.on_bar = ['loss'] + on_bar

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self.pl_module = pl_module
        logger = self.txt_logger
        from lightning.pytorch.utilities.model_summary import ModelSummary
        paramter_info = ModelSummary(pl_module, max_depth=self.summary_depth)
        logger.info('====================================================='*3)
        logger.info(f'Model Structure:\n{pl_module}')
        logger.info('====================================================='*3)
        logger.info(f'Parameter Summary:\n{paramter_info}')
        logger.info('====================================================='*3)
        if trainer.logger is None:
            logger.info('Tensorboard logger is disabled.')
        else:
            logger.info(f'Tensorboard logger is enabled with log folder {trainer.logger.log_dir}.')
        logger.info('====================================================='*3)


    def prefix_keys(self, x: dict, prefix: str='', ) -> dict:
        out = dict([(str(Path(prefix) / k), v) for k, v in x.items()])
        return out

    def current_stage(self, trainer):
        coded = lambda xs: ''.join([str(int(x)) for x in xs])
        status = [trainer.training, trainer.validating, trainer.sanity_checking]
        prefixes = {
            '100': 'train',
            '010': 'val',
            '001': 'sanity',
            '000': 'others',
        }
        state_code = coded(status)
        return prefixes[state_code]

    def log_to_txt(self, losses, log_step=True):
        epoch = self.pl_module.current_epoch
        max_epochs = self.pl_module.trainer.max_epochs

        msg = ', '.join([f'{k}: {v.item():.04f}' for k, v in losses.items()])

        e = f'{epoch + 1}/{max_epochs}'
        s = f'{self.cur_step}/{self.nsteps}'
        msg = f'Epoch [{e}], Step [{s}] : {msg}'
        self.txt_logger.info(msg)

    def log_to_all(self, losses, prefix=''):
        msg = self.prefix_keys(losses, prefix)
        self.log_to_txt(msg)

        on_bars = self.prefix_keys(dict([k, v] for k, v in losses.items() if k in self.on_bar), prefix)
        remains = self.prefix_keys(dict([k, v] for k, v in losses.items() if not k in self.on_bar), prefix)
        
        self.pl_module.log_dict(on_bars, batch_size=self.bs, sync_dist=self.sync, prog_bar=True)
        self.pl_module.log_dict(remains, batch_size=self.bs, sync_dist=self.sync)


    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.cur_step = 0
        self.nsteps = trainer.num_sanity_val_steps
    
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.txt_logger.info('Fit started.')

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.txt_logger.info('Fit finished.')

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.txt_logger.info('Training started.')

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.cur_step = 0
        self.nsteps = trainer.num_training_batches
        self.txt_logger.info(f'Train epoch {pl_module.current_epoch + 1} started.')
      
    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int) -> None:
        self.cur_step += 1
    
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        prefix = self.current_stage(trainer)
        self.log_to_all(outputs, prefix)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.txt_logger.info(f'Train epoch {pl_module.current_epoch + 1} finished.')

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.txt_logger.info('Training finished.')

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.sanity_checking:
            self.txt_logger.info('Sanity checking started.')
        else:
            self.txt_logger.info('Validating started.')

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.cur_step = 0
        if trainer.sanity_checking:
            self.nsteps = trainer.num_sanity_val_steps
        else:
            self.nsteps = sum(trainer.num_val_batches)
        
    def on_validation_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.cur_step += 1    

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        prefix = self.current_stage(trainer)
        self.log_to_all(outputs[0], prefix)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_to_all(pl_module.metrics if hasattr(pl_module, 'metrics') else {})

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.sanity_checking:
            self.txt_logger.info('Sanity checking finished.')
        else:
            self.txt_logger.info('Validating finished.')

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.txt_logger.info('Test started.')

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.cur_step = 0
        self.nsteps = sum(trainer.num_test_batches)
    
    def on_test_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.cur_step += 1

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        prefix = self.current_stage(trainer)
        self.log_to_all(outputs[0], prefix)
    
    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_to_all(pl_module.metrics if hasattr(pl_module, 'metrics') else {})
    
    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.txt_logger.info('Test finished.')    

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.cur_step = 0
        self.nsteps = sum(trainer.num_predict_batches)

    def on_predict_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.cur_step += 1
