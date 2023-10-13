'''
 # @ Author: Zhi Wu
 # @ Create Time: 2023-04-01 01:07:51
 # @ Modified by: Zhi Wu
 # @ Modified time: 2023-04-03 12:30:40
 # @ Description: Other utilities for custimizing lightning.
 '''

from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT


class StepCountCallback(Callback):
    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.cur_step = 0
        pl_module.nsteps = trainer.num_sanity_val_steps
        return super().on_sanity_check_start(trainer, pl_module)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.cur_step = 0
        if trainer.sanity_checking:
            pl_module.nsteps = trainer.num_sanity_val_steps
        else:
            pl_module.nsteps = sum(trainer.num_val_batches)
        return super().on_validation_epoch_start(trainer, pl_module)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.cur_step = 0
        pl_module.nsteps = trainer.num_training_batches
        return super().on_train_epoch_start(trainer, pl_module)

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.cur_step = 0
        pl_module.nsteps = sum(trainer.num_predict_batches)
        return super().on_predict_epoch_start(trainer, pl_module)

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.cur_step = 0
        pl_module.nsteps = sum(trainer.num_test_batches)
        return super().on_test_epoch_start(trainer, pl_module)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        pl_module.cur_step += 1
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        pl_module.cur_step += 1
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_predict_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        pl_module.cur_step += 1
        return super().on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        pl_module.cur_step += 1
        return super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
