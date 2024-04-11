'''
 # @ Author: Zhi Wu
 # @ Create Time: 2023-04-01 01:07:51
 # @ Modified by: Zhi Wu
 # @ Modified time: 2023-04-03 12:30:40
 # @ Description: CKPTFormatter implementation to customize tensorboard log format.
 '''

import logging
from pathlib import Path
from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.utilities.types import STEP_OUTPUT

from src import Registry


@Registry.register_callback()
class CKPTFormatter(ModelCheckpoint):
    def __init__(self, monitor='val/loss', filename=None, save_top_k=5, save_last=True, 
                        verbose=True, auto_insert_metric_name=False, **kwargs):
        if not monitor is None and filename is None:
            filename = f'e{{epoch}}-s{{step}}-{Path(monitor).stem}-{{{monitor}:.4f}}'
        super().__init__(monitor=monitor, filename=filename, save_top_k=save_top_k, 
                            save_last=save_last, verbose=verbose, 
                            auto_insert_metric_name=auto_insert_metric_name,
                            **kwargs)