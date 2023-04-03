'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-14 18:11:52
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-14 18:12:54
 # @ Description: Collection of some useful functions for running the whole project.
 '''

from collections import namedtuple
from copy import deepcopy
from pathlib import Path

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks import ModelSummary as SummaryCallback
from lightning.pytorch.callbacks import RichModelSummary as RichSummaryCallback
from lightning.pytorch.utilities.model_summary import ModelSummary as Summary
from models import StepCountCallback
from models.datasets import load_minist_dataset
from models.mynet import MyNet

from .config import ConfigDict


def build_model(name, **kwargs):
    """Build model object

    Args:
        cfg (dict): Model name and initialization parameters.

    Returns:
        LightningModule: Pytorch-lightning modules.
    """
    models = {
        'mynet' : MyNet
    }
    model = models[name.lower()](**ConfigDict(kwargs).to_dict())
    # Log model structure and parameter summary
    logger = kwargs.get('txtlogger', None)
    if not logger is None:
        log_model_info(logger, model, Summary(model, max_depth=3))
    return model
    
def load_dataset(name, **kwargs):
    _ = {
        'minist' : load_minist_dataset,
    }
    return _[name.lower()](**kwargs)

def log_model_info(logger, model_sturcture, paramter_info):
    logger.info('=====================================================')
    logger.info(f'Model Structure:\n{model_sturcture}')
    logger.info('=====================================================')
    logger.info(f'Parameter Summary:\n{paramter_info}')
    logger.info('=====================================================')

def build_callbacks(cb_names=['lr_monitor', 'ckpt_callback', 'summary']):
    if not isinstance(cb_names, list):
        cb_names = [cb_names]
    cb_names.append('step_cnt')
    cbs = {
        'lr_monitor' : LearningRateMonitor(logging_interval='step'),
        'ckpt_callback' : ModelCheckpoint(
            monitor='val/loss', save_top_k=5, 
            filename='e{epoch}-s{step}-loss{val/loss:.4f}',
            auto_insert_metric_name=False,
            save_last=True, verbose=True
        ),
        'summary' : SummaryCallback(max_depth=3),
        'rich_summary' : RichSummaryCallback(max_depth=3),
        'step_cnt' : StepCountCallback(),
    }
    return [cbs[x] for x in cb_names]

def purify_cfg(config, keys_to_filter=[]):
    cfg = deepcopy(config)
    for ktf in keys_to_filter:
        cfg.pop(ktf, None)
    return cfg
    