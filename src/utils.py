'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-14 18:11:52
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-14 18:12:54
 # @ Description: Collection of some useful functions for running the whole project.
 '''

import time
from collections import namedtuple
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks import ModelSummary as SummaryCallback
from lightning.pytorch.callbacks import RichModelSummary as RichSummaryCallback
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.utilities.model_summary import ModelSummary as Summary

from .datasets import load_minist_dataset
from .interface import PLModule
from .mynet import MyNet
from .tools import (ConfigDict, StepCountCallback, build_logger, load_config,
                    setup_logger_ddp)


def build_plmodule(**kwargs):
    logger = kwargs.get('txtlogger', None)
    plmodule = PLModule(**kwargs)
    if not logger is None:
        log_model_info(logger, plmodule, Summary(plmodule, max_depth=3))
    return plmodule

# def build_model(name, logger=None, **kwargs):
#     """Build model object

#     Args:
#         cfg (dict): Model name and initialization parameters.

#     Returns:
#         LightningModule: Pytorch-lightning modules.
#     """
#     models = {
#         'mynet' : MyNet
#     }
#     model = models[name.lower()](**ConfigDict(kwargs).to_dict())
#     # Log model structure and parameter summary
#     # logger = kwargs.get('txtlogger', None)
#     # if not logger is None:
#     #     log_model_info(logger, model, Summary(model, max_depth=3))
#     return model
    
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

def find_best_lr(trainer, model, train_loader, val_loader, logger, show=True, update_attr=False):
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, train_dataloaders=train_loader, 
                                val_dataloaders=val_loader, update_attr=update_attr)

    fig = lr_finder.plot(suggest=True)
    fig.savefig('lr_curve.jpg')
    if show:
        fig.show()
        plt.pause(10)
    try:
        logger.info('Best learning rate found %4f' % lr_finder.suggestion())
    except TypeError:
        logger.info('Best learning rate not found.')
    else:
        logger.info('Learning rate curve has been saved in lr_curve.jpg')
    return lr_finder
