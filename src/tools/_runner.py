'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-14 18:11:52
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-14 18:12:54
 # @ Description: Base class of Runner.
 '''

import importlib
import pprint

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner

# from .datasets import *
# from .tools import *
from .registry import Registry
from .logger import build_logger
from .config import load_config
from .dict import deep_update

import logging


class _Runner():
    CALLBACKS = {
        'LearningRateMonitor' : {'logging_interval':'step'},
        'ModelSummary' : {'max_depth':3},
    }

    def __init__(self, args) -> None:
        config = load_config_and_update(args.cfg)
        self.config = config
        self.args = args
        self.mode = args.mode if args.mode != 'find_lr' else 'train'
        self.tb_logger = TensorBoardLogger('checkpoints', self.config.exper)
        self.logger = build_logger(config.exper, args.mode, logger_name=config.model_name,
                                    root='checkpoints', v_num=self.log_version)
        
        self.logger.info(args)
        self.logger.info(f'Configuration:\n{pprint.pformat(config)}')
        self.trainer = None
        self.auto_import()
        self.register_offical_callbacks()

    def register_offical_callbacks(self):
        import inspect
        cb_mod = importlib.import_module('lightning.pytorch.callbacks')
        for cb_name, cb_cls in inspect.getmembers(cb_mod, inspect.isclass):
            Registry.register(cb_cls)

    def auto_import(self):
        from pathlib import Path
        ignore = ['__pycache__']
        sub_pkgs = [en for en in Path('src').iterdir() if en.is_dir() and en.name not in ignore]
        for pkg in sub_pkgs:
            try:
                importlib.import_module(f'src.{pkg.name}')
            except Exception as e:
                console = logging.getLogger("lightning.pytorch.core")
                console.warning(f'Failed to import {pkg.name} package: {e}')
    
    def get_logger(self):
        return self.logger

    @property
    def log_version(self):
        if not hasattr(self, 'tb_logger'):
            return None
        v = self.tb_logger.version
        v = v if v == 0 else (v if self.mode == 'train' else v-1)
        return f'version_{v}'

    @property
    def ckpt(self):
        ckpt = self.cfg.ckpt_path if hasattr(self.cfg, 'ckpt_path') else None
        return ckpt
    
    @property
    def cfg(self):
        return eval(f'self.config.{self.mode if self.mode != "val" else "test"}')
    
    @property
    def __PKG__(self):
        PKG = self.config.__PKG__
        return PKG if PKG else 'interface'
    
    def find_lr(self, model, dm, **kwargs):
        if not self.trainer is None:
            find_best_lr(self.trainer, model, self.logger, dm)

    def build_callbacks(self, cb_params):
        for k, v in cb_params.items():
            cb_params[k] = {} if v is None else v
        cb_params = deep_update(self.CALLBACKS, cb_params)
        cbs = [Registry.instantiate(k, **v) for k, v in cb_params.items()]
        return cbs


def find_best_lr(trainer, model, logger, datamodule, show=True, update_attr=False):
    import matplotlib.pyplot as plt
    logger.info('Seaching for best learning rate ...')
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule=datamodule, update_attr=update_attr)

    fig = lr_finder.plot(suggest=True)
    fig.suptitle(f'Suggested lr: {lr_finder.suggestion():.4f}', fontsize=12)
    fig.savefig('lr_curve.jpg')
    if show:
        fig.show()
        plt.pause(10)
    try:
        logger.info('Best learning rate found %4f' % lr_finder.suggestion())
        print('Best learning rate found %4f' % lr_finder.suggestion())
    except TypeError:
        logger.info('Best learning rate not found.')
        print('Best learning rate not found.')
    else:
        logger.info('Learning rate curve has been saved in lr_curve.jpg')
        print('Learning rate curve has been saved in lr_curve.jpg')
    return lr_finder

def load_config_and_update(cfg):
    cfgs = load_config(cfg)

    for k in ['train', 'test']:
        cfgs[k]['model'] = deep_update(cfgs.model, cfgs[k]['model'])
        for kk in cfgs[k]:
            if kk.find('set') != -1:
                cfgs[k][kk] = deep_update(cfgs.dataset, cfgs[k][kk])
    return cfgs