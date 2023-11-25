'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-14 18:11:52
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-14 18:12:54
 # @ Description: Collection of some useful functions for running the whole project.
 '''

import pprint
from copy import deepcopy

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import *
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner

from .datasets import build_minist_dataset
from .interface import PLModule
from .tools import *
from .tools.callbacks import *


def build_dataset(name, **kwargs):
    _ = {
        'minist' : build_minist_dataset,
    }
    return _[name.lower()](**kwargs)

def deep_update(raw, new):
    if new is None:
        return raw
    foo = deepcopy(raw)
    update_keys(foo, new)
    insert_keys(foo, new)
    return foo

def update_keys(raw, new):
    for key in raw:
        if key not in new.keys():
            continue
        if isinstance(raw[key], dict) and isinstance(new[key], dict):
            raw[key] = deep_update(raw[key], new[key])
        else:
            raw[key] = new[key]

def insert_keys(raw, new):
    update_dict = {}
    for key in new:
        if key not in raw.keys():
            update_dict[key] = new[key]
    raw.update(update_dict)


def find_best_lr(trainer, model, logger, train_dataloaders, val_dataloaders, show=True, update_attr=False):
    import matplotlib.pyplot as plt
    logger.info('Seaching for best learning rate ...')
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, train_dataloaders=train_dataloaders, 
                                val_dataloaders=val_dataloaders, update_attr=update_attr)

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

class ConfigParser():
    CALLBACKS = {
        'LearningRateMonitor' : {'logging_interval':'step'},
        # 'CKPTFormatter' : {},
        'ModelSummary' : {'max_depth':3},
        # 'TxtLogger' : {},
    }

    def __init__(self, args) -> None:
        config = self.load_config(args.cfg)
        self.config = config
        self.args = args
        self.mode = args.mode
        self.tb_logger = TensorBoardLogger('checkpoints', self.config.exper)
        self.logger = build_logger(config.exper, args.mode, logger_name=config.model_name,
                                    root='checkpoints', v_num=self.log_version)
        
        self.logger.info(args)
        self.logger.info(f'Configuration:\n{pprint.pformat(config)}')
    
    def load_config(self, cfg):
        cfgs = load_config(cfg)
        
        cfgs.train.model = deep_update(cfgs.model, cfgs.train.model)
        cfgs.train.trainset = deep_update(cfgs.dataset, cfgs.train.trainset)
        cfgs.train.valset = deep_update(cfgs.dataset, cfgs.train.valset)
        cfgs.test.model = deep_update(cfgs.model, cfgs.test.model)
        cfgs.test.dataset = deep_update(cfgs.dataset, cfgs.test.dataset)
        return cfgs
    
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
        return eval(f'self.config.{self.mode}')

    def build_plmodule(self):
        self.logger.info('Building model.')
        return PLModule(**self.cfg.model)

    def build_dataloader(self):
        self.logger.info(f'Building {self.mode.capitalize()} dataset.')
        if self.mode == 'train':
            trainset, train_loader = build_dataset(**self.cfg.trainset)
            valset, val_loader = build_dataset(**self.cfg.valset)
            loaders = {
                'train_dataloaders': train_loader,
                'val_dataloaders': val_loader,
            }
        else:
            dataset, data_loader = build_dataset(**self.cfg.dataset)
            loaders = {
                'dataloaders': data_loader,
            }
        return loaders

    def build_trainer(self):
        self.logger.info(f'Building {self.mode.capitalize()} phase Trainer.')
        cfg = deepcopy(self.cfg.trainer)
        for k in ['logger']:
            cfg.pop(k, None)
        cfg['callbacks'] = self.build_callbacks(cfg.get('callbacks', {}))
        if self.mode == 'train':
            cfg['logger'] = self.tb_logger
            trainer = Trainer(**cfg)
        else:
            cfg['enable_checkpointing'] = False
            cfg['logger'] = False
            trainer = Trainer(**cfg)
        return trainer

    def build_callbacks(self, cb_params):
        for k, v in cb_params.items():
            cb_params[k] = {} if v is None else v
        cb_params = deep_update(self.CALLBACKS, cb_params)
        cbs = [eval(k)(**v) for k, v in cb_params.items()]
        return cbs
