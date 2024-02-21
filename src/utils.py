'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-14 18:11:52
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-14 18:12:54
 # @ Description: Collection of some useful functions for running the whole project.
 '''

import importlib
import pprint

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import *
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner

from .datasets import *
from .tools import *


class Runner():
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
    
    def get_logger(self):
        return self.logger

    @property
    def train(self):
        return None if self.trainer is None else self.trainer.fit

    @property
    def test(self):
        return None if self.trainer is None else self.trainer.test
    
    @property
    def val(self):
        return None if self.trainer is None else self.trainer.validate

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
    
    def find_lr(self, model, train_dataloaders, val_dataloaders, **kwargs):
        if not self.trainer is None:
            find_best_lr(self.trainer, model, self.logger, train_dataloaders, val_dataloaders)

    def build_plmodule(self):
        self.logger.info('Building model.')
        PLModule = importlib.import_module(f'.{self.__PKG__}', package='src').PLModule
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

    def run(self):
        self.trainer = self.build_trainer() if self.trainer is None else self.trainer
        model = self.build_plmodule()
        data_loader = self.build_dataloader()
        eval(f'self.{self.args.mode}')(model, **data_loader, ckpt_path=self.ckpt)


def find_best_lr(trainer, model, logger, train_dataloaders, val_dataloaders, show=True, update_attr=False):
    import matplotlib.pyplot as plt
    logger.info('Seaching for best learning rate ...')
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, train_dataloaders=train_dataloaders, 
                                val_dataloaders=val_dataloaders, update_attr=update_attr)

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