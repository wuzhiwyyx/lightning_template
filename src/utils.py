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
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner

# from .datasets import *
# from .tools import *
from .tools import _Runner, PLData, KFoldLoop

from copy import deepcopy


class Runner(_Runner):
    
    def build_plmodule(self):
        self.logger.info('Building model.')
        PLModule = importlib.import_module(f'.{self.__PKG__}', package='src').PLModule
        return PLModule(**self.cfg.model)

    def build_datamodule(self):
        self.logger.info(f'Building {self.mode.capitalize()} dataset.')
        if self.mode == 'train':
            dm = PLData(train=self.cfg.trainset, val=self.cfg.valset, 
                            kfolds=self.cfg.kfold if hasattr(self.cfg, 'kfold') else None)
        elif self.mode == 'val':
            dm = PLData(val=self.cfg.dataset)
        else:
            dm = PLData(test=self.cfg.dataset)
        return dm

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
        if hasattr(self.cfg, 'kfold') and not self.cfg.kfold is None:
            default_ = trainer.fit_loop
            trainer.fit_loop = KFoldLoop(self.cfg.kfold, trainer, default_.min_epochs, default_.max_epochs)
            # trainer.fit_loop.connect(default_loop)
        return trainer

    def run(self):
        self.trainer = self.build_trainer() if self.trainer is None else self.trainer
        model = self.build_plmodule()
        dm = self.build_datamodule()
        fn = {
            'train': self.trainer.fit,
            'val': self.trainer.validate,
            'test': self.trainer.test,
            'find_lr': self.find_lr,
        }
        fn[self.args.mode](model, dm, ckpt_path=self.ckpt)
