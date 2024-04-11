'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-14 18:11:52
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-14 18:12:54
 # @ Description: Collection of some useful functions for running the whole project.
 '''

import importlib
from copy import deepcopy

from lightning.pytorch import Trainer

from .tools import PLData, _Runner


class Runner(_Runner):
    
    def build_plmodule(self):
        self.logger.info('Building model.')
        PLModule = importlib.import_module(f'.{self.__PKG__}', package='src').PLModule
        return PLModule(**self.cfg.model)

    def build_datamodule(self):
        self.logger.info(f'Building {self.mode.capitalize()} dataset.')
        if self.mode == 'train':
            dm = PLData(train=self.cfg.trainset, val=self.cfg.valset)
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
