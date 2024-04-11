'''
 # @ Author: Zhi Wu
 # @ Create Time: 2024-03-07 16:07:21
 # @ Modified by: Zhi Wu
 # @ Modified time: 2024-03-07 16:08:33
 # @ Description: Datamodule defenition.
 '''

from copy import deepcopy
from typing import Any, Optional
from abc import ABC

import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset, random_split

from .registry import Registry
from sklearn.model_selection import KFold


class PLData(pl.LightningDataModule):
    def __init__(self, train=None, val=None, test=None, predict=None, kfolds=None) -> None:
        super().__init__()
        self.train_cfg = train
        self.val_cfg = val
        self.test_cfg = test
        self.predict_cfg = predict
        self.kfolds = kfolds
        self.cur_fold = 0

    def _build_dataset_from_list(self, configs):
        configs = configs if isinstance(configs, list) else [configs]
        datasets = [self.build_dataset(cfg) for cfg in configs]
        datasets = datasets[0] if len(datasets) == 1 else datasets
        return datasets
    
    def _build_loader_from_list(self, datasets, configs):
        configs = configs if isinstance(configs, list) else [configs]
        datasets = datasets if isinstance(datasets, list) else [datasets]
        if len(datasets) != len(configs):
            assert len(configs) == 1, 'config should be a single dict or a list of dicts'
            configs = configs * len(datasets)
        loaders = [self.build_loader(dataset, cfg) for dataset, cfg in zip(datasets, configs)]
        loaders = loaders[0] if len(loaders) == 1 else loaders
        return loaders

    def prepare_data(self) -> None:
        # if kfolds is given, but single train_cfg is passed, conduct random kfold
        if not self.kfolds is None and isinstance(self.train_cfg, dict):
            # create random kfold dataset from train_cfg, val_cfg will be omitted
            self.build_kfold(self.train_cfg, self.kfolds)
        else:
            # Two cases:
            # Case1: kfolds is not given, create according to user config
            # Case2: kfolds is given, conduct kfolds according to user config
            self.train_sets = self._build_dataset_from_list(self.train_cfg)
            self.val_sets = self._build_dataset_from_list(self.val_cfg)

        # test/predict datasets is created as usual
        self.test_sets = self._build_dataset_from_list(self.test_cfg)
        self.predict_sets = self._build_dataset_from_list(self.predict_cfg)

    def build_kfold(self, config, kfold):
        assert not config is None, 'config should not be none when building kfold'
        dataset = self.build_dataset(config)
        nsamples = len(dataset)
        splits = [split for split in KFold(kfold).split(range(nsamples))]
        self.train_sets = [Subset(dataset, x) for x, _ in splits]

        self.val_sets = [Subset(dataset, x) for _, x in splits]
        
    # def setup_folds(self, num_folds: int) -> None:
    #     if self.kfolds is None:
    #         return
    #     self.num_folds = num_folds
    #     self.splits = [split for split in KFold(num_folds).split(range(len(self.train_dataset)))]

    def setup_fold_index(self, fold_index: int) -> None:
        if self.kfolds is None:
            return
        self.cur_fold = fold_index
    
    def build_dataset(self, config):
        if config is None:
            return None
        cfg = deepcopy(config)
        cfg.pop('loader', None)
        dataset = Registry.get_class(cfg.pop('name'))(*cfg)
        return dataset
    
    def build_loader(self, dataset, config):
        cfg = deepcopy(config)
        loader = DataLoader(dataset, **cfg.loader, drop_last=True)
        return loader
        
    def train_dataloader(self) -> Any:
        if self.kfolds is None:
            return self._build_loader_from_list(self.train_sets, self.train_cfg)
        else:
            if isinstance(self.train_cfg, list):
                return self.build_loader(self.train_sets[self.cur_fold], self.train_cfg[self.cur_fold])
            else:
                return self.build_loader(self.train_sets[self.cur_fold], self.train_cfg)
        
    def val_dataloader(self) -> Any:
        if self.kfolds is None:
            return self._build_loader_from_list(self.val_sets, self.val_cfg)
        else:
            if isinstance(self.val_cfg, list):
                return self.build_loader(self.val_sets[self.cur_fold], self.val_cfg[self.cur_fold])
            else:
                return self.build_loader(self.val_sets[self.cur_fold], self.val_cfg)
        
    def test_dataloader(self) -> Any:
        return self._build_loader_from_list(self.test_sets, self.test_cfg)
        
    def predict_dataloader(self) -> Any:
        return self._build_loader_from_list(self.predict_sets, self.predict_cfg)
