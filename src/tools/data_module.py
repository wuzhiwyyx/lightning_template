'''
 # @ Author: Zhi Wu
 # @ Create Time: 2024-03-07 16:07:21
 # @ Modified by: Zhi Wu
 # @ Modified time: 2024-03-07 16:08:33
 # @ Description: Datamodule defenition.
 '''

from copy import deepcopy
from typing import Any, Optional

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .registry import Registry


class PLData(pl.LightningDataModule):
    def __init__(self, train=None, val=None, test=None, predict=None) -> None:
        super().__init__()
        self.train_cfg = train
        self.val_cfg = val
        self.test_cfg = test
        self.predict_cfg = predict

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
        # train/val datasets are created as usual
        self.train_sets = self._build_dataset_from_list(self.train_cfg)
        self.val_sets = self._build_dataset_from_list(self.val_cfg)

        # test/predict datasets are created as usual
        self.test_sets = self._build_dataset_from_list(self.test_cfg)
        self.predict_sets = self._build_dataset_from_list(self.predict_cfg)
    
    def build_dataset(self, config):
        if config is None:
            return None
        cfg = deepcopy(config)
        cfg.pop('loader', None)
        dataset = Registry.get_class(cfg.pop('name'))(*cfg)
        return dataset
    
    def build_loader(self, dataset, config):
        cfg = deepcopy(config)
        collate = cfg.loader.pop('collate_fn', None)
        collate = None if collate is None else Registry.get_fn(collate)
        loader = DataLoader(dataset, **cfg.loader, drop_last=True, collate_fn=collate)
        return loader
        
    def train_dataloader(self) -> Any:
        return self._build_loader_from_list(self.train_sets, self.train_cfg)
        
    def val_dataloader(self) -> Any:
        return self._build_loader_from_list(self.val_sets, self.val_cfg)
        
    def test_dataloader(self) -> Any:
        return self._build_loader_from_list(self.test_sets, self.test_cfg)
        
    def predict_dataloader(self) -> Any:
        return self._build_loader_from_list(self.predict_sets, self.predict_cfg)
