'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-14 18:11:52
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-14 18:12:54
 # @ Description: Collection of some useful functions for running the whole project.
 '''

from model import MyNet
from model.datasets import load_minist_dataset

def build_model(cfg):
    """Build model object

    Args:
        cfg (dict): Model name and initialization parameters.

    Returns:
        LightningModule: Pytorch-lightning modules.
    """
    models = {
        'MyNet' : MyNet
    }
    return models[cfg.name](**cfg.params)
    
def load_dataset(config):
    _ = {
        'minist' : load_minist_dataset,
    }
    return _[config.name](config)
