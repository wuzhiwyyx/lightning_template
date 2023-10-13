import random
import sys

import torch
from tqdm import tqdm

from src import MyNet, build_plmodule, load_config, load_dataset

config = load_config('configs/config.yaml')
trainset, train_loader = load_dataset(config.train.trainset)
