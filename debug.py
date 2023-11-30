import random
import sys

import torch
from tqdm import tqdm

from src import MyNet, load_config, build_dataset

config = load_config('configs/config.yaml')
trainset, train_loader = build_dataset(**config.train.trainset)
net = MyNet()
