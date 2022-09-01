import sys
sys.path.append('.')
import random

from model import MyNet
from utils import load_config, build_model, load_dataset, ConfigDict

import torch
from tqdm import tqdm

config = load_config('configs/config.yaml')
trainset, train_loader = load_dataset(config.train.trainset)
