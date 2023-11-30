'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-23 22:23:46
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-25 00:54:42
 # @ Description: Train and evaluation script.
 '''
import argparse

import torch
from lightning.pytorch import seed_everything

from src import Runner

torch.set_float32_matmul_precision('medium')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/config.yaml', help='config file path')
    parser.add_argument('--mode', default='train', help='train/test/val/ model and find lr',
                                choices=['train', 'test', 'val', 'find_lr'])

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    seed_everything(1)

    Runner(args).run()


