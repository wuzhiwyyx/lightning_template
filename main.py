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

from src import ConfigParser, find_best_lr

torch.set_float32_matmul_precision('medium')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/config.yaml', help='config file path')
    parser.add_argument('--mode', default='train', choices=['train', 'test', 'val'], help='train or test model')
    parser.add_argument('--vis', action='store_true', help='visualized_result')
    parser.add_argument('--best_lr', action='store_true', help='auto find best learning rate')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cparser = ConfigParser(args)
    
    seed_everything(1)

    logger = cparser.get_logger()
    
    model = cparser.build_plmodule()
    data_loader = cparser.build_dataloader()
    trainer = cparser.build_trainer()

    if args.best_lr:
        find_best_lr(trainer, model, cparser.get_logger(), **data_loader)
    else:
        runs = {
            'train': trainer.fit,
            'test': trainer.test,
            'val': trainer.validate
        }
        runs[args.mode](model, **data_loader, ckpt_path=cparser.ckpt)


