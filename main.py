'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-23 22:23:46
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-25 00:54:42
 # @ Description: Train and evaluation script.
 '''
import argparse
import pprint

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler

from src import (build_callbacks, build_logger, build_plmodule, find_best_lr,
                 load_config, load_dataset, purify_cfg)

torch.set_float32_matmul_precision('medium')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/config.yaml', help='config file path')
    parser.add_argument('--mode', default='train', choices=['train', 'test', 'val'], help='train or test model')
    parser.add_argument('--vis', action='store_true', help='visualized_result')

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train_with_best_lr", action='store_true', help='auto find best learning rate and start training')
    group.add_argument('--best_lr', action='store_true', help='auto find best learning rate')

    return parser.parse_args()

def train(config, args, logger):
    logger.info('Building model.')
    model = build_plmodule(**config.model, txtlogger=logger)
    
    logger.info('Building Tensorboard logger.')
    tb_logger = TensorBoardLogger('checkpoints', config.model.exper)

    logger.info('Building Training dataset.')
    trainset, train_loader = load_dataset(**config.trainset)
    valset, val_loader = load_dataset(**config.valset)

    logger.info('Building Train phase Trainer.')
    cfg = purify_cfg(config.trainer, ['logger', 'callbacks'])
    trainer = Trainer(**cfg, logger=tb_logger, callbacks=build_callbacks(),
                        profiler=AdvancedProfiler(filename="perf_logs"))
    
    if args.best_lr:
        import matplotlib.pyplot as plt
        logger.info('Seaching for best learning rate ...')
        find_best_lr(trainer, model, train_loader, val_loader, logger)
    elif args.train_with_best_lr:
        lr_finder = find_best_lr(trainer, model, train_loader, val_loader, logger, update_attr=True)
        logger.info('Training with learning rate %4f' % lr_finder.suggestion())
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                    ckpt_path=config.ckpt_path)
    else:
        logger.info('Training started.')
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                    ckpt_path=config.ckpt_path)
        logger.info('Training finished.')

def test(config, args, logger):
    logger.info('Building model.')
    model = build_plmodule(**config.model, save_pred=args.save_pred, vis=args.vis, txtlogger=logger)
    
    
    logger.info('Building Test dataset.')
    dataset, data_loader = load_dataset(**config.valset)

    logger.info('Building Test phase Trainer.')
    cfg = purify_cfg(config.trainer, ['logger', 'enable_checkpointing', 'callbacks'])
    trainer = Trainer(**cfg, logger=False, enable_checkpointing=False, 
                        callbacks=build_callbacks('summary'))

    logger.info('Test started.')
    trainer.test(model, dataloaders=data_loader, ckpt_path=config.ckpt_path)
    logger.info('Test finished.')

def validate(config, args, logger):
    logger.info('Building model.')
    model = build_plmodule(**config.model, txtlogger=logger)
    

    logger.info('Building Validation dataset.')
    valset, val_loader = load_dataset(**config.valset)

    logger.info('Building Validate phase Trainer.')
    cfg = purify_cfg(config.trainer, ['logger', 'callbacks'])
    trainer = Trainer(**cfg, logger=False, callbacks=build_callbacks('summary'))
    
    logger.info('Validating started.')
    trainer.validate(model, dataloaders=val_loader)
    logger.info('Validating finished.')


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.cfg)
    
    logger = build_logger(config, args.mode, model_name=config.model_name)
    
    # pl.seed_everything(1)
    seed_everything(1)
    logger.info(args)
    logger.info(f'Configuration:\n{pprint.pformat(config)}')

    if args.mode == 'train':
        train(config.train, args, logger)
    elif args.mode == 'val':
        validate(config.test, args, logger)
    elif args.mode == 'test':
        test(config.test, args, logger)


