'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-24 21:57:44
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-25 00:47:12
 # @ Description: Pytorch-lightning model definition, controlling training strategy and train/val/test dataset.
 '''

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import optim
from .mynet import MyNet as MyNet_

class MyNet(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, batch_size=12):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = MyNet_()
        
    def forward(self, data, target=None):
        output = self.model(data)
        loss = F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        if not target is None:
            correct = pred.eq(target.view_as(pred)).sum().item()
            return loss, pred, correct
        return loss, pred
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        data, target = batch
        loss, pred, correct = self.forward(data, target)
        return loss, pred, correct

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        data, target = batch
        loss, pred, correct = self.forward(data, target)
            
        # Logging to TensorBoard by default
        self.log('loss', loss.item(), batch_size=self.batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, target = batch
        loss, pred, correct = self.forward(data, target)
            
        # Logging to TensorBoard by default
        self.log('loss', loss.item(), batch_size=self.batch_size, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        lr_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
        return lr_dict