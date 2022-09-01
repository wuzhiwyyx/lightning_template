'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-14 15:40:50
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-14 17:26:32
 # @ Description: HIBER dataset loader.
 '''

from torch.utils import data
import torch
from torchvision import datasets

class MINISTDataset(data.Dataset):
    """Example MINIST dataset definition."""

    def __init__(self, root, mode, download=True, transform=None):
        self.data = datasets.MNIST(root, train=mode, download=download, transform=transform)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]