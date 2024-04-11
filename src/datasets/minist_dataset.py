'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-14 15:40:50
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-14 17:26:32
 # @ Description: HIBER dataset loader.
 '''

import torch
from torch.utils import data
from torchvision import datasets, transforms

from src import Registry


@Registry.register_dataset()
class MINISTDataset(data.Dataset):
    """Example MINIST dataset definition."""

    def __init__(self, root, mode, download=True, transform=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data = datasets.MNIST(root, train=mode, download=download, transform=transform)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    

