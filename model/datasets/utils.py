'''
 # @ Author: Zhi Wu
 # @ Create Time: 1970-01-01 00:00:00
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-08-27 15:52:34
 # @ Description: Dataset loader.
 '''

from torch.utils.data import DataLoader
from torchvision import transforms
from .minist_dataset import MINISTDataset

def load_minist_dataset(config):
    """Example of load minist dataset from file

    Args:
        config (dict): Dict object containing dataset initial parameters and dataloader initial parameters.

    Returns:
        dataset (torch.utils.data.Dataset): Dataset object.
        loader (torch.utils.data.DataLoader): DataLoader object
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = MINISTDataset(**config.dataset, transform=transform)
    loader = DataLoader(dataset, **config.loader, drop_last=True)
    return dataset, loader
