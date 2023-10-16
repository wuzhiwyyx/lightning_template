'''
 # @ Author: Zhi Wu
 # @ Create Time: 1970-01-01 00:00:00
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-08-27 13:39:35
 # @ Description: RFMask main structure definition.
 '''

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    """Example model definition. Performing classification on MINIST."""

    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, target=None):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        loss = F.nll_loss(output, target, reduction='sum')  # sum up batch loss

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        
        correct = None
        if not target is None:
            correct = pred.eq(target.view_as(pred)).sum()
            
        return {'loss': loss}, pred, correct