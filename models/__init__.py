import torch
import torch.nn as nn
import torch.nn.functional as F

class DefaultModel(nn.Module):
    def __init__(self, config):
        self.cfg = config['model']
        super().__init__()
    def forward(self,x):
        return x