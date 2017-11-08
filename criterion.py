from torch import nn
import torch
from utils import to_var


class PatchGanLoss(nn.Module):
    """
    Takes discriminator output and label (1 for real 0 for fake)
    """
    def __init__(self, criterion=nn.BCEWithLogitsLoss()):
        super(PatchGanLoss, self).__init__()
        self.criterion = criterion

    def forward(self, output, target):
        real_idx = (target == 1).nonzero()
        target = torch.zeros(output.size())
        target[real_idx, ...] = 1
        return self.criterion(output, to_var(target))
