import torch
from torch import nn


class KLDLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()

    def forward(self, mu, logvar):
        kld_loss = torch.mean(-0.5 * \
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1), 
            dim = 0)
        return kld_loss
