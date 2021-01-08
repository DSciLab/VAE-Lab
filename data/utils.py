from torch import nn


class LinearNormalize(nn.Module):
    def forward(self, x):
        return x / 255.0
