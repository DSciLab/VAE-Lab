import torch
from torch import nn
import torch.nn.functional as F


class BasicClassifier(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.conv1 = nn.Conv2d(opt.image_chan, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.fc = nn.Sequential(
            nn.Dropout2d(0.25),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, opt.num_classes)
        )

    def forward(self, x):
        # imput resolution 28*28
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, dim=2)

        x = torch.flatten(x, start_dim=1)
        out = self.fc(x)
        prob_out = F.softmax(out)
        
        return prob_out, out
