import math
import torch
from torch import nn


class Flow(nn.Module):
    def __init__(self, width=512):
        super().__init__()
        self.register_buffer("freq", torch.exp(torch.linspace(0, math.log(1000), 16)))
        self.body = nn.Sequential(
            nn.Linear(3136 + 32, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.output = nn.Linear(width, 3136)

    def forward(self, x, t):
        a = t[:, None] * self.freq[None]
        return self.output(self.body(torch.cat([x, a.sin(), a.cos()], 1)))


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.embed = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU())
        self.head = nn.Linear(128, 10)

    def forward(self, x, features=False):
        z = self.embed(self.conv(x))
        return (
            (self.head(z), nn.functional.normalize(z, dim=1))
            if features
            else self.head(z)
        )
