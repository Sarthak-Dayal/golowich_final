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


class SharedResidualFlow(Flow):
    """Full-rank time-dependent input path plus a shared learned residual.

    The Gaussian conditional velocity is a stable full-rank starting field.
    Learned center/border scalar corrections can contract noise in every pixel
    direction, including directions outside the MLP output span.
    """

    def __init__(self, width=512):
        super().__init__(width)
        self.skip_mlp = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.skip_mlp[-1].weight)
        nn.init.zeros_(self.skip_mlp[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        mask = torch.zeros(56, 56)
        mask[14:42, 14:42] = 1
        self.register_buffer("center_mask", mask.flatten()[None])

    def forward(self, x, t):
        a = t[:, None] * self.freq[None]
        te = torch.cat([a.sin(), a.cos()], 1)
        delta = 8 * torch.tanh(self.skip_mlp(te) / 8)
        skip = delta[:, 0, None] * self.center_mask + delta[:, 1, None] * (
            1 - self.center_mask
        )
        gaussian = ((2 * t - 1) / ((1 - t).square() + t.square()))[:, None]
        return (gaussian + skip) * x + self.output(self.body(torch.cat([x, te], 1)))


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
