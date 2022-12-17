import torch
from torch import nn


class RelationClassifier(nn.Module):
    def __init__(self, inp_dim: int = 2048, hid_dim: int = 128):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(inp_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 4),
        )

    def forward(self, inp_a, inp_b):
        inp = torch.cat((inp_a, inp_b), dim=1)
        return self.seq(inp)
