# base MLP for Transformers

import torch
import torch.nn as nn
from functools import partial


class LinearMLP(nn.Module):
    """build linear layer mlp with GELU and optional bias"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features or in_features
        self.out_features = out_features or in_features
        self.bias = bias
        self.drop = drop

        self.linear1 = nn.Linear(self.in_features, self.hidden_features, bias=self.bias)
        self.act1 = nn.GELU()
        self.linear2 = nn.Linear(
            self.hidden_features, self.out_features, bias=self.bias
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x
