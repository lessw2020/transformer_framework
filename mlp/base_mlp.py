# base MLP for Transformers

import torch
import torch.nn as nn
from functools import partial


class LinearMLP(nn.Module):
    """build linear layer mlp with swiGLU and optional bias"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        use_swiglu=True,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.in_proj_split = [hidden_features, hidden_features]
        self.out_features = out_features or in_features
        self.bias = bias
        self.drop = drop
        self.activation_fn = act_layer()

        self.linear1 = nn.Linear(self.in_features, sum(self.in_proj_split), bias=self.bias)
        self.linear2 = nn.Linear(
            self.hidden_features, self.out_features, bias=self.bias
        )

    def forward(self, x):
        x = self.linear1(x)
        inner_mlp, gate = torch.split(x, self.in_proj_split, dim=-1)
        x = self.activation_fn(inner_mlp) * gate
        x = self.linear2(x)
        return x
