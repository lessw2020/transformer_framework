# dual patchnorm implementation
# based on https://arxiv.org/abs/2302.01327v2

from einops.layers.torch import Rearrange
import torch
import torch.nn as nn


class DualPatchNormEmbedding(nn.Module):
    """patch embedding via linear projection with dual pre and post LN"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        use_bias_proj: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        self.patch_size = patch_size
        self.patch_dim = in_chans * (patch_size) ** 2
        self.proj_layer = nn.Linear(
            self.patch_dim,
            embed_dim,
            bias=use_bias_proj,
        )
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.dual_patchnorm = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            nn.LayerNorm(self.patch_dim, norm_eps),
            self.proj_layer,
            nn.LayerNorm(embed_dim, norm_eps),
        )

    def forward(self, x):
        # incoming x = b,c,h,w
        # debug:
        # b,c, h,w = x.shape
        # assert h == self.img_size
        # assert w = self.img_size
        x = self.dual_patchnorm(x)
        return x
