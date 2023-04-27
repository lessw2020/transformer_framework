# Builds upon code from Ross Wightman/timm:
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

# adds DualPatchNormEmbedding, LinearMLP class
# Important updates - qk norm, parallelized block
# fused attention


import torch
import math
import torch.nn as nn
from typing import Optional, Callable, Tuple, Union

from mlp.base_mlp import LinearMLP
from functools import partial
from patch_embeddings.dual_patchnorm import DualPatchNormEmbedding
import logging
import torch.nn.functional as F
from collections import OrderedDict
from weight_init.weight_init import trunc_normal_, lecun_normal_


_logger = logging.getLogger(__name__)
import torch.distributed as dist


def _log(msg):
    rank = dist.get_rank()
    if rank == 0:
        _logger.warning(f"{msg}")


# from Ross Wightman layers:


def named_apply(
    fn: Callable,
    module: nn.Module,
    name="",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    return_indices: torch.jit.Final[bool]

    def __init__(
        self,
        prob: float = 0.5,
        num_prefix_tokens: int = 1,
        ordered: bool = False,
        return_indices: bool = False,
    ):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.num_prefix_tokens = (
            num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        )
        self.ordered = ordered
        self.return_indices = return_indices

    def forward(
        self, x
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if not self.training or self.prob == 0.0:
            if self.return_indices:
                return x, None
            return x

        if self.num_prefix_tokens:
            prefix_tokens, x = (
                x[:, : self.num_prefix_tokens],
                x[:, self.num_prefix_tokens :],
            )
        else:
            prefix_tokens = None

        B = x.shape[0]
        L = x.shape[1]
        num_keep = max(1, int(L * (1.0 - self.prob)))
        keep_indices = torch.argsort(torch.randn(B, L, device=x.device), dim=-1)[
            :, :num_keep
        ]
        if self.ordered:
            # NOTE does not need to maintain patch order in typical transformer use,
            # but possibly useful for debug / visualization
            keep_indices = keep_indices.sort(dim=-1)[0]
        x = x.gather(1, keep_indices.unsqueeze(-1).expand((-1, -1) + x.shape[2:]))

        if prefix_tokens is not None:
            x = torch.cat((prefix_tokens, x), dim=1)

        if self.return_indices:
            return x, keep_indices
        return x


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=True,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        use_fused_attention=True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attention

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            #_log(f"running fused attention")
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = LinearMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ResPostBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_fused_attention = True,
    ):
        super().__init__()
        self.init_values = init_values

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            use_fused_attention=use_fused_attention
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = LinearMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.init_weights()

    def init_weights(self):
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x):
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x

class ParallelLayersBlock(nn.Module):
    """ Process MLP and Attention in parallel
        Based on 'Scaling Vision Transformers to 22 Billion Parameters` - https://arxiv.org/abs/2302.05442
        We do not use qkv bias
        Do use mlp bias
        Do use qk normalization
        This code is heavily based on TIMM ParallelScalingBlock: 
        https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py


    """
    def __init__(self, dimension, num_heads, mlp_ratio=4.0, qk_normalization=True, projection_drop = 0.0, attention_drop = 0.0, init_values=None, 
                 drop_path = 0.0, activation_layer = nn.GELU, normalization_layer=nn.LayerNorm, use_scaled_dpa=True, use_attention_out_bias=True):
        super().__init__()
        assert dimension % num_heads==0, f"dimensions {dimension.shape} must be evenly divisible by num_heads {num_heads=}"
        self.num_heads = num_heads
        self.head_dim = dimension//num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attention = use_scaled_dpa
        mlp_hidden_dim = int(mlp_ratio * dimension)
        in_proj_out_dim = mlp_hidden_dim + 3*dimension

        self.in_normalization = normalization_layer(dimension)
        self.in_projection = nn.Linear(dimension, in_proj_out_dim, bias = False) # not using qkv bias
        self.in_proj_split = [mlp_hidden_dim] + 3 * [dimension]

        # setup no op for qkv bias, but real bias for mlp portion of common in projection
        self.register_buffer("qkv_bias", torch.zeros(3*dimension), persistent=False)
        self.mlp_bias = nn.Parameter(torch.zeros(mlp_hidden_dim))

        self.q_norm = normalization_layer(self.head_dim)
        self.k_norm = normalization_layer(self.head_dim)
        self.attention_drop = nn.Dropout(attention_drop)
        self.attention_out_proj = nn.Linear(dimension, dimension, bias= use_attention_out_bias)

        self.mlp_drop = nn.Dropout(projection_drop)
        self.mlp_act = activation_layer()
        self.mlp_out_proj = nn.Linear(mlp_hidden_dim, dimension, bias = True)

        self.layer_scale = (
            LayerScale(dimension, init_values=init_values)
            if init_values is not None
            else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        B, S, C = x.shape

        # single layernorm for all
        y = self.in_normalization(x)

        # process first full MLP layer for all (qkv bias is not trained)
        y = F.linear(y, self.in_projection.weight,torch.cat((self.qkv_bias, self.mlp_bias)) )

        # split
        core_mlp, q, k, v = torch.split(y, self.in_proj_split, dim=-1)

        # Dot product attention w/ qk norm
        q = self.q_norm(q.view(B, S, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(B, S, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if self.fused_attention:
            final_attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attention_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            final_attn = attn @ v
        final_attn = final_attn.transpose(1, 2).reshape(B, S, C)

        # final attention linear out
        final_attn = self.attention_out_proj(final_attn)

        # process MLP side
        core_mlp = self.mlp_act(core_mlp)
        core_mlp = self.mlp_drop(core_mlp)
        core_mlp = self.mlp_out_proj(core_mlp)

        #assert x_mlp == test_xmlp, f"mismatch using sequential {test_xmlp=}, {res=}"
        # join attention and mlp outs
        y = self.drop_path(self.layer_scale(final_attn + core_mlp))
        # add residual
        x = x + y
        return x

class ParallelThingsBlock(nn.Module):
    """Parallel ViT block (N parallel attention followed by N parallel MLP)
    Based on:
      `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    """

    def __init__(
        self,
        dim,
        num_heads,
        num_parallel=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        init_values=None,
        proj_drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("norm", norm_layer(dim)),
                            (
                                "attn",
                                Attention(
                                    dim,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_norm=qk_norm,
                                    attn_drop=attn_drop,
                                    proj_drop=proj_drop,
                                    norm_layer=norm_layer,
                                ),
                            ),
                            (
                                "ls",
                                LayerScale(dim, init_values=init_values)
                                if init_values
                                else nn.Identity(),
                            ),
                            (
                                "drop_path",
                                DropPath(drop_path)
                                if drop_path > 0.0
                                else nn.Identity(),
                            ),
                        ]
                    )
                )
            )
            self.ffns.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("norm", norm_layer(dim)),
                            (
                                "mlp",
                                LinearMLP(
                                    dim,
                                    hidden_features=int(dim * mlp_ratio),
                                    act_layer=act_layer,
                                    drop=proj_drop,
                                ),
                            ),
                            (
                                "ls",
                                LayerScale(dim, init_values=init_values)
                                if init_values
                                else nn.Identity(),
                            ),
                            (
                                "drop_path",
                                DropPath(drop_path)
                                if drop_path > 0.0
                                else nn.Identity(),
                            ),
                        ]
                    )
                )
            )

    def _forward_jit(self, x):
        x = x + torch.stack([attn(x) for attn in self.attns]).sum(dim=0)
        x = x + torch.stack([ffn(x) for ffn in self.ffns]).sum(dim=0)
        return x

    @torch.jit.ignore
    def _forward(self, x):
        x = x + sum(attn(x) for attn in self.attns)
        x = x + sum(ffn(x) for ffn in self.ffns)
        return x

    def forward(self, x):
        return self._forward(x)


class VisionTransformer(nn.Module):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: str = "",
        embed_layer: Callable = DualPatchNormEmbedding,
        norm_layer: Optional[Callable] = None,
        act_layer: Optional[Callable] = None,
        block_fn: Callable = Block,
        input_size=224,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layey.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            # bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        )
        embed_len = (
            num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        )
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        #def __init__(self, dimension, num_heads, mlp_ratio=4.0, gk_normalization=True, projection_drop = 0.0, attention_drop = 0.0, init_values=None, 
        #         drop_path = 0.0, activation_layer = nn.GELU, normalization_layer=nn.LayerNorm, use_scaled_dpa=True, use_attention_out_bias=True):
        if block_fn == ParallelLayersBlock:

            self.blocks = nn.Sequential(
                *[
                    block_fn(
                        dimension=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        #qkv_bias=qkv_bias,
                        qk_normalization=qk_norm,
                        init_values=init_values,
                        projection_drop=proj_drop_rate,
                        attention_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        activation_layer=act_layer,
                        normalization_layer=norm_layer,
                        use_scaled_dpa = True, 
                        use_attention_out_bias=True,
                        
                    )
                    for i in range(depth)
                ]
            )
        else:
            self.blocks = nn.Sequential(
                *[
                    block_fn(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        #qkv_bias=qkv_bias,
                        qk_norm=qk_norm,
                        init_values=init_values,
                        proj_drop=proj_drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        use_fused_attention = True, 
                        
                    )
                    for i in range(depth)
                ]
            )

        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if weight_init != "skip":
            self.init_weights(weight_init)

    def init_weights(self, mode=""):
        _log(f"init mode = {mode=}")
        assert mode in ("jax", "jax_nlhb", "moco", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    # @torch.jit.ignore()
    # def load_pretrained(self, checkpoint_path, prefix=""):
    #    _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "token")
            self.global_pool = global_pool
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #    x = checkpoint_seq(self.blocks, x)
        # else:
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = (
                x[:, self.num_prefix_tokens :].mean(dim=1)
                if self.global_pool == "avg"
                else x[:, 0]
            )
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def init_weights_vit_jax(module: nn.Module, name: str = "", head_bias: float = 0.0):
    """ViT weight initialization, matching JAX (Flax) impl"""
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(
                    module.bias, std=1e-6
                ) if "mlp" in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = ""):
    """ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed"""
    if isinstance(module, nn.Linear):
        if "qkv" in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(
                6.0 / float(module.weight.shape[0] // 3 + module.weight.shape[1])
            )
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def get_init_weights_vit(mode="jax", head_bias: float = 0.0):
    if "jax" in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif "moco" in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


def resize_pos_embed(
    posemb,
    posemb_new,
    num_prefix_tokens=1,
    gs_new=(),
    interpolation="bicubic",
    antialias=False,
):
    """Rescale the grid of position embeddings when loading from state_dict.

    *DEPRECATED* This function is being deprecated in favour of resample_abs_pos_embed

    Adapted from:
        https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    """
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = (
            posemb[:, :num_prefix_tokens],
            posemb[0, num_prefix_tokens:],
        )
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info(
        f"Resized position embedding: {posemb.shape} ({[gs_old, gs_old]}) to {posemb_new.shape} ({gs_new})."
    )
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid,
        size=gs_new,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


def build_smart_vit(model_params):
    use_parallel = model_params.get('use_parallel_attention', False)
    
    if use_parallel:
        print(f"Building with Parallel Layers Attention")
        block_function = ParallelLayersBlock
        del model_params['use_parallel_attention']  # models don't understand this
    else:
        print(f"Building with Sequential Attention")
        block_function = ResPostBlock

    model_kwargs = dict(
        #patch_size=16,
        #embed_dim=768,
        #depth=12,
        #num_heads=12,
        qkv_bias=False,
        qk_norm=True, 
        block_fn=block_function,
        no_embed_class=True,
        #norm_layer=RmsNorm,
    )
    
    merged_vals = {**model_kwargs, **model_params}

    model = VisionTransformer(**merged_vals)
    return model
