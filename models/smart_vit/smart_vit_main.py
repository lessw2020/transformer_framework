""" Smart ViT - this includes arch changes based on:
https://arxiv.org/abs/2302.05442  (Scaling VIT to 22B)
and leverages/builds on Ross Wightman's:
Vision_Transformer_RelPos
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer_relpos.py

Immediate changes:
Adds Dual Patchnorm and default qk_norm
"""

import torch
import torch.nn as nn
from typing import Optional

from mlp.base_mlp import LinearMLP
from functools import partial
from patch_embeddings.dual_patchnorm import DualPatchNormEmbedding
from positional_embedding.rel_pos_embd import RelPosBias, RelPosMlp

from .sar import checkpoint


# from Ross Wightman layers:


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


def build_smart_vit(model_params):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_bias=False,
        block_fn=ResPostRelPosBlock,
    )
    merged_vals = {**model_kwargs, **model_params}
    model = VisionTransformerRelPos(**merged_vals)
    return model


class ScaledAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=False,
        rel_pos_cls=None,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        scaled_attn=False,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim should be divisible by num_heads, got {dim=} and {num_heads=}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = scaled_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim)
        self.k_norm = norm_layer(self.head_dim)
        self.rel_pos = rel_pos_cls(num_heads=num_heads) if rel_pos_cls else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def checkpoint_forward(self, q, k, v):
        def cp_forward(q, k, v):
            # print(f"q shape {q.shape}")

            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            attn = attn.softmax(dim=-1)
            # attn = self.attn_drop(attn)
            x = attn @ v

            return x

        hidden_states = checkpoint(cp_forward, None, q, k, v)
        print(f"returning hidden states")

        return hidden_states

    '''def _checkpointed_attention_forward(self, query_layer, key_layer,
                                        value_layer, attention_mask):
        """Forward method with activation checkpointing."""
        def custom_forward(*inputs):
            query_layer = inputs[0]
            key_layer = inputs[1]
            value_layer = inputs[2]
            attention_mask = inputs[3]
            output_ = self.core_attention(query_layer, key_layer,
                                          value_layer, attention_mask)
            return output_

        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False, query_layer, key_layer, value_layer, attention_mask)

        return hidden_states
    '''

    def forward(
        self,
        x,
        shared_rel_pos: Optional[torch.Tensor] = None,
        use_ckp=False,
    ):
        # print(f"correct forward!!!!!!")
        if use_ckp:
            print(f"Using CHKP! ")
            B, N, C = x.shape
            # print(f"{B=}, {N=}, {C=}")
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)
            q = self.q_norm(q)
            k = self.k_norm(k)
            x = self.checkpoint_forward(q, k, v)

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            # x = self.proj_drop(x)
            return x
        else:
            B, N, C = x.shape
            # print(f"{B=}, {N=}, {C=}")
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)
            q = self.q_norm(q)
            k = self.k_norm(k)
            # print(f"q shape {q.shape}")

            if self.fused_attn:
                attn_bias = None
                if self.rel_pos is not None:
                    attn_bias = self.rel_pos.get_bias()
                elif shared_rel_pos is not None:
                    attn_bias = shared_rel_pos

                x = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_bias,
                    dropout_p=self.attn_drop.p,
                )
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                # if self.rel_pos is not None:
                #    print(f"self rel pos is valid")
                #    attn = self.rel_pos(attn, shared_rel_pos=shared_rel_pos)
                # elif shared_rel_pos is not None:
                #    attn = attn + shared_rel_pos
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


class RelPosBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        rel_pos_cls=None,
        init_values=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ScaledAttention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            rel_pos_cls=rel_pos_cls,
            attn_drop=attn_drop,
            proj_drop=drop,
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
            drop=drop,
            bias=True,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor] = None):
        x = x + self.drop_path1(
            self.ls1(self.attn(self.norm1(x), shared_rel_pos=shared_rel_pos))
        )
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ResPostRelPosBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        rel_pos_cls=None,
        init_values=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.init_values = init_values

        self.attn = ScaledAttention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            rel_pos_cls=rel_pos_cls,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = LinearMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.init_weights()

    def init_weights(self):
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x, shared_rel_pos: Optional[torch.Tensor] = None):
        x = x + self.drop_path1(self.norm1(self.attn(x, shared_rel_pos=shared_rel_pos)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class VisionTransformerRelPos(nn.Module):
    """Vision Transformer w/ Relative Position Bias

    Differing from classic vit, this impl
      * uses relative position index (swin v1 / beit) or relative log coord + mlp (swin v2) pos embed
      * defaults to no class token (can be enabled)
      * defaults to global avg pool for head (can be changed)
      * layer-scale (residual branch gain) enabled
    """

    def __init__(
        self,
        img_size=224,
        image_size=224,  # temp TODO resolve dupe image/img names
        input_size=(3, 224, 224),  # temp duplicate
        pool_size=None,
        crop_pct=None,
        interpolation=None,
        fixed_input_size=None,
        first_conv=None,
        classifier=None,  # end of dupes
        patch_size=16,
        in_chans=3,
        num_classes=3,
        global_pool="avg",
        embed_dim=1024,
        depth=12,
        num_heads=14,
        mlp_ratio=4.0,
        qkv_bias=True,
        init_values=1e-6,
        class_token=False,
        fc_norm=False,
        rel_pos_type="mlp",
        rel_pos_dim=None,
        shared_rel_pos=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="skip",
        embed_layer=DualPatchNormEmbedding,
        norm_layer=None,
        act_layer=None,
        block_fn=ResPostRelPosBlock,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'avg')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_norm (bool): Enable normalization of query and key in attention
            init_values: (float): layer-scale init values
            class_token (bool): use class token (default: False)
            fc_norm (bool): use pre classifier norm instead of pre-pool
            rel_pos_ty pe (str): type of relative position
            shared_rel_pos (bool): share relative pos across all blocks
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        feat_size = self.patch_embed.grid_size

        rel_pos_args = dict(window_size=feat_size, prefix_tokens=self.num_prefix_tokens)
        # if rel_pos_type.startswith("mlp"):
        #    if rel_pos_dim:
        #        rel_pos_args["hidden_dim"] = rel_pos_dim
        #    if "swin" in rel_pos_type:
        #        rel_pos_args["mode"] = "swin"
        rel_pos_cls = partial(RelPosMlp, **rel_pos_args)
        # else:
        # rel_pos_cls = partial(RelPosBias, **rel_pos_args)
        self.shared_rel_pos = None
        # if shared_rel_pos:
        #    self.shared_rel_pos = rel_pos_cls(num_heads=num_heads)
        # NOTE shared rel pos currently mutually exclusive w/ per-block, but could support both...
        #   rel_pos_cls = None

        self.cls_token = (
            nn.Parameter(torch.zeros(1, self.num_prefix_tokens, embed_dim))
            if class_token
            else None
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    rel_pos_cls=rel_pos_cls,
                    init_values=init_values,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim) if not fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if fc_norm else nn.Identity()
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if weight_init != "skip":
            self.init_weights(weight_init)

    def init_weights(self, mode=""):
        assert mode in ("jax", "moco", "")
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        # FIXME weight init scheme using PyTorch defaults curently
        # named_apply(get_init_weights_vit(mode, head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r"^cls_token|patch_embed",  # stem and embed
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

    def forward_features(self, x):
        x = self.patch_embed(x)
        # if self.cls_token is not None:
        #    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        # shared_rel_pos = (
        #    self.shared_rel_pos.get_bias() if self.shared_rel_pos is not None else None
        # )
        for blk in self.blocks:
            # if self.grad_checkpointing and not torch.jit.is_scripting():
            #    x = checkpoint(blk, x, shared_rel_pos=shared_rel_pos)
            # else:
            x = blk(x, shared_rel_pos=None)  # shared_rel_pos)
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
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


# def _create_vision_transformer_relpos(variant, pretrained=False, **kwargs):
#    model = build_model_with_cfg(VisionTransformerRelPos, variant, pretrained, **kwargs)
#    return model


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        # "mean": IMAGENET_INCEPTION_MEAN,
        # "std": IMAGENET_INCEPTION_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }
