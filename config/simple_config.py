# experimental - create a simple default config file for users with appropriate defaults
import functools
from dataclasses import dataclass

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    BackwardPrefetch,
    MixedPrecision,
)
import torch

# bf16 policy
bf16_mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# model specific -- users must supply this atm
from vit_pytorch.deepvit import DeepViT, Residual

model_layer_class = Residual


@dataclass
class fsdp_simple_config:

    # use mixed precision - will attempt bf16 mixed precision
    use_mixed_precision: bool = True
    active_mp_policy = bf16_mixed_precision_policy

    # for running in FP32 - use TF32 (recommend True)
    use_tf32: bool = True

    # prefetch settings
    forward_prefetch = False
    backward_prefetch: BackwardPrefetch = BackwardPrefetch.BACKWARD_PRE

    # sharding policy
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD

    # model wrapping policy
    use_non_recursive_wrapping_policy = False
    transformer_layer_class_set = {model_layer_class}

    # use activation checkpointing (default to True)
    use_fsdp_activation_checkpointing: bool = True
    transformer_layer_class = model_layer_class
