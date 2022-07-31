# Experimental - create default FSDP config for users

import functools
from dataclasses import dataclass

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing_wrapper,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    BackwardPrefetch,
    MixedPrecision
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

bf16_policy = MixedPrecision(
        # Param precision
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )

## FSDP Settings


    
  # activation checkpointing
    fsdp_activation_checkpointing: bool = True
      
    use_mixed_precision: bool = True
    mixed_precision_policy: MixedPrecision = bf16_policy
    # this is only for fp32 scenario...
    use_tf32: bool = False
      

## FSDP core functions -------------------

def init_fsdp(model):
    model = FSDP
      
def fsdp_apply_checkpointing(model, blocks):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    
    check_fn = lambda submodule: isinstance(submodule, blocks)
    
    apply_activation_checkpointing_wrapper(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )

    
   
