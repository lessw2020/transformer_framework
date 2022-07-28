# Experimental - create default FSDP config for users

import functools
from dataclasses import dataclass

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing_wrapper,
)
from torch.distributed.fsdp import (
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

## FSDP Settings

@dataclass
class fsdp_config:
   # disable forward_prefetch since it currently doesn't work with activation
   # checkpointing for several cases
  forward_prefetch = False
    
  backward_prefetch = None  # BackwardPrefetch.BACKWARD_PRE
  
  # sharding policy
  sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    
  # activation checkpointing
    fsdp_activation_checkpointing: bool = True
      
    use_mixed_precision: bool = True
    # this is only for fp32 scenario...
    use_tf32: bool = False
      

## FSDP core functions -------------------
      
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

    
   
