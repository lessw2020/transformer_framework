# experimental - house a number of 'boilerplate' code options in this file for good fsdp defaults.
import functools
from dataclasses import dataclass
from threading import local

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from config.simple_config import fsdp_simple_config as cfg_fsdp
from vit_pytorch.deepvit import DeepViT, Residual

# --- bfloat 16 checker
from pkg_resources import packaging
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist

# global flag that confirms ampere architecture, cuda version and
# nccl version to verify bfloat16 native support is ready

bfloat_native_available = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
)



def get_wrapping_policy(cfg, layer_class):
    recursive_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=layer_class,
    )
    if not cfg.use_non_recursive_wrapping_policy:
        return recursive_policy
    else:
        # The ParamExecOrderPolicy that is in development
        from torch.distributed.fsdp.wrap import (
            always_wrap_policy,
            ParamExecOrderPolicy,
            HandleInitMode,
        )
        return ParamExecOrderPolicy(
            handle_init_mode=HandleInitMode.MODULE_LEVEL,
            bucket_size=int(17000000 * 5 + 1),
            module_level_group_policy=always_wrap_policy,
        )


def apply_fsdp_checkpointing(model, blocks):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, blocks)
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )

def start_fsdp(model, local_rank):
    # a translation function that reads the config file and implements those fsdp settings 
    # into supplied model, returns model

    cfg = cfg_fsdp()

    # get mixed precision policy (and verify bfloat16)
    mp_policy = None # we default to fp32
    if cfg.use_mixed_precision:
        # verify bfloat16
        if bfloat_native_available:
            mp_policy = cfg.active_mp_policy
            if local_rank==0:
                print(f"BFloat16 mixed precision activated")
        else:
            if local_rank==0:
                print(f"WARNING: Mixed Precision requested in config, but BFloat not available...using FP32")    
    
    # if not using mixed precision, turn on TF32 for matmul?
    if not cfg.use_mixed_precision and cfg.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        if local_rank == 0:
            print(f"--> TF32 support for matmul enabled. ")

    # get wrapping policy
    current_layer_class = cfg.transformer_layer_class
    
    #if local_rank==0:
    print(f"wrapping model using the layer class {current_layer_class}")

    current_wrapping_policy = get_wrapping_policy(cfg, cfg.transformer_layer_class_set)
    print(f"wrapping policy ready") # = {current_wrapping_policy}")

    # apply fsdp
    model = FSDP(
        model,
        auto_wrap_policy=current_wrapping_policy,
        mixed_precision=mp_policy,
        backward_prefetch=cfg.backward_prefetch,
        sharding_strategy=cfg.sharding_strategy,
        device_id=torch.cuda.current_device(),
        forward_prefetch=cfg.forward_prefetch,
        # run_with_synch=True,
    )
    
    
    # apply checkpointing
    if cfg.use_fsdp_activation_checkpointing:
        apply_fsdp_checkpointing(model, current_layer_class)

    return model
 