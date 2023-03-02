import functools
from dataclasses import dataclass
import policies

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp import (
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch


@dataclass
class base_config:
    # seed
    seed: int = 2022
    verbose: bool = True  # how much info to show...
    # how many mini batches to time with
    total_steps_to_run: int = None

    # stats
    print_memory_summary: bool = False

    # training
    num_epochs: int = 8

    model_weights_bf16: bool = False  # warning, True will  move model weights to BF16...use BFF_AdamW optimizer

    # policies
    use_mixed_precision: bool = True
    mp_policy = policies.bf16_policy

    use_low_precision_gradient_policy: bool = False
    # this is only for fp32 scenario...
    use_tf32: bool = False

    label_smoothing_value = 0.0  # default to none, adjust in model config

    # add in tp support (default to false for base, activate in model)
    # generally change only in the model config, this is here for back compat.
    use_tp = False

    # optimizer config
    optimizer: str = (
        "dadapt_adam"  # [AdamW, AnyPrecision, int8] (fp32, bf16, int8 optimizers)
    )

    ap_momentum_dtype = torch.float32  # momentum and variance
    ap_variance_dtype = torch.float32  # variance

    ap_use_kahan_summation: bool = False

    # sharding policy
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    print_sharding_plan: bool = False

    run_profiler: bool = False
    profile_folder: str = "fsdp/profile_tracing"

    # disable forward_prefetch since it currently doesn't work with activation
    # checkpointing for several cases
    forward_prefetch = False

    # log
    log_every: int = 1

    # dataloaders
    num_workers_dataloader: int = 2

    # training
    batch_size_training: int = 48

    # activation checkpointing
    fsdp_activation_checkpointing: bool = False

    # validation
    run_validation: bool = True
    val_batch_size = 24

    # logging
    track_memory = True
    memory_report: bool = True
    nccl_debug_handler: bool = True
    distributed_debug: bool = True

    # use_non_recursive_wrapping: bool = True
    # backward_prefetch = None

    use_non_recursive_wrapping: bool = False
    backward_prefetch = BackwardPrefetch.BACKWARD_PRE


def get_policy_base(blocks):
    cfg = base_config()
    recursive_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=blocks,
    )
    if not cfg.use_non_recursive_wrapping:
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


def fsdp_checkpointing_base(model, blocks):
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
