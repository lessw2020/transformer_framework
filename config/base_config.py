from dataclasses import dataclass

from torch.distributed.fsdp import (
    ShardingStrategy,
    BackwardPrefetch,
)

@dataclass
class base_config:

    # seed
    seed: int = 2022
    verbose: bool = True  # how much info to show...
    # how many mini batches to time with
    total_steps_to_run: int = 5

    # training
    batch_size_training: int = 15
    num_epochs: int = 1

    # sharding policy
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    print_sharding_plan: bool = False

    run_profiler: bool = False

    # backward prefetch
    backward_prefetch = BackwardPrefetch.BACKWARD_PRE

    # log
    log_every: int = 1

    # dataloaders
    num_workers_dataloader: int = 2

    # policies
    use_mixed_precision: bool = True

    # activation checkpointing
    fsdp_activation_checkpointing: bool = True

    # validation
    run_validation: bool = False
    val_batch_size = 4

    # logging
    track_memory = True
    memory_report: bool = True
    nccl_debug_handler: bool = True
    distributed_debug: bool = True