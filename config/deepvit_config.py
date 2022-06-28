from dataclasses import dataclass
from torch.distributed.fsdp import ShardingStrategy


@dataclass
class train_config:

    # seed
    seed: int = 2022

    # model
    model_name = "500M"

    # available models - name is ~ num params
    # 60M
    # 500M
    # 1.5B
    # 3B
    # 8B

    # how many mini batches to time with
    total_steps_to_run: int = 5

    run_profiler: bool = False

    # log
    log_every: int = 1

    # save models
    save_model: bool = False
    save_folder = "training_checkpoints"
    checkpoint_max_save_count: int = (
        2  # number of 'best' checkpoints to save based on val loss
    )

    # sharding policy
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    print_sharding_plan: bool = False

    # dataloaders
    num_workers_dataloader: int = 0

    # policies
    use_mixed_precision: bool = True

    # activation checkpointing
    fsdp_activation_checkpointing: bool = True

    # datasets
    # dataset_train = "datasets_grammar/grammar_train.csv"
    # dataset_test = "datasets_grammar/grammar_validation.csv"

    # training
    batch_size_training: int = 5
    num_epochs: int = 1

    # validation
    run_validation: bool = True
    val_batch_size = 4

    # logging
    track_memory = True
    memory_report: bool = True
    nccl_debug_handler: bool = True
    distributed_debug: bool = True