import functools
import torch

from dataclasses import dataclass
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import Dataset
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from typing import Tuple
from vit_pytorch.deepvit import DeepViT, Residual

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    BackwardPrefetch,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)


@dataclass
class train_config:

    # seed
    seed: int = 2022

    # model
    model_name = "1.5B"

    # available models - name is ~ num params
    # 60M
    # 500M
    # 1.5B
    # 3B
    # 8B

    verbose: bool = True  # how much info to show...
    # how many mini batches to time with
    total_steps_to_run: int = 5

    run_profiler: bool = False

    # backward prefetch
    backward_prefetch = BackwardPrefetch.BACKWARD_PRE

    # log
    log_every: int = 1

    # checkpoint models
    save_model_checkpoint: bool = False
    load_model_checkpoint: bool = False
    checkpoint_type = StateDictType.FULL_STATE_DICT
    model_save_name = "t5-"
    checkpoint_folder = "training_checkpoints"
    checkpoint_max_save_count: int = (
        2  # number of 'best' checkpoints to save based on val loss
    )

    # optimizers load and save
    save_optimizer: bool = False
    load_optimizer: bool = False
    optimizer_name: str = "Adam"
    optimizer_checkpoint_file: str = "Adam-t5--1.pt"

    checkpoint_model_filename: str = "t5--1.pt"

    # sharding policy
    sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP
    print_sharding_plan: bool = False

    # dataloaders
    num_workers_dataloader: int = 8

    # policies
    use_mixed_precision: bool = True

    # activation checkpointing
    fsdp_activation_checkpointing: bool = True

    # datasets
    # dataset_train = "datasets_grammar/grammar_train.csv"
    # dataset_test = "datasets_grammar/grammar_validation.csv"

    # training
    batch_size_training: int = 4
    num_epochs: int = 1

    # validation
    run_validation: bool = False
    val_batch_size = 4

    # logging
    track_memory = True
    memory_report: bool = True
    nccl_debug_handler: bool = True
    distributed_debug: bool = True


def build_model(model_size: str):
    model_args = dict()
    if model_size == "60M":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 1,
            "heads": 1,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }
    if model_size == "500M":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 59,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }

    if model_size == "1.5B":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 177,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }
    if model_size == "3B":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 357,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }
    if model_size == "8B":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 952,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }
    model = DeepViT(**model_args)

    return model


class GeneratedDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        super()
        self._input_shape = kwargs.get("input_shape", [3, 256, 256])
        self._input_type = kwargs.get("input_type", torch.float32)
        self._len = kwargs.get("len", 1000000)
        self._num_classes = kwargs.get("num_classes", 1000)

    def __len__(self):
        return self._len

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rand_image = torch.randn(self._input_shape, dtype=self._input_type)
        label = torch.tensor(
            data=[index % self._num_classes], dtype=torch.int64)
        return rand_image, label


def get_policy():
    functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Residual,
        },
    )
