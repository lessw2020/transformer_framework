import functools
import time
import torch
from dataclasses import dataclass
from typing import Tuple

from torch.utils.data import Dataset
from torch.distributed.fsdp.wrap import (
    always_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.distributed.fsdp import StateDictType
from vit_pytorch.deepvit import DeepViT, Residual
from .base_config import base_config


@dataclass
class train_config(base_config):

    # model
    model_name = "500M"

    # available models - name is ~ num params
    # 60M
    # 500M
    # 750M
    # 1B
    # 1.5B
    # 2B
    # 2.5B
    # 3B
    # 8B

    # checkpoint models
    save_model_checkpoint: bool = False
    load_model_checkpoint: bool = False
    checkpoint_type = StateDictType.FULL_STATE_DICT
    model_save_name = "deepvit-"
    checkpoint_folder = "training_checkpoints"
    checkpoint_max_save_count: int = (
        2  # number of 'best' checkpoints to save based on val loss
    )

    # optimizers load and save
    save_optimizer: bool = False
    load_optimizer: bool = False
    optimizer_name: str = "Adam"
    optimizer_checkpoint_file: str = "Adam-deepvit--1.pt"

    checkpoint_model_filename: str = "deepvit--1.pt"


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
    if model_size == "750M":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 89,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }

    if model_size == "1B":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 118,
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
    if model_size == "2B":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 236,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }

    if model_size == "2.5B":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 296,
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
        label = torch.tensor(data=[index % self._num_classes], dtype=torch.int64)
        return rand_image, label


def get_dataset():
    return GeneratedDataset()


def get_policy():
    recursive_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Residual,
        },
    )
    return recursive_policy
    # The ParamExecOrderPolicy that is in development
    # from torch.distributed.fsdp.wrap import (
    #     ParamExecOrderPolicy,
    #     HandleInitMode,
    # )
    # return ParamExecOrderPolicy(
    #     handle_init_mode=HandleInitMode.MODULE_LEVEL,
    #     bucket_size=int(17000000 * 2 + 1),
    #     module_level_group_policy=recursive_policy,
    # )


def train(model, data_loader, torch_profiler, optimizer, memmax, local_rank, tracking_duration, total_steps_to_run):
    cfg = train_config()
    loss_function = torch.nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    for batch_index, (inputs, target) in enumerate(data_loader, start=1):
        inputs, targets = inputs.to(torch.cuda.current_device()), torch.squeeze(
            target.to(torch.cuda.current_device()), -1
        )
        if optimizer:
            optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        if optimizer:
            optimizer.step()

        # update durations and memory tracking
        if local_rank == 0:
            mini_batch_time = time.perf_counter() - t0
            tracking_duration.append(mini_batch_time)
            if memmax:
                memmax.update()

        if (
            batch_index % cfg.log_every == 0
            and torch.distributed.get_rank() == 0
            and batch_index > 1
        ):
            print(
                f"step: {batch_index-1}: time taken for the last {cfg.log_every} steps is {mini_batch_time}, loss is {loss}"
            )

        # reset timer
        t0 = time.perf_counter()
        if torch_profiler is not None:
            torch_profiler.step()
        if batch_index > total_steps_to_run:
            break