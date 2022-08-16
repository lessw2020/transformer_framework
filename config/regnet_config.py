import time
import torch
from dataclasses import dataclass
from typing import Tuple

from transformers import RegNetForImageClassification, RegNetConfig
from transformers.models.regnet.modeling_regnet import RegNetStage, RegNetYLayer
from torch.utils.data import Dataset
from torch.distributed.fsdp import StateDictType
from .base_config import base_config, fsdp_checkpointing_base, get_policy_base


@dataclass
class train_config(base_config):

    # model
    # model_name = "facebook/regnet-y-040" # 20M
    model_name = "3B"
    # 3B
    # 7B
    # 10B

    # checkpoint models
    save_model_checkpoint: bool = False
    load_model_checkpoint: bool = False
    checkpoint_type = StateDictType.FULL_STATE_DICT
    model_save_name = "regnet-"
    checkpoint_folder = "training_checkpoints"
    checkpoint_max_save_count: int = (
        2  # number of 'best' checkpoints to save based on val loss
    )

    # optimizers load and save
    save_optimizer: bool = False
    load_optimizer: bool = False
    
    optimizer_checkpoint_file: str = "Adam-regnet--1.pt"

    checkpoint_model_filename: str = "regnet--1.pt"


def build_model(model_name: str):
    if model_name == "1B":
        config = RegNetConfig(
            num_labels=1000,
            depths=[2, 7, 17, 1],
            hidden_sizes=[1010, 2020, 3333, 1414],
            groups_width=1010,
        )
    elif model_name == "3B":
        config = RegNetConfig(
            num_labels=1000,
            depths=[2, 7, 17, 1],
            hidden_sizes=[2020, 4040, 6666, 2828],
            groups_width=1010,
        )
    elif model_name == "7B":
        config = RegNetConfig(
            num_labels=1000,
            depths=[2, 7, 17, 1],
            hidden_sizes=[2020, 4040, 11110, 2828],
            groups_width=1010,
        )
    elif model_name == "10B":
        config = RegNetConfig(
            num_labels=1000,
            depths=[2, 7, 17, 1],
            hidden_sizes=[2020, 4040, 11110, 28280],
            groups_width=1010,
        )
    return RegNetForImageClassification(config)


class GeneratedDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        super()
        self._input_shape = kwargs.get("input_shape", [3, 224, 224])
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
    return get_policy_base({RegNetYLayer})


def fsdp_checkpointing(model):
    return fsdp_checkpointing_base(model, RegNetYLayer)


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
        outputs = model(inputs).logits
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