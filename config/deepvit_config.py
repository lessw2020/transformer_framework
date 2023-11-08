import time
from dataclasses import dataclass
from typing import Tuple
import os

import torch
import torchvision
import torchvision.transforms as transforms
import tqdm
from torch import distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from vit_pytorch.deepvit import DeepViT, Transformer
import torchvision.models as models

from .base_config import base_config, fsdp_checkpointing_base, get_policy_base

NUM_CLASSES = 10000


@dataclass
class train_config(base_config):
    # model
    model_name = "1B"

    # available models -name is ~ num params
    # 60M
    # 500M
    # 750M
    # 1B
    # 1.5B
    # 2B
    # 2.5B
    # 3B
    # 8B

    # training
    batch_size_training: int = 192

    # use synthetic data
    use_synthetic_data: bool = True
    train_data_path = "datasets_vision/imagenette320/train"
    val_data_path = "datasets_vision/imagenette320/val"

    # mixed precision
    use_mixed_precision: bool = True

    # activation checkpointing
    fsdp_activation_checkpointing: bool = True
    hf_t5_checkpointing: bool = False

    # torch.compile
    use_torch_compile: bool = True

    use_deferred_init: bool = False

    # checkpoint models
    save_model_checkpoint: bool = False
    # only for local dist
    single_file_per_rank = True

    load_model_checkpoint: bool = False
    checkpoint_type = StateDictType.SHARDED_STATE_DICT

    dist_checkpoint_root_folder = "distributed_checkpoints"
    dist_checkpoint_folder = "DeepVit_local_checkpoint"
    model_save_name = "deepvit-"
    checkpoint_folder = "training_checkpoints"
    checkpoint_max_save_count: int = (
        2  # number of 'best' checkpoints to save based on val loss
    )

    save_using_num_threads: int = 10

    # optimizers load and save
    save_optimizer: bool = False
    load_optimizer: bool = False

    optimizer_checkpoint_file: str = "Adam-deepvit--1.pt"

    checkpoint_model_filename: str = "deepvit--1.pt"


def build_model(model_size: str, **kwargs):
    model_args = dict()
    if model_size == "60M":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": NUM_CLASSES,
            "dim": 1024,
            "depth": 10,
            "heads": 8,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }
    if model_size == "120M":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": NUM_CLASSES,
            "dim": 1024,
            "depth": 2,
            "heads": 2,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }
    if model_size == "500M":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": NUM_CLASSES,
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
            "num_classes": NUM_CLASSES,
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
            "num_classes": NUM_CLASSES,
            "dim": 1024,
            "depth": 118,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.0,
            "emb_dropout": 0.1,
        }
    if model_size == "1.5B":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": NUM_CLASSES,
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
            "num_classes": NUM_CLASSES,
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
            "num_classes": NUM_CLASSES,
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
            "num_classes": NUM_CLASSES,
            "dim": 1024,
            "depth": 357,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }
    if model_size == "3.5B":
        model.args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": NUM_CLASSES,
            "dim": 1024,
            "depth": 404,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }
    if model_size == "8B":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": NUM_CLASSES,
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


def get_dataset(train=True):
    cfg = train_config()
    if cfg.use_synthetic_data:
        return GeneratedDataset()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if train:
        input_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        input_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    data_path = cfg.train_data_path if train else cfg.val_data_path
    return torchvision.datasets.ImageFolder(data_path, transform=input_transform)


def get_policy():
    return get_policy_base({Transformer})


def fsdp_checkpointing(model):
    return fsdp_checkpointing_base(model, Transformer)


def train(
    model,
    data_loader,
    torch_profiler,
    optimizer,
    memmax,
    local_rank,
    tracking_duration,
    total_steps_to_run,
    use_synthetic_data=True,
    use_label_singular=False,
    use_parallel_attention=False,
    use_fused_attention=True,
    stats=None,
    lr_scheduler=None,
):
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

        if batch_index % cfg.log_every == 0 and torch.distributed.get_rank() == 0:
            print(
                f"step: {batch_index}: time taken for the last {cfg.log_every} steps is {mini_batch_time}, loss is {loss}"
            )

        # reset timer
        t0 = time.perf_counter()
        if torch_profiler is not None:
            torch_profiler.step()
        if total_steps_to_run is not None and batch_index > total_steps_to_run:
            break


def validation(model, local_rank, rank, val_loader, world_size):
    epoch_val_accuracy = 0
    epoch_val_loss = 0
    model.eval()
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )

    loss_function = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(val_loader):
            inputs, target = inputs.to(torch.cuda.current_device()), target.to(
                torch.cuda.current_device()
            )
            output = model(inputs)
            loss = loss_function(output, target)

            # measure accuracy and record loss
            acc = (output.argmax(dim=1) == target).float().mean()
            epoch_val_accuracy += acc / len(val_loader)
            epoch_val_loss += loss / len(val_loader)

            if rank == 0:
                inner_pbar.update(1)

    metrics = torch.tensor([epoch_val_loss, epoch_val_accuracy]).to(
        torch.cuda.current_device()
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    metrics /= world_size
    epoch_val_loss, epoch_val_accuracy = metrics[0], metrics[1]
    if rank == 0:
        print(f"val_loss : {epoch_val_loss:.4f} :  val_acc: {epoch_val_accuracy:.4f}\n")
    return
