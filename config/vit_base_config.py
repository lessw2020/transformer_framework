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
import torchvision.transforms as tvt

# from vit_pytorch.deepvit import DeepViT, Residual
# import torchvision.models as models
from models.vit import ViT, ViTEncoderBlock

from .base_config import base_config, fsdp_checkpointing_base, get_policy_base

NUM_CLASSES = 1000  # default to imagenet, updates in dataset selection


@dataclass
class train_config(base_config):
    # model
    model_name = "250M"

    use_timm = False
    # model_name = (
    # "vit_relpos_medium_patch16_rpn_224"  #
    #    "vit_relpos_base_patch16_rpn_224"
    # "maxxvitv2_rmlp_base_rw_224"
    # )
    # model_num_heads = 16

    # use TP
    use_tp: bool = False

    # image size
    image_size: int = 256

    # use synthetic data
    use_synthetic_data: bool = True

    # todo - below needs to become dynamic since we are adding more datasets
    use_pokemon_dataset: bool = False
    if use_pokemon_dataset:
        NUM_CLASSES = 150

    use_beans_dataset: bool = True
    if use_beans_dataset:
        NUM_CLASSES = 3

    # real dset
    num_categories = NUM_CLASSES
    # train_data_path = "datasets_vision/pets/train"
    # val_data_path = "datasets_vision/pets/val"

    # mixed precision
    use_mixed_precision: bool = False

    # checkpoint models
    save_model_checkpoint: bool = False
    # only for local and sharded dist
    single_file_per_rank = True

    load_model_checkpoint: bool = False
    checkpoint_type = StateDictType.SHARDED_STATE_DICT

    dist_checkpoint_root_folder = "distributed_checkpoints"
    dist_checkpoint_folder = "vit_local_checkpoint"
    model_save_name = "vit-"
    checkpoint_folder = "training_checkpoints"
    checkpoint_max_save_count: int = (
        2  # number of 'best' checkpoints to save based on val loss
    )

    layernorm_eps = 1e-6

    # optimizers load and save
    save_optimizer: bool = False
    load_optimizer: bool = False

    optimizer_checkpoint_file: str = "Adam-vit--1.pt"

    checkpoint_model_filename: str = "vit--1.pt"

    # VIT specific
    """image_size": cfg.TRAIN.IM_SIZE,
            "patch_size": cfg.VIT.PATCH_SIZE,
            "stem_type": cfg.VIT.STEM_TYPE,
            "c_stem_kernels": cfg.VIT.C_STEM_KERNELS,
            "c_stem_strides": cfg.VIT.C_STEM_STRIDES,
            "c_stem_dims": cfg.VIT.C_STEM_DIMS,
            "n_layers": cfg.VIT.NUM_LAYERS,
            "n_heads": cfg.VIT.NUM_HEADS,
            "hidden_d": cfg.VIT.HIDDEN_DIM,
            "mlp_d": cfg.VIT.MLP_DIM,
            "cls_type": cfg.VIT.CLASSIFIER_TYPE,
            "num_classes": cfg.MODEL.NUM_CLASSES,
        }
    # Patch Size (TRAIN.IM_SIZE must be divisible by PATCH_SIZE)
_C.VIT.PATCH_SIZE = 16

# Type of stem select from {'patchify', 'conv'}
_C.VIT.STEM_TYPE = "patchify"

# C-stem conv kernel sizes (https://arxiv.org/abs/2106.14881)
_C.VIT.C_STEM_KERNELS = []

# C-stem conv strides (the product of which must equal PATCH_SIZE)
_C.VIT.C_STEM_STRIDES = []

# C-stem conv output dims (last dim must equal HIDDEN_DIM)
_C.VIT.C_STEM_DIMS = []

# Number of layers in the encoder
_C.VIT.NUM_LAYERS = 12

# Number of self attention heads
_C.VIT.NUM_HEADS = 12

# Hidden dimension
_C.VIT.HIDDEN_DIM = 768

# Dimension of the MLP in the encoder
_C.VIT.MLP_DIM = 3072

# Type of classifier select from {'token', 'pooled'}
_C.VIT.CLASSIFIER_TYPE = "token"

    """


def build_model(model_size: str, layernorm_eps_in: float = 1e-6):
    model_args = dict()
    model_args["layernorm_eps"] = layernorm_eps_in

    if model_size == "250M":
        model_args = {
            **model_args,
            "image_size": 224,
            "patch_size": 16,
            "num_classes": NUM_CLASSES,
            "mlp_dim": 3072,
            "dropout": 0.1,
            "emb_dropout": 0.1,
            "c_stem_kernels": [],
            "c_stem_strides": [],
            "c_stem_dims": [],
            "n_layers": 24,
            "n_heads": 16,
            "hidden_d": 1024,
            "mlp_d": 4096,
            "cls_type": "pooled",
            "stem_type": "patchify",
        }

    assert model_args.get(
        "image_size"
    ), f"failed to build model args for {model_size=}...is your model size listed in config?"
    model = ViT(params=model_args)

    return model


class GeneratedDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        super()
        image_size = kwargs.get("image_size", 256)
        self._input_shape = kwargs.get("input_shape", [3, image_size, image_size])
        self._input_type = kwargs.get("input_type", torch.float32)
        self._len = kwargs.get("len", 1000000)
        self._num_classes = kwargs.get("num_classes", 1000)

    def __len__(self):
        return self._len

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rand_image = torch.randn(self._input_shape, dtype=self._input_type)
        label = torch.tensor(data=[index % self._num_classes], dtype=torch.int64)
        return rand_image, label


def load_image_batch(batch, feature_extractor):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in batch["image"]], return_tensors="pt")

    inputs["labels"] = batch["labels"]
    return inputs


def get_hf_extractor(model_name=None):
    from transformers import ViTFeatureExtractor

    if not model_name:
        model_name = "google/vit-base-patch16-224-in21k"
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)


def get_dataset():
    """generate both train and val dataset"""
    cfg = train_config()
    if cfg.use_synthetic_data:
        image_size = 256
        if cfg.image_size:
            image_size = cfg.image_size
        return GeneratedDataset(image_size=cfg.image_size)


def get_beans_dataset():
    from dataset_classes.dataset_beans import get_datasets

    return get_datasets()


def get_pokemon_dataset():
    from dataset_classes.dataset_pokemon import get_datasets

    return get_datasets()


def get_policy():
    # todo - can't use autowrap policy with 2d
    return None  # get_policy_base({ViTEncoderBlock})


def fsdp_checkpointing(model):
    return fsdp_checkpointing_base(model, ViTEncoderBlock)


def train(
    model,
    data_loader,
    torch_profiler,
    optimizer,
    memmax,
    local_rank,
    tracking_duration,
    total_steps_to_run,
):
    cfg = train_config()
    loss_function = torch.nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    for batch_index, (batch) in enumerate(data_loader, start=1):
        inputs = batch["pixel_values"]
        targets = batch["labels"]
        inputs, targets = inputs.to(torch.cuda.current_device()), torch.squeeze(
            targets.to(torch.cuda.current_device()), -1
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


def validation(model, local_rank, rank, val_loader, world_size, stats=None):
    epoch_val_accuracy = 0
    epoch_val_loss = 0
    model.eval()
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )
        assert stats is not None, f"missing stats!"

    loss_function = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (batch) in enumerate(val_loader):
            inputs = batch["pixel_values"]
            targets = batch["labels"]

            inputs, targets = inputs.to(torch.cuda.current_device()), targets.to(
                torch.cuda.current_device()
            )
            output = model(inputs)
            loss = loss_function(output, targets)

            # measure accuracy and record loss
            acc = (output.argmax(dim=1) == targets).float().mean()
            # if acc > 0:
            #    print(f"********** success: {acc=}\n")
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
        if stats is not None:
            print(f"updating stats...")
            loss = f"{epoch_val_loss:.4f}"
            acc = f"{epoch_val_accuracy:.4f}"
            stats["loss"].append(loss)
            stats["accuracy"].append(acc)
            # print(f"{stats=}")
    return
