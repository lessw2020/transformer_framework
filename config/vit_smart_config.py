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
    # model_name = "90M"
    total_steps_to_run: int = 8
    use_timm = False
    model_name = (
        # "vit_relpos_medium_patch16_rpn_224"  #
        # "vit_relpos_base_patch16_rpn_224"
        # "maxxvitv2_rmlp_base_rw_224"
        #"smartvit90"
        #"631M"
        "1B"
        #"1.8B"
        #"4B"
        #"22B"

    )

    use_parallel_attention: bool = True

    # only relevant if using parallel_attention...
    # parallel attention outer projection fusion
    use_upper_fusion: bool = True  

    # use scaled dot product attention
    use_fused_attention: bool = True

    # profile
    run_profiler: bool = False
    profile_folder: str = "tp_fsdp/profile_tracing"

    # use TP
    use_tp: bool = False

    # image size
    image_size: int = 224

    batch_size_training: int = 20

    # use synthetic data
    use_synthetic_data: bool = False
    use_label_singular = False
    # todo - below needs to become dynamic since we are adding more datasets
    use_pokemon_dataset: bool = False
    if use_pokemon_dataset:
        NUM_CLASSES = 150

    use_beans_dataset: bool = False
    if use_beans_dataset:
        NUM_CLASSES = 3

    use_food = True

    if use_food:
        NUM_CLASSES = 101
        use_label_singular = True

    # real dset
    num_categories = NUM_CLASSES
    label_smoothing_value = 0.0
    # train_data_path = "datasets_vision/pets/train"
    # val_data_path = "datasets_vision/pets/val"

    # mixed precision
    use_mixed_precision: bool = True

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
    optimizer = "dadapt_adam"

    save_optimizer: bool = False
    load_optimizer: bool = False

    optimizer_checkpoint_file: str = "Adam-vit--1.pt"

    checkpoint_model_filename: str = "vit--1.pt"


def build_model(model_size: str, layernorm_eps_in: float = 1e-6, use_parallel_attention=True, use_upper_fusion=True, use_fused_attention=True):


    if model_size == "smartvit90":
        model_args = {
            "patch_size": 16,
            "embed_dim": 1320,
            "depth": 16,
            "num_heads": 12,
            "num_classes": NUM_CLASSES,
            "image_size": 224,
            
        }
    elif model_size == "631M":

        model_args = {
            "patch_size": 14,
            "embed_dim": 1280,
            "depth": 32,
            "num_heads": 16,
            "num_classes": NUM_CLASSES,
            "image_size": 224,
            
        }
    elif model_size == "1B":
        #model_args = dict(patch_size=14, embed_dim=1408, mlp_ratio=48/11, depth=40, num_heads=16)
        model_args = {
            "patch_size": 14,
            "embed_dim": 1408,
            "mlp_ratio":48/11,
            "depth": 40,
            "num_heads": 16,
            "num_classes": NUM_CLASSES,
            "image_size": 224,
            
        }
    elif model_size =="1.8B":
        # model_args = dict(patch_size=14, embed_dim=1664, mlp_ratio=64/13, depth=48, num_heads=16)
        model_args = {
            "patch_size": 14,
            "embed_dim": 1664,
            "mlp_ratio":64/13,
            "depth": 48,
            "num_heads": 16,
            "num_classes": NUM_CLASSES,
            "image_size": 224,
            
        }
    elif model_size == "4B":
        model_args = {
            "patch_size": 14,
            "embed_dim": 1792,
            "mlp_ratio":8.571428571428571,
            "depth": 56,
            "num_heads": 16,
            "num_classes": NUM_CLASSES,
            "image_size": 224,
            
        }
    elif model_size == "22B":
        model_args = {
            "patch_size": 14,
            "embed_dim": 6144,
            #"mlp_ratio":4.0,
            "depth": 48,
            "num_heads": 48,
            "num_classes": NUM_CLASSES,
            "image_size": 224,
            
        }



    # core model args
    #model_args["layernorm_eps"] = layernorm_eps_in

    # current control over parallel vs sequential attention blocks
    if use_parallel_attention:
        model_args["use_parallel_attention"]= True
    if use_fused_attention:
        model_args["use_fused_attention"]=True
    if use_upper_fusion:
        model_args["use_upper_fusion"]=True

    assert model_args.get(
        "image_size"
    ), f"failed to build model args for {model_size=}...is your model size listed in config?"

    from models.smart_vit.vit_main import build_smart_vit

    model = build_smart_vit(model_args)
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


def get_universal_dataset():
    from dataset_classes.hf_universal import get_datasets

    return get_datasets()


def get_policy():
    #return None
    cfg = train_config()
    # todo - can't use autowrap policy with 2d
    from models.smart_vit.vit_main import ParallelAttentionBlock, ResPostBlock
    if cfg.use_parallel_attention:
        return get_policy_base({ParallelAttentionBlock})
    else:
        return get_policy_base({ResPostBlock})


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
    use_synthetic_data=False,
    use_label_singular=False,
):
    cfg = train_config()
    label_smoothing_amount = cfg.label_smoothing_value
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing_amount)
    t0 = time.perf_counter()
    for batch_index, (batch) in enumerate(data_loader, start=1):
        # print(f"{batch=}")
        if use_synthetic_data:
            inputs, targets = batch
        elif use_label_singular:
            inputs = batch["pixel_values"]
            targets = batch["label"]

        else:
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


def validation(
    model,
    local_rank,
    rank,
    val_loader,
    world_size,
    stats=None,
    use_label_singular=False,
    metric_logger=None,
):
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
            if use_label_singular:
                inputs = batch["pixel_values"]
                targets = batch["label"]
            else:
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
            float_acc = float(acc)
            stats["best_accuracy"] = max(float_acc, stats["best_accuracy"])
            best_acc = stats["best_accuracy"]
            stats["loss"].append(loss)
            stats["accuracy"].append(acc)
            if metric_logger:
                epoch_results = f"accuracy: {acc}, best_acc: {best_acc}, loss: {loss}\n"
                try:
                    with open(metric_logger, "a") as fwriter:
                        fwriter.write(epoch_results)

                except OSError as oserr:
                    print("Error while writing stats to disc ", oserr)

            # print(f"{stats=}")
    return
