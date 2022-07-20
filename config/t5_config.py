import time
import tqdm
import torch
from dataclasses import dataclass

from torch.distributed.fsdp import StateDictType
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.models.t5.modeling_t5 import T5Block
import datasets_grammar as dg
from .base_config import base_config, fsdp_checkpointing_base, get_policy_base


@dataclass
class train_config(base_config):

    # model
    model_name = "google/t5-v1_1-small"
    # available models
    # t5-base
    # google/t5-v1_1-small
    # google/t5-v1_1-base
    # google/t5-v1_1-large
    # google/t5-v1_1-xl  #3b
    # google/t5-v1_1-xxl #11b
    tokenizer = "t5-small"

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

    # datasets
    dataset_train = "datasets_grammar/grammar_train.csv"  # grammar_13k.csv
    dataset_test = "datasets_grammar/grammar_validation.csv"


def build_model(model_name: str):
    return AutoModelForSeq2SeqLM.from_pretrained(model_name)


def get_dataset():
    cfg = train_config()
    train_name = cfg.dataset_train
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer, model_max_length=512)
    train_dataset = dg.get_dataset(tokenizer, train_name, 512, 512, True)
    return train_dataset


def get_policy():
    return get_policy_base({T5Block})


def fsdp_checkpointing(model):
    return fsdp_checkpointing_base(model, T5Block)


def train(model, data_loader, torch_profiler, optimizer, memmax, local_rank, tracking_duration, total_steps_to_run):
    cfg = train_config()
    model.train()
    if local_rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(data_loader)), colour="blue", desc="r0 Training Epoch"
        )
    batch_index = 0
    t0 = time.perf_counter()
    for batch in data_loader:
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)
        if optimizer:
            optimizer.zero_grad()
        output = model(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=batch["target_ids"],
        )
        loss = output["loss"]
        loss.backward()
        if optimizer:
            optimizer.step()

        if local_rank == 0:
            inner_pbar.update(1)
        if torch_profiler:
            torch_profiler.step()
        batch_index += 1
        mini_batch_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        if local_rank == 0:
            tracking_duration.append(mini_batch_time)
        if (
            batch_index % cfg.log_every == 0
            and torch.distributed.get_rank() == 0
            and batch_index > 1
        ):
            print(
                f"step: {batch_index-1}: time taken for the last {cfg.log_every} steps is {mini_batch_time}"
            )

        if batch_index > total_steps_to_run:
            break
    if local_rank == 0:
        inner_pbar.close()
        print("tracking_duration", tracking_duration)
