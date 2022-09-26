import time
import tqdm
import torch
from dataclasses import dataclass

from torch.distributed.fsdp import StateDictType
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    T5Config,
)
from transformers.models.t5.modeling_t5 import T5Block
import datasets_grammar as dg
from .base_config import base_config, fsdp_checkpointing_base, get_policy_base


@dataclass
class train_config(base_config):

    # model
    model_name = "t5-base"
    # available models
    # t5-small / base / large  - 1.0 pretrained
    # or
    # google/t5-v1_1-small
    # google/t5-v1_1-base
    # google/t5-v1_1-large
    # google/t5-v1_1-xl  #2b
    # google/t5-v1_1-xxl #8b
    # t5-11b
    tokenizer = "t5-large"

    # important - if you want trackable loss stats, please ensure you use real data:
    use_real_data = True

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

    optimizer_checkpoint_file: str = "Adam-t5--1.pt"

    checkpoint_model_filename: str = "t5--1.pt"

    # datasets
    dataset_train = "datasets_grammar/grammar_train.csv"  # grammar_13k.csv
    dataset_test = "datasets_grammar/grammar_validation.csv"


def build_model(model_name: str):
    cfg = train_config()
    if cfg.use_real_data:
        return AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if model_name == "google/t5-v1_1-small":
        configure = T5Config(
            d_ff=1024,
            d_kv=64,
            d_model=512,
            decoder_start_token_id=0,
            dropout_rate=0.1,
            eos_token_id=1,
            initializer_factor=1.0,
            is_encoder_decoder=True,
            layer_norm_epsilon=1e-06,
            num_decoder_layers=8,
            num_heads=6,
            num_layers=8,
            pad_token_id=0,
            relative_attention_num_buckets=32,
            vocab_size=32128,
        )
    elif model_name == "google/t5-v1_1-base":
        configure = T5Config(
            d_ff=2048,
            d_kv=64,
            d_model=768,
            decoder_start_token_id=0,
            dropout_rate=0.1,
            eos_token_id=1,
            initializer_factor=1.0,
            is_encoder_decoder=True,
            layer_norm_epsilon=1e-06,
            num_decoder_layers=12,
            num_heads=12,
            num_layers=12,
            pad_token_id=0,
            relative_attention_num_buckets=32,
            vocab_size=32128,
        )
    elif model_name == "google/t5-v1_1-large":
        configure = T5Config(
            d_ff=2816,
            d_kv=64,
            d_model=1024,
            decoder_start_token_id=0,
            dropout_rate=0.1,
            eos_token_id=1,
            initializer_factor=1.0,
            is_encoder_decoder=True,
            layer_norm_epsilon=1e-06,
            num_decoder_layers=24,
            num_heads=16,
            num_layers=24,
            pad_token_id=0,
            relative_attention_num_buckets=32,
            vocab_size=32128,
        )
    elif model_name == "google/t5-v1_1-xl":
        configure = T5Config(
            d_ff=5120,
            d_kv=64,
            d_model=2048,
            decoder_start_token_id=0,
            dropout_rate=0.1,
            eos_token_id=1,
            initializer_factor=1.0,
            is_encoder_decoder=True,
            layer_norm_epsilon=1e-06,
            num_decoder_layers=24,
            num_heads=32,
            num_layers=24,
            pad_token_id=0,
            relative_attention_num_buckets=32,
            vocab_size=32128,
        )
    elif model_name == "google/t5-v1_1-xxl":
        configure = T5Config(
            d_ff=10240,
            d_kv=64,
            d_model=4096,
            decoder_start_token_id=0,
            dropout_rate=0.1,
            eos_token_id=1,
            initializer_factor=1.0,
            is_encoder_decoder=True,
            layer_norm_epsilon=1e-06,
            num_decoder_layers=24,
            num_heads=64,
            num_layers=24,
            pad_token_id=0,
            relative_attention_num_buckets=32,
            vocab_size=32128,
        )
    elif model_name == "t5-11b":
        configure = T5Config(
            d_ff=65536,
            d_kv=128,
            d_model=1024,
            decoder_start_token_id=0,
            dropout_rate=0.1,
            eos_token_id=1,
            initializer_factor=1.0,
            is_encoder_decoder=True,
            layer_norm_epsilon=1e-06,
            n_positions=512,
            num_heads=128,
            num_layers=24,
            pad_token_id=0,
            relative_attention_num_buckets=32,
            vocab_size=32128,
        )
    return T5ForConditionalGeneration(configure)


class GeneratedDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        super()
        self._input_shape = kwargs.get("input_shape", (1024,))
        self._input_type = kwargs.get("input_type", torch.long)
        self._len = kwargs.get("len", 1000000)
        self._num_classes = kwargs.get("num_classes", 32000)

    def __len__(self):
        return self._len

    def __getitem__(self, index: int):
        return {
            "source_ids": torch.randint(
                1, self._num_classes, self._input_shape, dtype=self._input_type
            ),
            "source_mask": torch.randint(
                1, self._num_classes, self._input_shape, dtype=self._input_type
            ),
            "target_ids": torch.randint(
                1, self._num_classes, self._input_shape, dtype=self._input_type
            ),
            "target_mask": torch.randint(
                1, self._num_classes, self._input_shape, dtype=self._input_type
            ),
        }


def get_dataset():
    cfg = train_config()
    if cfg.use_real_data:
        train_name = cfg.dataset_train
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer, model_max_length=512)
        train_dataset = dg.get_dataset(tokenizer, train_name, 512, 512, True)
        return train_dataset
    return GeneratedDataset()


def get_policy():
    return get_policy_base({T5Block})


def fsdp_checkpointing(model):
    return fsdp_checkpointing_base(model, T5Block)


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
        assert optimizer
        if optimizer:
            optimizer.step()
            if hasattr(model, '_averager'):
                print(" -- averaging --")
                model._averager.average_parameters(model.parameters())
            else:
                print(" --- NOT averaging --")

        if local_rank == 0:
            inner_pbar.update(1)
        if torch_profiler:
            torch_profiler.step()
        batch_index += 1
        mini_batch_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        if local_rank == 0:
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

        if batch_index > total_steps_to_run:
            break
    if local_rank == 0:
        inner_pbar.close()
        print("tracking_duration", tracking_duration)
