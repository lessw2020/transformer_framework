from pathlib import Path
from datetime import datetime
import torch

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)

from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    save_state_dict,
    load_state_dict,
)


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


def save_checkpoint(
    model, rank, cfg, epoch=1, curr_metric=None, best_metric=None, verbose=True
):
    """saving model via rank0 cpu streaming or distributed checkpointing"""
    # if a metric is passed in, confirm if we want to checkpoint based on new low or not
    # and update best metric
    if curr_metric and best_metric:
        if curr_metric > best_metric:
            if verbose and rank == 0:
                print(
                    f"--> New record for curr_metric not established.  Not saving checkpoint "
                )
            return
        best_metric = curr_metric

    # confirm type of checkpoint and save
    if cfg.checkpoint_type == StateDictType.LOCAL_STATE_DICT:
        # create writer to current path
        save_dir = Path.cwd() / cfg.checkpoint_folder
        writer = FileSystemWriter(save_dir)

        with FSDP.state_dict_type(
            model,
            StateDictType.LOCAL_STATE_DICT,
        ):
            state_dict = model.state_dict()

        save_state_dict(state_dict, writer)
        if rank == 0:
            print(f"--> distributed checkpoint saved at {save_dir}")

        return  # we break to avoid hitting default full state

    # saving with rank0 cpu
    if not cfg.checkpoint_type == StateDictType.FULL_STATE_DICT:
        print(f" unable to handle checkpoint type {cfg.checkpoint_type}, aborting")

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state = model.state_dict()

    if verbose:
        print(f"saving process: rank {rank}  done w state_dict")

    if rank == 0:
        print(f"--> saving model ...")
        save_dir = Path.cwd() / cfg.checkpoint_folder
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = cfg.model_save_name + "-" + str(epoch) + ".pt"
        save_full_path = str(save_dir) + "/" + save_name
        torch.save(cpu_state, save_full_path)

        print(f"--> saved {save_full_path} to disk")


def load_checkpoint(model, rank, cfg, verbose=True):
    """load local checkpoint to rank0 cpu"""
    if cfg.checkpoint_type == StateDictType.LOCAL_STATE_DICT:
        checkdir = Path.cwd() / "dist_checkpoint"
        reader = FileSystemReader(checkdir)

        with FSDP.state_dict_type(
            model,
            StateDictType.LOCAL_STATE_DICT,
        ):
            state_dict = model.state_dict()
            load_state_dict(state_dict, reader)
            model.load_state_dict(state_dict)

        print(f"--> local state loaded on rank {rank}")
        return

    if rank != 0:
        return

    full_state_dict_model_path = (
        Path.cwd() / cfg.checkpoint_folder / cfg.checkpoint_model_filename
    )

    model_checkpoint = torch.load(full_state_dict_model_path)
    model.load_state_dict(model_checkpoint)
    if verbose:
        print(f"checkpoint loaded to rank0 cpu")
