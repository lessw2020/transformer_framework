from pathlib import Path
from datetime import datetime 
import torch

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig, # general model non-sharded, non-flattened params
    LocalStateDictConfig, # flattened params, usable only by FSDP
    #ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
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

def save_checkpoint(model, rank, cfg, epoch=1, verbose=True):
    """ saving model via rank0 cpu streaming"""
    #nonlocal fullstate_save_policy
    #fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    if cfg.checkpoint_type == StateDictType.LOCAL_STATE_DICT:
        # create writer to current path
        save_dir = Path.cwd()/'dist_checkpoint'
        writer = FileSystemWriter(save_dir)
            
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT, ):
            state_dict = model.state_dict()

        save_state_dict(state_dict, writer)
        
        return  # we break to avoid hitting default full state

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state = model.state_dict()

    if verbose:
        print(f"saving process: rank {rank}  done w state_dict")

    if rank == 0:
        print(f"--> saving model ...")
        
        save_name = "test" + "-" + ".pt"

        torch.save(cpu_state, save_name)

        print(f"--> saved {save_name} to disk")

                
def load_checkpoint(model, rank, cfg, verbose=True):
    """load local checkpoint to rank0 cpu"""
    if cfg.checkpoint_type==StateDictType.LOCAL_STATE_DICT:
        checkdir = Path.cwd()/'dist_checkpoint'
        reader = FileSystemReader(checkdir)

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT,) :
            state_dict = model.state_dict()
            load_state_dict(state_dict, reader)
            model.load_state_dict(state_dict)

        print(f"--> local state loaded on rank {rank}")
    return

    if rank !=0:
        return
    
    full_state_dict = torch.load(cfg.checkpoint_name)
    model.load_state_dict(full_state_dict)
    if verbose:
        print(f"checkpoint loaded to rank0 cpu")
