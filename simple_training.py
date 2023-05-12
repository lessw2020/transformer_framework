import argparse
import os
from threading import local
import time


import colorama
import torch


from colorama import Fore

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
)

import model_checkpointing

import torch.distributed as dist

import environment


colorama.init(autoreset=True)  # reset after every line

import performance
from config.simple_config import fsdp_simple_config
from config.simple_translator import start_fsdp


def print_model(model, file_name, rank):
    if rank != 0:
        return

    fn = file_name
    with open(fn, "w") as external_file:
        print(f"model wrapping = \n{model}\n\n", file=external_file)

        external_file.close()


def setup():
    """we use torchrun for init so no params needed here"""
    dist.init_process_group("nccl")


def setup_environ_flags(cfg, rank):
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    if cfg.nccl_debug_handler:
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    if cfg.distributed_debug:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        if rank == 0:
            print(f"--> running with torch dist debug set to detail")


def cleanup():
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    if rank == 0:
        print(f"clearing gpu cache for all ranks")
    torch.cuda.empty_cache()


def setup_tasks(rank, world_size, cfg):
    """keep the basic setup list here"""
    setup()
    clear_gpu_cache(rank)  # need to call torch set device first?
    # set_printing()
    setup_environ_flags(cfg, rank)


# ------ main code loop -----------------
def fsdp_main():
    """main process,  within each rank process"""

    cfg = config.train_config()  # loads from defaults

    cfg_fsdp = fsdp_simple_config()

    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"\nusing simple config = current settings are {cfg_fsdp}\n")
        print(f"--> World Size = {world_size}\n")
        print(f"--> Device_count = {torch.cuda.device_count()}")
        print(f"--> running with these defaults {cfg}")
        # time_of_run = get_date_of_run()

    setup_tasks(rank, world_size, cfg)

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)

    # setup memory tracking for perf
    if local_rank == 0:
        memmax = performance.Memory_Maximizer()
    else:
        memmax = None

    # ====   use new transformer wrapper

    my_auto_wrap_policy = config.get_policy()
    if rank == 0:
        print(f"policy is {my_auto_wrap_policy}")
    dataset = config.get_dataset()

    if local_rank == 0:
        print(f"\n--> Prepping {cfg.model_name} model ...\n")
    model = config.build_model(cfg.model_name)

    if local_rank == 0:
        print(f"--> {cfg.model_name} built.")
        num_params = (sum(p.numel() for p in model.parameters())) / 1e6
        print(f"built model with {num_params}M params")

    if local_rank == 0:
        init_start = time.perf_counter()

    # preload checkpoint if desired
    if (
        cfg.load_model_checkpoint
        and cfg.checkpoint_type == StateDictType.FULL_STATE_DICT
    ):
        model_checkpointing.load_model_checkpoint(model, rank, cfg)

    if rank == 0:
        print(f"backward prefetch set to {cfg_fsdp.backward_prefetch}")
        print(f"sharding set to {cfg_fsdp.sharding_strategy}")
        print(f"--> Batch Size = {cfg.batch_size_training}")

    # init FSDP
    model = start_fsdp(model, local_rank)

    # safety check...
    if model is None:
        print("Error: received none model from fsdp_start...")
        return

    # print sharding plan?
    if rank == 0 and cfg.print_sharding_plan:
        print(model)

    # postload checkpoint if desired
    if (
        cfg.load_model_checkpoint
        and cfg.checkpoint_type == StateDictType.LOCAL_STATE_DICT
    ):
        model_checkpointing.load_distributed_model_checkpoint(model, rank, cfg)

    if local_rank == 0:
        init_time = time.perf_counter() - init_start
        print(f"local rank {local_rank} init time = {init_time}")

    # data loader -------------
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size_training,
        num_workers=cfg.num_workers_dataloader,
        pin_memory=False,
    )

    # memory and timing tracking
    if local_rank == 0:
        memmax.start()
        # torch.cuda.reset_peak_memory_stats()
        tracking_duration = []
    else:
        tracking_duration = None

    # warmup, this is only used in the non-recursive ParamExecOrderPolicy
    config.train(
        model, data_loader, None, None, memmax, local_rank, tracking_duration, 1
    )
    if rank == 0:
        print("Finish warm up")
    model.zero_grad()

    # optimizer ----------

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0, amsgrad=True
    )
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    if rank == 0:
        print(f"==> optimizer = Adam\n")

    # load optimizer checkpoint
    if cfg.load_optimizer:
        model_checkpointing.load_optimizer_checkpoint(model, optimizer, rank, cfg)

    torch_profiler = None
    if cfg.run_profiler and rank == 0:
        print(f"Profiling active.  Traces will be saved at {cfg.profile_folder}")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.profile_folder),
            profile_memory=True,
            with_stack=False,
            record_shapes=True,
        ) as torch_profiler:
            config.train(
                model,
                data_loader,
                torch_profiler,
                optimizer,
                memmax,
                local_rank,
                tracking_duration,
                cfg.total_steps_to_run,
            )
    else:
        config.train(
            model,
            data_loader,
            None,
            optimizer,
            memmax,
            local_rank,
            tracking_duration,
            cfg.total_steps_to_run,
        )
        # checkpointing for model and optimizer
        if cfg.save_model_checkpoint:
            if cfg.checkpoint_type == StateDictType.FULL_STATE_DICT:
                model_checkpointing.save_model_checkpoint(
                    model, optimizer, rank, cfg, epoch=1
                )
            elif cfg.checkpoint_type == StateDictType.LOCAL_STATE_DICT:
                model_checkpointing.save_distributed_model_checkpoint(model, rank, cfg)

        if cfg.save_optimizer:
            model_checkpointing.save_optimizer_checkpoint(
                model, optimizer, rank, cfg, epoch=1
            )

    # memory summary
    if local_rank == 0:
        # memory monitor
        memmax.stop()  # stop and display info

        stable_sum = sum(tracking_duration[1:])
        stable_avg = stable_sum / cfg.total_steps_to_run
        stable_avg = round(stable_avg, 4)
        print(
            Fore.GREEN
            + f"\n--> Step avg speed based on {cfg.total_steps_to_run} steps: {stable_avg} seconds"
        )
        print(Fore.LIGHTBLUE_EX + f"\n--> Model Size =  {num_params} M Params")
        print(f"\nCUDA Memory Summary After Training:\n {torch.cuda.memory_summary()}")

    cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch experiments with FSDP")
    parser.add_argument(
        "--model",
        default="deepvit",
        metavar="string",
        choices=["deepvit", "t5", "regnet"],
        help="choose model to run, available: `deepvit`, `t5`, `regnet` (default: deepvit)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    assert args.model in ["deepvit", "t5", "regnet"]
    if args.model == "deepvit":
        import config.deepvit_config as config
    elif args.model == "t5":
        import config.t5_config as config
    elif args.model == "regnet":
        import config.regnet_config as config

    fsdp_main()
