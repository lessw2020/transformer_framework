import os
import time
from distutils.version import LooseVersion

import colorama
import torch

# bfloat16 support verification imports (network and gpu native support)
import torch.cuda.nccl as nccl
import torch.distributed as dist
from colorama import Fore

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
)

import model_checkpointing

bf16_ready = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and LooseVersion(torch.version.cuda) >= "11.0"
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
)

colorama.init(autoreset=True)  # reset after every line

# some globals
g_gigabyte_unit_size = 1024**3

import performance


def print_model(model, file_name, local_rank):
    if local_rank != 0:
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
    print(f"clearing cache for rank {rank}")
    torch.cuda.empty_cache()


def setup_tasks(rank, world_size, cfg):
    """keep the basic setup list here"""
    setup()
    clear_gpu_cache(rank)  # need to call torch set device first?
    # set_printing()
    setup_environ_flags(cfg, rank)


def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / g_gigabyte_unit_size
    metric_num = round(metric_num, ndigits=4)
    return metric_num


# ------ main code loop -----------------
def fsdp_main():
    """main process,  within each rank process"""

    cfg = config.train_config()  # loads from defaults

    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"--> World Size = {world_size}\n")
        print(f"--> running with these defaults {cfg}")
        # time_of_run = get_date_of_run()

    setup_tasks(rank, world_size, cfg)

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)

    # setup memory tracking for perf
    if rank == 0:
        memmax = performance.Memory_Maximizer()

    # ====   use new transformer wrapper

    my_auto_wrap_policy = config.get_policy()
    dataset = config.GeneratedDataset()

    log_every = cfg.log_every
    model = config.build_model(cfg.model_name)

    if local_rank == 0:
        num_params = (sum(p.numel() for p in model.parameters())) / 1e6
        print(f"built model with {num_params}M params")

    #   Setup Mixed Precision --------------
    # === leverage FSDP Mixed Precision
    bfSixteen = MixedPrecision(
        # Param precision
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )

    mp_policy = None

    if bf16_ready:
        mp_policy = bfSixteen  # set to None to run with fp32
        if local_rank == 0:
            print(f"--> Running with bfloat16 mixed precision")
        else:
            if local_rank == 0:
                print(f"--> Warning - bf16 support not available.  Reverting to fp32")

    log_every = cfg.log_every

    if local_rank == 0:
        init_start = time.perf_counter()

    # preload checkpoint if desired
    if (
        cfg.load_model_checkpoint
        and cfg.checkpoint_type == StateDictType.FULL_STATE_DICT
    ):
        model_checkpointing.load_model_checkpoint(model, rank, cfg)

    prefetch_policy = cfg.backward_prefetch
    print(f"backward prefetch set to {prefetch_policy}")

    # ----- main FSDP init -----------
    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        mixed_precision=mp_policy,
        backward_prefetch=prefetch_policy,
        device_id=torch.cuda.current_device(),
        sharding_strategy=cfg.sharding_strategy,
        forward_prefetch=True,
    )

    """if cfg.fsdp_activation_checkpointing:
        policies.fsdp_checkpointing(model)l
        if local_rank==0:
            print(f"--> FSDP activation checkpointing in use")
    """
    # postload checkpoint if desired
    if (
        cfg.load_model_checkpoint
        and cfg.checkpoint_type == StateDictType.LOCAL_STATE_DICT
    ):
        model_checkpointing.load_model_checkpoint(model, rank, cfg)

    if local_rank == 0:
        init_time = time.perf_counter() - init_start
        print(f"local rank {local_rank} init time = {init_time}")

    # optimizer ----------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0, amsgrad=True
    )
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    if local_rank == 0:
        print(f"==> optimizer = AdamW\n")

    # load optimizer checkpoint
    if cfg.load_optimizer:
        model_checkpointing.load_optimizer_checkpoint(model, optimizer, rank, cfg)

    # data loader -------------

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size_training, num_workers=8, pin_memory=False
    )
    loss_function = torch.nn.CrossEntropyLoss()

    # memory and timing tracking
    if local_rank == 0:
        memmax.start()

        # torch.cuda.reset_peak_memory_stats()
        tracking_duration = []

    torch_profiler = None
    if cfg.run_profiler:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1, warmup=2, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                "fsdp_a100/profile_traces"
            ),
            profile_memory=True,
            with_stack=False,
            record_shapes=True,
        ) as torch_profiler:
            for batch_index, (inputs, target) in enumerate(data_loader, start=1):
                inputs, targets = inputs.to(torch.cuda.current_device()), torch.squeeze(
                    target.to(torch.cuda.current_device()), -1
                )
                optimizer.zero_grad()
                # with torch.cuda.amp.autocast(mixed_precision):
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                torch_profiler.step()

                if batch_index > cfg.total_steps_to_run:
                    break

    else:
        t0 = time.perf_counter()
        for batch_index, (inputs, target) in enumerate(data_loader, start=1):
            inputs, targets = inputs.to(torch.cuda.current_device()), torch.squeeze(
                target.to(torch.cuda.current_device()), -1
            )
            optimizer.zero_grad()
            # with torch.cuda.amp.autocast(mixed_precision):
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            # update durations and memory tracking
            if local_rank == 0:
                mini_batch_time = time.perf_counter() - t0
                tracking_duration.append(mini_batch_time)

                memmax.update()

            if (
                batch_index % log_every == 0
                and torch.distributed.get_rank() == 0
                and batch_index > 1
            ):
                print(
                    f"step: {batch_index-1}: time taken for the last {log_every} steps is {mini_batch_time}"
                )

            # reset timer
            t0 = time.perf_counter()
            if batch_index > cfg.total_steps_to_run:
                break

        # checkpointing for model and optimizer
        if (
            cfg.save_model_checkpoint
            and cfg.checkpoint_type == StateDictType.FULL_STATE_DICT
        ):
            model_checkpointing.save_model_checkpoint(
                model, optimizer, rank, cfg, epoch=1
            )

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
            # print(Fore.YELLOW + f"\nSagemaker Time to Beat: 8 seconds")
            # gain = (8.00 - stable_avg) / stable_avg
            # gain = round(gain, 4)
            # print(
            #    Fore.LIGHTGREEN_EX + f"Net FSDP Speed Gain over SageMaker: {gain*100}%"
            # )
            print(Fore.LIGHTBLUE_EX + f"\n--> Model Size =  {num_params} M Params")

            print(
                f"\nCUDA Memory Summary After Training:\n {torch.cuda.memory_summary()}"
            )

    cleanup()


if __name__ == "__main__":
    import config.deepvit_config as config
    fsdp_main()
