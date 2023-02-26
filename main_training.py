import argparse
import os
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

bf16_ready = environment.verify_bfloat_support

from torch.utils.data import DistributedSampler


colorama.init(autoreset=True)  # reset after every line

import performance

# import optimizers


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


# wrapper to avoid cluttering with if rank==0...
def rank_print(rank, x):
    if rank == 0:
        print(x)


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

    # todo - clean this up...temp bridge for testing pokemon dataset
    if cfg.use_synthetic_data == False:
        use_pokemon = False
        use_beans = False
    try:
        use_pokemon = cfg.use_pokemon_dataset
        use_beans = cfg.use_beans_dataset
    except:
        print(f"pokemon nor beans set not enabled")
        pass

    val_dataset = None
    _stats = None
    if use_pokemon:
        dataset, val_dataset = config.get_pokemon_dataset()

    elif use_beans:
        dataset, val_dataset = config.get_beans_dataset()
    else:
        dataset = config.get_dataset()

    if use_beans or use_pokemon:
        if rank == 0:
            import collections

            _stats = collections.defaultdict(list)

    # samplers ----

    train_sampler = DistributedSampler(
        dataset, rank=dist.get_rank(), num_replicas=dist.get_world_size(), shuffle=True
    )

    if cfg.run_validation:
        if not val_dataset:
            val_dataset = config.get_dataset(train=False)
        val_sampler = DistributedSampler(
            val_dataset, rank=dist.get_rank(), num_replicas=dist.get_world_size()
        )

    if local_rank == 0:
        print(f"\n--> Prepping {cfg.model_name} model ...\n")
        print(f"stats is ready....? {_stats=}, {local_rank=}, {rank=}")

    # ---  build model
    use_timm = False
    try:
        use_timm = cfg.use_timm
    except:
        pass  # means older config w/o timm support flag

    if not use_timm:
        model = config.build_model(cfg.model_name)

    elif use_timm:
        import timm
        import torch.nn as nn

        model = timm.create_model(
            cfg.timm_model_name,
            act_layer=nn.GELU,
            qk_norm=True,
            num_classes=cfg.num_categories,
        )

    if local_rank == 0:
        print(f"--> {cfg.model_name} built.")
        num_params = (sum(p.numel() for p in model.parameters())) / 1e6
        print(f"built model with {num_params}M params")

    mp_policy = None

    if cfg.use_mixed_precision and bf16_ready:
        mp_policy = cfg.mp_policy

        if rank == 0:
            print(f"bf16 check passed")
            print(f"\n--> Running with mixed precision {cfg.mp_policy} policy")

    else:
        if rank == 0:
            print(f"--> Warning - bf16 support not available.  Using fp32")

    # if not using mixed precision, turn on TF32 for matmul?
    if not cfg.use_mixed_precision and cfg.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        if rank == 0:
            print(f"--> TF32 support for matmul enabled. ")

    if local_rank == 0:
        init_start = time.perf_counter()

    # preload checkpoint if desired
    if cfg.load_model_checkpoint:
        if cfg.checkpoint_type == StateDictType.FULL_STATE_DICT:
            model_checkpointing.load_model_checkpoint(model, rank, cfg)

        elif cfg.checkpoint_type == StateDictType.LOCAL_STATE_DICT:
            model_checkpointing.load_distributed_model_checkpoint(model, rank, cfg)

    prefetch_policy = cfg.backward_prefetch
    if rank == 0:
        print(f"backward prefetch set to {prefetch_policy}")
        print(f"sharding set to {cfg.sharding_strategy}")
        print(f"--> Batch Size = {cfg.batch_size_training}")

    # model weights to BF16?
    if cfg.model_weights_bf16:
        model = model.to(torch.bfloat16)
        mp_policy = None
        if rank == 0:
            print(f"--> Model converted to BF16.\nRunning in ** PURE ** BFloat mode")

    # ----- Add 2D Tensor Parallel if activated (in config)
    if cfg.use_tp:
        print(f"Tensor Parallel activated - init start\n")

        from torch.distributed.tensor.parallel.fsdp import enable_2d_with_fsdp

        TP_AVAILABLE = False
        try:
            from torch.distributed._tensor import (
                DeviceMesh,
            )
            from torch.distributed.tensor.parallel import (
                PairwiseParallel,
                parallelize_module,
                # get_parallelization_fqn,
            )

            # need to setup hooks for TP

            fsdp_is_available = enable_2d_with_fsdp()

            TP_AVAILABLE = fsdp_is_available

        except BaseException as e:
            print(f"Exception during TP init - {e=}\n")
            pass

        assert TP_AVAILABLE, f"fsdp did not init"
        print(f"tp_initialized - rank {rank}\n")

        # Init TP
        _tp = int(os.environ.get("RANK", -1)) != -1  # verify distributed run

        assert (
            _tp and TP_AVAILABLE
        ), "this config assumes setup for Tensor Parallel - distributed not ready here."

        # rank_print(f"TP is available = {TP_AVAILABLE}\n")
        model_parallel_size = 2

        # 2-D mesh is [dp, tp]
        twod_mesh = DeviceMesh(
            device_type="cuda",
            mesh=torch.arange(0, world_size).view(model_parallel_size, -1),
        )
        rank_print(rank, f"{twod_mesh=}")

        for i in range(12):
            block = model.get_submodule(f"encoder.block_{i}")
            parallelized_block = parallelize_module(
                module=block,
                device_mesh=twod_mesh,
                parallelize_plan={
                    "self_attention": PairwiseParallel(),
                    "mlp_block": PairwiseParallel(),
                },
                tp_mesh_dim=1,
            )
            block = parallelized_block
        """
        if rank == 0:
            print(f"&&&&&&&&&&&\n {model=}")
        model = parallelize_module(
            model,
            twod_mesh,
            {"self_attention": PairwiseParallel(), "mlp_block": PairwiseParallel()},
            tp_mesh_dim=1,
        )

        """
        # print(f"{tp_model=}")

        fsdp_pg = twod_mesh.get_dim_groups()[0]

        # todo - add back main code later for resume
        device = "cuda"
        model.to(device)
        model = FSDP(model, process_group=fsdp_pg)

    process_group_fsdp = None

    if cfg.use_tp:
        fsdp_pg = twod_mesh.get_dim_groups()[0]
        process_group_fsdp = fsdp_pg
    # ----- main FSDP init -----------
    model = FSDP(
        model,
        process_group=process_group_fsdp,
        auto_wrap_policy=my_auto_wrap_policy,
        mixed_precision=mp_policy,
        backward_prefetch=prefetch_policy,
        sharding_strategy=cfg.sharding_strategy,
        device_id=torch.cuda.current_device(),
        forward_prefetch=cfg.forward_prefetch,
        limit_all_gathers=False,
    )

    if (
        cfg.load_model_checkpoint
        and cfg.checkpoint_type == StateDictType.SHARDED_STATE_DICT
    ):
        model_checkpointing.load_model_sharded(model, rank, cfg)

    if cfg.fsdp_activation_checkpointing:
        config.fsdp_checkpointing(model)
        if rank == 0:
            print(f"--> FSDP activation checkpointing in use")

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
        sampler=train_sampler,
    )

    if cfg.run_validation:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.val_batch_size,
            num_workers=cfg.num_workers_dataloader,
            pin_memory=False,
            sampler=val_sampler,
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
    optimizer = None
    lr = 8e-4
    weight_decay = 0.002

    if cfg.optimizer == "int8":
        import bitsandbytes as bnb

        optimizer = bnb.optim.Adam8bit(
            model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False
        )
        if rank == 0:
            print(f"Running with 8 bit optimizer")

    elif cfg.optimizer == "AnyPrecision":
        import optimizers

        optimizer = optimizers.AnyPrecisionAdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum_dtype=cfg.ap_momentum_dtype,
            variance_dtype=cfg.ap_variance_dtype,
            use_kahan_summation=cfg.ap_use_kahan_summation,
        )
        if rank == 0:
            print(
                f"Running with AnyPrecision Optimizer, momo={cfg.ap_momentum_dtype}, var = {cfg.ap_variance_dtype}, kahan summation =  {cfg.ap_use_kahan_summation}"
            )

    else:
        from dadaptation import DAdaptAdam

        # optimizer = torch.optim.AdamW(
        optimizer = DAdaptAdam(
            model.parameters(),
            lr=1.0,
            weight_decay=weight_decay,
            amsgrad=False,
            decouple=True,
            log_every=4,
        )
        if rank == 0:
            print(f"Running with DAdapt optimizer")

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

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
        for i in range(1, cfg.num_epochs + 1):
            if rank == 0:
                print(f"Epoch: {i} starting...")
                assert _stats is not None, "missing stats in main"
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
            if cfg.total_steps_to_run is not None:
                break

            if cfg.run_validation:
                if rank == 0:
                    assert _stats is not None, "no stats in main"
                config.validation(
                    model, local_rank, rank, val_loader, world_size, stats=_stats
                )

        # checkpointing for model and optimizer
        if cfg.save_model_checkpoint:
            if cfg.checkpoint_type == StateDictType.FULL_STATE_DICT:
                model_checkpointing.save_model_checkpoint(
                    model, optimizer, rank, cfg, epoch=1
                )
            elif cfg.checkpoint_type == StateDictType.LOCAL_STATE_DICT:
                print(f"Saving Model via Distributed Checkpoint")
                model_checkpointing.save_distributed_model_checkpoint(model, rank, cfg)

            elif cfg.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                model_checkpointing.save_model_sharded(model, rank, cfg)

        if cfg.save_optimizer:
            model_checkpointing.save_optimizer_checkpoint(
                model, optimizer, rank, cfg, epoch=1
            )

    # memory summary
    if local_rank == 0:
        # memory monitor
        memmax.stop()  # stop and display info
        # print(f"{tracking_duration=}, {cfg.total_steps_to_run=}")
        if _stats:
            total_loss_curve = _stats["loss"]
            total_acc_curve = _stats["accuracy"]
            for loss, acc in zip(total_loss_curve, total_acc_curve):
                print(f"{loss=}, {acc=}")

        stable_sum = sum(
            tracking_duration[3:]
        )  # this is b/c of 2 warmup steps, plus remove first actual step
        if cfg.total_steps_to_run is not None:
            stable_avg = stable_sum / cfg.total_steps_to_run
            stable_avg = round(stable_avg, 4)
            print(
                Fore.GREEN
                + f"\n--> Step avg speed based on {cfg.total_steps_to_run} steps: {stable_avg} seconds"
            )
        print(Fore.LIGHTBLUE_EX + f"\n--> Model Size =  {num_params} M Params")
        if cfg.print_memory_summary:
            print(
                f"\nCUDA Memory Summary After Training:\n {torch.cuda.memory_summary()}"
            )

    cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch experiments with FSDP")
    parser.add_argument(
        "--model",
        default="deepvit",
        metavar="string",
        choices=["deepvit", "t5", "regnet", "vit"],
        help="choose model to run, available: `deepvit`, `t5`, `regnet`, `vit` (default: deepvit)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"******* loading model {args.model=}")
    assert args.model in ["deepvit", "t5", "regnet", "vit"]
    if args.model == "deepvit":
        import config.deepvit_config as config
    elif args.model == "t5":
        import config.t5_config as config
    elif args.model == "regnet":
        import config.regnet_config as config
    elif args.model == "vit":
        import config.vit_config as config

    fsdp_main()
