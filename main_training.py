import argparse
import os
import time


import colorama
import torch

import torch
import torch.nn as nn
from colorama import Fore

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
)

import model_checkpointing

import torch.distributed as dist

import environment
from contextlib import contextmanager

bf16_ready = environment.verify_bfloat_support

from torch.utils.data import DistributedSampler
from torch.distributed.fsdp._common_utils import _is_fsdp_flattened

colorama.init(autoreset=True)  # reset after every line

import performance
import contextlib

_none_context = contextlib.nullcontext()

# add DDP support
from torch.nn.parallel import DistributedDataParallel as DDP


# import optimizers


@contextmanager
def init_empty_weights(include_buffers: bool = False):
    """
    A context manager under which models are initialized with all parameters on the meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.
    Args:
        include_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to also put all buffers on the meta device while initializing.
    Example:
    ```pyton
    import torch.nn as nn
    from accelerate import init_empty_weights
    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```
    <Tip warning={true}>
    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].
    </Tip>
    """
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(
                module._parameters[name].to(torch.device("meta")), **kwargs
            )

    def register_empty_buffer(module, name, buffer):
        old_register_buffer(module, name, buffer)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(torch.device("meta"))

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer


@torch.no_grad()
def my_init_fn(module: nn.Module):
    for submodule in module.modules():
        for param_name, param in submodule.named_parameters(recurse=False):
            if not _is_fsdp_flattened(param) and param.is_meta:
                materialized_param = nn.Parameter(
                    torch.empty_like(param, device=torch.device("cuda"))
                )
                # nn.init.uniform_(materialized_param)
                setattr(submodule, param_name, materialized_param)


def print_model(model, file_name, rank):
    if rank != 0:
        return

    fn = file_name
    with open(fn, "w") as external_file:
        print(f"model wrapping = \n{model}\n\n", file=external_file)

        external_file.close()


def print_memory_summary(prefix, device):
    rank = int(os.getenv("RANK"))
    if rank == 0:
        peak_memory_active = torch.cuda.memory_stats().get("active_bytes.all.peak", 0)
        print(
            f"{prefix}, GPU peak memory allocation: {torch.cuda.max_memory_allocated(device) // 1e9}GB, "
            f"GPU peak memory reserved: {torch.cuda.max_memory_reserved(device) // 1e9}GB, "
            f"GPU peak memory active: {peak_memory_active // 1e9}GB"
        )
        torch.cuda.reset_peak_memory_stats(device)


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
    dist.barrier()
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


def zero_print(rank, x):
    if rank == 0:
        print(x)


# ------ main code loop -----------------
def fsdp_main():
    """main process,  within each rank process"""

    cfg = config.train_config()  # loads from defaults

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.manual_seed(cfg.seed + local_rank)
    torch.manual_seed(cfg.seed + local_rank)

    if rank == 0:
        print(f"--> World Size = {world_size}\n")
        print(f"--> Device_count = {torch.cuda.device_count()}")
        print(f"--> running with these defaults {cfg}")
        # time_of_run = get_date_of_run()

    setup_tasks(rank, world_size, cfg)

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)

    from functools import partial

    _zero_print = partial(zero_print, local_rank)

    # setup memory tracking for perf
    if local_rank == 0:
        memmax = performance.Memory_Maximizer()
    else:
        memmax = None

    # ====   use new transformer wrapper

    my_auto_wrap_policy = config.get_policy()
    if rank == 0:
        print(f"wrapping policy is {my_auto_wrap_policy}")

    use_pokemon = False
    use_beans = False
    use_food = False
    use_label_singular = False
    # todo - clean this up...temp bridge for testing pokemon dataset
    if cfg.use_synthetic_data == False:
        use_pokemon = False
        use_beans = False
        use_food = False
    try:
        use_pokemon = cfg.use_pokemon_dataset
        use_beans = cfg.use_beans_dataset
        use_food = cfg.use_food
    except:
        print(f"pokemon nor beans set not enabled")
        pass

    val_dataset = None
    _stats = None
    if use_pokemon:
        dataset, val_dataset = config.get_pokemon_dataset()

    elif use_beans:
        assert not use_food and not use_pokemon, f"multiple datasets enabled."
        dataset, val_dataset = config.get_beans_dataset()
    elif use_food:
        assert not use_beans and not use_pokemon, f"multiple datasets enabled."
        dataset, val_dataset = config.get_universal_dataset()
        use_label_singular = True
    else:
        dataset = config.get_dataset()

    if not cfg.use_synthetic_data:
        if rank == 0:
            import collections

            _stats = collections.defaultdict(list)
            _stats["best_accuracy"] = 0.00

    # samplers ----

    train_sampler = DistributedSampler(
        dataset, rank=dist.get_rank(), num_replicas=dist.get_world_size(), shuffle=True
    )

    if cfg.run_validation:
        if not val_dataset:
            val_dataset = config.get_dataset()  # train=False)
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
        print("***** building the model  ******")
        use_deferred_init = False
        try:
            use_deferred_init = cfg.use_deferred_init
        except:
            pass

        with init_empty_weights() if cfg.use_deferred_init else _none_context:
            _zero_print(f"using deferred? {use_deferred_init}")
            use_parallel = False
            use_upper_fusion = False
            use_fused_attention = cfg.use_fused_attention
            use_mqa = False
            try:
                use_parallel = cfg.use_parallel_attention
                # use_upper_fusion = cfg.use_upper_fusion
                use_mqa = cfg.use_multi_query_attention
                print(f"**** Use MQA = {use_mqa}")
            except:
                # TODO - make this error appropriate per model ...print(f"failed to load pattn blocks params!")
                pass
            if use_parallel:
                model = config.build_model(
                    cfg.model_name,
                    use_parallel_attention=use_parallel,
                    # use_upper_fusion=use_upper_fusion,
                    use_fused_attention=use_fused_attention,
                    use_multi_query_attention=use_mqa,
                )
            else:
                model = config.build_model(
                    cfg.model_name,
                    use_parallel_attention=False,
                    use_fused_attention=use_fused_attention,
                )
        print_memory_summary("vit", "cuda")
        time.sleep(2)

        # TODO - we used to run HF checkpointing generically...adding this for now.
        if cfg.hf_t5_checkpointing:
            model.decoder.gradient_checkpointing = True
            model.encoder.gradient_checkpointing = True

    elif use_timm:
        # if you are here and this import fails - run:
        # git clone https://github.com/huggingface/pytorch-image-models.git
        # and then in the cloned main dir, run 'python setup.py develop'

        import timm
        import torch.nn as nn

        model = timm.create_model(
            cfg.model_name,
            # num_heads=cfg.model_num_heads,
            pretrained=False,
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
                ColwiseParallel,
                RowwiseParallel,
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
        # rank_print(rank, f"{model=}")

        # this is for parallelized vit - need to dynamically locate blocks
        # rank_print(rank, f"{model=}")

        # assert False, "remove"
        # tp parallelized block
        # col
        # in proj
        # row
        # mlp_out_proj
        # attn_out_proj

        blocks = model.get_submodule(f"blocks")
        total_blocks = len(blocks)
        # print(f"len block {total_blocks}")
        for i, block in enumerate(blocks):
            try:
                rank_print(rank, f"\nparallelization of block {i}")

                parallelized_block = parallelize_module(
                    module=block,
                    device_mesh=twod_mesh,
                    parallelize_plan={
                        "attn.qkv": ColwiseParallel(),
                        "attn.out_proj": RowwiseParallel(),
                        "mlp.linear1": ColwiseParallel(),
                        "mlp.linear2": RowwiseParallel(),
                    },
                    tp_mesh_dim=1,
                )
                # print(f"\nSuccess - {blocks[i]}\n")
                block = parallelized_block
                # rank_print(rank, f"{parallelized_block=}")

            except e:
                print(f"{e=}")
                assert False, f"failed to TP"
            # rank_print(rank, f"{blocks=}")
        # rank_print(rank, f"{model=}")
        """
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
        # model.to(device)
        # model = FSDP(model, process_group=fsdp_pg)

    process_group_fsdp = None

    if cfg.use_tp:
        fsdp_pg = twod_mesh.get_dim_groups()[0]
        process_group_fsdp = fsdp_pg

    # ----- main FSDP or DDP init -----------
    if cfg.use_ddp:
        model.to("cuda")
        model = DDP(
            model,
            device_ids=[local_rank],
            bucket_cap_mb=cfg.ddp_bucket_size,
            gradient_as_bucket_view=cfg.ddp_use_gradient_view,
        )

    else:
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
            param_init_fn=my_init_fn,
        )
    print_memory_summary("vit", "cuda")

    time.sleep(2)

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
    """config.train(
        model,
        data_loader,
        None,
        None,
        memmax,
        local_rank,
        tracking_duration,
        1,
        use_synthetic_data=cfg.use_synthetic_data,
    )
    if rank == 0:
        print("Finish warm up")
    model.zero_grad()
    """

    # optimizer ----------
    optimizer = None
    lr = 9e-4
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
    elif cfg.optimizer == "dadapt_adanip":
        from adanip_exp import DAdaptAdanIP

        optimizer = DAdaptAdanIP(  # DAdaptAdam(
            model.parameters(),
            lr=1.0,
            weight_decay=weight_decay,
            # amsgrad=False,
            # decouple=True,
            # log_every=4,
        )
        if rank == 0:
            print(f"Running with DAdapt AdanIP optimizer")

    elif cfg.optimizer == "dadapt_adam":
        from dadaptation import DAdaptAdam

        # optimizer = torch.optim.AdamW(
        optimizer = DAdaptAdanIP(  # DAdaptAdam(
            model.parameters(),
            lr=1.0,
            weight_decay=weight_decay,
            # amsgrad=False,
            # decouple=True,
            # log_every=4,
        )
        if rank == 0:
            print(f"Running with DAdapt optimizer")
    elif cfg.optimizer == "AdamW":
        use_fused_optimizer = cfg.use_fused_optimizer

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0005,
            weight_decay=weight_decay,
            fused=use_fused_optimizer,
        )
        if rank == 0:
            print(
                f"Running with AdamW optimizer, with fusion set to {use_fused_optimizer}"
            )

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # linear warmup
    from torch.optim.lr_scheduler import LinearLR

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=50)
    # (optimizer, start_factor=0.3333333333333333, end_factor=1.0, total_iters=5, last_epoch=- 1, verbose=False)

    # start adding in logged metrics...
    _metric_logger = None
    if cfg.run_validation:
        from metric_logging.metric_logger import get_date_time

        curr_time = get_date_time()
        file_description = "stats_" + curr_time + ".txt"
        _metric_logger = file_description

    # load optimizer checkpoint
    if cfg.load_optimizer:
        model_checkpointing.load_optimizer_checkpoint(model, optimizer, rank, cfg)

    torch_profiler = None
    total_steps = None
    if cfg.total_steps_to_run:
        total_steps = cfg.total_steps_to_run - 1  # fix off by one for step count

    @contextlib.contextmanager
    def maybe_run_profiler(cfg, *args, **kwargs):
        use_profiler: bool = cfg.run_profiler

        if use_profiler:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    cfg.profile_folder
                ),
                profile_memory=True,
                with_stack=False,
                record_shapes=True,
            ) as torch_profiler:
                yield torch_profiler
        else:
            torch_profiler = contextlib.nullcontext()
            yield None

    if cfg.run_profiler:
        print(f"Profiling active.  Traces will be saved at {cfg.profile_folder}")

    with maybe_run_profiler(cfg):
        for i in range(cfg.num_epochs):
            if rank == 0:
                print(f"Epoch: {i} starting...")
                if not cfg.use_synthetic_data:
                    assert _stats is not None, "missing stats in main"
            config.train(
                model,
                data_loader,
                torch_profiler,
                optimizer,
                memmax,
                local_rank,
                tracking_duration,
                total_steps,
                use_synthetic_data=cfg.use_synthetic_data,
                use_label_singular=use_label_singular,
                stats=_stats,
                lr_scheduler=warmup_scheduler,
            )
            if cfg.total_steps_to_run is not None:
                break

            if cfg.run_validation:
                if rank == 0:
                    assert _stats is not None, "no stats in main"
                with torch.no_grad():
                    config.validation(
                        model,
                        local_rank,
                        rank,
                        val_loader,
                        world_size,
                        stats=_stats,
                        use_label_singular=use_label_singular,
                        metric_logger=_metric_logger,
                    )
        # print(f"rank {local_rank} in front of barrier...")
        # dist.barrier()
        # print(f"rank {local_rank} past barrier...")
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
    print(f"** exit loop - rank {local_rank} reporting....")
    if local_rank == 0:
        # memory monitor
        memmax.stop()  # stop and display info
        # print(f"{tracking_duration=}, {cfg.total_steps_to_run=}")
        if _stats:
            total_loss_curve = _stats["loss"]
            total_acc_curve = _stats["accuracy"]
            training_loss_curve = _stats["training_loss"]

            if cfg.print_training_loss_data:
                print(f"Training loss data")
                for i, loss in enumerate(training_loss_curve):
                    print(f"{loss}")

            print(f"\nValidation loss data")
            for i, loss in enumerate(total_loss_curve):
                print(f"{loss}")

            print(f"\nAccuracy validation")
            for i, accuracy in enumerate(total_acc_curve):
                print(f"{accuracy}")

            # print(f"Training time average iter")
            total_training_iter_times = _stats["training_iter_time"]
            denom = len(total_training_iter_times)
            # total_times = sum(total_training_iter_times)
            # average_iter = round(total_times / denom, 5)
            # print(f"\nAverage iter = {average_iter}")

            best_val_acc = 0
            if total_acc_curve:
                best_val_acc = 100 * float(max(total_acc_curve))
            print(Fore.GREEN + f"\n--> Highest Val Accuracy =  {best_val_acc}\n")

            warmup_steps = cfg.warmup_steps
            iters_to_avg = total_training_iter_times[warmup_steps:]

            stable_sum = sum(iters_to_avg)
            # print(f"len iters_to_avg = {len(iters_to_avg)}")
            total_steps_measured = denom - warmup_steps
            stable_avg = stable_sum / total_steps_measured
            stable_avg = round(stable_avg, 4)
            print(
                Fore.GREEN
                + f"\n--> Step avg speed (in seconds) based on {total_steps_measured} steps: {stable_avg}\nexcluding {warmup_steps} steps as warmup"
            )

        if cfg.total_steps_to_run is not None:
            warmup_steps = cfg.warmup_steps
            iters_to_avg = tracking_duration[warmup_steps:]

            stable_sum = sum(iters_to_avg)
            # print(f"len iters_to_avg = {len(iters_to_avg)}")
            total_steps_measured = cfg.total_steps_to_run - warmup_steps
            stable_avg = stable_sum / total_steps_measured
            stable_avg = round(stable_avg, 4)
            print(
                Fore.GREEN
                + f"\n--> Step avg speed based on {total_steps_measured} steps: {stable_avg} seconds"
            )
        try:
            if cfg.use_deferred_init:
                print(
                    Fore.LIGHTBLUE_EX
                    + f"\n ==>> This run used deferred init! \nIf you are training and seeing no/poor training results, \n pls set this to False in the config file.**\n"
                )
        except:
            pass
        training_framework = "DDP" if cfg.use_ddp else "FSDP"
        print(Fore.GREEN + f"\nDist Training Framework used = {training_framework}\n")
        if cfg.use_ddp:
            print(
                f"DDP settings:  \nddp_bucket_size={cfg.ddp_bucket_size},\nddp_use_gradient_view={cfg.ddp_use_gradient_view}\n"
            )
        print(f"This was run with TensorParallel? = {cfg.use_tp}\n")
        try:
            print(f"Run with Parallel Attention? {cfg.use_parallel_attention}")
            print(f"Run with MQA? {cfg.use_multi_query_attention}\n")
        except:
            pass
        print(f"Batch size used = {cfg.batch_size_training}\n")
        if not cfg.use_ddp:
            print(
                f"FSDP Activation Checkpointing? = {cfg.fsdp_activation_checkpointing}"
            )
        if cfg.hf_t5_checkpointing:
            print(f"HF Activation Checkpointing? = {cfg.hf_t5_checkpointing}")

        print(Fore.LIGHTBLUE_EX + f"\n--> Model Size =  {num_params} M Params\n")
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
        choices=["deepvit", "t5", "regnet", "vitbase", "vitsmart"],
        help="choose model to run, available: `deepvit`, `t5`, `regnet`, `vitbase`, 'vitsmart' (default: vitbase)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"******* loading model {args.model=}")
    assert args.model in ["deepvit", "t5", "regnet", "vitbase", "vitsmart"]
    if args.model == "deepvit":
        import config.deepvit_config as config
    elif args.model == "t5":
        import config.t5_config as config
    elif args.model == "regnet":
        import config.regnet_config as config
    elif args.model == "vitbase":
        import config.vit_base_config as config
    elif args.model == "vitsmart":
        import config.vit_smart_config as config

    fsdp_main()
