# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor, mesh_resources, Replicate, sharding_prop
from torch.distributed._tensor.device_mesh import init_device_mesh
import os


def create_device_mesh(
    replica_groups_size=None,
    sharding_group_size=None,
    device=None,
):
    """wrapper function for creating a device mesh - partially for educational purposes, you can simply
    use *init_device_mesh* api directly if desired.

    Provides some automation:
    1 - If no sharding group size, will default to gpus per node (same as HSDP default)
    2 - If no replica group size, will auto-create replica group size based on sharding group size and available world GPUs.

    Usage:
    1 - generate device mesh
    If your model fits on 4 GPUS, and you have 3 nodes of 8 GPUs, then:
        Sharding_Group_Size = 4
        Replica_Groups_Size = (24 total gpus, 4 per sharding group) = 6 Replica Groups

    2 - pass into FSDP init via:
    sharded_model = FSDP(model, device_mesh = device_mesh ... )
    Note that this requires FSDP.ShardingStrategy to be in [HYBRID_SHARD, _HYBRID_SHARD_ZERO2]


    """

    _rank = int(os.environ["RANK"])
    _local_rank = int(os.environ["LOCAL_RANK"])
    _world_size = int(os.environ["WORLD_SIZE"])
    _local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    if device is None:
        device = f"cuda"  # :{_local_rank}"

    if sharding_group_size is None:
        sharding_group_size = _local_world_size

    if replica_groups_size is None:
        assert (
            _world_size % sharding_group_size == 0
        ), f"unable to evenly shard {_world_size=} by {sharding_group_size}"
        replica_groups_size = _world_size // sharding_group_size

    device_mesh = None
    device_mesh = init_device_mesh(device, (replica_groups_size, sharding_group_size))
    assert device_mesh is not None, "unable to create valid device mesh"

    return device_mesh
