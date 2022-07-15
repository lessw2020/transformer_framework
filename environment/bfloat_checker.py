from pkg_resources import packaging
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist

# global flag that confirms ampere architecture, cuda version and
# nccl version to verify bfloat16 native support is ready

verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
)
