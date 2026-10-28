"""DDP setup and synchronized failures for CPU tests and eight-GPU execution."""
import os
import torch
import torch.distributed as dist


def configure_execution():
    # Set before CUDA initialization, including non-DDP campaign workers.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup(config, cpu=False):
    configure_execution()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world != config["world_size"]:
        raise ValueError(f"Launch world size {world} != configured {config['world_size']}")
    device = torch.device("cpu" if cpu else f"cuda:{local}")
    if not cpu:
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("gloo" if cpu else "nccl")
    torch.manual_seed(config["seed"])
    torch.set_num_threads(1)
    return rank, world, device


def phase(action, device):
    result, failure = None, None
    try:
        result = action()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    if dist.is_initialized():
        errors = [None]*dist.get_world_size()
        dist.all_gather_object(errors, failure)
    else:
        errors = [failure]
    if any(errors):
        raise RuntimeError("AI distributed phase failed: " + repr(errors))
    return result


def close():
    if dist.is_initialized():
        dist.destroy_process_group()
