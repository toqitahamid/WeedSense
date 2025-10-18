"""
Training utilities for WeedSense

Includes functions for:
- Seed fixing for reproducibility
- CUDA/cuDNN setup
- Distributed training setup
- Model profiling
"""

import torch
import numpy as np
import random
import time
import os
from pathlib import Path
from torch.backends import cudnn
from torch import nn, distributed as dist
from typing import Union


def fix_seeds(seed: int = 3407, deterministic: bool = False) -> None:
    """
    Fix random seeds for reproducibility
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic operations (slower but reproducible)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    setup_cudnn(deterministic)


def setup_cudnn(deterministic: bool = False) -> None:
    """
    Setup cuDNN backend
    
    Args:
        deterministic: If True, use deterministic operations (slower but reproducible)
    """
    cudnn.benchmark = not deterministic
    cudnn.deterministic = deterministic


def time_sync() -> float:
    """Synchronized time measurement across CUDA devices"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def get_model_size(model: Union[nn.Module, torch.jit.ScriptModule]) -> float:
    """
    Get model size in MB
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size in MB
    """
    tmp_model_path = Path('temp.p')
    if isinstance(model, torch.jit.ScriptModule):
        torch.jit.save(model, tmp_model_path)
    else:
        torch.save(model.state_dict(), tmp_model_path)
    size = tmp_model_path.stat().st_size
    os.remove(tmp_model_path)
    return size / 1e6


def count_parameters(model: nn.Module) -> float:
    """
    Count trainable parameters in millions
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of parameters in millions
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def setup_ddp() -> int:
    """
    Setup Distributed Data Parallel training
    
    Returns:
        GPU rank/id
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)
        dist.init_process_group('nccl', init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
    else:
        gpu = 0
    return gpu


def cleanup_ddp():
    """Cleanup Distributed Data Parallel"""
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reduce tensor across all processes in DDP
    
    Args:
        tensor: Input tensor
    
    Returns:
        Averaged tensor
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


@torch.no_grad()
def measure_throughput(dataloader, model: nn.Module, times: int = 30, device: str = 'cuda'):
    """
    Measure model throughput (images/second)
    
    Args:
        dataloader: Data loader
        model: PyTorch model
        times: Number of iterations to average
        device: Device to use
    """
    model.eval()
    images, *_ = next(iter(dataloader))
    images = images.to(device)
    B = images.shape[0]
    
    print(f"Measuring throughput over {times} iterations...")
    start = time_sync()
    for _ in range(times):
        model(images)
    end = time_sync()
    
    throughput = times * B / (end - start)
    print(f"Batch Size: {B}, Throughput: {throughput:.2f} images/s")
    return throughput

