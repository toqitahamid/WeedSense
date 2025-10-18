"""
Optimizers for WeedSense

Provides optimizer factory with weight decay handling
"""

from torch import nn
from torch.optim import AdamW, SGD


def get_optimizer(model: nn.Module, optimizer: str, lr: float, weight_decay: float = 0.01):
    """
    Get optimizer with proper weight decay handling
    
    Args:
        model: PyTorch model
        optimizer: Optimizer name ('adamw' or 'sgd')
        lr: Learning rate
        weight_decay: Weight decay factor
    
    Returns:
        Optimizer instance
    """
    # Separate parameters with and without weight decay
    wd_params, nwd_params = [], []
    for p in model.parameters():
        if p.dim() == 1:  # Bias and normalization parameters
            nwd_params.append(p)
        else:  # Weight parameters
            wd_params.append(p)
    
    params = [
        {"params": wd_params},
        {"params": nwd_params, "weight_decay": 0}
    ]
    
    if optimizer == 'adamw':
        return AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    else:
        return SGD(params, lr, momentum=0.9, weight_decay=weight_decay)

