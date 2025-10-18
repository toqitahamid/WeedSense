"""
Loss functions for WeedSense multi-task learning

Includes:
- CrossEntropy with auxiliary head support
- MultiTaskLoss for joint training
"""

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class CrossEntropy(nn.Module):
    """
    Cross Entropy Loss with auxiliary head support
    
    Args:
        ignore_label: Label to ignore in loss calculation
        weight: Class weights
        aux_weights: Weights for auxiliary heads [main, aux1, aux2, ...]
    """
    
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, aux_weights: list = [1, 0.4, 0.4, 0.4, 0.4]) -> None:
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        return self.criterion(preds, labels)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for WeedSense
    
    Combines:
    - Segmentation loss (with auxiliary heads)
    - Height regression loss (MSE)
    - Growth stage classification loss (CrossEntropy)
    
    Args:
        ignore_label: Label to ignore in segmentation
        seg_weight: Weight for segmentation loss
        height_weight: Weight for height loss
        week_weight: Weight for week classification loss
        aux_weights: Weights for auxiliary segmentation heads
    """
    
    def __init__(
        self, 
        ignore_label: int = 255,
        seg_weight: float = 1.0,
        height_weight: float = 1.0,
        week_weight: float = 1.0,
        aux_weights: list = None
    ):
        super().__init__()
        self.seg_criterion = CrossEntropy(
            ignore_label=ignore_label,
            aux_weights=aux_weights if aux_weights is not None else [1]
        )
        self.height_criterion = nn.MSELoss()
        self.week_criterion = nn.CrossEntropyLoss()
        self.seg_weight = seg_weight
        self.height_weight = height_weight
        self.week_weight = week_weight

    def forward(self, preds, targets):
        """
        Forward pass
        
        Args:
            preds: Tuple of (seg_preds, height_pred, week_pred)
            targets: Tuple of (seg_target, height_target, week_target)
        
        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary with individual loss values
        """
        seg_preds, height_pred, week_pred = preds
        seg_target, height_target, week_target = targets
        
        # Segmentation loss (handles both tuple and single tensor)
        seg_loss = self.seg_criterion(seg_preds, seg_target)
        
        # Height regression loss
        height_loss = self.height_criterion(height_pred.squeeze(), height_target)
        
        # Week classification loss
        week_loss = self.week_criterion(week_pred, week_target)
        
        # Combined loss
        total_loss = (
            self.seg_weight * seg_loss +
            self.height_weight * height_loss +
            self.week_weight * week_loss
        )
        
        return total_loss, {
            'seg_loss': seg_loss.item(),
            'height_loss': height_loss.item(),
            'week_loss': week_loss.item()
        }


def get_loss(
    loss_name: str = 'MultiTaskLoss',
    ignore_label: int = 255,
    seg_weight: float = 1.0,
    height_weight: float = 1.0,
    week_weight: float = 1.0,
    aux_weights: list = None
):
    """
    Factory function to get loss
    
    Args:
        loss_name: Name of loss function
        ignore_label: Label to ignore
        seg_weight: Segmentation loss weight
        height_weight: Height loss weight  
        week_weight: Week classification loss weight
        aux_weights: Auxiliary head weights
    
    Returns:
        Loss function
    """
    if loss_name == 'MultiTaskLoss':
        return MultiTaskLoss(
            ignore_label=ignore_label,
            seg_weight=seg_weight,
            height_weight=height_weight,
            week_weight=week_weight,
            aux_weights=aux_weights
        )
    elif loss_name == 'CrossEntropy':
        return CrossEntropy(ignore_label=ignore_label, aux_weights=aux_weights)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

