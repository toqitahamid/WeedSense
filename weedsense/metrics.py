"""
Evaluation metrics for WeedSense

Includes metrics for:
- Semantic segmentation (IoU, F1, Accuracy)
- Height regression (MAE, RMSE, R²)
- Growth stage classification (Accuracy, F1, Precision, Recall)
"""

import torch
from torch import Tensor
import numpy as np
from typing import Tuple
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class SegmentationMetrics:
    """Metrics for semantic segmentation"""
    
    def __init__(self, num_classes: int, ignore_label: int, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).to(device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Update metrics with batch predictions"""
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(
            target[keep] * self.num_classes + pred[keep],
            minlength=self.num_classes**2
        ).view(self.num_classes, self.num_classes)

    def compute_iou(self) -> Tuple[list, float]:
        """Compute IoU per class and mean IoU"""
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[list, float]:
        """Compute F1 per class and mean F1"""
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[list, float]:
        """Compute pixel accuracy per class and mean accuracy"""
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)

    def compute_all(self):
        """Compute all segmentation metrics"""
        ious, miou = self.compute_iou()
        acc, macc = self.compute_pixel_acc()
        f1, mf1 = self.compute_f1()
        
        return {
            'per_class_iou': ious,
            'mean_iou': miou,
            'per_class_acc': acc,
            'mean_acc': macc,
            'per_class_f1': f1,
            'mean_f1': mf1
        }


class HeightMetrics:
    """Metrics for height regression"""
    
    def __init__(self):
        self.predictions = []
        self.targets = []
    
    def update(self, preds, targets):
        """Update with batch predictions"""
        self.predictions.extend(preds)
        self.targets.extend(targets)
    
    def compute(self):
        """Compute all height metrics"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        errors = predictions - targets
        abs_errors = np.abs(errors)
        
        return {
            'mae': mean_absolute_error(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'r2': r2_score(targets, predictions),
            'max_error': np.max(abs_errors),
            'error_counts': {
                'within_1cm': np.mean(abs_errors <= 1.0) * 100,
                'within_2cm': np.mean(abs_errors <= 2.0) * 100,
                'within_5cm': np.mean(abs_errors <= 5.0) * 100
            },
            'error_distribution': {
                'mean': np.mean(errors),
                'std': np.std(errors),
                'median': np.median(errors),
                'percentiles': {
                    '25th': np.percentile(errors, 25),
                    '75th': np.percentile(errors, 75),
                    '95th': np.percentile(errors, 95)
                }
            }
        }


class WeekMetrics:
    """Metrics for growth stage classification"""
    
    def __init__(self):
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, preds, targets, probs=None):
        """Update with batch predictions"""
        self.predictions.extend(preds)
        self.targets.extend(targets)
        if probs is not None:
            self.probabilities.extend(probs)
    
    def compute(self):
        """Compute all classification metrics"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        return {
            'accuracy': accuracy_score(targets, predictions),
            'macro_f1': f1_score(targets, predictions, average='macro', zero_division=0),
            'weighted_f1': f1_score(targets, predictions, average='weighted', zero_division=0),
            'macro_precision': precision_score(targets, predictions, average='macro', zero_division=0),
            'macro_recall': recall_score(targets, predictions, average='macro', zero_division=0),
            'per_class_metrics': {
                f'week_{i}': {
                    'precision': precision_score(targets, predictions, labels=[i], average='micro', zero_division=0),
                    'recall': recall_score(targets, predictions, labels=[i], average='micro', zero_division=0),
                    'f1': f1_score(targets, predictions, labels=[i], average='micro', zero_division=0),
                    'support': int(np.sum(targets == i))
                } for i in sorted(set(targets))
            }
        }

