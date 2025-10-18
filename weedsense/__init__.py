"""
WeedSense: Multi-Task Learning for Weed Segmentation, Height Estimation, and Growth Stage Classification

Official PyTorch implementation of WeedSense accepted at ICCV 2025.
"""

__version__ = "1.0.0"

from .models import WeedSense
from .datasets import WeedDataset

__all__ = ['WeedSense', 'WeedDataset', '__version__']

