"""
Dual-path UIB Encoder for WeedSense

Consists of:
- Detail Branch: Captures fine-grained spatial details
- Semantic Branch: Extracts semantic context using UIB blocks
- Aggregation Layer: Fuses detail and semantic features
"""

import torch
from torch import nn
from .blocks import ConvModule, UIBBlock


class StemBlock(nn.Module):
    """
    Stem block for efficient initial feature extraction
    
    Inspired by Inception architecture for balanced efficiency and richness.
    """
    
    def __init__(self):
        super().__init__()
        self.conv_3x3 = ConvModule(3, 16, 3, 2, 1)
        self.left = nn.Sequential(
            ConvModule(16, 8, 1, 1, 0),
            ConvModule(8, 16, 3, 2, 1)
        )
        self.right = nn.MaxPool2d(3, 2, 1, ceil_mode=False)
        self.fuse = ConvModule(32, 16, 3, 1, 1)
    
    def forward(self, x):
        x = self.conv_3x3(x)
        x_left = self.left(x)
        x_right = self.right(x)
        y = torch.cat([x_left, x_right], dim=1)
        return self.fuse(y)


class ContextEmbeddingBlock(nn.Module):
    """
    Context embedding block for global context enhancement
    
    Captures global statistical information through adaptive pooling.
    """
    
    def __init__(self, channels=128):
        super().__init__()
        self.inner = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(channels),
            ConvModule(channels, channels, 1, 1, 0)
        )
        self.conv = ConvModule(channels, channels, 3, 1, 1)
    
    def forward(self, x):
        y = self.inner(x)
        out = x + y
        return self.conv(out)


class DetailBranch(nn.Module):
    """
    Detail Branch for fine-grained spatial detail preservation
    
    Follows shallow-wide architecture (VGGNet-inspired)
    with three stages: S1 (H/2), S2 (H/4), S3 (H/8)
    """
    
    def __init__(self):
        super().__init__()
        self.S1 = nn.Sequential(
            ConvModule(3, 64, 3, 2, 1),
            ConvModule(64, 64, 3, 1, 1)
        )
        self.S2 = nn.Sequential(
            ConvModule(64, 64, 3, 2, 1),
            ConvModule(64, 64, 3, 1, 1),
            ConvModule(64, 64, 3, 1, 1)
        )
        self.S3 = nn.Sequential(
            ConvModule(64, 128, 3, 2, 1),
            ConvModule(128, 128, 3, 1, 1),
            ConvModule(128, 128, 3, 1, 1)
        )
    
    def forward(self, x):
        return self.S3(self.S2(self.S1(x)))


class SemanticBranch(nn.Module):
    """
    Semantic Branch for semantic context extraction
    
    Follows deep-narrow architecture with aggressive downsampling.
    Uses UIB blocks with SE for enhanced feature representation:
    - Stem Block: Initial feature extraction (H/4, 16 channels)
    - S3: Two UIB blocks (H/8, 32 channels)
    - S4: Two UIB blocks (H/16, 64 channels)
    - S5: Four UIB blocks + Context Embedding (H/32, 128 channels)
    """
    
    def __init__(self):
        super().__init__()
        self.S1S2 = StemBlock()
        
        # Stage 3: UIB blocks with expansion ratio 6
        self.S3 = nn.Sequential(
            UIBBlock(16, 32, stride=2, exp_ratio=6),  # Downsample
            UIBBlock(32, 32, exp_ratio=6)             # No downsample
        )
        
        # Stage 4: UIB blocks
        self.S4 = nn.Sequential(
            UIBBlock(32, 64, stride=2, exp_ratio=6),  # Downsample
            UIBBlock(64, 64, exp_ratio=6)             # No downsample
        )
        
        # Stage 5: Four UIB blocks for deeper semantic understanding
        self.S5_1 = nn.Sequential(
            UIBBlock(64, 128, stride=2, exp_ratio=6),  # Downsample
            UIBBlock(128, 128, exp_ratio=6),
            UIBBlock(128, 128, exp_ratio=6),
            UIBBlock(128, 128, exp_ratio=6)
        )
        
        # Context embedding for global context
        self.S5_2 = ContextEmbeddingBlock(128)
    
    def forward(self, x):
        """
        Returns:
            Tuple of (aux2, aux3, aux4, aux5_1, aux5_2)
            for auxiliary supervision during training
        """
        x2 = self.S1S2(x)      # H/4, 16 channels
        x3 = self.S3(x2)       # H/8, 32 channels
        x4 = self.S4(x3)       # H/16, 64 channels
        x5_1 = self.S5_1(x4)   # H/32, 128 channels
        x5_2 = self.S5_2(x5_1) # H/32, 128 channels
        return x2, x3, x4, x5_1, x5_2


class AggregationLayer(nn.Module):
    """
    Aggregation Layer for fusing detail and semantic features
    
    Uses semantic features as intelligent guides to direct where 
    detail features should focus, creating a unified representation
    that preserves spatial precision while incorporating semantic understanding.
    
    Input:
        - x_d: Detail features (128 channels, H/8)
        - x_s: Semantic features (128 channels, H/32)
    Output:
        - Fused features (128 channels, H/8)
    """
    
    def __init__(self):
        super().__init__()
        # Left path: Process detail features
        self.left1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False),  # Depthwise
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 1, 1, 0, bias=False)  # Pointwise
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(3, 2, 1, ceil_mode=False)
        )
        
        # Right path: Process semantic features for attention
        self.right1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=4),
            nn.Sigmoid()  # Attention weights
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False),  # Depthwise
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 1, 1, 0, bias=False),  # Pointwise
            nn.Sigmoid()  # Attention weights
        )
        
        self.up = nn.Upsample(scale_factor=4)
        self.conv = ConvModule(128, 128, 3, 1, 1)
    
    def forward(self, x_d, x_s):
        """
        Args:
            x_d: Detail branch features (batch, 128, H/8, W/8)
            x_s: Semantic branch features (batch, 128, H/32, W/32)
        
        Returns:
            Aggregated features (batch, 128, H/8, W/8)
        """
        x1 = self.left1(x_d)
        x2 = self.left2(x_d)
        x3 = self.right1(x_s)
        x4 = self.right2(x_s)
        
        # Apply semantic attention to detail features
        left = x1 * x3
        right = x2 * x4
        right = self.up(right)
        out = left + right
        
        return self.conv(out)

