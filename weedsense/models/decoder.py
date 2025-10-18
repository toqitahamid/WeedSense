"""
Multi-Task Bifurcated Decoder for WeedSense

Consists of:
- SegmentationHead: For semantic segmentation
- TemporalGrowthDecoder: For height estimation and growth stage classification
- Auxiliary heads for deep supervision during training
"""

import torch
from torch import nn
from torch.nn import functional as F
from .blocks import ConvModule, TransformerBlock


class SegmentationHead(nn.Module):
    """
    Segmentation head with progressive upsampling
    
    Uses PixelShuffle for parameter-free 8x upsampling to recover
    full spatial resolution for pixel-level predictions.
    
    Args:
        in_channels: Input channels (default: 128)
        mid_channels: Intermediate channels (default: 1024)
        num_classes: Number of segmentation classes (default: 17)
        upscale_factor: Upsampling factor (default: 8)
        is_aux: Whether this is an auxiliary head (default: False)
    """
    
    def __init__(self, in_channels, mid_channels, num_classes, upscale_factor=8, is_aux=False):
        super().__init__()
        out_channels = num_classes * upscale_factor * upscale_factor
        
        self.conv_3x3 = ConvModule(in_channels, mid_channels, 3, 1, 1)
        self.drop = nn.Dropout(0.1)
        
        if is_aux:
            self.conv_out = nn.Sequential(
                ConvModule(mid_channels, upscale_factor * upscale_factor, 3, 1, 1),
                nn.Conv2d(upscale_factor * upscale_factor, out_channels, 1, 1, 0),
                nn.PixelShuffle(upscale_factor)
            )
        else:
            self.conv_out = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, 1, 1, 0),
                nn.PixelShuffle(upscale_factor)
            )
    
    def forward(self, x):
        out = self.conv_3x3(x)
        out = self.drop(out)
        return self.conv_out(out)


class TemporalGrowthDecoder(nn.Module):
    """
    Temporal Growth Decoder for height and growth stage prediction
    
    Processes aggregated features through:
    1. Global average pooling to 128-dim representation
    2. Linear projection to 512-dim embeddings
    3. Transformer block with multi-head self-attention
    4. Parallel task-specific heads for height and week prediction
    
    Architecture employs hard parameter sharing in feature processing
    while maintaining separate head weights for task-specific predictions.
    
    Args:
        embed_dim: Transformer embedding dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        num_weeks: Number of growth stages/weeks (default: 11)
    """
    
    def __init__(self, embed_dim=512, num_heads=8, num_weeks=11):
        super().__init__()
        
        # Project from 128 to embed_dim
        self.project = nn.Linear(128, embed_dim)
        
        # Transformer for feature enhancement
        self.transformer = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dim_feedforward=2048,
            dropout=0.1
        )
        
        # Shared task-specific feature processing
        self.task_head = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
        )
        
        # Task-specific output heads
        self.height_output = nn.Linear(512, 1)  # Regression
        self.week_output = nn.Linear(512, num_weeks)  # Classification
        
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, fused_features):
        """
        Args:
            fused_features: Aggregated features (batch, 128, H/8, W/8)
        
        Returns:
            Tuple of (height_prediction, week_logits)
            - height_prediction: (batch, 1) - Normalized height in cm
            - week_logits: (batch, 11) - Logits for week 1-11 classification
        """
        # Global average pooling
        global_features = F.adaptive_avg_pool2d(fused_features, 1).squeeze(-1).squeeze(-1)
        global_features = self.dropout(global_features)
        
        # Project and transform features
        projected_features = self.project(global_features)  # (batch, embed_dim)
        projected_features = projected_features.unsqueeze(1)  # (batch, 1, embed_dim)
        transformed_features = self.transformer(projected_features)
        transformed_features = transformed_features.squeeze(1)  # (batch, embed_dim)
        
        # Task-specific processing
        task_features = self.task_head(transformed_features)
        height_out = self.height_output(task_features)
        week_out = self.week_output(task_features)
        
        return height_out, week_out

