"""
Building blocks for WeedSense model

Includes:
- UIB (Universal Inverted Bottleneck) blocks with Squeeze-and-Excitation
- ConvModule for efficient convolution operations
"""

import torch
from torch import nn
import math
from functools import partial


class ConvModule(nn.Sequential):
    """Standard Conv-BN-ReLU block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


def make_divisible(v, divisor=8, min_value=None):
    """Ensure channels are divisible by divisor"""
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration"""
    
    def __init__(self, in_channels, rd_ratio=0.25):
        super().__init__()
        rd_channels = make_divisible(in_channels * rd_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, rd_channels, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, in_channels, 1, bias=True)
        self.gate = nn.Sigmoid()
    
    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class LayerScale2d(nn.Module):
    """Layer scale for stabilizing training"""
    
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    
    def forward(self, x):
        return x * self.gamma.view(1, -1, 1, 1)


class UIBBlock(nn.Module):
    """
    Universal Inverted Bottleneck (UIB) Block with SE Layer
    
    Configuration S0-M3-E0 uses only middle 3x3 depthwise convolution.
    This is the optimal configuration from our ablation studies.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels  
        stride: Stride for downsampling (default: 1)
        exp_ratio: Expansion ratio for bottleneck (default: 6)
        se_ratio: Squeeze-Excitation ratio (default: 0.25)
        layer_scale_init: Initial value for layer scale (default: 1e-5)
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        exp_ratio=6,
        se_ratio=0.25,
        layer_scale_init=1e-5,
    ):
        super().__init__()
        self.has_skip = (stride == 1 and in_channels == out_channels)
        mid_channels = make_divisible(in_channels * exp_ratio)
        
        # Point-wise expansion (1x1 conv)
        self.pw_exp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Depth-wise 3x3 convolution (S0-M3-E0 configuration)
        self.dw_mid = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Squeeze-and-Excitation
        self.se = SqueezeExcite(mid_channels, rd_ratio=se_ratio)
        
        # Point-wise projection (1x1 conv)
        self.pw_proj = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Layer scale for training stability
        self.layer_scale = LayerScale2d(out_channels, layer_scale_init)
    
    def forward(self, x):
        shortcut = x
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.se(x)
        x = self.pw_proj(x)
        x = self.layer_scale(x)
        if self.has_skip:
            x = x + shortcut
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block for temporal growth decoder
    
    Uses multi-head self-attention and feed-forward network
    with layer normalization and residual connections.
    """
    
    def __init__(self, embed_dim=512, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

