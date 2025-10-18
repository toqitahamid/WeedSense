"""
WeedSense: Multi-Task Learning for Weed Analysis

Main model class integrating dual-path encoder and multi-task decoder
for simultaneous:
- Semantic segmentation of 16 weed species
- Plant height estimation (0.2-155 cm)
- Growth stage classification (weeks 1-11)
"""

import torch
from torch import nn
import math
from typing import Tuple, Union

from .encoder import DetailBranch, SemanticBranch, AggregationLayer
from .decoder import SegmentationHead, TemporalGrowthDecoder


class WeedSense(nn.Module):
    """
    WeedSense Multi-Task Model
    
    Architecture achieves:
    - 89.78% mIoU for segmentation
    - 1.67 cm MAE for height estimation
    - 99.99% accuracy for growth stage classification
    - 160 FPS real-time inference on V100 GPU
    
    Args:
        num_classes: Number of segmentation classes (default: 17 = 16 species + background)
        num_weeks: Number of growth stages (default: 11 for weeks 1-11)
    
    Example:
        >>> model = WeedSense(num_classes=17, num_weeks=11)
        >>> x = torch.randn(2, 3, 512, 512)
        >>> seg, height, week = model(x)
        >>> print(f"Seg: {seg.shape}, Height: {height.shape}, Week: {week.shape}")
    """
    
    def __init__(self, num_classes=17, num_weeks=11):
        super().__init__()
        
        # Dual-path UIB Encoder
        self.detail_branch = DetailBranch()
        self.semantic_branch = SemanticBranch()
        self.aggregation_layer = AggregationLayer()
        
        # Segmentation heads (main + auxiliary for deep supervision)
        self.output_head = SegmentationHead(128, 1024, num_classes, upscale_factor=8, is_aux=False)
        
        # Auxiliary heads for training only (discarded during inference)
        self.aux2_head = SegmentationHead(16, 128, num_classes, upscale_factor=4, is_aux=True)
        self.aux3_head = SegmentationHead(32, 128, num_classes, upscale_factor=8, is_aux=True)
        self.aux4_head = SegmentationHead(64, 128, num_classes, upscale_factor=16, is_aux=True)
        self.aux5_head = SegmentationHead(128, 128, num_classes, upscale_factor=32, is_aux=True)
        
        # Temporal Growth Decoder
        self.temporal_decoder = TemporalGrowthDecoder(
            embed_dim=512,
            num_heads=8,
            num_weeks=num_weeks
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def forward(self, x: torch.Tensor) -> Union[Tuple, Tuple[Tuple, torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input images (batch, 3, H, W)
        
        Returns:
            Training mode:
                Tuple of ((seg_main, seg_aux2, seg_aux3, seg_aux4, seg_aux5), height, week)
            Inference mode:
                Tuple of (seg_out, height, week)
        """
        # Extract features through dual-path encoder
        x_d = self.detail_branch(x)
        aux2, aux3, aux4, aux5, x_s = self.semantic_branch(x)
        
        # Aggregate detail and semantic features
        fused_features = self.aggregation_layer(x_d, x_s)
        
        # Segmentation output
        seg_out = self.output_head(fused_features)
        
        # Temporal growth predictions
        height_out, week_out = self.temporal_decoder(fused_features)
        
        if self.training:
            # During training, return auxiliary outputs for deep supervision
            aux2 = self.aux2_head(aux2)
            aux3 = self.aux3_head(aux3)
            aux4 = self.aux4_head(aux4)
            aux5 = self.aux5_head(aux5)
            return (seg_out, aux2, aux3, aux4, aux5), height_out, week_out
        
        # During inference, return only main predictions
        return seg_out, height_out, week_out
    
    def load_pretrained(self, checkpoint_path: str):
        """
        Load pretrained weights
        
        Args:
            checkpoint_path: Path to checkpoint file (.pth)
        """
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(state_dict, strict=True)
        print(f"Loaded pretrained weights from {checkpoint_path}")
    
    def _init_weights(self, m: nn.Module):
        """Initialize network weights"""
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def get_num_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_flops(self, input_size=(1, 3, 512, 512)):
        """
        Estimate FLOPs (requires thop package)
        
        Args:
            input_size: Input tensor size
        
        Returns:
            Number of FLOPs in billions (GFLOPs)
        """
        try:
            from thop import profile
            input_tensor = torch.randn(input_size)
            flops, params = profile(self, inputs=(input_tensor,), verbose=False)
            return flops / 1e9, params / 1e6
        except ImportError:
            print("Please install thop: pip install thop")
            return None, None


if __name__ == '__main__':
    # Test the model
    model = WeedSense(num_classes=17, num_weeks=11)
    x = torch.randn(2, 3, 512, 512)
    
    # Test training mode
    model.train()
    train_out = model(x)
    print("Training output shapes:")
    print(f"  Seg main: {train_out[0][0].shape}")
    print(f"  Seg aux2: {train_out[0][1].shape}")
    print(f"  Seg aux3: {train_out[0][2].shape}")
    print(f"  Seg aux4: {train_out[0][3].shape}")
    print(f"  Seg aux5: {train_out[0][4].shape}")
    print(f"  Height: {train_out[1].shape}")
    print(f"  Week: {train_out[2].shape}")
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        eval_out = model(x)
        print("\nInference output shapes:")
        print(f"  Seg: {eval_out[0].shape}")
        print(f"  Height: {eval_out[1].shape}")
        print(f"  Week: {eval_out[2].shape}")
    
    # Print model statistics
    print(f"\nModel statistics:")
    print(f"  Parameters: {model.get_num_params() / 1e6:.2f}M")

