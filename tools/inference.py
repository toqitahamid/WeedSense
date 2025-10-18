"""
Inference script for WeedSense model

Usage:
    python tools/inference.py --config configs/weedsense_default.yaml \
                              --checkpoint pretrained/weedsense_iccv2025.pth \
                              --image path/to/image.jpg \
                              --output outputs/prediction.png
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from weedsense import WeedSense


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_image(image_path, image_size=(512, 512)):
    """
    Prepare image for inference
    
    Args:
        image_path: Path to input image
        image_size: Target size (height, width)
    
    Returns:
        Tuple of (input_tensor, original_image)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    return input_tensor, original_image


def visualize_results(original_image, seg_mask, height, week, output_path=None):
    """
    Visualize segmentation results with predictions
    
    Args:
        original_image: Original PIL image
        seg_mask: Segmentation mask (H, W) numpy array
        height: Predicted height in cm
        week: Predicted week (1-11)
        output_path: Optional path to save visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Segmentation mask
    axes[1].imshow(seg_mask, cmap='tab20')
    axes[1].set_title(f'Segmentation\nHeight: {height:.2f} cm | Week: {week}')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='WeedSense Inference')
    parser.add_argument('--config', type=str, default='configs/weedsense_default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output visualization')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = WeedSense(
        num_classes=config['MODEL']['NUM_CLASSES'],
        num_weeks=config['MODEL']['NUM_WEEKS']
    )
    model.load_pretrained(args.checkpoint)
    model = model.to(device)
    model.eval()
    
    # Prepare image
    print(f"Processing image: {args.image}")
    input_tensor, original_image = prepare_image(
        args.image,
        image_size=tuple(config['TEST']['IMAGE_SIZE'])
    )
    input_tensor = input_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        seg_out, height_out, week_out = model(input_tensor)
        
        # Post-process predictions
        seg_mask = seg_out.argmax(dim=1).squeeze().cpu().numpy()
        height_pred = height_out.squeeze().cpu().item()
        week_pred = week_out.argmax(dim=1).item() + 1  # Convert to 1-11
    
    # Print results
    print("\n" + "="*50)
    print("PREDICTIONS:")
    print("="*50)
    print(f"Predicted Height: {height_pred:.2f} cm (normalized)")
    print(f"Predicted Growth Stage: Week {week_pred}")
    print("="*50 + "\n")
    
    # Visualize
    visualize_results(
        original_image,
        seg_mask,
        height_pred,
        week_pred,
        output_path=args.output
    )
    
    print("Inference completed successfully!")


if __name__ == '__main__':
    main()

