"""
Quick Start Example for WeedSense

This script demonstrates how to:
1. Load a pre-trained WeedSense model
2. Run inference on a single image
3. Visualize the results
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from weedsense import WeedSense


def main():
    # Configuration
    checkpoint_path = 'pretrained/weedsense_iccv2025.pth'
    image_path = 'path/to/your/weed_image.jpg'  # Change this to your image
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # 1. Load pre-trained model
    print("\n1. Loading WeedSense model...")
    model = WeedSense(num_classes=17, num_weeks=11)
    
    # Load pretrained weights if available
    if Path(checkpoint_path).exists():
        model.load_pretrained(checkpoint_path)
        print(f"   Loaded weights from {checkpoint_path}")
    else:
        print(f"   Warning: Checkpoint not found at {checkpoint_path}")
        print(f"   Using randomly initialized weights")
    
    model = model.to(device)
    model.eval()
    
    # Print model statistics
    num_params = model.get_num_params()
    print(f"   Model parameters: {num_params / 1e6:.2f}M")
    
    # 2. Prepare input image
    print("\n2. Preparing input image...")
    
    if not Path(image_path).exists():
        print(f"   Error: Image not found at {image_path}")
        print(f"   Please provide a valid image path")
        return
    
    image = Image.open(image_path).convert('RGB')
    print(f"   Image size: {image.size}")
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    print(f"   Input tensor shape: {input_tensor.shape}")
    
    # 3. Run inference
    print("\n3. Running inference...")
    with torch.no_grad():
        seg_mask, height, week = model(input_tensor)
    
    # Post-process predictions
    pred_mask = seg_mask.argmax(dim=1).squeeze().cpu().numpy()
    pred_height = height.squeeze().cpu().item()
    pred_week = week.argmax(dim=1).item() + 1  # Convert to 1-11
    
    # 4. Display results
    print("\n" + "="*50)
    print("RESULTS:")
    print("="*50)
    print(f"Segmentation shape: {pred_mask.shape}")
    print(f"Predicted Height: {pred_height:.2f} cm (normalized)")
    print(f"Predicted Growth Stage: Week {pred_week}")
    print(f"Segmentation classes found: {len(set(pred_mask.flatten()))}")
    print("="*50)
    
    # 5. Optional: Visualize (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        
        print("\n5. Creating visualization...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Segmentation
        axes[1].imshow(pred_mask, cmap='tab20')
        axes[1].set_title(f'Segmentation\nHeight: {pred_height:.2f} cm | Week: {pred_week}')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = 'outputs/quick_start_result.png'
        Path('outputs').mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   Saved visualization to {output_path}")
        
        plt.show()
    except ImportError:
        print("\n   Note: Install matplotlib to enable visualization")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

