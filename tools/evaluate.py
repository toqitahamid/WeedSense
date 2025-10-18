"""
Evaluation script for WeedSense model

Usage:
    python tools/evaluate.py --config configs/weedsense_default.yaml \
                             --checkpoint pretrained/weedsense_iccv2025.pth \
                             --data_root path/to/data \
                             --split test
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from weedsense import WeedSense, WeedDataset


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_iou(pred, target, num_classes):
    """
    Compute Intersection over Union (IoU) for segmentation
    
    Args:
        pred: Predicted masks (N,)
        target: Ground truth masks (N,)
        num_classes: Number of classes
    
    Returns:
        Array of IoU per class
    """
    ious = []
    pred = pred.flatten()
    target = target.flatten()
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return np.array(ious)


def evaluate(model, dataloader, device, num_classes):
    """
    Evaluate model on dataset
    
    Args:
        model: WeedSense model
        dataloader: Data loader
        device: Device to use
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    # Initialize metrics
    all_ious = []
    height_errors = []
    week_correct = 0
    week_total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images, masks, heights, weeks, _ = batch
            images = images.to(device)
            masks = masks.to(device)
            heights = heights.to(device)
            weeks = weeks.to(device)
            
            # Forward pass
            seg_out, height_out, week_out = model(images)
            
            # Segmentation metrics
            pred_masks = seg_out.argmax(dim=1)
            for i in range(len(images)):
                iou = compute_iou(
                    pred_masks[i].cpu().numpy(),
                    masks[i].cpu().numpy(),
                    num_classes
                )
                all_ious.append(iou)
            
            # Height metrics (MAE)
            height_mae = torch.abs(height_out.squeeze() - heights).cpu().numpy()
            height_errors.extend(height_mae.tolist())
            
            # Week classification accuracy
            week_pred = week_out.argmax(dim=1)
            week_correct += (week_pred == weeks).sum().item()
            week_total += len(weeks)
    
    # Compute statistics
    all_ious = np.array(all_ious)
    mean_iou = np.nanmean(all_ious, axis=0)
    
    results = {
        'segmentation': {
            'mIoU': np.nanmean(mean_iou) * 100,
            'per_class_iou': mean_iou * 100,
        },
        'height': {
            'MAE': np.mean(height_errors),
            'RMSE': np.sqrt(np.mean(np.array(height_errors)**2)),
            'median_error': np.median(height_errors),
        },
        'week': {
            'accuracy': (week_correct / week_total) * 100,
        }
    }
    
    return results


def print_results(results):
    """Print evaluation results"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\nSegmentation:")
    print(f"  mIoU: {results['segmentation']['mIoU']:.2f}%")
    
    print("\nHeight Estimation:")
    print(f"  MAE: {results['height']['MAE']:.2f} cm")
    print(f"  RMSE: {results['height']['RMSE']:.2f} cm")
    print(f"  Median Error: {results['height']['median_error']:.2f} cm")
    
    print("\nGrowth Stage Classification:")
    print(f"  Accuracy: {results['week']['accuracy']:.2f}%")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='WeedSense Evaluation')
    parser.add_argument('--config', type=str, default='configs/weedsense_default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of dataset')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading {args.split} dataset from {args.data_root}")
    dataset = WeedDataset(
        root=args.data_root,
        split=args.split,
        normalize_height=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = WeedSense(
        num_classes=config['MODEL']['NUM_CLASSES'],
        num_weeks=config['MODEL']['NUM_WEEKS']
    )
    model.load_pretrained(args.checkpoint)
    model = model.to(device)
    
    # Evaluate
    print("Starting evaluation...")
    results = evaluate(
        model,
        dataloader,
        device,
        num_classes=config['MODEL']['NUM_CLASSES']
    )
    
    # Print results
    print_results(results)
    
    # Save results
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'evaluation_{args.split}.txt'
    
    with open(output_file, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.data_root}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Dataset size: {len(dataset)}\n\n")
        
        f.write("Segmentation:\n")
        f.write(f"  mIoU: {results['segmentation']['mIoU']:.2f}%\n\n")
        
        f.write("Height Estimation:\n")
        f.write(f"  MAE: {results['height']['MAE']:.2f} cm\n")
        f.write(f"  RMSE: {results['height']['RMSE']:.2f} cm\n")
        f.write(f"  Median Error: {results['height']['median_error']:.2f} cm\n\n")
        
        f.write("Growth Stage Classification:\n")
        f.write(f"  Accuracy: {results['week']['accuracy']:.2f}%\n")
    
    print(f"Results saved to {output_file}")
    print("\nEvaluation completed successfully!")


if __name__ == '__main__':
    main()

