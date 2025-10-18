"""
Training script for WeedSense

Usage:
    python tools/train.py --config configs/weedsense_default.yaml

Features:
- Multi-task learning (segmentation + height + week)
- Auxiliary supervision for deep learning
- Mixed precision training (AMP)
- Distributed training (DDP)
- WandB logging (optional)
"""

import torch
import argparse
import yaml
import time
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from weedsense import WeedSense, WeedDataset
from weedsense.augmentations import get_train_augmentation, get_val_augmentation
from weedsense.losses import get_loss
from weedsense.optimizers import get_optimizer
from weedsense.schedulers import get_scheduler
from weedsense.metrics import SegmentationMetrics, HeightMetrics, WeekMetrics
from weedsense.utils.training_utils import fix_seeds, setup_cudnn, setup_ddp, cleanup_ddp


def train_epoch(model, trainloader, loss_fn, optimizer, scheduler, scaler, device, use_amp=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    seg_loss_sum = 0.0
    height_loss_sum = 0.0
    week_loss_sum = 0.0
    
    pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc="Training")
    
    for iter_idx, (images, masks, heights, weeks, _) in pbar:
        optimizer.zero_grad(set_to_none=True)
        
        images = images.to(device)
        masks = masks.to(device)
        heights = heights.to(device).float()
        weeks = weeks.to(device).long()
        
        with autocast(device_type='cuda', enabled=use_amp):
            outputs = model(images)
            
            # Handle model outputs
            if isinstance(outputs[0], tuple):
                seg_outputs, height_pred, week_pred = outputs
            else:
                seg_pred, height_pred, week_pred = outputs
                seg_outputs = seg_pred
            
            loss, loss_dict = loss_fn(
                (seg_outputs, height_pred, week_pred),
                (masks, heights, weeks)
            )
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        seg_loss_sum += loss_dict['seg_loss']
        height_loss_sum += loss_dict['height_loss']
        week_loss_sum += loss_dict['week_loss']
        
        lr = scheduler.get_lr()[0]
        
        pbar.set_postfix({
            'loss': f"{total_loss/(iter_idx+1):.4f}",
            'seg': f"{seg_loss_sum/(iter_idx+1):.4f}",
            'height': f"{height_loss_sum/(iter_idx+1):.4f}",
            'week': f"{week_loss_sum/(iter_idx+1):.4f}",
            'lr': f"{lr:.6f}"
        })
    
    return {
        'total_loss': total_loss / len(trainloader),
        'seg_loss': seg_loss_sum / len(trainloader),
        'height_loss': height_loss_sum / len(trainloader),
        'week_loss': week_loss_sum / len(trainloader),
        'lr': lr
    }


@torch.no_grad()
def validate(model, valloader, device, num_classes):
    """Validate model"""
    model.eval()
    
    seg_metrics = SegmentationMetrics(num_classes, 255, device)
    height_metrics = HeightMetrics()
    week_metrics = WeekMetrics()
    
    for images, masks, heights, weeks, _ in tqdm(valloader, desc="Validating"):
        images = images.to(device)
        masks = masks.to(device)
        heights = heights.to(device).float()
        weeks = weeks.to(device).long()
        
        outputs = model(images)
        
        if isinstance(outputs[0], tuple):
            seg_preds = outputs[0][0]
            height_preds = outputs[1]
            week_preds = outputs[2]
        else:
            seg_preds, height_preds, week_preds = outputs
        
        # Update metrics
        seg_metrics.update(seg_preds.softmax(dim=1), masks)
        
        # Denormalize height for metrics
        height_preds_denorm = valloader.dataset.denormalize_height(height_preds)
        heights_denorm = valloader.dataset.denormalize_height(heights)
        height_metrics.update(
            height_preds_denorm.cpu().numpy().flatten().tolist(),
            heights_denorm.cpu().numpy().flatten().tolist()
        )
        
        # Week metrics
        week_preds_class = torch.argmax(week_preds, dim=1)
        week_metrics.update(
            week_preds_class.cpu().numpy(),
            weeks.cpu().numpy()
        )
    
    results = {
        'segmentation': seg_metrics.compute_all(),
        'height': height_metrics.compute(),
        'week': week_metrics.compute()
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train WeedSense')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Setup
    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp() if cfg['TRAIN'].get('DDP', False) else 0
    device = torch.device(cfg['DEVICE'])
    
    # Create save directory
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    
    # Prepare datasets
    train_transform = get_train_augmentation(cfg['TRAIN']['IMAGE_SIZE'], seg_fill=cfg['DATASET']['IGNORE_LABEL'])
    val_transform = get_val_augmentation(cfg['EVAL']['IMAGE_SIZE'])
    
    trainset = WeedDataset(cfg['DATASET']['ROOT'], 'train', train_transform, normalize_height=True)
    valset = WeedDataset(cfg['DATASET']['ROOT'], 'val', val_transform, normalize_height=True)
    
    # Create dataloaders
    if cfg['TRAIN'].get('DDP', False):
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
    else:
        sampler = RandomSampler(trainset)
    
    trainloader = DataLoader(
        trainset,
        batch_size=cfg['TRAIN']['BATCH_SIZE'],
        sampler=sampler,
        num_workers=cfg['TRAIN']['NUM_WORKERS'],
        drop_last=True,
        pin_memory=True
    )
    
    valloader = DataLoader(
        valset,
        batch_size=cfg['EVAL'].get('BATCH_SIZE', 8),
        shuffle=False,
        num_workers=cfg['EVAL'].get('NUM_WORKERS', 4),
        pin_memory=True
    )
    
    # Create model
    model = WeedSense(
        num_classes=cfg['MODEL']['NUM_CLASSES'],
        num_weeks=cfg['MODEL']['NUM_WEEKS']
    )
    model = model.to(device)
    
    if cfg['TRAIN'].get('DDP', False):
        model = DDP(model, device_ids=[gpu])
    
    # Setup training
    loss_fn = get_loss(
        loss_name=cfg['LOSS']['NAME'],
        ignore_label=cfg['DATASET']['IGNORE_LABEL'],
        seg_weight=cfg['LOSS']['SEG_WEIGHT'],
        height_weight=cfg['LOSS']['HEIGHT_WEIGHT'],
        week_weight=cfg['LOSS']['WEEK_WEIGHT'],
        aux_weights=cfg['LOSS'].get('AUX_WEIGHTS')
    )
    
    optimizer = get_optimizer(
        model,
        cfg['OPTIMIZER']['NAME'],
        cfg['OPTIMIZER']['LR'],
        cfg['OPTIMIZER']['WEIGHT_DECAY']
    )
    
    iters_per_epoch = len(trainloader)
    scheduler = get_scheduler(
        cfg['SCHEDULER']['NAME'],
        optimizer,
        max_iter=cfg['TRAIN']['EPOCHS'] * iters_per_epoch,
        eta_ratio=cfg['SCHEDULER'].get('ETA_RATIO', 0),
        power=cfg['SCHEDULER'].get('POWER', 0.9),
        warmup_iter=cfg['SCHEDULER']['WARMUP_ITER'],
        warmup_ratio=cfg['SCHEDULER']['WARMUP_RATIO']
    )
    
    scaler = GradScaler(enabled=cfg['TRAIN']['AMP'])
    
    # Training loop
    best_miou = 0.0
    start_time = time.time()
    
    print(f"\nStarting training for {cfg['TRAIN']['EPOCHS']} epochs...")
    print(f"Training samples: {len(trainset)}, Validation samples: {len(valset)}")
    
    for epoch in range(cfg['TRAIN']['EPOCHS']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{cfg['TRAIN']['EPOCHS']}")
        print(f"{'='*60}")
        
        if cfg['TRAIN'].get('DDP', False):
            sampler.set_epoch(epoch)
        
        # Train
        train_metrics = train_epoch(
            model, trainloader, loss_fn, optimizer,
            scheduler, scaler, device, cfg['TRAIN']['AMP']
        )
        
        print(f"\nTrain Loss: {train_metrics['total_loss']:.4f}")
        print(f"  Seg: {train_metrics['seg_loss']:.4f} | "
              f"Height: {train_metrics['height_loss']:.4f} | "
              f"Week: {train_metrics['week_loss']:.4f}")
        
        # Validate
        if (epoch + 1) % cfg['TRAIN']['EVAL_INTERVAL'] == 0 or (epoch + 1) == cfg['TRAIN']['EPOCHS']:
            results = validate(
                model.module if cfg['TRAIN'].get('DDP', False) else model,
                valloader,
                device,
                cfg['MODEL']['NUM_CLASSES']
            )
            
            miou = results['segmentation']['mean_iou']
            height_mae = results['height']['mae']
            week_acc = results['week']['accuracy']
            
            print(f"\nValidation Results:")
            print(f"  mIoU: {miou:.2f}% | Height MAE: {height_mae:.2f} cm | Week Acc: {week_acc*100:.2f}%")
            
            # Save best model
            if miou > best_miou:
                best_miou = miou
                model_path = save_dir / "weedsense_best.pth"
                torch.save(
                    model.module.state_dict() if cfg['TRAIN'].get('DDP', False) else model.state_dict(),
                    model_path
                )
                print(f"  ✓ Saved best model (mIoU: {best_miou:.2f}%)")
    
    # Training complete
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best mIoU: {best_miou:.2f}%")
    print(f"Total time: {elapsed/3600:.2f} hours")
    print(f"{'='*60}\n")
    
    cleanup_ddp()


if __name__ == '__main__':
    main()

