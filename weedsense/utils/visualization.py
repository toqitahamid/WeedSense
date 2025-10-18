"""
Visualization utilities for WeedSense predictions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def visualize_prediction(image, seg_mask, height, week, species_names=None):
    """
    Visualize model predictions
    
    Args:
        image: Input image (PIL Image or numpy array)
        seg_mask: Segmentation mask (H, W) numpy array
        height: Predicted height in cm
        week: Predicted week (1-11)
        species_names: Optional list of species names
    
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Segmentation with predictions
    axes[1].imshow(seg_mask, cmap='tab20', interpolation='nearest')
    title = f'Segmentation Prediction\n'
    title += f'Height: {height:.2f} cm | Growth Stage: Week {week}'
    axes[1].set_title(title, fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig


def overlay_mask(image, mask, alpha=0.5, colormap='tab20'):
    """
    Overlay segmentation mask on image
    
    Args:
        image: Input image (PIL Image or numpy array)
        mask: Segmentation mask (H, W) numpy array
        alpha: Transparency for overlay (0-1)
        colormap: Matplotlib colormap name
    
    Returns:
        Overlayed image as numpy array
    """
    # Convert image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize mask to match image if needed
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    # Create colored mask
    cmap = plt.get_cmap(colormap)
    mask_colored = cmap(mask / mask.max())[:, :, :3]  # RGB only
    mask_colored = (mask_colored * 255).astype(np.uint8)
    
    # Overlay
    overlayed = cv2.addWeighted(image, 1-alpha, mask_colored, alpha, 0)
    
    return overlayed


def plot_height_distribution(heights, predictions):
    """
    Plot height distribution and predictions
    
    Args:
        heights: Ground truth heights
        predictions: Predicted heights
    
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Scatter plot
    axes[0].scatter(heights, predictions, alpha=0.5, s=10)
    axes[0].plot([heights.min(), heights.max()], 
                 [heights.min(), heights.max()], 
                 'r--', lw=2, label='Perfect prediction')
    axes[0].set_xlabel('Ground Truth Height (cm)')
    axes[0].set_ylabel('Predicted Height (cm)')
    axes[0].set_title('Height Prediction Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error distribution
    errors = predictions - heights
    axes[1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Prediction Error (cm)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Error Distribution (MAE: {np.abs(errors).mean():.2f} cm)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plot confusion matrix for classification
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names
    
    Returns:
        Figure object
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Set ticks
    if class_names is not None:
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, str(cm[i, j]),
                          ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white")
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    return fig

