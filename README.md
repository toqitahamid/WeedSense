# WeedSense: Multi-Task Learning for Weed Segmentation, Height Estimation, and Growth Stage Classification

[![ICCV 2025](https://img.shields.io/badge/ICCV-2025-blue)](https://iccv2025.org/)
[![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b.svg)](#)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Official PyTorch implementation** of WeedSense accepted at ICCV 2025.

> 📄 **Paper**: Coming soon  
> 🌐 **Project Page**: weedsense.github.io  
> 🎬 **Demo**: See project page for demonstrations

## 📋 Overview

WeedSense is a novel multi-task learning architecture for comprehensive weed analysis that jointly performs semantic segmentation, height estimation, and growth stage classification. The model addresses critical challenges in agriculture by enabling automated weed monitoring and analysis for sustainable agricultural practices and site-specific management approaches.

### Key Features

✨ **State-of-the-art Performance**: **89.78% mIoU** and **94.54% mF1** for segmentation  
📏 **Accurate Height Estimation**: **1.67 cm MAE** across 0.2-155 cm range  
📅 **Growth Stage Classification**: **99.99% accuracy** for temporal analysis  
⚡ **Real-time Inference**: **160 FPS** on NVIDIA V100 GPU  
🎯 **Efficient Architecture**: **30.50M parameters** with **16.73 GFLOPs**  
🌱 **Comprehensive Dataset**: **16 weed species** over **11-week growth cycle**  
📦 **Easy to Use**: Standalone PyTorch implementation with minimal dependencies

### Architecture Highlights

- **Dual-Path UIB Encoder**: Detail and Semantic branches for complementary feature extraction
- **Universal Inverted Bottleneck (UIB) Blocks**: S0-M3-E0 configuration with Squeeze-and-Excitation
- **Multi-Task Bifurcated Decoder**: Parallel pathways for segmentation and temporal growth prediction
- **Transformer-Based Feature Fusion**: Multi-head self-attention for temporal growth analysis
- **Auxiliary Supervision**: Deep supervision during training (zero inference overhead)

## 🚀 Getting Started

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.0 (for GPU training/inference)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/toqitahamid/weedsense.git
cd weedsense
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install WeedSense**
```bash
pip install -e .
```

### Quick Start

#### Inference with Pre-trained Model

```python
import torch
from weedsense import WeedSense
from PIL import Image
import torchvision.transforms as transforms

# Load pre-trained model
model = WeedSense(num_classes=17, num_weeks=11)
model.load_pretrained('pretrained/weedsense_iccv2025.pth')
model.eval()
model.cuda()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

image = Image.open('path/to/your/weed_image.jpg')
input_tensor = transform(image).unsqueeze(0).cuda()

# Run inference
with torch.no_grad():
    seg_mask, height, week = model(input_tensor)
    
    # Get predictions
    pred_mask = seg_mask.argmax(dim=1)  # (1, H, W)
    pred_height = height.item()  # Scalar (normalized)
    pred_week = week.argmax(dim=1).item() + 1  # Week 1-11
    
print(f"Predicted height: {pred_height:.2f} cm")
print(f"Predicted growth stage: Week {pred_week}")
```

#### Training

```bash
# Single GPU
python tools/train.py --config configs/weedsense_default.yaml

# Multi-GPU with DDP (4 GPUs)
torchrun --nproc_per_node=4 tools/train.py --config configs/weedsense_default.yaml
```

#### Evaluation

```bash
python tools/evaluate.py \
    --config configs/weedsense_default.yaml \
    --checkpoint pretrained/weedsense_iccv2025.pth \
    --data_root ./data \
    --split test
```

#### Benchmarking

```bash
python tools/benchmark.py \
    --checkpoint pretrained/weedsense_iccv2025.pth \
    --input-size 512 512 \
    --batch-size 1 \
    --runs 3
```

## 📊 Model Architecture

### Dual-Path UIB Encoder

The encoder efficiently balances spatial detail preservation and semantic context extraction:

**Detail Branch** (Shallow-Wide):
- Three sequential stages (S1, S2, S3)
- Progressive downsampling: H/2 → H/4 → H/8
- Channel expansion: 64 → 64 → 128
- VGGNet-inspired architecture for fine-grained details

**Semantic Branch** (Deep-Narrow):
- Stem Block for efficient initial feature extraction
- UIB blocks with S0-M3-E0 configuration (only middle 3×3 depthwise conv)
- Four-stage hierarchy: 16 → 32 → 64 → 128 channels
- Squeeze-and-Excitation for adaptive channel recalibration
- Context Embedding Block for global context enhancement

**Aggregation Layer**:
- Fuses detail and semantic features at H/8 resolution
- Uses semantic features as attention guides
- Preserves spatial precision while incorporating semantic understanding

### Multi-Task Bifurcated Decoder

**Segmentation Head**:
- Progressive upsampling with PixelShuffle (8× parameter-free)
- 128 → 1024 → num_classes pathway
- Auxiliary heads at multiple scales for deep supervision

**Temporal Growth Decoder**:
- Global average pooling: 128-dim representation
- Linear projection to 512-dim embeddings
- Transformer block with 8-head self-attention
- Parallel task-specific heads:
  - Height regression: 512 → 1024 → 512 → 1
  - Growth stage classification: 512 → 1024 → 512 → 11

## 📁 Dataset

### Weed Species Dataset

Our dataset contains **120,341 annotated images** of **16 weed species** captured over **11 weeks**:

**Species** (categorized by growth rate):

*Fast-growing (>10 cm/week)*:
- AMATU (Amaranthus tuberculatus) - 13.72 cm/week, max 155 cm
- SORHA (Sorghum halepense) - 14.06 cm/week, max 121 cm
- SETFA (Setaria faberi) - 11.75 cm/week, max 124 cm

*Medium-growing (5-10 cm/week)*:
- SORVU, PANDI, SETPU, DIGSA, ECHCG, SIDSP, AMARE, ABUTH, AMBEL

*Slow-growing (<5 cm/week)*:
- AMAPA, CYPES, CHEAL, ERICA (slowest: 1.70 cm/week, max 17.3 cm)

### Dataset Structure

```
data/
├── train/
│   ├── images/           # RGB images (720×960)
│   ├── mmseg_masks/      # Segmentation masks
│   └── train_data.csv    # Height and week labels
├── val/
│   ├── images/
│   ├── mmseg_masks/
│   └── val_data.csv
└── test/
    ├── images/
    ├── mmseg_masks/
    └── test_data.csv
```

**CSV Format**:
```csv
img,species,height,week
ABUTH_week_1_IMG_0001_frame_001.jpg,ABUTH,2.5,1
AMAPA_week_5_IMG_0234_frame_042.jpg,AMAPA,28.3,5
...
```

**Dataset Statistics**:
- **Total Images**: 120,341 frames
- **Split**: 80% train, 10% val, 10% test
- **Species**: 16 weed species
- **Height Range**: 0.2 - 155 cm
- **Growth Stages**: 11 weeks (BBCH 11-60)
- **Resolution**: 720 × 960 pixels
- **Annotations**: Pixel-level masks + height + growth stage

## 📈 Results

### Quantitative Results

| Model | mIoU (%) ↑ | mF1 (%) ↑ | Height MAE (cm) ↓ | R² ↑ | Week Acc (%) ↑ | Params (M) ↓ | GFLOPs ↓ | FPS ↑ |
|-------|-----------|----------|------------------|------|---------------|-------------|---------|------|
| MTL-SegFormer | 56.17 | 70.56 | 5.10 | 0.9289 | 97.96 | 8.19 | 7.94 | 138 |
| MTL-UNet | 51.51 | 66.30 | 2.15 | 0.9899 | 99.97 | 35.67 | 233.23 | 94 |
| MTL-PoolFormer | 72.45 | 83.53 | 4.20 | 0.9499 | 98.70 | 18.77 | 22.86 | 101 |
| MTL-BiSeNetV1 | 87.25 | 93.08 | 1.97 | 0.9913 | 99.99 | 17.31 | 13.30 | 249 |
| MTL-BiSeNetV2 | 89.29 | 94.26 | 1.75 | 0.9930 | 100.00 | 29.62 | 16.84 | 185 |
| MTL-SFNet | 86.27 | 92.51 | 4.21 | 0.9511 | 98.45 | 18.62 | 30.77 | 151 |
| **WeedSense** | **89.78** | **94.54** | **1.67** | **0.9941** | **99.99** | **30.50** | **16.73** | **160** |

### Height Estimation by Plant Size

| Model | Small (0-20cm) ↓ | Medium (20-50cm) ↓ | Large (50-100cm) ↓ | Very Large (>100cm) ↓ |
|-------|-----------------|-------------------|-------------------|----------------------|
| MTL-BiSeNetV2 | **1.20** | 2.53 | 2.32 | 3.12 |
| **WeedSense** | **1.20** | **2.28** | **2.28** | **2.60** |

**Key Achievements**:
- 🏆 **Best Overall**: 89.78% mIoU, 94.54% mF1 for segmentation
- 📏 **Accurate Heights**: 1.67 cm MAE, 70.37% predictions within 2 cm
- 📊 **Consistent Performance**: Maintains accuracy across all plant sizes
- ⚡ **Efficient**: 3× faster than sequential single-task execution
- 💡 **Parameter Efficient**: 32.4% fewer parameters than separate models

## 🔧 Configuration

Model configuration through YAML files. See `configs/weedsense_default.yaml` for full options.

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 2×10⁻⁴ | Base learning rate (AdamW) |
| Weight Decay | 0.0001 | L2 regularization |
| Batch Size | 8 | Training batch size |
| Epochs | 50 | Total training epochs |
| Input Size | 512×512 | Image resolution |
| Warmup Iterations | 1,500 | Linear warmup |
| Loss Weights | [1.0, 1.0, 1.0] | Seg, Height, Week |
| Aux Weights | [1.0, 0.4, 0.4, 0.4, 0.4] | Main + 4 auxiliary heads |

## 🛠️ Advanced Usage

### Custom Dataset

```python
from weedsense.datasets import WeedDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = WeedDataset(
    root='path/to/data',
    split='train',
    normalize_height=True
)

# Create dataloader
dataloader = DataLoader(
    dataset, 
    batch_size=8,
    shuffle=True,
    num_workers=4
)

# Iterate
for image, mask, height, week, species in dataloader:
    # Training code here
    pass
```

### Model Modifications

```python
from weedsense import WeedSense

# Custom number of classes/weeks
model = WeedSense(num_classes=20, num_weeks=15)

# Get model statistics
print(f"Parameters: {model.get_num_params() / 1e6:.2f}M")

# Load pretrained weights
model.load_pretrained('path/to/checkpoint.pth')
```

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{sarker2025weedsense,
  title={WeedSense: Multi-Task Learning for Weed Segmentation, Height Estimation, and Growth Stage Classification},
  author={Sarker, Toqi Tahamid and Ahmed, Khaled R and Islam, Taminul and Rankrape, Cristiana Bernardi and Gage, Karla},
  journal={arXiv preprint arXiv:2508.14486},
  year={2025}
}
```

**Paper**: [arXiv:2508.14486](https://arxiv.org/abs/2508.14486) | **Project Page**: [https://weedsense.github.io](https://weedsense.github.io)

## 🙏 Acknowledgements

This research was conducted at the SIU Horticulture Research Center. We thank:

- **[sithu31296/semantic-segmentation](https://github.com/sithu31296/semantic-segmentation)** - This codebase is built upon this excellent semantic segmentation library
- The greenhouse facility staff for supporting data collection
- The open-source community for excellent tools:
  - [PyTorch](https://pytorch.org/)
  - [timm](https://github.com/huggingface/pytorch-image-models)
  - [SAM2](https://github.com/facebookresearch/segment-anything-2) for annotation assistance

## 📄 License

This project is released under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📧 Contact

For questions and feedback:
- **Email**: toqitahamid.sarker@siu.edu
- **Project Page**: weedsense.github.io
- **GitHub Issues**: For bug reports and feature requests

---

**Note**: This is research code provided for reproducibility and further research. For production use, additional testing and validation are recommended.

## 🔗 Related Projects

- **[semantic-segmentation](https://github.com/sithu31296/semantic-segmentation)**: Base library for semantic segmentation models
- **BiSeNetV2**: Bilateral segmentation network for real-time semantic segmentation
- **MobileNetV4**: Universal Inverted Bottleneck blocks for efficient architectures

## 📝 TODO

- [ ] Add visualization tools
- [ ] Add ONNX export support
- [ ] Add TensorRT optimization guide
- [ ] Add data collection guidelines
- [ ] Add pre-trained models on HuggingFace

