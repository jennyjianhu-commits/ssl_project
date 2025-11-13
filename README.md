# Semi-Supervised Learning for CIFAR-10 with Limited Labels

A comprehensive implementation of semi-supervised learning techniques progressing through 5 phases, from supervised baseline to advanced methods (Mean Teacher, FixMatch, FlexMatch) with evaluation framework.

* **Dataset**: CIFAR-10 (32x32 RGB images, 10 classes)
* **Model**: Modified ResNet-18 (adapted for small image sizes)
* **Framework**: PyTorch
  
---
## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Phases Overview](#phases-overview)
- [Quick Start](#quick-start)
- [File Structure](#file-structure)
- [Usage Guide](#usage-guide)
- [Key Components](#key-components)
- [Results & Evaluation](#results--evaluation)
- [Hyperparameters](#hyperparameters)

---

## Project Overview

This project implements a progressive pipeline for semi-supervised learning research:

1. **Phase 1**: Supervised baseline with limited labeled data
2. **Phase 2**: Mean Teacher - self-ensembling approach
3. **Phase 3**: FixMatch - confidence thresholding with consistency regularization
4. **Phase 4**: FlexMatch - curriculum pseudo-labeling with adaptive thresholds
5. **Phase 5**: Comprehensive evaluation and comparison

The project explores how semi-supervised learning methods adapt to low-label scenarios (1%, 5%, 10%, and 100% labeled data).

---

## Architecture

### Model: Modified ResNet-18

Adapted for CIFAR-10 small image size (32x32):
- **Initial Layer**: 3x3 convolution (stride=1) instead of 7x7 (stride=2)
- **Removed**: MaxPooling layer after initial convolution
- **Layers**: 4 residual blocks with 2 blocks each
- **Classifier**: Global Average Pooling Linear (512 x 10 classes)

Key files:
- `utils/common_utils.py:103-175` - ModifiedResNet18 implementation

### Data Processing

- **Normalization**: CIFAR-10 standard (mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
- **Augmentation Types**:
  - **Weak**: Random horizontal flip + random crop (for Mean Teacher)
  - **Strong**: Weak augmentation + RandAugment (for FixMatch/FlexMatch student)
  - **Standard**: Used in supervised learning

---

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.7.1
- TorchVision >= 0.22.1
- NumPy >= 2.0.2
- Matplotlib >= 3.9.4
- scikit-learn >= 1.6.1

### Setup

```bash
# Clone repository
git clone <repo-url>
cd projectA

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install torch torchvision numpy matplotlib scikit-learn
```
---

## Phases Overview

### Phase 1: Supervised Baseline (`phase1_baseline.py`)

**Goal**: Establish supervised learning performance baseline with varying labeled data ratios.

**Key Features**:
- Trains on full dataset (100% labeled) and limited labeled scenarios
- Labeled ratios: 1%, 5%, 10%, 100%
- Cross-entropy loss only
- Learning rate scheduler with decay

**Output**:
- Training metrics plot
- Detailed results JSON with per-ratio statistics
- Test accuracy comparison

**Run**:
```bash
python phase1_baseline.py --epochs 100 --ratios 0.01 0.05 0.1 1.0 --seeds 42 123 456
```

---

### Phase 2: Mean Teacher (`phase2_mean_teacher.py`)

**Goal**: Implement self-ensembling approach using teacher-student framework.

**Key Features**:
- **Teacher Model**: Exponential Moving Average (EMA) of student
- **Student Model**: Regular model trained on labeled + unlabeled data
- **Loss Function**:
  - Supervised loss (labeled data)
  - Consistency regularization loss (unlabeled data)
- **Dual Augmentation**: Weak augmentation for teacher, strong for student
- **EMA Decay**: 0.999 (configured in code)

**Components**:
- `models/ema.py` - ModelEMA implementation

**Output**:
- Tracking metrics per epoch
- Performance comparison across labeled ratios
- Training visualizations

**Run**:
```bash
python phase2_mean_teacher.py --epochs 100 --ratios 0.01 0.05 0.1 --seeds 42 123 456
```
---
