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

### Phase 3: FixMatch (`phase3_fixmatch.py`)

**Goal**: Implement FixMatch with confidence thresholding and consistency regularization.

**Key Features**:
- **Weak + Strong Augmentation**: Uses RandAugment for unlabeled data
- **Pseudo-labeling**:
  - Thresholding on maximum confidence
  - Only uses predictions with confidence > threshold (default: 0.95)
- **Consistency Loss**: MSE between predictions of weakly and strongly augmented versions
- **Cosine Learning Rate Schedule**: With warmup phase
- **RandAugment Integration**: `randaugment.py` provides augmentation operations

**Components**:
- `randaugment.py` - RandAugmentMC implementation with FixMatch augmentation pool
  - 14 augmentation operations: rotation, translation, shear, brightness, contrast, etc.
  - Configured for FixMatch with specific parameter ranges

**Loss Formulation**:
```
L_total = L_sup + lambda * L_unsup
L_sup = cross_entropy(pred_labeled, label)
L_unsup = mse(pred_strong_aug, pseudo_label)
```

**Run**:
```bash
python phase3_fixmatch.py --epochs 280 --ratios 0.01 0.05 0.1 --seeds 42 123 456
```
---

### Phase 4: FlexMatch (`phase4_flexmatch.py`)

**Goal**: Implement FlexMatch with curriculum pseudo-labeling and adaptive thresholds.

**Key Differences from FixMatch**:
- **Class-wise Adaptive Thresholds**: Instead of fixed threshold
  - Adjusts per-class threshold based on learning status
  - Uses beta mapping function (convex, concave, or linear)
- **Curriculum Pseudo-labeling**:
  - Gradually increases labeling as training progresses
  - Tracks which unlabeled samples have been assigned pseudo-labels
- **Learning Status Tracking**: Monitors confidence distribution per class

**Features**:
- Dynamic threshold adjustment during training
- Beta mapping functions for curriculum scheduling
- Extended logging of pseudo-labeling statistics

**Run**:
```bash
python phase4_flexmatch.py --epochs 280 --ratios 0.01 0.05 0.1 --seeds 42 123 456
```

---

### Phase 5: Evaluation & Comparison (`phase5_evaluation.py`)

**Goal**: Comprehensive evaluation and visualization of all methods.

**Functionality**:
- **Aggregation**: Collects results from all 4 methods
- **Statistics**: Computes mean, standard deviation, and 95% confidence intervals
- **Visualization**:
  - Comparative bar chart with error bars
  - Shows all methods across different labeled ratios
- **Output Formats**:
  - Markdown table for reports
  - High-quality PNG visualization (300 DPI)

**Automatically Locates**:
```
results/phase1_baseline_results_*/detailed_results.json
results/phase2_mean_teacher_results_*/detailed_results.json
results/phase3_fixmatch_results_*/detailed_results.json
results/phase4_flexmatch_results_*/detailed_results.json
```

**Run**:
```bash
python phase5_evaluation.py
```

**Output**:
```
# Phase 5 Comprehensive Evaluation Results Comparison

| labeled ratio | Supervised | Mean Teacher | FixMatch | FlexMatch |
|---|---|---|---|---|
| 100.0% | 92.34±0.45 | 91.98±0.52 | 91.21±0.89 | 91.87±0.61 |
| 10.0% | 78.92±1.23 | 85.34±0.98 | 87.45±1.12 | 88.92±0.87 |
| 5.0% | 68.43±2.15 | 78.92±1.45 | 81.23±1.34 | 83.45±1.02 |
| 1.0% | 42.12±3.42 | 58.34±2.87 | 62.45±2.12 | 68.92±1.98 |

=± comparison figure generated: results/phase5_comparison.png
```
---

## Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision numpy matplotlib scikit-learn

# 2. Run all phases sequentially
python phase1_baseline.py --epochs 100 --seeds 42 123 456
python phase2_mean_teacher.py --epochs 100 --seeds 42 123 456
python phase3_fixmatch.py --epochs 280 --seeds 42 123 456
python phase4_flexmatch.py --epochs 280 --seeds 42 123 456

# 3. Generate comprehensive evaluation
python phase5_evaluation.py

# 4. Results are saved in results/ directory
# Logs are saved in logs/ directory
```

---

---

## Usage Guide

### Basic Training (Phase 1)

```bash
python phase1_baseline.py
```

### Custom Labeled Ratios

```bash
# Train only on 1% and 10% labeled data
python phase1_baseline.py --ratios 0.01 0.1 --epochs 150

# Single seed for quick testing
python phase1_baseline.py --seeds 42 --epochs 50
```

### Multiple Seeds for Robustness

```bash
# Train with 3 different random seeds
python phase1_baseline.py --seeds 42 123 456 --epochs 200

# Results are aggregated: mean ± std
```

### Logging and Monitoring

```bash
# Set logging level (DEBUG, INFO, WARNING, ERROR)
python phase1_baseline.py --log-level DEBUG

# Logs are saved in ./logs/ with timestamps
# Example: logs/phase1_baseline_20240113_143022.log
```

### Viewing Results

```bash
# After all phases complete
python phase5_evaluation.py

# This generates:
# - results/phase5_comparison.png (high-quality chart)
# - Console markdown table with statistics
```

---

## Key Components

### 1. Common Utilities (`utils/common_utils.py`)

**Data Processing**:
- `prepare_supervised_data()` (lines 233-295): Creates train/val/test loaders for supervised learning
- `prepare_semi_supervised_data()` (lines 297-383): Splits data into labeled/unlabeled/val with dual augmentation
- `get_cifar10_transforms()` (lines 181-231): Returns appropriate augmentation pipelines

**Model & Training**:
- `ModifiedResNet18` (lines 103-175): Custom ResNet architecture for 32x32 images
- `evaluate_model()` (lines 389-414): Computes accuracy with optional detailed predictions
- `create_visualizations()` (lines 416-469): Generates training metric plots

**Infrastructure**:
- `setup_logging()` (lines 28-53): Configures file + console logging with timestamps
- `set_seed()` (lines 55-61): Sets PyTorch, NumPy, and CUDA seeds for reproducibility
- `create_common_parser()` (lines 475-493): Shared argument parser for all phases

### 2. EMA Implementation (`models/ema.py`)

**ModelEMA Class** (lines 6-38):
- Deep copy of model for teacher network
- Exponential moving average update: `theta_t = alpha*theta_t_1 + (1-alpha)*theta_t`
- No gradient computation (eval mode)
- Handles both standard and DataParallel models

**Usage** (from Phase 2):
```python
ema_model = ModelEMA(args, model, decay=0.999)
# ... training loop ...
ema_model.update(model)  # Update after each batch
predictions = ema_model.ema(unlabeled_data)  # Use for consistency
```

### 3. RandAugment (`randaugment.py`)

**14 Augmentation Operations**:
- **Geometric**: Rotate (line 80), ShearX (line 92), ShearY (line 99), TranslateX (line 123), TranslateY (line 131)
- **Color**: Brightness (line 24), Color (line 29), Contrast (line 34), Sharpness (line 87)
- **Pixel-level**: Posterize (line 75), Solarize (line 106), SolarizeAdd (line 111), Equalize (line 63)
- **Structure**: AutoContrast (line 20), Invert (line 71), Cutout (line 39)

**Classes**:
- `RandAugmentMC` (lines 209-224): FixMatch variant (randomly applies 2 ops)
- `RandAugmentPC` (lines 191-206): Custom variant with different operation pool

**Integration**:
- Automatically applies cutout after augmentation (50% patch)
- Magnitude-controlled intensity (1-10 scale)
- Probability-based application

---

## Results & Evaluation

### Expected Performance (CIFAR-10, ResNet-18)

| Labeled Ratio | Supervised | Mean Teacher | FixMatch | FlexMatch |
|---|---|---|---|---|
| 100% | ~92-93% | ~92-93% | ~91-92% | ~91-93% |
| 10% | ~79-81% | ~84-86% | ~87-89% | ~88-90% |
| 5% | ~68-71% | ~78-80% | ~80-83% | ~83-85% |
| 1% | ~42-45% | ~58-60% | ~62-65% | ~68-70% |

*Note: Exact values depend on seeds, hyperparameters, and computational variations*

### Output Files

**Per Phase**:
- results/
   - phase1_baseline_results_timestamp
      - detailed_results.json      
   - phase2_mean_teacher_results_timestamp
   - phase3_fixmatch_results_timestamp
   - phase4_flexmatch_results_timestamp
```

**Phase 5 Evaluation**:
```
results/
   phase5_comparison.png
```

**Logs**:
```
logs/
   phase1_baseline_timestamp.log
   phase2_mean_teacher_timestamp.log
   phase3_fixmatch_timestamp.log
   phase4_flexmatch_timestamp.log
```
---

## Hyperparameters

### Default Training Settings

| Parameter | Value | Adjustable |
|---|---|---|
| Optimizer | SGD | phase-specific |
| Learning Rate | 0.03 | code |
| Momentum | 0.9 | code |
| Weight Decay | 5e-4 | code |
| Batch Size | 64 | code |
| Epochs | 200 | `--epochs` |
| Labeled Ratios | [0.01, 0.05, 0.1, 1.0] | `--ratios` |
| Seeds | [42, 123, 456] | `--seeds` |

### Method-Specific Parameters

**Mean Teacher**:
- EMA decay: 0.999
- Consistency weight: 100.0
- Unlabeled loss ratio: scales with epoch

**FixMatch**:
- Confidence threshold: 0.95
- Unlabeled loss weight: 1.0
- Warmup epochs: 5
- Cosine schedule with 7/16 cycles

**FlexMatch**:
- Initial threshold: 0.95
- Beta mapping: 'convex'
- Curriculum learning enabled
- Class-wise threshold adaptation

---

## Development Notes

### Adding New Phases

1. Create `phaseX_<method>.py` following existing pattern
2. Import from `utils/common_utils.py`
3. Save results with unique timestamp in `results/` directory
4. Update `PHASE_RESULT_PATTERNS` in `phase5_evaluation.py` (lines 33-38)

### Extending Augmentation

Add to `randaugment.py`:
```python
# Add new operation (e.g., around line 190 with other classes)
def CustomAug(img, v, max_v, bias=0):
    # Implementation
    return modified_img

# Register in augmentation pool (my_augment_pool function)
augs.append((CustomAug, max_value, bias_value))
```

### Modifying Model Architecture

Edit `ModifiedResNet18` in `utils/common_utils.py:103-175`:
- Change channel dimensions in `_make_layer()`
- Adjust block depths in `__init__()`
- Modify stride settings for different resolution handling
  
---

## License

This project is for educational purposes as part of coursework.

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size in code
# Default: batch_size=64
# Edit in phase files to use batch_size=32 or smaller
```

### Slow Augmentation
- RandAugment uses PIL which can be slow
- Consider reducing `num_ops` or `magnitude` parameters
- Use GPU augmentation libraries (Albumentations) for faster training

### Missing Data
```bash
# CIFAR-10 auto-downloads on first run
# Check ./data/ directory after first run
# Manual download: https://www.cs.toronto.edu/~kriz/cifar.html
```

### Reproducibility Issues
- Set all seeds: `python phase1_baseline.py --seeds 42`
- Disable CUDA non-determinism (slower):
  - Add `torch.backends.cudnn.deterministic = True` in code

---

## Contact & Support

For issues or questions, please:
1. Check logs in `logs/` directory for detailed error messages
2. Verify all dependencies are installed: `pip list | grep -E 'torch|numpy|matplotlib'`
3. Test with small dataset: `python phase1_baseline.py --epochs 10 --seeds 42`

---

**Last Updated**: 2025
**Python Version**: 3.9+
**PyTorch Version**: 2.7.1+
