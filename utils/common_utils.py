#!/usr/bin/env python3
"""
Common utilities for semi-supervised learning experiments
Shared functions and classes used across different phases
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
import time
from datetime import datetime
import argparse
import logging
import copy

# =============================================================================
# Common setup functions
# =============================================================================

def setup_logging(phase_name, log_dir="./logs", log_level=logging.INFO):
    """setup logging configuration"""
    # create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{phase_name}_{timestamp}.log"
    
    # configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # also print to console
        ]
    )
    
    # log initial setup
    logging.info("=" * 80)
    logging.info(f"{phase_name} Experiment")
    logging.info("=" * 80)
    logging.info(f"Log file: {log_file}")
    
    return log_file

def set_seed(seed=42):
    """set random seed to ensure reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Global device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Model definition: modified ResNet-18
# =============================================================================

class BasicBlock(nn.Module):
    """ResNet basic residual block"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class ModifiedResNet18(nn.Module):
    """
    modified ResNet-18 for CIFAR-10:
    1. replace initial 7×7 convolution layer (stride 2) with 3×3 layer (stride 1)
    2. remove MaxPool2d layer to maintain spatial resolution
    3. suitable for CIFAR-10 (32×32) small size images
    """
    
    def __init__(self, num_classes=10):
        super(ModifiedResNet18, self).__init__()
        
        # modified initial layer: 3×3 convolution (stride 1) instead of 7×7 (stride 2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # note: removed MaxPool2d layer from original ResNet-18
        
        # four main layers of ResNet-18
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # weight initialization
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """build ResNet layers"""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # modified initial layer processing (no MaxPool)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# =============================================================================
# Data processing functions
# =============================================================================

def get_cifar10_transforms(augmentation_type='standard'):
    """
    Get CIFAR-10 data transforms
    
    Args:
        augmentation_type: 'standard', 'weak', 'strong', 'test'
    
    Returns:
        transform pipeline
    """
    
    # CIFAR-10 normalization values
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    if augmentation_type == 'standard':
        # Standard augmentation for supervised learning
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augmentation_type == 'weak':
        # Weak augmentation for teacher network
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augmentation_type == 'strong':
        # Strong augmentation for student network
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.RandAugment(num_ops=2, magnitude=10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augmentation_type == 'test':
        # No augmentation for testing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
    
    return transform

def prepare_supervised_data(labeled_ratio=1.0, batch_size=128, validation_split=0.1):
    """
    Prepare CIFAR-10 dataset for supervised learning
    
    Args:
        labeled_ratio: labeled data ratio (0.01, 0.05, 0.1, 1.0)
        batch_size: batch size
        validation_split: validation set ratio
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Get transforms
    train_transform = get_cifar10_transforms('standard')
    test_transform = get_cifar10_transforms('test')
    
    # Download dataset
    full_trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    total_size = len(full_trainset)
    indices = np.random.permutation(total_size)
    
    if labeled_ratio < 1.0:
        # Limited labeled data experiment
        labeled_size = int(total_size * labeled_ratio)
        val_size = int(total_size * validation_split)
        
        labeled_indices = indices[:labeled_size]
        val_indices = indices[labeled_size:labeled_size + val_size]
        
        labeled_set = Subset(full_trainset, labeled_indices)
        val_set = Subset(full_trainset, val_indices)
        
        logging.info(f"labeled data: {len(labeled_set)} ({labeled_ratio*100:.1f}%)")
        logging.info(f"validation data: {len(val_set)} ({validation_split*100:.1f}%)")
    else:
        # Fully supervised learning (100% labeled data)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        labeled_set = Subset(full_trainset, train_indices)
        val_set = Subset(full_trainset, val_indices)
        
        logging.info(f"training data: {len(labeled_set)} ({train_size/total_size*100:.1f}%)")
        logging.info(f"validation data: {len(val_set)} ({validation_split*100:.1f}%)")
    
    logging.info(f"test data: {len(testset)}")
    
    # Create data loaders
    train_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

def prepare_semi_supervised_data(labeled_ratio=0.1, batch_size=128, validation_split=0.1):
    """
    Prepare CIFAR-10 dataset for semi-supervised learning with dual augmentation
    
    Args:
        labeled_ratio: labeled data ratio (0.01, 0.05, 0.1)
        batch_size: batch size
        validation_split: validation set ratio
    
    Returns:
        labeled_loader, unlabeled_loader, val_loader, test_loader
    """
    
    # Get transforms
    weak_transform = get_cifar10_transforms('weak')
    strong_transform = get_cifar10_transforms('strong')
    test_transform = get_cifar10_transforms('test')
    
    # Download dataset
    full_trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=None  # no transform initially
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    total_size = len(full_trainset)
    indices = np.random.permutation(total_size)
    
    # Split data: labeled, unlabeled, validation
    labeled_size = int(total_size * labeled_ratio)
    val_size = int(total_size * validation_split)
    unlabeled_size = total_size - labeled_size - val_size
    
    labeled_indices = indices[:labeled_size]
    val_indices = indices[labeled_size:labeled_size + val_size]
    unlabeled_indices = indices[labeled_size + val_size:]
    
    # Create datasets with appropriate transforms
    class DualTransformDataset:
        """Dataset that applies different transforms to the same data"""
        def __init__(self, dataset, indices, weak_transform, strong_transform):
            self.dataset = dataset
            self.indices = indices
            self.weak_transform = weak_transform
            self.strong_transform = strong_transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            data, target = self.dataset[self.indices[idx]]
            weak_data = self.weak_transform(data)
            strong_data = self.strong_transform(data)
            return weak_data, strong_data, target
    
    class SingleTransformDataset:
        """Dataset with single transform"""
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            data, target = self.dataset[self.indices[idx]]
            return self.transform(data), target
    
    # Create datasets
    labeled_set = SingleTransformDataset(full_trainset, labeled_indices, strong_transform)
    unlabeled_set = DualTransformDataset(full_trainset, unlabeled_indices, weak_transform, strong_transform)
    val_set = SingleTransformDataset(full_trainset, val_indices, test_transform)
    
    logging.info(f"labeled data: {len(labeled_set)} ({labeled_ratio*100:.1f}%)")
    logging.info(f"unlabeled data: {len(unlabeled_set)} ({(1-labeled_ratio-validation_split)*100:.1f}%)")
    logging.info(f"validation data: {len(val_set)} ({validation_split*100:.1f}%)")
    logging.info(f"test data: {len(testset)}")
    
    # Create data loaders
    labeled_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True, num_workers=0)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return labeled_loader, unlabeled_loader, val_loader, test_loader

# =============================================================================
# Training and evaluation functions
# =============================================================================

def evaluate_model(model, data_loader, detailed=False):
    """evaluate model performance"""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if detailed:
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    
    if detailed:
        return accuracy, all_predictions, all_targets
    else:
        return accuracy

def create_visualizations(all_results, save_dir, phase_name):
    """create visualization results"""
    
    # 1. performance comparison chart
    ratios = sorted(all_results.keys())
    means = [all_results[ratio]['mean'] for ratio in ratios]
    stds = [all_results[ratio]['std'] for ratio in ratios]
    
    ratio_labels = []
    for r in ratios:
        if r == 1.0:
            ratio_labels.append("100%\n(fully supervised)")
        else:
            ratio_labels.append(f"{r*100:.1f}%")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if phase_name == "phase1":
        colors = ['red' if r==1.0 else 'skyblue' for r in ratios]
        title = f'{phase_name}: ModifiedResNet18 on CIFAR-10\nsupervised learning baseline performance'
    elif phase_name == "phase2":
        colors = 'lightgreen'
        title = f'{phase_name}: Mean Teacher on CIFAR-10\nsemi-supervised learning performance'
    else:
        colors = 'orange'
        title = f'{phase_name}: Semi-supervised learning performance'
    
    bars = ax.bar(ratio_labels, means, yerr=stds, capsize=5, alpha=0.8, 
                  color=colors, edgecolor='navy')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('labeled data ratio', fontsize=12)
    ax.set_ylabel('test accuracy (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # add numerical labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.5, f'{mean:.1f}±{std:.1f}%', 
               ha='center', va='bottom', fontweight='bold')
    
    # add legend for phase1
    if phase_name == "phase1":
        full_patch = plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.8, label='fully supervised baseline')
        limited_patch = plt.Rectangle((0,0),1,1, facecolor='skyblue', alpha=0.8, label='limited labeled data')
        ax.legend(handles=[full_patch, limited_patch], loc='upper left')
    elif phase_name == "phase2":
        mean_teacher_patch = plt.Rectangle((0,0),1,1, facecolor='lightgreen', alpha=0.8, label='Mean Teacher')
        ax.legend(handles=[mean_teacher_patch], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{phase_name}_performance_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info("📈 visualization results generated")

# =============================================================================
# Common argument parser
# =============================================================================

def create_common_parser(description, default_epochs=100):
    """Create common argument parser"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--ratios', nargs='+', type=float, 
                       default=[0.1, 0.05, 0.01],
                       help='labeled data ratio')
    parser.add_argument('--seeds', nargs='+', type=int, 
                       default=[42, 123, 456],
                       help='random seeds list')
    parser.add_argument('--epochs', type=int, default=default_epochs,
                       help='training epochs')
    parser.add_argument('--no-save', action='store_true',
                       help='do not save results')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='directory to save log files')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='logging level')
    return parser 