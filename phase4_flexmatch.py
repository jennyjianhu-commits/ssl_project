#!/usr/bin/env python3
"""
Phase 4: FlexMatch implementation - CIFAR-10

experiment settings:
- dataset: CIFAR-10 (32×32, 3 channels, 10 classes)
- labeled ratio: 10%, 5%, 1%
- model: modified ResNet-18 (same as baseline for consistency)
- goal: implement FlexMatch with curriculum pseudo-labeling and dynamic threshold

Key differences from FixMatch:
- Uses class-wise adaptive thresholds based on learning status
- Implements curriculum pseudo-labeling with beta mapping functions
- Tracks which unlabeled samples have been assigned pseudo-labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import json
import os
import time
from datetime import datetime
import logging
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Import from parent directory (outside FlexMatch/)
import sys
sys.path.append("..")
from randaugment import RandAugmentMC
from ema import EMA
from optim import get_optimizer

# Import common utilities from current directory
from utils.common_utils import (
    setup_logging,
    set_seed,
    device,
    ModifiedResNet18,  # Keep using the same ResNet18 as other phases
    evaluate_model,
    create_visualizations,
    create_common_parser,
)

# Set high DPI for better figure quality
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12


# =============================================================================
# FlexMatch specific: Beta mapping functions
# =============================================================================
def get_mapping_function(mapping_type="convex"):
    """
    Get the beta mapping function for FlexMatch threshold adjustment

    Args:
        mapping_type: 'convex', 'concave', or 'linear'

    Returns:
        mapping function
    """
    if mapping_type == "convex":
        return lambda beta: beta / (2.0 - beta) if beta < 2.0 else beta
    elif mapping_type == "concave":
        return lambda beta: 2.0 - beta
    elif mapping_type == "linear":
        return lambda beta: beta
    else:
        raise ValueError(f"Unknown mapping type: {mapping_type}")


# =============================================================================
# FlexMatch data preparation
# =============================================================================


def prepare_flexmatch_data(
    labeled_ratio=0.1, batch_size=64, mu=7, validation_split=0.1
):
    """
    Prepare CIFAR-10 dataset for FlexMatch with dual augmentation

    Args:
        labeled_ratio: labeled data ratio
        batch_size: batch size for labeled data
        mu: ratio of unlabeled to labeled batch size
        validation_split: validation set ratio

    Returns:
        labeled_loader, unlabeled_loader, val_loader, test_loader
    """
    # CIFAR-10 normalization
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    # Weak augmentation
    weak_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Strong augmentation with RandAugment
    strong_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode="reflect"),
        RandAugmentMC(n=2, m=10),  # Use RandAugmentMC from parent directory
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Test transform
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Download datasets
    full_trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=None
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    total_size = len(full_trainset)
    indices = np.random.permutation(total_size)

    # Split data
    labeled_size = int(total_size * labeled_ratio)
    val_size = int(total_size * validation_split)
    unlabeled_size = total_size - labeled_size - val_size

    labeled_indices = indices[:labeled_size]
    val_indices = indices[labeled_size : labeled_size + val_size]
    unlabeled_indices = indices[labeled_size + val_size :]

    # Custom dataset classes
    class LabeledDataset:
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            data, target = self.dataset[self.indices[idx]]
            return self.transform(data), target

    class UnlabeledDataset:
        """Unlabeled dataset returns weak aug, strong aug, and index"""
        def __init__(self, dataset, indices, weak_transform, strong_transform):
            self.dataset = dataset
            self.indices = indices
            self.weak_transform = weak_transform
            self.strong_transform = strong_transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            data, _ = self.dataset[self.indices[idx]]
            weak_aug = self.weak_transform(data)
            strong_aug = self.strong_transform(data)
            return weak_aug, strong_aug, idx  # Return index for tracking

    class ValidationDataset:
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
    labeled_set = LabeledDataset(full_trainset, labeled_indices, weak_transform)
    unlabeled_set = UnlabeledDataset(
        full_trainset, unlabeled_indices, weak_transform, strong_transform
    )
    val_set = ValidationDataset(full_trainset, val_indices, test_transform)

    logging.info(f"labeled data: {len(labeled_set)} ({labeled_ratio * 100:.1f}%)")
    logging.info(f"unlabeled data: {len(unlabeled_set)} ({(unlabeled_size / total_size) * 100:.1f}%)")
    logging.info(f"validation data: {len(val_set)} ({validation_split * 100:.1f}%)")
    logging.info(f"test data: {len(testset)}")

    # Create data loaders
    labeled_loader = DataLoader(
        labeled_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    unlabeled_loader = DataLoader(
        unlabeled_set, batch_size=batch_size * mu, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return labeled_loader, unlabeled_loader, val_loader, test_loader


# =============================================================================
# FlexMatch training tracker
# =============================================================================
class FlexMatchTracker:
    """Training tracker for FlexMatch"""

    def __init__(self):
        self.train_losses = []
        self.train_losses_x = []
        self.train_losses_u = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epoch_times = []
        self.mask_ratios = []

    def update(self, train_loss, train_loss_x, train_loss_u, train_acc, val_acc, lr, epoch_time, mask_ratio):
        self.train_losses.append(train_loss)
        self.train_losses_x.append(train_loss_x)
        self.train_losses_u.append(train_loss_u)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
        self.mask_ratios.append(mask_ratio)

    def plot_metrics(self, title="FlexMatch Training Metrics"):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=18, fontweight="bold")

        # Total loss
        axes[0, 0].plot(self.train_losses, color="blue", linewidth=2)
        axes[0, 0].set_title("Total Loss", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("Epoch", fontsize=12)
        axes[0, 0].set_ylabel("Loss", fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)

        # Loss components
        axes[0, 1].plot(self.train_losses_x, label="Supervised", color="green", linewidth=2)
        axes[0, 1].plot(self.train_losses_u, label="Unsupervised", color="red", linewidth=2)
        axes[0, 1].set_title("Loss Components", fontsize=14, fontweight="bold")
        axes[0, 1].set_xlabel("Epoch", fontsize=12)
        axes[0, 1].set_ylabel("Loss", fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Accuracy curves
        axes[0, 2].plot(self.train_accuracies, label="Train", color="blue", linewidth=2)
        axes[0, 2].plot(self.val_accuracies, label="Validation", color="red", linewidth=2)
        axes[0, 2].set_title("Accuracy", fontsize=14, fontweight="bold")
        axes[0, 2].set_xlabel("Epoch", fontsize=12)
        axes[0, 2].set_ylabel("Accuracy (%)", fontsize=12)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Learning rate
        axes[1, 0].plot(self.learning_rates, color="orange", linewidth=2)
        axes[1, 0].set_title("Learning Rate", fontsize=14, fontweight="bold")
        axes[1, 0].set_xlabel("Epoch", fontsize=12)
        axes[1, 0].set_ylabel("LR", fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)

        # Mask ratio
        axes[1, 1].plot(self.mask_ratios, color="brown", linewidth=2)
        axes[1, 1].set_title("Pseudo-label Mask Ratio", fontsize=14, fontweight="bold")
        axes[1, 1].set_xlabel("Epoch", fontsize=12)
        axes[1, 1].set_ylabel("Mask Ratio", fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)

        # Training time
        axes[1, 2].plot(self.epoch_times, color="gray", linewidth=2)
        axes[1, 2].set_title("Training Time per Epoch", fontsize=14, fontweight="bold")
        axes[1, 2].set_xlabel("Epoch", fontsize=12)
        axes[1, 2].set_ylabel("Time (s)", fontsize=12)
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# =============================================================================
# FlexMatch training function
# =============================================================================
def train_flexmatch(
    model,
    labeled_loader,
    unlabeled_loader,
    val_loader,
    epochs=280,
    lr=0.03,
    weight_decay=5e-4,
    momentum=0.9,
    nesterov=True,
    lambda_u=1.0,
    threshold=0.95,
    mapping_type="convex",
    ema_decay=0.999,
    num_classes=10,
    early_stopping_patience=50,
):
    """
    Train model using FlexMatch algorithm

    Key Features:
    - Uses class-wise adaptive thresholds
    - Tracks learning status for curriculum learning
    - References external optimizer and EMA implementations
    """
    model = model.to(device)

    # Calculate total iterations for scheduler
    total_steps = len(unlabeled_loader) * epochs

    # Use external optimizer (from optim.py)
    optimizer, scheduler = get_optimizer(
        model=model,
        lr=lr,
        momentum=momentum,
        nesterov=nesterov,
        weight_decay=weight_decay,
        iterations=total_steps,
    )

    # Use external EMA (from ema.py)
    ema = EMA(model=model, decay=ema_decay, device=device)

    # FlexMatch: learning status tracking
    num_unlabeled = len(unlabeled_loader.dataset)
    learning_status = [-1] * num_unlabeled  # -1 means no pseudo-label yet

    # Get beta mapping function
    mapping_func = get_mapping_function(mapping_type)

    # Loss function
    criterion = nn.CrossEntropyLoss(reduction="none")

    # Tracking
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    tracker = FlexMatchTracker()

    logging.info(f"Start FlexMatch training, {epochs} epochs")
    logging.info(f"Threshold: {threshold}, λ_u: {lambda_u}, mapping: {mapping_type}")
    logging.info(f"Unlabeled samples: {num_unlabeled}")
    logging.info(f"Total steps: {total_steps}")
    logging.info("-" * 60)

    # Training loop
    model.train()
    labeled_iter = iter(labeled_loader)

    global_step = 0

    for epoch in range(epochs):
        start_time = time.time()

        epoch_loss = 0.0
        epoch_loss_x = 0.0
        epoch_loss_u = 0.0
        epoch_mask_ratio = 0.0
        epoch_correct = 0
        epoch_total = 0
        batch_count = 0

        # Calculate class-wise thresholds based on learning status
        cls_thresholds = torch.zeros(num_classes, device=device)
        counter = Counter(learning_status)
        num_unused = counter[-1]

        # FlexMatch curriculum learning
        if num_unused != num_unlabeled:
            max_counter = max([counter[c] for c in range(num_classes)])

            if max_counter < num_unused:
                sum_counter = sum([counter[c] for c in range(num_classes)])
                denominator = max(max_counter, num_unlabeled - sum_counter)
            else:
                denominator = max_counter

            for c in range(num_classes):
                beta = counter[c] / denominator if denominator > 0 else 0
                cls_thresholds[c] = mapping_func(beta) * threshold
        else:
            cls_thresholds = torch.ones(num_classes, device=device) * threshold

        # Train for one epoch
        for batch_idx, (inputs_u_w, inputs_u_s, u_idx) in enumerate(unlabeled_loader):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                inputs_x, targets_x = next(labeled_iter)

            # Move to device
            inputs_x = inputs_x.to(device)
            targets_x = targets_x.to(device)
            inputs_u_w = inputs_u_w.to(device)
            inputs_u_s = inputs_u_s.to(device)

            batch_size_x = inputs_x.size(0)

            # Forward pass
            all_inputs = torch.cat([inputs_x, inputs_u_w, inputs_u_s], dim=0)
            all_logits = model(all_inputs)

            logits_x = all_logits[:batch_size_x]
            logits_u_w = all_logits[batch_size_x : batch_size_x + inputs_u_w.size(0)]
            logits_u_s = all_logits[batch_size_x + inputs_u_w.size(0):]

            # Supervised loss
            loss_x = criterion(logits_x, targets_x).mean()

            # Calculate training accuracy on labeled data
            with torch.no_grad():
                _, predicted = logits_x.max(1)
                epoch_correct += predicted.eq(targets_x).sum().item()
                epoch_total += targets_x.size(0)

            # Generate pseudo-labels
            with torch.no_grad():
                pseudo_probs = F.softmax(logits_u_w, dim=1)
                max_probs, pseudo_labels = torch.max(pseudo_probs, dim=1)

                # Update learning status
                over_base_threshold = max_probs > threshold
                if over_base_threshold.any():
                    # Move to CPU for indexing
                    over_base_threshold_cpu = over_base_threshold.cpu()
                    sample_indices = u_idx[over_base_threshold_cpu].tolist()
                    pseudo_label_list = pseudo_labels[over_base_threshold].cpu().tolist()
                    for idx, label in zip(sample_indices, pseudo_label_list):
                        learning_status[idx] = label

            # FlexMatch: class-wise thresholds
            batch_threshold = torch.index_select(cls_thresholds, 0, pseudo_labels)
            mask = max_probs > batch_threshold

            # Unsupervised loss
            loss_u = (criterion(logits_u_s, pseudo_labels) * mask).mean()

            # Total loss
            loss = loss_x + lambda_u * loss_u

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update()

            # Track metrics
            epoch_loss += loss.item()
            epoch_loss_x += loss_x.item()
            epoch_loss_u += loss_u.item()
            epoch_mask_ratio += mask.float().mean().item()
            batch_count += 1
            global_step += 1

        # Validation
        avg_loss = epoch_loss / batch_count
        avg_loss_x = epoch_loss_x / batch_count
        avg_loss_u = epoch_loss_u / batch_count
        avg_mask_ratio = epoch_mask_ratio / batch_count
        train_acc = 100.0 * epoch_correct / epoch_total if epoch_total > 0 else 0.0

        val_acc = evaluate_model(ema.shadow, val_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - start_time

        tracker.update(avg_loss, avg_loss_x, avg_loss_u, train_acc, val_acc, current_lr, epoch_time, avg_mask_ratio)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logging.info(
                f"Epoch {epoch + 1:4d}/{epochs} | Iter {global_step:7d}/{total_steps} | "
                f"Loss: {avg_loss:.4f} | Lx: {avg_loss_x:.4f} | Lu: {avg_loss_u:.4f} | "
                f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
                f"Mask: {avg_mask_ratio*100:.1f}% | Unused: {num_unused}/{num_unlabeled} | "
                f"LR: {current_lr:.5f}"
            )

        # Early stopping
        if patience_counter >= early_stopping_patience:
            logging.info(
                f"Early stopping triggered! Validation accuracy hasn't improved for {early_stopping_patience} epochs"
            )
            break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        ema.shadow.load_state_dict(best_model_state)

    logging.info("-" * 60)
    logging.info(f"Training completed! Best val accuracy: {best_val_acc:.2f}%")

    # Print final learning status
    final_counter = Counter(learning_status)
    logging.info("\nFinal learning status:")
    for c in range(num_classes):
        logging.info(f"  Class {c}: {final_counter[c]} samples")
    logging.info(f"  Unused: {final_counter[-1]} samples")

    return ema.shadow, tracker, best_val_acc


# =============================================================================
# Phase 4 main experiment
# =============================================================================
def run_phase4_experiments(
    labeled_ratios=[0.1, 0.05, 0.01],
    seeds=[42, 123, 456],
    epochs=280,
    save_results=True,
):
    """Run Phase 4 FlexMatch experiments"""

    logging.info("🚀 Phase 4: CIFAR-10 FlexMatch experiment")
    logging.info("=" * 80)
    logging.info("Configuration:")
    logging.info("   Dataset: CIFAR-10")
    logging.info("   Model: Modified ResNet-18 (same as baseline)")
    logging.info("   Algorithm: FlexMatch (curriculum pseudo-labeling)")
    logging.info(f"   Labeled ratios: {[f'{r*100:.1f}%' for r in labeled_ratios]}")
    logging.info(f"   Random seeds: {seeds}")
    logging.info(f"   Training epochs: {epochs}")
    logging.info("=" * 80)

    all_results = defaultdict(list)
    detailed_results = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./results/phase4_flexmatch_results_{timestamp}"
    if save_results:
        os.makedirs(results_dir, exist_ok=True)

    # FlexMatch hyperparameters (from reference implementation)
    lambda_u = 1.0
    threshold = 0.95
    mu = 7
    mapping_type = "convex"

    total_experiments = len(labeled_ratios) * len(seeds)
    experiment_count = 0

    for labeled_ratio in labeled_ratios:
        ratio_name = f"{labeled_ratio * 100:.1f}%"
        logging.info(f"\n📊 Labeled ratio: {ratio_name}")
        logging.info("-" * 60)

        ratio_accuracies = []

        for seed in seeds:
            experiment_count += 1
            logging.info(f"\n🎯 Experiment {experiment_count}/{total_experiments} - seed: {seed}")

            try:
                set_seed(seed)

                # Prepare data
                logging.info("🔄 Preparing data...")
                labeled_loader, unlabeled_loader, val_loader, test_loader = prepare_flexmatch_data(
                    labeled_ratio=labeled_ratio, batch_size=64, mu=mu, validation_split=0.1
                )

                # Create model (ModifiedResNet18 from common_utils)
                logging.info("🏗️  Creating model...")
                model = ModifiedResNet18(num_classes=10)
                total_params = sum(p.numel() for p in model.parameters())
                logging.info(f"    Model parameters: {total_params:,}")

                # Train with FlexMatch
                logging.info("🏋️  Training...")
                trained_model, tracker, best_val_acc = train_flexmatch(
                    model=model,
                    labeled_loader=labeled_loader,
                    unlabeled_loader=unlabeled_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    lr=0.03,
                    weight_decay=5e-4,
                    momentum=0.9,
                    nesterov=True,
                    lambda_u=lambda_u,
                    threshold=threshold,
                    mapping_type=mapping_type,
                    ema_decay=0.999,
                    num_classes=10,
                    early_stopping_patience=50,
                )

                # Test evaluation
                test_acc, predictions, targets = evaluate_model(trained_model, test_loader, detailed=True)
                logging.info(f"✅ Test accuracy: {test_acc:.2f}%")

                ratio_accuracies.append(test_acc)
                experiment_result = {
                    "labeled_ratio": labeled_ratio,
                    "seed": seed,
                    "test_accuracy": test_acc,
                    "best_val_accuracy": best_val_acc,
                    "final_train_accuracy": tracker.train_accuracies[-1] if tracker.train_accuracies else 0.0,
                    "total_epochs": len(tracker.train_losses),
                    "total_params": total_params,
                    "lambda_u": lambda_u,
                    "threshold": threshold,
                    "mapping_type": mapping_type,
                }
                detailed_results.append(experiment_result)

                # Save training curve
                if save_results:
                    fig = tracker.plot_metrics(f"FlexMatch - CIFAR-10 - {ratio_name} - Seed {seed}")
                    plt.savefig(
                        f"{results_dir}/training_curve_{labeled_ratio:.3f}_{seed}.png",
                        dpi=150, bbox_inches="tight"
                    )
                    plt.close()

            except Exception as e:
                logging.error(f"❌ Experiment failed: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
                continue

        # Calculate statistics
        if ratio_accuracies:
            mean_acc = np.mean(ratio_accuracies)
            std_acc = np.std(ratio_accuracies)
            all_results[labeled_ratio] = {
                "mean": mean_acc,
                "std": std_acc,
                "all_scores": ratio_accuracies,
            }
            logging.info(f"\n📈 {ratio_name} result: {mean_acc:.2f}% ± {std_acc:.2f}%")

    # Final report
    logging.info("\n" + "=" * 80)
    logging.info("📊 Phase 4 FlexMatch Results")
    logging.info("=" * 80)

    if all_results:
        logging.info(f"\n{'Labeled Ratio':<15} | {'Mean Accuracy':<12} | {'Std':<6} | {'95% CI':<8}")
        logging.info("-" * 50)

        for labeled_ratio in labeled_ratios:
            if labeled_ratio in all_results:
                result = all_results[labeled_ratio]
                mean_acc = result["mean"]
                std_acc = result["std"]
                ci_95 = 1.96 * std_acc / np.sqrt(len(seeds))
                ratio_name = f"{labeled_ratio * 100:.1f}%"
                logging.info(f"{ratio_name:<15} | {mean_acc:10.2f}% | {std_acc:5.2f}% | ±{ci_95:4.2f}%")

        if save_results:
            create_visualizations(all_results, results_dir, "phase4_flexmatch")
            with open(f"{results_dir}/detailed_results.json", "w") as f:
                json.dump(detailed_results, f, indent=2)
            logging.info(f"\n💾 Results saved to: {results_dir}/")

    return all_results, detailed_results


def main():
    """Main function"""
    parser = create_common_parser(
        description="Phase 4: CIFAR-10 FlexMatch experiment",
        default_epochs=280,
    )

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())
    log_file = setup_logging("phase4_flexmatch", log_dir=args.log_dir, log_level=log_level)

    logging.info("Parameters:")
    logging.info(f"  Labeled ratios: {args.ratios}")
    logging.info(f"  Seeds: {args.seeds}")
    logging.info(f"  Epochs: {args.epochs}")
    logging.info(f"  Save results: {not args.no_save}")
    logging.info(f"  Log file: {log_file}")

    all_results, detailed_results = run_phase4_experiments(
        labeled_ratios=args.ratios,
        seeds=args.seeds,
        epochs=args.epochs,
        save_results=not args.no_save,
    )

    logging.info("\n🎉 Phase 4 completed!")

    for ratio in args.ratios:
        if ratio in all_results:
            result = all_results[ratio]
            logging.info(f"• {ratio*100:.1f}% labeled: {result['mean']:.2f}% ± {result['std']:.2f}%")


if __name__ == "__main__":
    main()
