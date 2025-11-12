#!/usr/bin/env python3
"""
Phase 3: FixMatch implementation - CIFAR-10

experiment settings:
- dataset: CIFAR-10 (32×32, 3 channels, 10 classes)
- labeled ratio: 10%, 5%, 1%
- model: modified ResNet-18 (same as baseline for consistency)
- goal: implement FixMatch with confidence thresholding and consistency regularization
"""

from cProfile import label
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
import time
from datetime import datetime
import logging
import math
from torch.optim.lr_scheduler import LambdaLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from randaugment import RandAugmentMC

# Import common utilities
from utils.common_utils import (
    setup_logging,
    set_seed,
    device,
    ModifiedResNet18,
    evaluate_model,
    create_visualizations,
    create_common_parser,
)

# Import EMA
import sys

sys.path.append("..")
from models.ema import ModelEMA

# Set high DPI and larger font size for better figure quality
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 14

# =============================================================================
# Improved learning rate scheduler with warmup
# =============================================================================
def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=7.0 / 16.0,
    last_epoch=-1,
):
    """
    Create a learning rate scheduler with warmup and cosine decay
    This is the same scheduler used in the original FixMatch implementation
    """

    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


# =============================================================================
# FixMatch specific data preparation
# =============================================================================
def prepare_fixmatch_data(labeled_ratio=0.1, batch_size=64, mu=7, validation_split=0.1):
    """
    Prepare CIFAR-10 dataset for FixMatch with proper augmentation strategy

    Args:
        labeled_ratio: labeled data ratio (0.01, 0.05, 0.1)
        batch_size: batch size for labeled data
        mu: ratio of unlabeled to labeled batch size
        validation_split: validation set ratio

    Returns:
        labeled_loader, unlabeled_loader, val_loader, test_loader
    """
    # CIFAR-10 normalization values
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    # Weak augmentation for labeled data and teacher predictions (same as original FixMatch)
    weak_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32, padding=int(32 * 0.125), padding_mode="reflect"
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Strong augmentation for student consistency training (same as original FixMatch)
    strong_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32, padding=int(32 * 0.125), padding_mode="reflect"
            ),
            RandAugmentMC(n=2, m=10),  # Use professional RandAugmentMC
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Test transform (no augmentation)
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    # Download datasets
    full_trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=None
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    total_size = len(full_trainset)
    indices = np.random.permutation(total_size)

    # Split data: labeled, unlabeled, validation
    labeled_size = int(total_size * labeled_ratio)
    val_size = int(total_size * validation_split)
    unlabeled_size = total_size - labeled_size - val_size

    labeled_indices = indices[:labeled_size]
    val_indices = indices[labeled_size : labeled_size + val_size]
    unlabeled_indices = indices[labeled_size + val_size :]

    # Custom dataset classes for FixMatch dual augmentation
    class FixMatchLabeledDataset:
        """Labeled dataset with weak augmentation"""

        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            data, target = self.dataset[self.indices[idx]]
            return self.transform(data), target

    class FixMatchUnlabeledDataset:
        """Unlabeled dataset with dual augmentation for FixMatch"""

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
            return weak_aug, strong_aug

    class ValidationDataset:
        """Validation dataset"""

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
    labeled_set = FixMatchLabeledDataset(full_trainset, labeled_indices, weak_transform)
    unlabeled_set = FixMatchUnlabeledDataset(
        full_trainset, unlabeled_indices, weak_transform, strong_transform
    )
    val_set = ValidationDataset(full_trainset, val_indices, test_transform)

    logging.info(f"labeled data: {len(labeled_set)} ({labeled_ratio * 100:.1f}%)")
    logging.info(
        f"unlabeled data: {len(unlabeled_set)} ({(unlabeled_size / total_size) * 100:.1f}%)"
    )
    logging.info(f"validation data: {len(val_set)} ({validation_split * 100:.1f}%)")
    logging.info(f"test data: {len(testset)}")

    # Calculate batch sizes for logging
    unlabeled_batch_size = batch_size * mu
    logging.info(f"labeled batch size: {batch_size}")
    logging.info(f"unlabeled batch size: {unlabeled_batch_size} (mu={mu})")

    # Create data loaders
    labeled_loader = DataLoader(
        labeled_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    unlabeled_loader = DataLoader(
        unlabeled_set, batch_size=batch_size * mu, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return labeled_loader, unlabeled_loader, val_loader, test_loader


# =============================================================================
# FixMatch training tracker
# =============================================================================
class FixMatchTracker:
    """FixMatch training process tracker"""

    def __init__(self):
        self.train_losses = []
        self.train_losses_x = []  # supervised loss
        self.train_losses_u = []  # unsupervised loss
        self.val_accuracies = []
        self.train_accuracies = []
        self.learning_rates = []
        self.epoch_times = []
        self.mask_ratios = []  # ratio of pseudo-labels above threshold

    def update(
        self,
        train_loss,
        train_loss_x,
        train_loss_u,
        val_acc,
        train_acc,
        lr,
        epoch_time,
        mask_ratio,
    ):
        self.train_losses.append(train_loss)
        self.train_losses_x.append(train_loss_x)
        self.train_losses_u.append(train_loss_u)
        self.val_accuracies.append(val_acc)
        self.train_accuracies.append(train_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
        self.mask_ratios.append(mask_ratio)

    def plot_metrics(self, title="FixMatch Training Metrics"):
        """plot training metrics specific to FixMatch"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=18, fontweight="bold")

        # total loss
        axes[0, 0].plot(self.train_losses, color="blue", linewidth=2)
        axes[0, 0].set_title("Total Loss", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[0, 0].set_ylabel("Loss", fontsize=12, fontweight="bold")
        axes[0, 0].grid(True, alpha=0.3)

        # supervised vs unsupervised loss
        axes[0, 1].plot(
            self.train_losses_x, label="Supervised Loss", color="green", linewidth=2
        )
        axes[0, 1].plot(
            self.train_losses_u, label="Unsupervised Loss", color="red", linewidth=2
        )
        axes[0, 1].set_title("Loss Components", fontsize=14, fontweight="bold")
        axes[0, 1].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[0, 1].set_ylabel("Loss", fontsize=12, fontweight="bold")
        axes[0, 1].legend(fontsize=11, frameon=True, shadow=True)
        axes[0, 1].grid(True, alpha=0.3)

        # Accuracy curve
        axes[0, 2].plot(self.val_accuracies, color="purple", label="Validation Accuracy", linewidth=2)
        axes[0, 2].plot(self.train_accuracies, color="blue", label="Training Accuracy", linewidth=2)
        axes[0, 2].set_title("Accuracy", fontsize=14, fontweight="bold")
        axes[0, 2].set_xlabel("Epoch", fontsize=12)
        axes[0, 2].set_ylabel("Accuracy (%)", fontsize=12)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # learning rate
        axes[1, 0].plot(self.learning_rates, color="orange", linewidth=2)
        axes[1, 0].set_title("Learning Rate", fontsize=14, fontweight="bold")
        axes[1, 0].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("LR", fontsize=12, fontweight="bold")
        axes[1, 0].grid(True, alpha=0.3)

        # mask ratio (confidence threshold effectiveness)
        axes[1, 1].plot(self.mask_ratios, color="brown", linewidth=2)
        axes[1, 1].set_title(
            "Pseudo-label Confidence Mask Ratio", fontsize=14, fontweight="bold"
        )
        axes[1, 1].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[1, 1].set_ylabel("Mask Ratio", fontsize=12, fontweight="bold")
        axes[1, 1].grid(True, alpha=0.3)

        # training time per epoch
        axes[1, 2].plot(self.epoch_times, color="gray", linewidth=2)
        axes[1, 2].set_title("Training Time per Epoch", fontsize=14, fontweight="bold")
        axes[1, 2].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[1, 2].set_ylabel("Time (s)", fontsize=12, fontweight="bold")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# =============================================================================
# FixMatch training function
# =============================================================================
def train_fixmatch(
    model,
    labeled_loader,
    unlabeled_loader,
    val_loader,
    epochs=280,
    lr=0.03,
    weight_decay=5e-4,
    lambda_u=1.0,
    threshold=0.95,
    early_stopping_patience=50,
    use_ema=True,
    ema_decay=0.999,
):
    """
    Train model using FixMatch algorithm with EMA support

    Args:
        model: Modified ResNet-18 model
        labeled_loader: labeled data loader
        unlabeled_loader: unlabeled data loader
        val_loader: validation data loader
        epochs: training epochs
        lr: learning rate
        weight_decay: weight decay
        lambda_u: coefficient for unsupervised loss
        threshold: confidence threshold for pseudo-labels
        early_stopping_patience: early stopping patience
        use_ema: whether to use EMA model
        ema_decay: EMA decay rate

    Returns:
        trained model, tracker, best validation accuracy
    """

    model = model.to(device)

    # FixMatch uses SGD with momentum as in the original paper
    no_decay = ["bias", "bn"]
    grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.SGD(grouped_parameters, lr=lr, momentum=0.9, nesterov=True)

    # Calculate total steps for scheduler (same as original FixMatch)
    total_steps = len(unlabeled_loader) * epochs
    warmup_steps = int(0.05 * total_steps)  # 5% warmup for better training stability

    # Use the same scheduler as original FixMatch implementation
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Initialize EMA model if enabled
    ema_model = None
    if use_ema:
        # Create a simple args object for EMA
        class Args:
            def __init__(self):
                self.device = device

        args = Args()
        ema_model = ModelEMA(args, model, ema_decay)
        logging.info(f"✅ EMA enabled with decay rate: {ema_decay}")
    else:
        logging.info("⚠️  EMA disabled")

    # early stopping mechanism
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None

    # training tracker
    tracker = FixMatchTracker()

    logging.info(f"Start FixMatch training, {epochs} epochs")
    logging.info(f"Confidence threshold: {threshold}, λ_u: {lambda_u}")
    logging.info("-" * 60)

    # Create iterators
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    for epoch in range(epochs):
        start_time = time.time()

        # Training phase
        model.train()
        epoch_loss = 0.0
        epoch_loss_x = 0.0  # supervised loss
        epoch_loss_u = 0.0  # unsupervised loss
        epoch_mask_ratio = 0.0
        epoch_correct = 0
        epoch_total = 0
        batch_count = 0

        # Training loop - iterate through unlabeled data
        for batch_idx in range(len(unlabeled_loader)):
            try:
                # Get labeled data
                inputs_x, targets_x = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                inputs_x, targets_x = next(labeled_iter)

            try:
                # Get unlabeled data (weak and strong augmentations)
                (inputs_u_w, inputs_u_s) = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                (inputs_u_w, inputs_u_s) = next(unlabeled_iter)

            # Move to device
            inputs_x = inputs_x.to(device)
            targets_x = targets_x.to(device)
            inputs_u_w = inputs_u_w.to(device)
            inputs_u_s = inputs_u_s.to(device)

            batch_size = inputs_x.size(0)

            # Interleave labeled and unlabeled data for efficient computation
            all_inputs = torch.cat([inputs_x, inputs_u_w, inputs_u_s], dim=0)
            all_logits = model(all_inputs)

            # Split logits
            logits_x = all_logits[:batch_size]
            logits_u_w = all_logits[batch_size : batch_size + inputs_u_w.size(0)]
            logits_u_s = all_logits[batch_size + inputs_u_w.size(0) :]

            # Supervised loss
            loss_x = F.cross_entropy(logits_x, targets_x)

            # Calculate training accuracy for supervised data
            with torch.no_grad():
                _, predicted = logits_x.max(1)
                epoch_correct += predicted.eq(targets_x).sum().item()
                epoch_total += targets_x.size(0)

            # Generate pseudo-labels from weakly augmented unlabeled data
            with torch.no_grad():
                pseudo_probs = F.softmax(logits_u_w, dim=1)
                max_probs, pseudo_labels = torch.max(pseudo_probs, dim=1)
                mask = max_probs.ge(threshold).float()

                # Note: We cannot calculate true pseudo-label accuracy without true labels
                # The mask ratio serves as a proxy for confidence threshold effectiveness

            # Unsupervised loss (consistency regularization)
            loss_u = (
                F.cross_entropy(logits_u_s, pseudo_labels, reduction="none") * mask
            ).mean()

            # Total loss
            loss = loss_x + lambda_u * loss_u

            # Check for NaN or infinite values
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(
                    f"⚠️  NaN/Inf loss detected at epoch {epoch + 1}, batch {batch_idx}"
                )
                continue

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update EMA model if enabled
            if ema_model is not None:
                ema_model.update(model)

            # Track metrics
            epoch_loss += loss.item()
            epoch_loss_x += loss_x.item()
            epoch_loss_u += loss_u.item()
            epoch_mask_ratio += mask.mean().item()
            batch_count += 1

        # Check if we have valid batches
        if batch_count == 0:
            logging.error(f"❌ No valid batches in epoch {epoch + 1}")
            continue

        # Average metrics over batches
        avg_loss = epoch_loss / batch_count
        avg_loss_x = epoch_loss_x / batch_count
        avg_loss_u = epoch_loss_u / batch_count
        avg_mask_ratio = epoch_mask_ratio / batch_count
        train_acc = 100.0 * epoch_correct / epoch_total if epoch_total > 0 else 0.0
        # Convert mask ratio to percentage (what % of unlabeled data exceeded threshold)
        mask_ratio_percent = avg_mask_ratio * 100

        # Validation phase - use EMA model if available
        if ema_model is not None:
            val_acc = evaluate_model(ema_model.ema, val_loader)
        else:
            val_acc = evaluate_model(model, val_loader)

        # Update learning rate (step per batch for step-based scheduler)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Record metrics
        epoch_time = time.time() - start_time
        tracker.update(
            avg_loss,
            avg_loss_x,
            avg_loss_u,
            val_acc,
            train_acc,
            current_lr,
            epoch_time,
            avg_mask_ratio,
        )

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Print progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logging.info(
                f"Epoch {epoch + 1:4d}/{epochs} | "
                f"Total Loss: {avg_loss:.4f} | "
                f"Sup Loss: {avg_loss_x:.4f} | "
                f"Unsup Loss: {avg_loss_u:.4f} | "
                f"λ_u: {lambda_u:.3f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Acc: {val_acc:.2f}% | "
                f"Mask: {mask_ratio_percent:.1f}% | "
                f"LR: {current_lr:.5f} | "
                f"Time: {epoch_time:.2f}s"
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
        if ema_model is not None:
            # Also update EMA model with best state
            ema_model.ema.load_state_dict(best_model_state)

    logging.info("-" * 60)
    logging.info(
        f"FixMatch training completed! Best validation accuracy: {best_val_acc:.2f}%"
    )

    # Return EMA model if available, otherwise return regular model
    final_model = ema_model.ema if ema_model is not None else model
    return final_model, tracker, best_val_acc


# =============================================================================
# Phase 3 main experiment
# =============================================================================


def run_phase3_experiments(
    labeled_ratios=[0.1, 0.05, 0.01],
    seeds=[42, 123, 456],
    epochs=280,
    save_results=True,
):
    """run Phase 3 FixMatch complete experiment"""

    logging.info("🚀 Phase 3: CIFAR-10 FixMatch semi-supervised learning experiment")
    logging.info("=" * 80)
    logging.info("experiment configuration:")
    logging.info("   dataset: CIFAR-10")
    logging.info("   model: Modified ResNet-18 (consistent with baseline)")
    logging.info(
        "   algorithm: FixMatch (consistency regularization + pseudo-labeling)"
    )
    logging.info(f"   labeled ratio: {[f'{r * 100:.1f}%' for r in labeled_ratios]}")
    logging.info(f"   random seeds: {seeds}")
    logging.info(f"   training epochs: {epochs}")
    logging.info("=" * 80)

    # results storage
    all_results = defaultdict(list)
    detailed_results = []

    # create results save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./results/phase3_fixmatch_results_{timestamp}"
    if save_results:
        os.makedirs(results_dir, exist_ok=True)

    total_experiments = len(labeled_ratios) * len(seeds)
    experiment_count = 0

    # FixMatch hyperparameters (from paper)
    lambda_u = 1.0  # unsupervised loss weight
    threshold = 0.95  # confidence threshold
    mu = 7  # ratio of unlabeled to labeled batch size

    for labeled_ratio in labeled_ratios:
        ratio_name = f"{labeled_ratio * 100:.1f}%"
        logging.info(f"\n📊 labeled ratio: {ratio_name}")
        logging.info("-" * 60)

        ratio_accuracies = []

        for seed in seeds:
            experiment_count += 1
            logging.info(
                f"\n🎯 experiment {experiment_count}/{total_experiments} - seed: {seed}"
            )

            try:
                # set random seed
                set_seed(seed)

                # prepare data
                logging.info("🔄 prepare FixMatch data...")
                labeled_loader, unlabeled_loader, val_loader, test_loader = (
                    prepare_fixmatch_data(
                        labeled_ratio=labeled_ratio,
                        batch_size=64,
                        mu=mu,
                        validation_split=0.1,
                    )
                )

                # create model (same as baseline for consistency)
                logging.info("🏗️  create ModifiedResNet18...")
                model = ModifiedResNet18(num_classes=10)
                total_params = sum(p.numel() for p in model.parameters())
                logging.info(f"    model parameters: {total_params:,}")

                # train model with FixMatch
                logging.info("🏋️  start FixMatch training...")
                trained_model, tracker, best_val_acc = train_fixmatch(
                    model=model,
                    labeled_loader=labeled_loader,
                    unlabeled_loader=unlabeled_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    lr=0.03,
                    weight_decay=5e-4,
                    lambda_u=lambda_u,
                    threshold=threshold,
                    early_stopping_patience=50,
                    use_ema=True,  # Enable EMA
                    ema_decay=0.999,  # Standard EMA decay rate
                )

                # test evaluation
                test_acc, predictions, targets = evaluate_model(
                    trained_model, test_loader, detailed=True
                )

                logging.info(f"✅ final test accuracy: {test_acc:.2f}%")

                # save results
                ratio_accuracies.append(test_acc)
                experiment_result = {
                    "labeled_ratio": labeled_ratio,
                    "seed": seed,
                    "test_accuracy": test_acc,
                    "best_val_accuracy": best_val_acc,
                    "total_epochs": len(tracker.train_losses),
                    "total_params": total_params,
                    "lambda_u": lambda_u,
                    "threshold": threshold,
                    "final_mask_ratio": tracker.mask_ratios[-1]
                    if tracker.mask_ratios
                    else 0.0,
                }
                detailed_results.append(experiment_result)

                # save training curve
                if save_results:
                    fig = tracker.plot_metrics(
                        f"FixMatch - CIFAR-10 - {ratio_name} - Seed {seed}"
                    )
                    plt.savefig(
                        f"{results_dir}/fixmatch_training_curve_{labeled_ratio:.3f}_{seed}.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    plt.close()

            except Exception as e:
                logging.error(
                    f"❌ Experiment failed for {ratio_name} labeled ratio, seed {seed}: {str(e)}"
                )
                logging.error(f"   Error type: {type(e).__name__}")
                import traceback

                logging.error(f"   Traceback: {traceback.format_exc()}")
                continue

        # calculate statistics only if we have results
        if ratio_accuracies:
            mean_acc = np.mean(ratio_accuracies)
            std_acc = np.std(ratio_accuracies)
            all_results[labeled_ratio] = {
                "mean": mean_acc,
                "std": std_acc,
                "all_scores": ratio_accuracies,
            }

            logging.info(
                f"\n📈 {ratio_name} average result: {mean_acc:.2f}% ± {std_acc:.2f}%"
            )
        else:
            logging.warning(
                f"⚠️  No successful experiments for {ratio_name} labeled ratio"
            )

    # generate final report
    logging.info("\n" + "=" * 80)
    logging.info("📊 Phase 3 FixMatch final results summary")
    logging.info("=" * 80)

    if all_results:
        logging.info(
            f"\n{'Labeled Ratio':<15} | {'Mean Accuracy':<12} | {'Std':<6} | {'95% CI':<8}"
        )
        logging.info("-" * 50)

        for labeled_ratio in labeled_ratios:
            if labeled_ratio in all_results:
                result = all_results[labeled_ratio]
                mean_acc = result["mean"]
                std_acc = result["std"]
                ci_95 = 1.96 * std_acc / np.sqrt(len(seeds))

                ratio_name = f"{labeled_ratio * 100:.1f}%"
                logging.info(
                    f"{ratio_name:<15} | {mean_acc:10.2f}% | {std_acc:5.2f}% | ±{ci_95:4.2f}%"
                )
            else:
                ratio_name = f"{labeled_ratio * 100:.1f}%"
                logging.info(
                    f"{ratio_name:<15} | {'FAILED':<12} | {'N/A':<6} | {'N/A':<8}"
                )

        # generate visualization
        if save_results:
            create_visualizations(all_results, results_dir, "phase3_fixmatch")

            # save detailed results
            with open(f"{results_dir}/detailed_results.json", "w") as f:
                json.dump(detailed_results, f, indent=2)

            logging.info(f"\n💾 results saved to: {results_dir}/")
    else:
        logging.error("❌ No successful experiments completed!")

    return all_results, detailed_results


def main():
    """main function"""
    parser = create_common_parser(
        description="Phase 3: CIFAR-10 FixMatch semi-supervised learning experiment",
        default_epochs=280,
    )

    args = parser.parse_args()

    # setup logging
    log_level = getattr(logging, args.log_level.upper())
    log_file = setup_logging(
        "phase3_fixmatch", log_dir=args.log_dir, log_level=log_level
    )

    # log experiment parameters
    logging.info("Experiment parameters:")
    logging.info(f"  labeled ratios: {args.ratios}")
    logging.info(f"  random seeds: {args.seeds}")
    logging.info(f"  training epochs: {args.epochs}")
    logging.info(f"  save results: {not args.no_save}")
    logging.info(f"  log file: {log_file}")

    # run experiment
    all_results, detailed_results = run_phase3_experiments(
        labeled_ratios=args.ratios,
        seeds=args.seeds,
        epochs=args.epochs,
        save_results=not args.no_save,
    )

    logging.info("\n🎉 Phase 3 FixMatch experiment completed!")
    logging.info("\n📝 key findings:")

    # print key conclusions
    for ratio in args.ratios:
        if ratio in all_results:
            result = all_results[ratio]
            logging.info(
                f"• {ratio * 100:.1f}% labeled data: {result['mean']:.2f}% ± {result['std']:.2f}%"
            )


if __name__ == "__main__":
    main()
