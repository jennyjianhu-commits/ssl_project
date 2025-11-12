#!/usr/bin/env python3
"""
Phase 2: Mean Teacher experiment - CIFAR-10

experiment settings:
- dataset: CIFAR-10 (32×32, 3 channels, 10 classes)
- labeled ratio: 10%, 5%, 1%
- model: modified ResNet-18 (student + teacher with EMA)
- goal: implement and evaluate Mean Teacher semi-supervised learning
"""

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

# Import common utilities
from utils.common_utils import (
    setup_logging,
    set_seed,
    device,
    ModifiedResNet18,
    prepare_semi_supervised_data,
    evaluate_model,
    create_visualizations,
    create_common_parser,
)

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
# Improved Learning Rate Schedulers
# =============================================================================
class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine decay

    Args:
        optimizer: optimizer
        warmup_epochs: number of warmup epochs
        total_epochs: total training epochs
        min_lr: minimum learning rate (default: 1e-6)
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, epoch):
        """Update learning rate"""
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


class OneCycleScheduler:
    """
    OneCycle learning rate scheduler for better convergence

    Args:
        optimizer: optimizer
        total_epochs: total training epochs
        max_lr: maximum learning rate
        min_lr: minimum learning rate (default: 1e-6)
        pct_start: percentage of epochs for warmup (default: 0.3)
    """

    def __init__(self, optimizer, total_epochs, max_lr, min_lr=1e-6, pct_start=0.3):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.pct_start = pct_start
        self.warmup_epochs = int(total_epochs * pct_start)

    def step(self, epoch):
        """Update learning rate"""
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.min_lr + (self.max_lr - self.min_lr) * epoch / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


def create_improved_scheduler(
    optimizer, epochs, scheduler_type="warmup_cosine", warmup_epochs=15, max_lr=None
):
    """
    Create improved learning rate scheduler

    Args:
        optimizer: optimizer
        epochs: total epochs
        scheduler_type: 'warmup_cosine', 'onecycle', or 'cosine'
        warmup_epochs: warmup epochs (for warmup_cosine)
        max_lr: maximum learning rate (for onecycle)

    Returns:
        scheduler
    """
    if scheduler_type == "warmup_cosine":
        return WarmupCosineScheduler(optimizer, warmup_epochs, epochs)
    elif scheduler_type == "onecycle":
        if max_lr is None:
            max_lr = optimizer.param_groups[0]["lr"] * 10  # 10x base lr
        return OneCycleScheduler(optimizer, epochs, max_lr)
    elif scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# =============================================================================
# Mean Teacher specific components
# =============================================================================
def update_ema_variables(student_model, teacher_model, alpha, global_step):
    """
    Update teacher model weights using exponential moving average

    Args:
        student_model: student model
        teacher_model: teacher model (EMA)
        alpha: EMA smoothing coefficient
        global_step: current training step
    """
    # Use true average until exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)

    for teacher_param, student_param in zip(
        teacher_model.parameters(), student_model.parameters()
    ):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)


def get_consistency_weight(epoch, labeled_ratio=None, rampup_length=80):
    """
    Get consistency weight with ramp-up, adjusted for different labeled ratios

    Args:
        epoch: current epoch
        labeled_ratio: labeled data ratio (e.g., 0.1 for 10%)
        rampup_length: ramp-up length in epochs (default: 80, will be overridden by labeled_ratio if provided)

    Returns:
        consistency weight (0 to 1)
    """
    # Adjust rampup_length based on labeled ratio if provided
    if labeled_ratio is not None:
        # More labeled data -> shorter rampup (faster convergence)
        # Less labeled data -> longer rampup (more careful training)
        if labeled_ratio >= 0.1:  # 10% or more
            rampup_length = 50
        elif labeled_ratio >= 0.05:  # 5%
            rampup_length = 70
        elif labeled_ratio >= 0.01:  # 1%
            rampup_length = 80
        else:  # less than 1%
            rampup_length = 100

    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(epoch, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def softmax_mse_loss(student_logits, teacher_logits):
    """
    MSE loss between softmax outputs

    Args:
        student_logits: student model logits
        teacher_logits: teacher model logits

    Returns:
        MSE loss
    """
    student_softmax = F.softmax(student_logits, dim=1)
    teacher_softmax = F.softmax(teacher_logits, dim=1)
    return F.mse_loss(student_softmax, teacher_softmax)


# =============================================================================
# Mean Teacher training tracker
# =============================================================================
class MeanTeacherTracker:
    """training process tracker for Mean Teacher"""

    def __init__(self):
        self.train_losses = []
        self.supervised_losses = []
        self.consistency_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.teacher_val_accuracies = []
        self.learning_rates = []
        self.consistency_weights = []
        self.epoch_times = []

    def update(
        self,
        train_loss,
        sup_loss,
        cons_loss,
        train_acc,
        val_acc,
        teacher_val_acc,
        lr,
        cons_weight,
        epoch_time,
    ):
        self.train_losses.append(train_loss)
        self.supervised_losses.append(sup_loss)
        self.consistency_losses.append(cons_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.teacher_val_accuracies.append(teacher_val_acc)
        self.learning_rates.append(lr)
        self.consistency_weights.append(cons_weight)
        self.epoch_times.append(epoch_time)

    def plot_metrics(self, title="Mean Teacher Training Metrics"):
        """plot training metrics"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=18, fontweight="bold")

        # loss curves
        axes[0, 0].plot(
            self.train_losses, color="blue", label="Total Loss", linewidth=2
        )
        axes[0, 0].plot(
            self.supervised_losses, color="red", label="Supervised Loss", linewidth=2
        )
        axes[0, 0].plot(
            self.consistency_losses,
            color="green",
            label="Consistency Loss",
            linewidth=2,
        )
        axes[0, 0].set_title("Training Losses", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[0, 0].set_ylabel("Loss", fontsize=12, fontweight="bold")
        axes[0, 0].legend(fontsize=11, frameon=True, shadow=True)
        axes[0, 0].grid(True, alpha=0.3)

        # accuracy curves
        axes[0, 1].plot(self.train_accuracies, label="Train", color="blue", linewidth=2)
        axes[0, 1].plot(
            self.val_accuracies, label="Student Val", color="red", linewidth=2
        )
        axes[0, 1].plot(
            self.teacher_val_accuracies,
            label="Teacher Val",
            color="orange",
            linewidth=2,
        )
        axes[0, 1].set_title("Accuracy", fontsize=14, fontweight="bold")
        axes[0, 1].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[0, 1].set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
        axes[0, 1].legend(fontsize=11, frameon=True, shadow=True)
        axes[0, 1].grid(True, alpha=0.3)

        # learning rate
        axes[1, 0].plot(self.learning_rates, color="green", linewidth=2)
        axes[1, 0].set_title("Learning Rate", fontsize=14, fontweight="bold")
        axes[1, 0].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("LR", fontsize=12, fontweight="bold")
        axes[1, 0].grid(True, alpha=0.3)

        # consistency weight
        axes[1, 1].plot(self.consistency_weights, color="purple", linewidth=2)
        axes[1, 1].set_title("Consistency Weight", fontsize=14, fontweight="bold")
        axes[1, 1].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[1, 1].set_ylabel("Weight", fontsize=12, fontweight="bold")
        axes[1, 1].grid(True, alpha=0.3)

        # training time per epoch
        axes[2, 0].plot(self.epoch_times, color="orange", linewidth=2)
        axes[2, 0].set_title("Training Time per Epoch", fontsize=14, fontweight="bold")
        axes[2, 0].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[2, 0].set_ylabel("Time (s)", fontsize=12, fontweight="bold")
        axes[2, 0].grid(True, alpha=0.3)

        # loss comparison
        axes[2, 1].plot(
            self.supervised_losses, color="red", label="Supervised", linewidth=2
        )
        axes[2, 1].plot(
            self.consistency_losses, color="green", label="Consistency", linewidth=2
        )
        axes[2, 1].set_title("Loss Components", fontsize=14, fontweight="bold")
        axes[2, 1].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[2, 1].set_ylabel("Loss", fontsize=12, fontweight="bold")
        axes[2, 1].legend(fontsize=11, frameon=True, shadow=True)
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# =============================================================================
# Mean Teacher training function
# =============================================================================
def train_mean_teacher(
    student_model,
    teacher_model,
    labeled_loader,
    unlabeled_loader,
    val_loader,
    epochs=100,
    lr=0.001,
    weight_decay=5e-4,
    ema_alpha=0.999,
    consistency_weight_max=10.0,
    early_stopping_patience=15,
    scheduler_type="warmup_cosine",
    warmup_epochs=10,
    max_lr=None,
    labeled_ratio=None,
):
    """
    Train Mean Teacher model

    Args:
        student_model: student model
        teacher_model: teacher model (EMA)
        labeled_loader: labeled data loader
        unlabeled_loader: unlabeled data loader
        val_loader: validation data loader
        epochs: number of training epochs
        lr: learning rate
        weight_decay: weight decay
        ema_alpha: EMA smoothing coefficient
        consistency_weight_max: maximum consistency weight
        early_stopping_patience: early stopping patience
        labeled_ratio: labeled data ratio for adaptive ramp-up

    Returns:
        trained student model, tracker, best validation accuracy
    """

    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)

    # Initialize teacher with student weights
    for teacher_param, student_param in zip(
        teacher_model.parameters(), student_model.parameters()
    ):
        teacher_param.data.copy_(student_param.data)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = create_improved_scheduler(
        optimizer,
        epochs,
        scheduler_type=scheduler_type,
        warmup_epochs=warmup_epochs,
        max_lr=max_lr,
    )

    # add mixed precision training
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # early stopping mechanism
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None

    # training tracker
    tracker = MeanTeacherTracker()

    logging.info(f"start Mean Teacher training, {epochs} epochs")
    logging.info(
        f"EMA alpha: {ema_alpha}, max consistency weight: {consistency_weight_max}"
    )
    logging.info(f"Learning rate scheduler: {scheduler_type}")
    if scheduler_type == "warmup_cosine":
        logging.info(f"Warmup epochs: {warmup_epochs}")
    elif scheduler_type == "onecycle":
        logging.info(f"Max learning rate: {max_lr}")
    if scaler:
        logging.info("Using mixed precision training")
    logging.info("-" * 60)

    global_step = 0

    for epoch in range(epochs):
        start_time = time.time()

        # training phase
        student_model.train()
        teacher_model.train()

        train_loss = 0.0
        supervised_loss = 0.0
        consistency_loss = 0.0
        train_correct = 0
        train_total = 0

        # get consistency weight
        cons_weight = (
            get_consistency_weight(epoch, labeled_ratio=labeled_ratio)
            * consistency_weight_max
        )

        # combine labeled and unlabeled data
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        # determine number of batches (use the longer one)
        labeled_batches = len(labeled_loader)
        unlabeled_batches = len(unlabeled_loader)
        total_batches = max(labeled_batches, unlabeled_batches)

        for batch_idx in range(total_batches):
            # get labeled batch
            try:
                labeled_data, labeled_target = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_data, labeled_target = next(labeled_iter)

            # get unlabeled batch
            try:
                weak_data, strong_data, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                weak_data, strong_data, _ = next(unlabeled_iter)

            labeled_data, labeled_target = (
                labeled_data.to(device),
                labeled_target.to(device),
            )
            weak_data, strong_data = weak_data.to(device), strong_data.to(device)

            optimizer.zero_grad()

            # use mixed precision training
            if scaler:
                with torch.amp.autocast("cuda"):
                    # supervised loss on labeled data
                    labeled_output = student_model(labeled_data)
                    sup_loss = criterion(labeled_output, labeled_target)

                    # consistency loss on unlabeled data
                    with torch.no_grad():
                        teacher_weak_output = teacher_model(weak_data)

                    student_strong_output = student_model(strong_data)
                    cons_loss = softmax_mse_loss(
                        student_strong_output, teacher_weak_output
                    )

                    # total loss
                    total_loss = sup_loss + cons_weight * cons_loss

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # supervised loss on labeled data
                labeled_output = student_model(labeled_data)
                sup_loss = criterion(labeled_output, labeled_target)

                # consistency loss on unlabeled data
                with torch.no_grad():
                    teacher_weak_output = teacher_model(weak_data)

                student_strong_output = student_model(strong_data)
                cons_loss = softmax_mse_loss(student_strong_output, teacher_weak_output)

                # total loss
                total_loss = sup_loss + cons_weight * cons_loss

                total_loss.backward()
                optimizer.step()

            # update teacher model
            global_step += 1
            update_ema_variables(student_model, teacher_model, ema_alpha, global_step)

            # record metrics
            train_loss += total_loss.item()
            supervised_loss += sup_loss.item()
            consistency_loss += cons_loss.item()

            _, predicted = labeled_output.max(1)
            train_total += labeled_target.size(0)
            train_correct += predicted.eq(labeled_target).sum().item()

        # validation phase
        student_val_acc = evaluate_model(student_model, val_loader)
        teacher_val_acc = evaluate_model(teacher_model, val_loader)
        train_acc = 100.0 * train_correct / train_total

        # update learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(epoch)

        # record metrics
        epoch_time = time.time() - start_time
        avg_train_loss = train_loss / total_batches
        avg_sup_loss = supervised_loss / total_batches
        avg_cons_loss = consistency_loss / total_batches

        tracker.update(
            avg_train_loss,
            avg_sup_loss,
            avg_cons_loss,
            train_acc,
            student_val_acc,
            teacher_val_acc,
            current_lr,
            cons_weight,
            epoch_time,
        )

        # early stopping check (use teacher validation accuracy)
        if teacher_val_acc > best_val_acc:
            best_val_acc = teacher_val_acc
            patience_counter = 0
            best_model_state = student_model.state_dict().copy()
        else:
            patience_counter += 1

        # print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logging.info(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Total Loss: {avg_train_loss:.4f} | "
                f"Sup Loss: {avg_sup_loss:.4f} | "
                f"Cons Loss: {avg_cons_loss:.4f} | "
                f"Cons Weight: {cons_weight:.2f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Student Val: {student_val_acc:.2f}% | "
                f"Teacher Val: {teacher_val_acc:.2f}% | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.2f}s"
            )

        # early stopping
        if patience_counter >= early_stopping_patience:
            logging.info(
                f"Early stop trigger! Teacher validation accuracy hasn't improved for {early_stopping_patience} epochs"
            )
            break

    # load best model
    if best_model_state:
        student_model.load_state_dict(best_model_state)

    logging.info("-" * 60)
    logging.info(
        f"Training completed! Best teacher validation accuracy: {best_val_acc:.2f}%"
    )

    return student_model, tracker, best_val_acc


# =============================================================================
# Phase 2 main experiment
# =============================================================================
def run_phase2_experiments(
    labeled_ratios=[0.1, 0.05, 0.01],
    seeds=[42, 123, 456],
    epochs=100,
    save_results=True,
):
    """run Phase 2 complete experiment"""

    logging.info(
        "🚀 Phase 2: CIFAR-10 Mean Teacher semi-supervised learning experiment"
    )
    logging.info("=" * 80)
    logging.info("experiment configuration:")
    logging.info("   dataset: CIFAR-10")
    logging.info("   model: Modified ResNet-18 (Student + Teacher)")
    logging.info("   method: Mean Teacher with EMA")
    logging.info(f"   labeled ratio: {[f'{r * 100:.1f}%' for r in labeled_ratios]}")
    logging.info(f"   random seeds: {seeds}")
    logging.info(f"   training epochs: {epochs}")
    logging.info("=" * 80)

    # results storage
    all_results = defaultdict(list)
    detailed_results = []

    # create results save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./results/phase2_mean_teacher_results_{timestamp}"
    if save_results:
        os.makedirs(results_dir, exist_ok=True)

    total_experiments = len(labeled_ratios) * len(seeds)
    experiment_count = 0

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

            # set random seed
            set_seed(seed)

            # prepare data
            logging.info("🔄 prepare semi-supervised data...")
            labeled_loader, unlabeled_loader, val_loader, test_loader = (
                prepare_semi_supervised_data(
                    labeled_ratio=labeled_ratio, batch_size=128, validation_split=0.1
                )
            )

            # create student and teacher models
            logging.info("🏗️  create student and teacher models...")
            student_model = ModifiedResNet18(num_classes=10)
            teacher_model = ModifiedResNet18(num_classes=10)

            total_params = sum(p.numel() for p in student_model.parameters())
            logging.info(f"    model parameters: {total_params:,}")

            # train Mean Teacher
            logging.info("🏋️  start Mean Teacher training...")
            trained_student, tracker, best_val_acc = train_mean_teacher(
                student_model=student_model,
                teacher_model=teacher_model,
                labeled_loader=labeled_loader,
                unlabeled_loader=unlabeled_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=0.001,
                weight_decay=1e-3,  # add weight decay
                ema_alpha=0.999,
                consistency_weight_max=10.0,
                early_stopping_patience=25,  # add patience
                scheduler_type="warmup_cosine",
                warmup_epochs=10,
                max_lr=None,
                labeled_ratio=labeled_ratio,
            )

            # test evaluation (use teacher model)
            test_acc, predictions, targets = evaluate_model(
                teacher_model, test_loader, detailed=True
            )

            logging.info(f"✅ final teacher test accuracy: {test_acc:.2f}%")

            # save results
            ratio_accuracies.append(test_acc)
            experiment_result = {
                "labeled_ratio": labeled_ratio,
                "seed": seed,
                "test_accuracy": test_acc,
                "best_val_accuracy": best_val_acc,
                "final_train_accuracy": tracker.train_accuracies[-1],
                "final_student_val_accuracy": tracker.val_accuracies[-1],
                "final_teacher_val_accuracy": tracker.teacher_val_accuracies[-1],
                "total_epochs": len(tracker.train_losses),
                "total_params": total_params,
            }
            detailed_results.append(experiment_result)

            # save training curve
            if save_results:
                fig = tracker.plot_metrics(
                    f"Mean Teacher - CIFAR-10 - {ratio_name} - Seed {seed}"
                )
                plt.savefig(
                    f"{results_dir}/training_curve_{labeled_ratio:.3f}_{seed}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()

        # calculate statistics
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

    # generate final report
    logging.info("\n" + "=" * 80)
    logging.info("📊 Phase 2 final results summary")
    logging.info("=" * 80)

    logging.info(
        f"\n{'Labeled Ratio':<20} | {'Mean Accuracy':<12} | {'Std':<6} | {'95% CI':<8}"
    )
    logging.info("-" * 55)

    for labeled_ratio in labeled_ratios:
        result = all_results[labeled_ratio]
        mean_acc = result["mean"]
        std_acc = result["std"]
        ci_95 = 1.96 * std_acc / np.sqrt(len(seeds))

        ratio_name = f"{labeled_ratio * 100:.1f}%"
        logging.info(
            f"{ratio_name:<20} | {mean_acc:10.2f}% | {std_acc:5.2f}% | ±{ci_95:4.2f}%"
        )

    # generate visualization
    if save_results:
        create_visualizations(all_results, results_dir, "phase2")

        # save detailed results
        with open(f"{results_dir}/detailed_results.json", "w") as f:
            json.dump(detailed_results, f, indent=2)

        logging.info(f"\n💾 results saved to: {results_dir}/")

    return all_results, detailed_results


def main():
    """main function"""
    parser = create_common_parser(
        description="Phase 2: CIFAR-10 Mean Teacher semi-supervised learning experiment",
        default_epochs=100,
    )

    args = parser.parse_args()

    # setup logging
    log_level = getattr(logging, args.log_level.upper())
    log_file = setup_logging(
        "phase2_mean_teacher", log_dir=args.log_dir, log_level=log_level
    )

    # log experiment parameters
    logging.info("Experiment parameters:")
    logging.info(f"  labeled ratios: {args.ratios}")
    logging.info(f"  random seeds: {args.seeds}")
    logging.info(f"  training epochs: {args.epochs}")
    logging.info(f"  save results: {not args.no_save}")
    logging.info(f"  log file: {log_file}")

    # run experiment
    all_results, detailed_results = run_phase2_experiments(
        labeled_ratios=args.ratios,
        seeds=args.seeds,
        epochs=args.epochs,
        save_results=not args.no_save,
    )

    logging.info("\n🎉 Phase 2 experiment completed!")
    logging.info("\n📝 key findings:")

    # print key conclusions
    for ratio in sorted(args.ratios):
        if ratio in all_results:
            acc = all_results[ratio]["mean"]
            std = all_results[ratio]["std"]
            logging.info(f"• {ratio * 100:.1f}% labeled data: {acc:.2f}% ± {std:.2f}%")


if __name__ == "__main__":
    main()
