#!/usr/bin/env python3
"""
Phase 1: supervised learning baseline experiment - CIFAR-10

experiment settings:
- dataset: CIFAR-10 (32×32, 3 channels, 10 classes)
- labeled ratio: 100%, 10%, 5%, 1%
- model: modified ResNet-18 (3×3 initial convolution, no MaxPool)
- goal: establish supervised learning performance baseline
"""

import torch
import torch.nn as nn
import torch.optim as optim
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

# Import common utilities
from utils.common_utils import (
    setup_logging,
    set_seed,
    device,
    ModifiedResNet18,
    prepare_supervised_data,
    evaluate_model,
    create_visualizations,
    create_common_parser,
)

# =============================================================================
# Training tracker
# =============================================================================
class BaselineTracker:
    """training process tracker"""

    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epoch_times = []

    def update(self, train_loss, train_acc, val_acc, lr, epoch_time):
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)

    def plot_metrics(self, title="Training Metrics"):
        """plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=18, fontweight="bold")

        # loss curve
        axes[0, 0].plot(self.train_losses, color="blue", linewidth=2)
        axes[0, 0].set_title("Training Loss", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[0, 0].set_ylabel("Loss", fontsize=12, fontweight="bold")
        axes[0, 0].tick_params(axis="both", labelsize=11)
        axes[0, 0].grid(True, alpha=0.3)

        # accuracy curve
        axes[0, 1].plot(self.train_accuracies, label="Train", color="blue", linewidth=2)
        axes[0, 1].plot(
            self.val_accuracies, label="Validation", color="red", linewidth=2
        )
        axes[0, 1].set_title("Accuracy", fontsize=14, fontweight="bold")
        axes[0, 1].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[0, 1].set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
        axes[0, 1].legend(fontsize=11, frameon=True, shadow=True)
        axes[0, 1].tick_params(axis="both", labelsize=11)
        axes[0, 1].grid(True, alpha=0.3)

        # learning rate
        axes[1, 0].plot(self.learning_rates, color="green", linewidth=2)
        axes[1, 0].set_title("Learning Rate", fontsize=14, fontweight="bold")
        axes[1, 0].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("LR", fontsize=12, fontweight="bold")
        axes[1, 0].tick_params(axis="both", labelsize=11)
        axes[1, 0].grid(True, alpha=0.3)

        # training time per epoch
        axes[1, 1].plot(self.epoch_times, color="orange", linewidth=2)
        axes[1, 1].set_title("Training Time per Epoch", fontsize=14, fontweight="bold")
        axes[1, 1].set_xlabel("Epoch", fontsize=12, fontweight="bold")
        axes[1, 1].set_ylabel("Time (s)", fontsize=12, fontweight="bold")
        axes[1, 1].tick_params(axis="both", labelsize=11)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# =============================================================================
# Training function
# =============================================================================
def train_model(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=0.001,
    weight_decay=5e-4,
    early_stopping_patience=15,
):
    """train model"""

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # early stopping mechanism
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None

    # training tracker
    tracker = BaselineTracker()

    logging.info(f"start training, {epochs} epochs")
    logging.info("-" * 60)

    for epoch in range(epochs):
        start_time = time.time()

        # training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

        # validation phase
        val_acc = evaluate_model(model, val_loader)
        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # update learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # record metrics
        epoch_time = time.time() - start_time
        tracker.update(avg_train_loss, train_acc, val_acc, current_lr, epoch_time)

        # early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logging.info(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.2f}s"
            )

        # early stopping
        if patience_counter >= early_stopping_patience:
            logging.info(
                f"Early stop trigger! Verification accuracy is {early_stopping_patience} epochs without improvement"
            )
            break

    # load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    logging.info("-" * 60)
    logging.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")

    return model, tracker, best_val_acc


# =============================================================================
# Phase 1 main experiment
# =============================================================================
def run_phase1_experiments(
    labeled_ratios=[1.0, 0.1, 0.05, 0.01],
    seeds=[42, 123, 456],
    epochs=100,
    save_results=True,
):
    """run Phase 1 complete experiment"""

    logging.info("🚀 Phase 1: CIFAR-10 supervised learning baseline experiment")
    logging.info("=" * 80)
    logging.info("experiment configuration:")
    logging.info("   dataset: CIFAR-10")
    logging.info("   model: Modified ResNet-18")
    logging.info(
        f"   labeled ratio: {[f'{r * 100:.1f}%' if r < 1 else '100% (fully supervised)' for r in labeled_ratios]}"
    )
    logging.info(f"   random seeds: {seeds}")
    logging.info(f"   training epochs: {epochs}")
    logging.info("=" * 80)

    # results storage
    all_results = defaultdict(list)
    detailed_results = []

    # create results save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./results/phase1_cifar10_results_{timestamp}"
    if save_results:
        os.makedirs(results_dir, exist_ok=True)

    total_experiments = len(labeled_ratios) * len(seeds)
    experiment_count = 0

    for labeled_ratio in labeled_ratios:
        ratio_name = (
            f"{labeled_ratio * 100:.1f}%"
            if labeled_ratio < 1.0
            else "100% (fully supervised)"
        )
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
            logging.info("🔄 prepare data...")
            train_loader, val_loader, test_loader = prepare_supervised_data(
                labeled_ratio=labeled_ratio, batch_size=128, validation_split=0.1
            )

            # create model
            logging.info("🏗️  create ModifiedResNet18...")
            model = ModifiedResNet18(num_classes=10)
            total_params = sum(p.numel() for p in model.parameters())
            logging.info(f"    model parameters: {total_params:,}")

            # train model
            logging.info("🏋️  start training...")
            trained_model, tracker, best_val_acc = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=0.001,
                weight_decay=5e-4,
                early_stopping_patience=15,
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
                "final_train_accuracy": tracker.train_accuracies[-1],
                "total_epochs": len(tracker.train_losses),
                "total_params": total_params,
            }
            detailed_results.append(experiment_result)

            # save training curve
            if save_results:
                fig = tracker.plot_metrics(
                    f"ModifiedResNet18 - CIFAR-10 - {ratio_name} - Seed {seed}"
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
    logging.info("📊 Phase 1 final results summary")
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

        ratio_name = (
            f"{labeled_ratio * 100:.1f}%"
            if labeled_ratio < 1.0
            else "100% (fully supervised)"
        )
        logging.info(
            f"{ratio_name:<20} | {mean_acc:10.2f}% | {std_acc:5.2f}% | ±{ci_95:4.2f}%"
        )

    # generate visualization
    if save_results:
        create_visualizations(all_results, results_dir, "phase1")

        # save detailed results
        with open(f"{results_dir}/detailed_results.json", "w") as f:
            json.dump(detailed_results, f, indent=2)

        logging.info(f"\n💾 results saved to: {results_dir}/")

    return all_results, detailed_results


def main():
    """main function"""
    parser = create_common_parser(
        description="Phase 1: CIFAR-10 supervised learning baseline experiment",
        default_epochs=100,
    )

    args = parser.parse_args()

    # setup logging
    log_level = getattr(logging, args.log_level.upper())
    log_file = setup_logging(
        "phase1_baseline", log_dir=args.log_dir, log_level=log_level
    )

    # log experiment parameters
    logging.info("Experiment parameters:")
    logging.info(f"  labeled ratios: {args.ratios}")
    logging.info(f"  random seeds: {args.seeds}")
    logging.info(f"  training epochs: {args.epochs}")
    logging.info(f"  save results: {not args.no_save}")
    logging.info(f"  log file: {log_file}")

    # run experiment
    all_results, detailed_results = run_phase1_experiments(
        labeled_ratios=args.ratios,
        seeds=args.seeds,
        epochs=args.epochs,
        save_results=not args.no_save,
    )

    logging.info("\n🎉 Phase 1 experiment completed!")
    logging.info("\n📝 key findings:")

    # print key conclusions
    full_supervised = all_results.get(1.0, {}).get("mean", 0)
    if full_supervised > 0:
        logging.info(f"• fully supervised baseline accuracy: {full_supervised:.2f}%")
        for ratio in [0.1, 0.05, 0.01]:
            if ratio in all_results:
                limited_acc = all_results[ratio]["mean"]
                retention = (limited_acc / full_supervised) * 100
                logging.info(
                    f"• {ratio * 100:.1f}% labeled data: {limited_acc:.2f}% (maintain {retention:.1f}% performance)"
                )


if __name__ == "__main__":
    main()
