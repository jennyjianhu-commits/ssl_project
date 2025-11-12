#!/usr/bin/env python3
"""
Phase 4: Comprehensive Evaluation and Comparison Analysis
- Summarize the performance of Supervised Baseline, Mean Teacher, and FixMatch under different labeled ratios
- Output mean, variance, and confidence intervals
- Generate comparative visualizations
- Output markdown/table reports
"""

import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
# Set high DPI and larger font size for better figure quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
from collections import defaultdict

# =====================
# configuration
# =====================
# automatically find results files in results directory
PHASE_RESULT_PATTERNS = {
    'Supervised': 'results/phase1_baseline_results_*/detailed_results.json',
    'MeanTeacher': 'results/phase2_mean_teacher_results_*/detailed_results.json',
    'FixMatch': 'results/phase3_fixmatch_results_*/detailed_results.json',
    'FlexMatch': 'results/phase4_flexmatch_results_*/detailed_results.json'
}

# labeled ratio (can be adjusted based on actual experiments)
LABELED_RATIOS = [1.0, 0.1, 0.05, 0.01]

# =====================
# utility functions
# =====================
# find the latest results file matching the pattern
def find_latest_result(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    # get the latest file
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def load_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def aggregate_results(detailed_results, method_name):
    """
    aggregate test accuracy for each labeled ratio
    return: {ratio: [acc1, acc2, ...]}
    """
    ratio_accs = defaultdict(list)
    for entry in detailed_results:
        ratio = entry.get('labeled_ratio', 1.0)
        acc = entry.get('test_accuracy', None)
        if acc is not None:
            ratio_accs[ratio].append(acc)
    return ratio_accs

# compute mean, std, ci95
def compute_stats(acc_list):
    arr = np.array(acc_list)
    mean = arr.mean() if len(arr) > 0 else 0.0
    std = arr.std() if len(arr) > 0 else 0.0
    ci95 = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 0 else 0.0
    return mean, std, ci95

# =====================
# main flow
# =====================
def main():
    # 1. read all methods results
    all_stats = {}
    for method, pattern in PHASE_RESULT_PATTERNS.items():
        file_path = find_latest_result(pattern)
        if not file_path:
            print(f"❌ no results file found for {method}!")
            continue
        print(f"✅ read {method} results: {file_path}")
        detailed_results = load_results(file_path)
        ratio_accs = aggregate_results(detailed_results, method)
        all_stats[method] = {}
        for ratio in LABELED_RATIOS:
            accs = ratio_accs.get(ratio, [])
            mean, std, ci95 = compute_stats(accs)
            all_stats[method][ratio] = {
                'mean': mean,
                'std': std,
                'ci95': ci95,
                'n': len(accs),
                'all': accs
            }
    # 2. output markdown table
    print("\n# Phase 5 comprehensive evaluation results comparison\n")
    header = "| labeled ratio | Supervised | Mean Teacher | FixMatch | FlexMatch |"
    sep = "|---|---|---|---|---|"
    print(header)
    print(sep)
    for ratio in LABELED_RATIOS:
        row = f"| {ratio*100:.1f}% "
        for method in ['Supervised', 'MeanTeacher', 'FixMatch', 'FlexMatch']:
            stat = all_stats.get(method, {}).get(ratio, None)
            if stat and stat['n'] > 0:
                row += f"| {stat['mean']:.2f}±{stat['ci95']:.2f} "
            else:
                row += "| - "
        row += "|"
        print(row)
    # 3. generate visualization
    fig, ax = plt.subplots(figsize=(10,6))
    width = 0.22
    x = np.arange(len(LABELED_RATIOS))
    methods = ['Supervised', 'MeanTeacher', 'FixMatch', 'FlexMatch']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#FFFF00']
    for i, method in enumerate(methods):
        means = [all_stats.get(method, {}).get(r, {}).get('mean', 0) for r in LABELED_RATIOS]
        ci95s = [all_stats.get(method, {}).get(r, {}).get('ci95', 0) for r in LABELED_RATIOS]
        bars = ax.bar(x+i*width, means, width, yerr=ci95s, capsize=5, label=method, color=colors[i], alpha=0.85)
        
        # Add value labels on top of bars
        for j, (bar, mean_val) in enumerate(zip(bars, means)):
            if mean_val > 0:  # Only show label if there's a value
                height = bar.get_height()
                # Position text above the error bar
                y_pos = height + ci95s[j] + 1.5
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{mean_val:.1f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{r*100:.1f}%" for r in LABELED_RATIOS])
    ax.set_xlabel("labeled ratio", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("Baseline, Mean Teacher, FixMatch, FlexMatch performance comparison (mean±95%CI)", fontsize=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/phase5_comparison.png")  # Uses the default 300 dpi from rcParams
    print("\n📊 comparison figure generated: results/phase5_comparison.png")
    print("\n🎉 Phase 5 comprehensive evaluation completed!")

if __name__ == "__main__":
    main() 