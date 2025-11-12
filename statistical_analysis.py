#!/usr/bin/env python3
"""
Statistical Analysis for Semi-Supervised Learning Project
=========================================================

This script performs statistical significance testing and effect size calculation
for comparing baseline and FixMatch performance across different labeled data ratios.

Author: Generated for ProjectA Semi-Supervised Learning Report
Date: 2025-08-14
"""

import json
import numpy as np
from scipy import stats
import pandas as pd
from pathlib import Path


def load_results(baseline_path, fixmatch_path, meanteacher_path=None):
    """Load experimental results from JSON files."""
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    
    with open(fixmatch_path, 'r') as f:
        fixmatch = json.load(f)
    
    meanteacher = None
    if meanteacher_path and Path(meanteacher_path).exists():
        with open(meanteacher_path, 'r') as f:
            meanteacher = json.load(f)
    
    return baseline, fixmatch, meanteacher


def extract_accuracies(results, ratio):
    """Extract test accuracies for a specific labeled ratio."""
    return [x['test_accuracy'] for x in results if x['labeled_ratio'] == ratio]


def calculate_statistics(group1, group2, group1_name="Group1", group2_name="Group2"):
    """Calculate comprehensive statistics between two groups."""
    # Basic statistics
    mean1, std1 = np.mean(group1), np.std(group1, ddof=1)
    mean2, std2 = np.mean(group2), np.std(group2, ddof=1)
    
    # Statistical tests
    t_stat, p_value = stats.ttest_ind(group2, group1)  # group2 vs group1
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    cohens_d = (mean2 - mean1) / pooled_std
    
    # Confidence intervals (95%)
    n1, n2 = len(group1), len(group2)
    se1 = std1 / np.sqrt(n1)
    se2 = std2 / np.sqrt(n2)
    
    # t-critical for 95% CI with small samples
    df = n1 + n2 - 2
    t_critical = stats.t.ppf(0.975, df)
    
    ci1 = (mean1 - t_critical * se1, mean1 + t_critical * se1)
    ci2 = (mean2 - t_critical * se2, mean2 + t_critical * se2)
    
    return {
        'group1_name': group1_name,
        'group2_name': group2_name,
        'group1_mean': mean1,
        'group1_std': std1,
        'group1_ci': ci1,
        'group2_mean': mean2,
        'group2_std': std2,
        'group2_ci': ci2,
        'difference': mean2 - mean1,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant_001': p_value < 0.001,
        'significant_01': p_value < 0.01,
        'significant_05': p_value < 0.05,
        'large_effect': abs(cohens_d) > 0.8,
        'very_large_effect': abs(cohens_d) > 2.0
    }


def print_statistical_summary(stats_dict, ratio):
    """Print formatted statistical summary."""
    print(f"\n{'='*50}")
    print(f"Statistical Analysis: {ratio*100:.0f}% Labeled Data")
    print(f"{'='*50}")
    
    print(f"{stats_dict['group1_name']}: {stats_dict['group1_mean']:.2f} ± {stats_dict['group1_std']:.2f}")
    print(f"{stats_dict['group2_name']}: {stats_dict['group2_mean']:.2f} ± {stats_dict['group2_std']:.2f}")
    print(f"Difference: {stats_dict['difference']:.2f} percentage points")
    
    print(f"\nStatistical Test Results:")
    print(f"  t-statistic: {stats_dict['t_statistic']:.3f}")
    print(f"  p-value: {stats_dict['p_value']:.6f}")
    print(f"  Effect size (Cohen's d): {stats_dict['cohens_d']:.3f}")
    
    print(f"\nSignificance Levels:")
    print(f"  p < 0.001: {stats_dict['significant_001']}")
    print(f"  p < 0.01:  {stats_dict['significant_01']}")
    print(f"  p < 0.05:  {stats_dict['significant_05']}")
    
    print(f"\nEffect Size Interpretation:")
    print(f"  Large effect (d > 0.8): {stats_dict['large_effect']}")
    print(f"  Very large effect (d > 2.0): {stats_dict['very_large_effect']}")


def create_summary_table(all_stats):
    """Create a summary table of all statistical comparisons."""
    df_data = []
    for ratio, stats_dict in all_stats.items():
        df_data.append({
            'Label_Ratio': f"{ratio*100:.0f}%",
            'Baseline_Mean': f"{stats_dict['group1_mean']:.2f}",
            'Baseline_Std': f"{stats_dict['group1_std']:.2f}",
            'FixMatch_Mean': f"{stats_dict['group2_mean']:.2f}",
            'FixMatch_Std': f"{stats_dict['group2_std']:.2f}",
            'Improvement': f"{stats_dict['difference']:.2f}",
            'p_value': f"{stats_dict['p_value']:.1e}",
            'Cohens_d': f"{stats_dict['cohens_d']:.1f}",
            'Significant': "***" if stats_dict['significant_001'] else "**" if stats_dict['significant_01'] else "*" if stats_dict['significant_05'] else "ns"
        })
    
    df = pd.DataFrame(df_data)
    return df


def main():
    """Main analysis function."""
    print("Statistical Analysis for Semi-Supervised Learning Project")
    print("=" * 60)
    
    # Paths to result files
    baseline_path = "results/phase1_baseline_results/detailed_results.json"
    fixmatch_path = "results/phase3_fixmatch_results_20250808_153102/detailed_results.json"
    meanteacher_path = "results/phase2_mean_teacher_results_20250801_141310/detailed_results.json"
    
    # Load results
    try:
        baseline, fixmatch, meanteacher = load_results(baseline_path, fixmatch_path, meanteacher_path)
        print(f"✓ Loaded baseline results: {len(baseline)} experiments")
        print(f"✓ Loaded FixMatch results: {len(fixmatch)} experiments")
        if meanteacher:
            print(f"✓ Loaded Mean Teacher results: {len(meanteacher)} experiments")
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        return
    
    # Analyze different label ratios
    ratios = [0.1, 0.05, 0.01]  # 10%, 5%, 1%
    all_stats = {}
    
    for ratio in ratios:
        baseline_acc = extract_accuracies(baseline, ratio)
        fixmatch_acc = extract_accuracies(fixmatch, ratio)
        
        if len(baseline_acc) == 0 or len(fixmatch_acc) == 0:
            print(f"⚠️  No data found for {ratio*100}% labeled ratio")
            continue
        
        stats_dict = calculate_statistics(
            baseline_acc, fixmatch_acc, 
            "Baseline", "FixMatch"
        )
        
        all_stats[ratio] = stats_dict
        print_statistical_summary(stats_dict, ratio)
    
    # Create and display summary table
    if all_stats:
        print(f"\n{'='*80}")
        print("SUMMARY TABLE")
        print(f"{'='*80}")
        summary_df = create_summary_table(all_stats)
        print(summary_df.to_string(index=False))
        
        print(f"\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        
        # Save summary table
        summary_df.to_csv("statistical_analysis_summary.csv", index=False)
        print(f"\n✓ Summary table saved to: statistical_analysis_summary.csv")
    
    # Additional analysis with Mean Teacher if available
    if meanteacher:
        print(f"\n{'='*80}")
        print("MEAN TEACHER vs BASELINE COMPARISON")
        print(f"{'='*80}")
        
        for ratio in ratios:
            baseline_acc = extract_accuracies(baseline, ratio)
            meanteacher_acc = extract_accuracies(meanteacher, ratio)
            
            if len(baseline_acc) == 0 or len(meanteacher_acc) == 0:
                continue
                
            stats_dict = calculate_statistics(
                baseline_acc, meanteacher_acc,
                "Baseline", "Mean Teacher"
            )
            print_statistical_summary(stats_dict, ratio)


if __name__ == "__main__":
    main()