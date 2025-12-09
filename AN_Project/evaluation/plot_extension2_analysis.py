#!/usr/bin/env python3

"""
Extension 2 Analysis and Visualization

Generates 4 publication-quality graphs comparing Baseline vs Extension 2:
1. Grouped bar chart - Mean Utility Score comparison (3 traffic scenarios)
2. Grouped bar chart - Throughput comparison (3 traffic scenarios)
3. Utility improvement in wireless scenario
4. Cumulative packet loss rate comparison across all scenarios
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Configure plotting style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.autolayout'] = True

# Color palette
COLORS = {
    'baseline': '#3498db',      # Blue
    'extension2': '#e74c3c',    # Red
}

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results" / "extensions_1_2"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
DPI = 300


def load_scenario_results(scenario_name):
    """Load results for a specific scenario"""
    filepath = RESULTS_DIR / f"{scenario_name}_results.json"
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_metrics(results, extension='extension2'):
    """Extract key metrics from results"""
    baseline = results['baseline'][0]
    ext_data = results.get(extension, [{}])[0]

    return {
        'baseline': {
            'throughput': baseline['avg_throughput']['mean'],
            'throughput_std': baseline['avg_throughput']['std'],
            'latency': baseline['avg_latency']['mean'],
            'latency_p99': baseline['latency_p99']['mean'],
            'utility': baseline['avg_utility']['mean'],
            'loss': baseline['loss_rate']['mean'],
        },
        'extension2': {
            'throughput': ext_data.get('avg_throughput', {}).get('mean', 0),
            'throughput_std': ext_data.get('avg_throughput', {}).get('std', 0),
            'latency': ext_data.get('avg_latency', {}).get('mean', 0),
            'latency_p99': ext_data.get('latency_p99', {}).get('mean', 0),
            'utility': ext_data.get('avg_utility', {}).get('mean', 0),
            'loss': ext_data.get('loss_rate', {}).get('mean', 0),
        }
    }


def plot_utility_comparison():
    """Plot 1: Grouped bar chart comparing Mean Utility Score"""
    print("\n📊 Generating Plot 1: Utility Score Comparison...")

    scenarios = ['bulk_transfer', 'video_streaming', 'realtime_gaming']
    scenario_labels = ['Bulk Transfer', 'Streaming', 'Real-time']

    baseline_utilities = []
    extension2_utilities = []

    for scenario in scenarios:
        results = load_scenario_results(scenario)
        metrics = extract_metrics(results)
        baseline_utilities.append(metrics['baseline']['utility'])
        extension2_utilities.append(metrics['extension2']['utility'])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(scenario_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_utilities, width,
                   label='Baseline', color=COLORS['baseline'], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, extension2_utilities, width,
                   label='Extension 2', color=COLORS['extension2'], alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Traffic Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Utility Score', fontsize=12, fontweight='bold')
    ax.set_title('Extension 2: Mean Utility Score Comparison Across Traffic Scenarios',
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'EXT2_01_utility_score_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    print("   ✅ Plot 1 saved: EXT2_01_utility_score_comparison.png")


def plot_throughput_comparison():
    """Plot 2: Grouped bar chart comparing Throughput"""
    print("📊 Generating Plot 2: Throughput Comparison...")

    scenarios = ['bulk_transfer', 'video_streaming', 'realtime_gaming']
    scenario_labels = ['Bulk Transfer', 'Streaming', 'Real-time']

    baseline_throughputs = []
    extension2_throughputs = []

    for scenario in scenarios:
        results = load_scenario_results(scenario)
        metrics = extract_metrics(results)
        baseline_throughputs.append(metrics['baseline']['throughput'])
        extension2_throughputs.append(metrics['extension2']['throughput'])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(scenario_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_throughputs, width,
                   label='Baseline', color=COLORS['baseline'], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, extension2_throughputs, width,
                   label='Extension 2', color=COLORS['extension2'], alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Traffic Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Throughput (Mbps)', fontsize=12, fontweight='bold')
    ax.set_title('Extension 2: Throughput Comparison Across Traffic Scenarios',
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'EXT2_02_throughput_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    print("   ✅ Plot 2 saved: EXT2_02_throughput_comparison.png")


def plot_wireless_utility_improvement():
    """Plot 4: Utility improvement in wireless scenario"""
    print("📊 Generating Plot 4: Wireless Utility Improvement...")

    results = load_scenario_results('wireless_2pct_loss')
    metrics = extract_metrics(results)

    baseline_util = metrics['baseline']['utility']
    extension2_util = metrics['extension2']['utility']

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(2)
    values = [baseline_util, extension2_util]
    labels = ['Baseline', 'Extension 2']
    colors = [COLORS['baseline'], COLORS['extension2']]

    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5, width=0.4)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., val,
               f'{val:.4f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Utility Score', fontsize=12, fontweight='bold')
    ax.set_title('Extension 2: Utility Score in Wireless Scenario (2% Loss)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'EXT2_04_wireless_utility_improvement.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    print("   ✅ Plot 4 saved: EXT2_04_wireless_utility_improvement.png")


def plot_packet_loss_rate_cumulative():
    """Plot 5: Cumulative packet loss rate comparison (like Plot 4 format)"""
    print("📊 Generating Plot 5: Cumulative Packet Loss Rate Comparison...")

    all_scenarios = [
        'bulk_transfer', 'video_streaming', 'realtime_gaming',
        'wireless_2pct_loss', 'moderate_latency', 'high_bandwidth'
    ]

    baseline_losses = []
    extension2_losses = []

    for scenario in all_scenarios:
        results = load_scenario_results(scenario)
        metrics = extract_metrics(results)
        baseline_losses.append(metrics['baseline']['loss'])
        extension2_losses.append(metrics['extension2']['loss'])

    # Calculate cumulative/average loss rate
    cumulative_baseline = np.mean(baseline_losses)
    cumulative_extension2 = np.mean(extension2_losses)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(2)
    values = [cumulative_baseline, cumulative_extension2]
    labels = ['Baseline', 'Extension 2']
    colors = [COLORS['baseline'], COLORS['extension2']]

    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5, width=0.4)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., val,
               f'{val:.6f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Average Packet Loss Rate', fontsize=12, fontweight='bold')
    ax.set_title('Extension 2: Cumulative Packet Loss Rate Across All Scenarios',
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'EXT2_05_packet_loss_cumulative.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    print("   ✅ Plot 5 saved: EXT2_05_packet_loss_cumulative.png")


def generate_summary_statistics():
    """Generate summary statistics table"""
    print("\n📊 Generating Summary Statistics...")

    scenarios = ['bulk_transfer', 'video_streaming', 'realtime_gaming']
    scenario_labels = ['Bulk Transfer', 'Streaming', 'Real-time']

    print("\n" + "="*140)
    print("EXTENSION 2 ANALYSIS - DETAILED METRICS COMPARISON")
    print("="*140)

    for i, scenario in enumerate(scenarios):
        results = load_scenario_results(scenario)
        metrics = extract_metrics(results)

        print(f"\n{scenario_labels[i].upper()}:")
        print("-" * 140)

        print(f"{'Metric':<30} {'Baseline':<20} {'Extension 2':<20} {'Change':<20}")
        print("-" * 140)

        # Utility
        baseline_util = metrics['baseline']['utility']
        ext2_util = metrics['extension2']['utility']
        change = ((ext2_util - baseline_util) / baseline_util * 100) if baseline_util != 0 else 0
        print(f"{'Mean Utility Score':<30} {baseline_util:<20.4f} {ext2_util:<20.4f} {change:+.2f}%")

        # Throughput
        baseline_tput = metrics['baseline']['throughput']
        ext2_tput = metrics['extension2']['throughput']
        change = ((ext2_tput - baseline_tput) / baseline_tput * 100) if baseline_tput != 0 else 0
        print(f"{'Avg Throughput (Mbps)':<30} {baseline_tput:<20.4f} {ext2_tput:<20.4f} {change:+.2f}%")

        # Loss Rate
        baseline_loss = metrics['baseline']['loss']
        ext2_loss = metrics['extension2']['loss']
        change = ((ext2_loss - baseline_loss) / (baseline_loss + 1e-10) * 100) if baseline_loss != 0 else 0
        print(f"{'Loss Rate':<30} {baseline_loss:<20.6f} {ext2_loss:<20.6f} {change:+.2f}%")

        # Latency
        baseline_lat = metrics['baseline']['latency']
        ext2_lat = metrics['extension2']['latency']
        change = ((ext2_lat - baseline_lat) / baseline_lat * 100) if baseline_lat != 0 else 0
        print(f"{'Avg Latency (ms)':<30} {baseline_lat:<20.4f} {ext2_lat:<20.4f} {change:+.2f}%")

    # Wireless scenario detail
    print("\n" + "="*140)
    print("WIRELESS SCENARIO (2% LOSS) - DETAILED ANALYSIS")
    print("="*140)

    results = load_scenario_results('wireless_2pct_loss')
    metrics = extract_metrics(results)

    print(f"\n{'Metric':<30} {'Baseline':<20} {'Extension 2':<20} {'Change':<20}")
    print("-" * 140)

    baseline_util = metrics['baseline']['utility']
    ext2_util = metrics['extension2']['utility']
    change = ((ext2_util - baseline_util) / baseline_util * 100) if baseline_util != 0 else 0
    print(f"{'Utility Improvement':<30} {baseline_util:<20.4f} {ext2_util:<20.4f} {change:+.2f}%")

    baseline_loss = metrics['baseline']['loss']
    ext2_loss = metrics['extension2']['loss']
    change = ((ext2_loss - baseline_loss) / (baseline_loss + 1e-10) * 100)
    print(f"{'Loss Rate Improvement':<30} {baseline_loss:<20.6f} {ext2_loss:<20.6f} {change:+.2f}%")

    print("\n" + "="*140)


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("EXTENSION 2 ANALYSIS AND VISUALIZATION")
    print("="*80)

    try:
        # Generate all plots
        plot_utility_comparison()
        plot_throughput_comparison()
        plot_wireless_utility_improvement()
        plot_packet_loss_rate_cumulative()

        # Generate summary statistics
        generate_summary_statistics()

        print("\n" + "="*80)
        print("✅ ALL PLOTS GENERATED SUCCESSFULLY")
        print("="*80)
        print(f"\n📁 Plots saved to: {PLOTS_DIR}")
        print("\nGenerated files:")
        print("  1. EXT2_01_utility_score_comparison.png")
        print("  2. EXT2_02_throughput_comparison.png")
        print("  3. EXT2_04_wireless_utility_improvement.png")
        print("  4. EXT2_05_packet_loss_cumulative.png")
        print("\n" + "="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
