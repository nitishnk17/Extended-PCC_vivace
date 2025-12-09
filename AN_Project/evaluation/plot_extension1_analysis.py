#!/usr/bin/env python3

"""
Extension 1 Analysis and Visualization

Generates 4 publication-quality graphs comparing Baseline vs Extension 1:
1. Grouped bar chart - Mean Utility Score comparison
2. Grouped bar chart - Throughput comparison
3. Scatter plot - Performance trade-off (Throughput vs 99th Percentile Latency)
4. Grouped bar chart - Throughput Standard Deviation comparison
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
    'extension1': '#2ecc71',    # Green
}

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results" / "extension1"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
DPI = 300


def load_scenario_results(scenario_name):
    """Load results for a specific scenario"""
    filepath = RESULTS_DIR / f"{scenario_name}_results.json"
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_metrics(results):
    """Extract key metrics from results"""
    baseline = results['baseline'][0]
    extension1 = results['extension1'][0]

    return {
        'baseline': {
            'throughput': baseline['avg_throughput']['mean'],
            'throughput_std': baseline['avg_throughput']['std'],
            'latency': baseline['avg_latency']['mean'],
            'latency_p99': baseline['latency_p99']['mean'],
            'utility': baseline['avg_utility']['mean'],
            'loss': baseline['loss_rate']['mean'],
        },
        'extension1': {
            'throughput': extension1['avg_throughput']['mean'],
            'throughput_std': extension1['avg_throughput']['std'],
            'latency': extension1['avg_latency']['mean'],
            'latency_p99': extension1['latency_p99']['mean'],
            'utility': extension1['avg_utility']['mean'],
            'loss': extension1['loss_rate']['mean'],
        }
    }


def plot_utility_comparison():
    """Plot 1: Grouped bar chart comparing Mean Utility Score"""
    print("\n📊 Generating Plot 1: Utility Score Comparison...")

    scenarios = ['bulk_transfer', 'video_streaming', 'realtime_gaming']
    scenario_labels = ['Bulk Transfer', 'Streaming', 'Real-time']

    baseline_utilities = []
    extension1_utilities = []

    for scenario in scenarios:
        results = load_scenario_results(scenario)
        metrics = extract_metrics(results)
        baseline_utilities.append(metrics['baseline']['utility'])
        extension1_utilities.append(metrics['extension1']['utility'])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(scenario_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_utilities, width,
                   label='Baseline', color=COLORS['baseline'], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, extension1_utilities, width,
                   label='Extension 1', color=COLORS['extension1'], alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Traffic Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Utility Score', fontsize=12, fontweight='bold')
    ax.set_title('Extension 1: Mean Utility Score Comparison Across Traffic Scenarios',
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'EXT1_01_utility_score_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    print("   ✅ Plot 1 saved: EXT1_01_utility_score_comparison.png")


def plot_throughput_comparison():
    """Plot 2: Grouped bar chart comparing Throughput"""
    print("📊 Generating Plot 2: Throughput Comparison...")

    scenarios = ['bulk_transfer', 'video_streaming', 'realtime_gaming']
    scenario_labels = ['Bulk Transfer', 'Streaming', 'Real-time']

    baseline_throughputs = []
    extension1_throughputs = []

    for scenario in scenarios:
        results = load_scenario_results(scenario)
        metrics = extract_metrics(results)
        baseline_throughputs.append(metrics['baseline']['throughput'])
        extension1_throughputs.append(metrics['extension1']['throughput'])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(scenario_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_throughputs, width,
                   label='Baseline', color=COLORS['baseline'], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, extension1_throughputs, width,
                   label='Extension 1', color=COLORS['extension1'], alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Traffic Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Throughput (Mbps)', fontsize=12, fontweight='bold')
    ax.set_title('Extension 1: Throughput Comparison Across Traffic Scenarios',
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'EXT1_02_throughput_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    print("   ✅ Plot 2 saved: EXT1_02_throughput_comparison.png")


def plot_performance_tradeoff():
    """Plot 3: Scatter plot - Performance trade-off (Throughput vs 99th Percentile Latency)"""
    print("📊 Generating Plot 3: Performance Trade-off Scatter Plot...")

    scenarios = ['bulk_transfer', 'video_streaming', 'realtime_gaming']
    scenario_labels = ['Bulk Transfer', 'Streaming', 'Real-time']

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, scenario in enumerate(scenarios):
        results = load_scenario_results(scenario)
        metrics = extract_metrics(results)

        # Plot baseline point
        baseline_latency = metrics['baseline']['latency_p99']
        baseline_throughput = metrics['baseline']['throughput']
        ax.scatter(baseline_latency, baseline_throughput,
                  s=250, color=COLORS['baseline'], marker='o',
                  alpha=0.8, edgecolor='black', linewidth=2,
                  label=f'Baseline' if i == 0 else '')
        ax.text(baseline_latency, baseline_throughput - 0.5,
               scenario_labels[i], ha='center', fontsize=9, fontweight='bold')

        # Plot extension1 point
        ext1_latency = metrics['extension1']['latency_p99']
        ext1_throughput = metrics['extension1']['throughput']
        ax.scatter(ext1_latency, ext1_throughput,
                  s=250, color=COLORS['extension1'], marker='s',
                  alpha=0.8, edgecolor='black', linewidth=2,
                  label=f'Extension 1' if i == 0 else '')
        ax.text(ext1_latency, ext1_throughput + 0.5,
               scenario_labels[i], ha='center', fontsize=9, fontweight='bold')

        # Draw arrow showing improvement direction
        ax.annotate('', xy=(ext1_latency, ext1_throughput),
                   xytext=(baseline_latency, baseline_throughput),
                   arrowprops=dict(arrowstyle='->', lw=1.5,
                                 color='gray', alpha=0.5, linestyle='--'))

    ax.set_xlabel('99th Percentile Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Throughput (Mbps)', fontsize=12, fontweight='bold')
    ax.set_title('Extension 1: Performance Trade-off Analysis\n(Throughput vs Latency)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'EXT1_03_performance_tradeoff.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    print("   ✅ Plot 3 saved: EXT1_03_performance_tradeoff.png")


def generate_summary_statistics():
    """Generate summary statistics table"""
    print("\n📊 Generating Summary Statistics...")

    scenarios = ['bulk_transfer', 'video_streaming', 'realtime_gaming']
    scenario_labels = ['Bulk Transfer', 'Streaming', 'Real-time']

    print("\n" + "="*120)
    print("EXTENSION 1 ANALYSIS - DETAILED METRICS COMPARISON")
    print("="*120)

    for i, scenario in enumerate(scenarios):
        results = load_scenario_results(scenario)
        metrics = extract_metrics(results)

        print(f"\n{scenario_labels[i].upper()}:")
        print("-" * 120)

        print(f"{'Metric':<30} {'Baseline':<20} {'Extension 1':<20} {'Change':<20}")
        print("-" * 120)

        # Utility
        baseline_util = metrics['baseline']['utility']
        ext1_util = metrics['extension1']['utility']
        change = ((ext1_util - baseline_util) / baseline_util * 100) if baseline_util != 0 else 0
        print(f"{'Mean Utility Score':<30} {baseline_util:<20.4f} {ext1_util:<20.4f} {change:+.2f}%")

        # Throughput
        baseline_tput = metrics['baseline']['throughput']
        ext1_tput = metrics['extension1']['throughput']
        change = ((ext1_tput - baseline_tput) / baseline_tput * 100) if baseline_tput != 0 else 0
        print(f"{'Avg Throughput (Mbps)':<30} {baseline_tput:<20.4f} {ext1_tput:<20.4f} {change:+.2f}%")

        # Throughput Std Dev
        baseline_std = metrics['baseline']['throughput_std']
        ext1_std = metrics['extension1']['throughput_std']
        change = ((ext1_std - baseline_std) / (baseline_std + 1e-10) * 100) if baseline_std != 0 else 0
        print(f"{'Throughput Std Dev (Mbps)':<30} {baseline_std:<20.6f} {ext1_std:<20.6f} {change:+.2f}%")

        # Latency
        baseline_lat = metrics['baseline']['latency']
        ext1_lat = metrics['extension1']['latency']
        change = ((ext1_lat - baseline_lat) / baseline_lat * 100) if baseline_lat != 0 else 0
        print(f"{'Avg Latency (ms)':<30} {baseline_lat:<20.4f} {ext1_lat:<20.4f} {change:+.2f}%")

        # 99th Percentile Latency
        baseline_p99 = metrics['baseline']['latency_p99']
        ext1_p99 = metrics['extension1']['latency_p99']
        change = ((ext1_p99 - baseline_p99) / baseline_p99 * 100) if baseline_p99 != 0 else 0
        print(f"{'99th Percentile Latency (ms)':<30} {baseline_p99:<20.4f} {ext1_p99:<20.4f} {change:+.2f}%")

        # Loss Rate
        baseline_loss = metrics['baseline']['loss']
        ext1_loss = metrics['extension1']['loss']
        change = ((ext1_loss - baseline_loss) / (baseline_loss + 1e-10) * 100) if baseline_loss != 0 else 0
        print(f"{'Loss Rate':<30} {baseline_loss:<20.6f} {ext1_loss:<20.6f} {change:+.2f}%")

    print("\n" + "="*120)


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("EXTENSION 1 ANALYSIS AND VISUALIZATION")
    print("="*80)

    try:
        # Generate all plots
        plot_utility_comparison()
        plot_throughput_comparison()
        plot_performance_tradeoff()

        # Generate summary statistics
        generate_summary_statistics()

        print("\n" + "="*80)
        print("✅ ALL PLOTS GENERATED SUCCESSFULLY")
        print("="*80)
        print(f"\n📁 Plots saved to: {PLOTS_DIR}")
        print("\nGenerated files:")
        print("  1. EXT1_01_utility_score_comparison.png")
        print("  2. EXT1_02_throughput_comparison.png")
        print("  3. EXT1_03_performance_tradeoff.png")
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
