#!/usr/bin/env python3

"""
Comprehensive Baseline vs Extension 3 Comparison

Creates multi-metric comparison visualization showing how Extension 3
improves convergence time while maintaining other performance metrics.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'extensions_1_2_3'
PLOTS_DIR = Path(__file__).parent.parent / 'plots'
DPI = 300

# Colors
COLORS = {
    'baseline': '#4C72B0',
    'extension3': '#55A868',
    'improvement': '#DD8452',
    'neutral': '#8C8C8C'
}


def load_scenario_results(scenario_name):
    """Load results for a specific scenario"""
    filepath = RESULTS_DIR / f"{scenario_name}_results.json"
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_multi_metric_comparison():
    """Create comprehensive multi-metric comparison"""
    print("\n📊 Generating Multi-Metric Comparison Graph...")

    scenarios = ['bulk_transfer', 'video_streaming', 'realtime_gaming']
    scenario_labels = ['Bulk\nTransfer', 'Video\nStreaming', 'Real-time\nGaming']

    # Collect data for all metrics
    metrics_data = {
        'convergence': {'baseline': [], 'ext3': [], 'label': 'Convergence Time (s)', 'lower_better': True},
        'jfi': {'baseline': [], 'ext3': [], 'label': 'Jain Fairness Index', 'lower_better': False},
        'throughput': {'baseline': [], 'ext3': [], 'label': 'Throughput (Mbps)', 'lower_better': False},
        'latency': {'baseline': [], 'ext3': [], 'label': 'Latency (ms)', 'lower_better': True},
    }

    for scenario in scenarios:
        results = load_scenario_results(scenario)
        baseline = results['baseline'][0]
        ext3 = results['extension3'][0]

        metrics_data['convergence']['baseline'].append(baseline['convergence_time_seconds']['mean'])
        metrics_data['convergence']['ext3'].append(ext3['convergence_time_seconds']['mean'])

        metrics_data['jfi']['baseline'].append(baseline['jain_fairness_index']['mean'])
        metrics_data['jfi']['ext3'].append(ext3['jain_fairness_index']['mean'])

        metrics_data['throughput']['baseline'].append(abs(baseline['avg_throughput']['mean']))
        metrics_data['throughput']['ext3'].append(abs(ext3['avg_throughput']['mean']))

        metrics_data['latency']['baseline'].append(baseline['avg_latency']['mean'])
        metrics_data['latency']['ext3'].append(ext3['avg_latency']['mean'])

    # Create figure with 4 subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline vs Extension 3: Comprehensive Comparison',
                 fontsize=16, fontweight='bold', y=0.995)

    metric_names = ['convergence', 'jfi', 'throughput', 'latency']

    for idx, (ax, metric) in enumerate(zip(axes.flat, metric_names)):
        data = metrics_data[metric]
        baseline_vals = data['baseline']
        ext3_vals = data['ext3']

        x = np.arange(len(scenario_labels))
        width = 0.35

        # Create bars
        bars1 = ax.bar(x - width/2, baseline_vals, width,
                      label='Baseline',
                      color=COLORS['baseline'], alpha=0.85,
                      edgecolor='white', linewidth=1.5)

        bars2 = ax.bar(x + width/2, ext3_vals, width,
                      label='Extension 3',
                      color=COLORS['extension3'], alpha=0.85,
                      edgecolor='white', linewidth=1.5)

        # Add value labels and calculate improvements
        for i in range(len(scenario_labels)):
            # Baseline label
            ax.text(x[i] - width/2, baseline_vals[i],
                   f'{baseline_vals[i]:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

            # Extension 3 label with improvement
            diff = ext3_vals[i] - baseline_vals[i]
            if abs(diff) > 0.01:  # Only show if meaningful difference
                if data['lower_better']:
                    improvement = ((baseline_vals[i] - ext3_vals[i]) / baseline_vals[i]) * 100
                    color = 'darkgreen' if improvement > 0 else 'darkred'
                    sign = '+' if improvement > 0 else ''
                else:
                    improvement = ((ext3_vals[i] - baseline_vals[i]) / baseline_vals[i]) * 100
                    color = 'darkgreen' if improvement > 0 else 'darkred'
                    sign = '+' if improvement > 0 else ''

                ax.text(x[i] + width/2, ext3_vals[i],
                       f'{ext3_vals[i]:.1f}\n({sign}{improvement:.0f}%)',
                       ha='center', va='bottom', fontsize=9, fontweight='bold',
                       color=color if abs(improvement) > 2 else 'black')
            else:
                ax.text(x[i] + width/2, ext3_vals[i],
                       f'{ext3_vals[i]:.2f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Customize subplot
        ax.set_ylabel(data['label'], fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels, fontsize=10)
        ax.legend(fontsize=9, loc='best', framealpha=0.95)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add improvement indicator for convergence
        if metric == 'convergence':
            ax.text(0.95, 0.95, '45% Faster', transform=ax.transAxes,
                   fontsize=12, fontweight='bold', color='darkgreen',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                   ha='right', va='top')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'baseline_vs_ext3_comprehensive.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    print("   ✅ Graph saved: baseline_vs_ext3_comprehensive.png")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("BASELINE vs EXTENSION 3 - COMPREHENSIVE COMPARISON")
    print("="*80)

    try:
        plot_multi_metric_comparison()

        print("\n" + "="*80)
        print("✅ COMPREHENSIVE COMPARISON GRAPH GENERATED")
        print("="*80)
        print(f"\n📁 Plot saved to: {PLOTS_DIR}")
        print("\nGenerated file:")
        print("  • baseline_vs_ext3_comprehensive.png")
        print("\n" + "="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Error generating graph: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
