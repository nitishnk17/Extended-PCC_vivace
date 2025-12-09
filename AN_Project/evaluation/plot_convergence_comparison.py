#!/usr/bin/env python3

"""
Extension 3 Convergence Time Comparison

Generates a bar graph comparing convergence times between Baseline and Extension 3,
showing the 45% improvement from cooperative exploration mechanism.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'extensions_1_2_3'
PLOTS_DIR = Path(__file__).parent.parent / 'plots'
DPI = 300

# Seaborn-style colors
COLORS = {
    'baseline': '#4C72B0',
    'extension3': '#55A868'
}

def load_scenario_results(scenario_name):
    """Load results for a specific scenario"""
    filepath = RESULTS_DIR / f"{scenario_name}_results.json"
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_convergence_comparison():
    """Create bar graph comparing convergence times"""
    print("\n📊 Generating Convergence Time Comparison Graph...")

    scenarios = ['bulk_transfer', 'video_streaming', 'realtime_gaming']
    scenario_labels = ['Bulk Transfer', 'Video Streaming', 'Real-time Gaming']

    baseline_conv = []
    extension3_conv = []
    baseline_std = []
    extension3_std = []

    # Extract convergence data
    for scenario in scenarios:
        results = load_scenario_results(scenario)

        baseline = results['baseline'][0]
        extension3 = results['extension3'][0]

        baseline_conv.append(baseline['convergence_time_seconds']['mean'])
        extension3_conv.append(extension3['convergence_time_seconds']['mean'])
        baseline_std.append(baseline['convergence_time_seconds']['std'])
        extension3_std.append(extension3['convergence_time_seconds']['std'])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(scenario_labels))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, baseline_conv, width,
                   label='Baseline',
                   color=COLORS['baseline'], alpha=0.85,
                   yerr=baseline_std, capsize=5, edgecolor='white', linewidth=1.5)

    bars2 = ax.bar(x + width/2, extension3_conv, width,
                   label='Extension 3',
                   color=COLORS['extension3'], alpha=0.85,
                   yerr=extension3_std, capsize=5, edgecolor='white', linewidth=1.5)

    # Add value labels on bars with improvement percentage
    for i in range(len(scenario_labels)):
        improvement = ((baseline_conv[i] - extension3_conv[i]) / baseline_conv[i]) * 100

        # Label on baseline bar
        ax.text(x[i] - width/2, baseline_conv[i] + baseline_std[i] + 0.3,
               f'{baseline_conv[i]:.1f}s',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Label on extension3 bar with improvement
        ax.text(x[i] + width/2, extension3_conv[i] + extension3_std[i] + 0.3,
               f'{extension3_conv[i]:.1f}s\n(-{improvement:.0f}%)',
               ha='center', va='bottom', fontsize=10, fontweight='bold',
               color='darkgreen')

    # Customize plot
    ax.set_xlabel('Traffic Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('Convergence Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Time Comparison',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=11)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(baseline_conv) * 1.25)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'EXT3_convergence_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close()

    print("   ✅ Graph saved: EXT3_convergence_comparison.png")

    # Print summary statistics
    print("\n" + "="*80)
    print("CONVERGENCE TIME COMPARISON - BASELINE VS EXTENSION 3")
    print("="*80)

    for i, scenario in enumerate(scenario_labels):
        improvement = ((baseline_conv[i] - extension3_conv[i]) / baseline_conv[i]) * 100
        print(f"\n{scenario.upper()}:")
        print(f"  Baseline:    {baseline_conv[i]:.1f}s (±{baseline_std[i]:.2f}s)")
        print(f"  Extension 3: {extension3_conv[i]:.1f}s (±{extension3_std[i]:.2f}s)")
        print(f"  Improvement: {improvement:.1f}% faster")

    print("\n" + "="*80)


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("EXTENSION 3 - CONVERGENCE TIME COMPARISON")
    print("="*80)

    try:
        plot_convergence_comparison()

        print("\n" + "="*80)
        print("✅ CONVERGENCE COMPARISON GRAPH GENERATED SUCCESSFULLY")
        print("="*80)
        print(f"\n📁 Plot saved to: {PLOTS_DIR}")
        print("\nGenerated file:")
        print("  • EXT3_convergence_comparison.png")
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
