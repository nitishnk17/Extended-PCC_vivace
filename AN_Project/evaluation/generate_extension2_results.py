#!/usr/bin/env python3

"""
Generate Extension 2 Results with Real Simulations

Extension 2: Wireless Loss Differentiation
- Loss Classifier (distinguishes congestion loss from wireless loss)
- Adaptive Loss Coefficient (adjusts response based on loss type)

Usage:
    python generate_extension2_results.py
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.network_simulator import NetworkSimulator
from src.pcc_vivace_baseline import PccVivaceBaseline
from src.pcc_vivace_extension1 import PccVivaceExtension1
from src.pcc_vivace_extension2 import PccVivaceExtension2


def run_simulation(network_config, extension_class, num_runs=5, duration=30.0):
    """Run actual simulations and collect real metrics"""
    all_runs = []

    for run_idx in range(num_runs):
        # Set random seed for reproducibility (baseline should be consistent!)
        np.random.seed(42 + run_idx)

        config = Config()
        config.network.bandwidth_mbps = network_config['bandwidth_mbps']
        config.network.delay_ms = network_config['delay_ms']
        config.network.queue_size = network_config['queue_size']
        config.network.loss_rate = network_config['loss_rate']

        if 'expected_traffic_type' in network_config:
            config.classifier.expected_traffic_type = network_config['expected_traffic_type']

        network = NetworkSimulator(config.network)
        algo = extension_class(network, config)

        results = algo.run(duration=duration)
        all_runs.append(results)

    # Aggregate metrics across runs
    metrics = {}

    throughput_values = [r['avg_throughput'] for r in all_runs]
    metrics['avg_throughput'] = {
        'mean': float(np.mean(throughput_values)),
        'std': float(np.std(throughput_values)),
        'min': float(np.min(throughput_values)),
        'max': float(np.max(throughput_values)),
        'values': throughput_values
    }

    latency_values = [r['avg_latency'] for r in all_runs]
    metrics['avg_latency'] = {
        'mean': float(np.mean(latency_values)),
        'std': float(np.std(latency_values)),
        'min': float(np.min(latency_values)),
        'max': float(np.max(latency_values)),
        'values': latency_values
    }

    loss_values = [r['loss_rate'] for r in all_runs]
    metrics['loss_rate'] = {
        'mean': float(np.mean(loss_values)),
        'std': float(np.std(loss_values)),
        'min': float(np.min(loss_values)),
        'max': float(np.max(loss_values)),
        'values': loss_values
    }

    utility_values = [r['avg_utility'] for r in all_runs]
    metrics['avg_utility'] = {
        'mean': float(np.mean(utility_values)),
        'std': float(np.std(utility_values)),
        'min': float(np.min(utility_values)),
        'max': float(np.max(utility_values)),
        'values': utility_values
    }

    final_rate_values = [r.get('final_rate', r['avg_throughput']) for r in all_runs]
    metrics['final_rate'] = {
        'mean': float(np.mean(final_rate_values)),
        'std': float(np.std(final_rate_values)),
        'min': float(np.min(final_rate_values)),
        'max': float(np.max(final_rate_values)),
        'values': final_rate_values
    }

    metrics['latency_p50'] = {
        'mean': float(np.percentile(latency_values, 50)),
        'std': float(np.std(latency_values))
    }
    metrics['latency_p95'] = {
        'mean': float(np.percentile(latency_values, 95)),
        'std': float(np.std(latency_values))
    }
    metrics['latency_p99'] = {
        'mean': float(np.percentile(latency_values, 99)),
        'std': float(np.std(latency_values))
    }

    # Extension 2 specific metrics
    if extension_class == PccVivaceExtension2:
        # Get loss classification data from results
        loss_stats = [r.get('loss_classification', {}) for r in all_runs]
        if loss_stats and loss_stats[0]:
            wireless_probs = [s.get('p_wireless', 0.5) for s in loss_stats]
            metrics['wireless_loss_detected'] = {
                'mean': float(np.mean(wireless_probs)),
                'std': float(np.std(wireless_probs)),
                'values': wireless_probs
            }

            confidences = [s.get('confidence', 0.0) for s in loss_stats]
            metrics['loss_type_accuracy'] = {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'values': confidences
            }

            lambdas = [s.get('current_lambda', 1.0) for s in loss_stats]
            metrics['adaptive_coefficient_mean'] = {
                'mean': float(np.mean(lambdas)),
                'std': float(np.std(lambdas)),
                'values': lambdas
            }

    # Rate history
    if 'history' in all_runs[0] and 'rate' in all_runs[0]['history']:
        rate_history = all_runs[0]['history']['rate']
        sample_rate = max(1, len(rate_history) // 100)
        metrics['rate_history'] = [float(r) for r in rate_history[::sample_rate]][:100]
    else:
        metrics['rate_history'] = []

    return metrics


def create_scenario(scenario_name, description, network_config):
    """Create a single scenario with baseline, extension1, and extension2 results"""

    print(f"\n  Running scenario: {scenario_name}")

    print(f"    - Running baseline...")
    baseline_metrics = run_simulation(network_config, PccVivaceBaseline, num_runs=5, duration=30.0)

    print(f"    - Running Extension 1...")
    extension1_metrics = run_simulation(network_config, PccVivaceExtension1, num_runs=5, duration=30.0)

    print(f"    - Running Extension 2...")
    extension2_metrics = run_simulation(network_config, PccVivaceExtension2, num_runs=5, duration=30.0)

    scenario = {
        'scenario': scenario_name,
        'description': description,
        'network_config': network_config,
        'baseline': [baseline_metrics],
        'extension1': [extension1_metrics],
        'extension2': [extension2_metrics],
        'statistics': {
            'total_runs': 5,
            'timestamp': datetime.now().isoformat(),
            'version': '2.0',
            'simulation_type': 'real'
        }
    }

    # Print improvement summary
    baseline_util = baseline_metrics['avg_utility']['mean']
    ext1_util = extension1_metrics['avg_utility']['mean']
    ext2_util = extension2_metrics['avg_utility']['mean']
    imp1 = ((ext1_util - baseline_util) / baseline_util) * 100
    imp2 = ((ext2_util - baseline_util) / baseline_util) * 100
    print(f"    ✓ Baseline: {baseline_util:.3f}")
    print(f"    ✓ Extension1: {ext1_util:.3f} ({imp1:+.1f}%)")
    print(f"    ✓ Extension2: {ext2_util:.3f} ({imp2:+.1f}%)")

    return scenario


def generate_all_scenarios():
    """Generate all network scenarios for Extension 2"""

    scenarios = {}

    scenarios['bulk_transfer'] = create_scenario(
        'bulk_transfer',
        'Bulk transfer: Large file download, maximize throughput',
        {
            'bandwidth_mbps': 10.0,
            'delay_ms': 50.0,
            'queue_size': 100,
            'loss_rate': 0.0,
            'expected_traffic_type': 'bulk'
        }
    )

    scenarios['video_streaming'] = create_scenario(
        'video_streaming',
        'Video streaming: 4K video streaming, consistent latency',
        {
            'bandwidth_mbps': 15.0,
            'delay_ms': 30.0,
            'queue_size': 50,
            'loss_rate': 0.002,
            'expected_traffic_type': 'streaming'
        }
    )

    scenarios['realtime_gaming'] = create_scenario(
        'realtime_gaming',
        'Real-time gaming: FPS gaming, minimize latency',
        {
            'bandwidth_mbps': 5.0,
            'delay_ms': 20.0,
            'queue_size': 30,
            'loss_rate': 0.001,
            'expected_traffic_type': 'realtime'
        }
    )

    scenarios['wireless_2pct_loss'] = create_scenario(
        'wireless_2pct_loss',
        'Wireless network: 2% packet loss, lossy conditions',
        {
            'bandwidth_mbps': 8.0,
            'delay_ms': 60.0,
            'queue_size': 80,
            'loss_rate': 0.02,
            'expected_traffic_type': 'bulk'
        }
    )

    scenarios['moderate_latency'] = create_scenario(
        'moderate_latency',
        'High latency network: Cable modem or mobile network',
        {
            'bandwidth_mbps': 5.0,
            'delay_ms': 150.0,
            'queue_size': 200,
            'loss_rate': 0.005,
            'expected_traffic_type': 'bulk'
        }
    )

    scenarios['high_bandwidth'] = create_scenario(
        'high_bandwidth',
        'High bandwidth network: Datacenter interconnect',
        {
            'bandwidth_mbps': 100.0,
            'delay_ms': 5.0,
            'queue_size': 2000,
            'loss_rate': 0.0,
            'expected_traffic_type': 'bulk'
        }
    )

    return scenarios


def save_results(scenarios, output_dir=None):
    """Save results to JSON files"""

    if output_dir is None:
        # Default to parent directory's results folder
        parent_dir = Path(__file__).parent.parent
        output_dir = parent_dir / 'results'

    output_path = Path(output_dir) / 'extensions_1_2'
    output_path.mkdir(parents=True, exist_ok=True)

    # Save complete results
    complete_file = output_path / 'complete_results.json'
    with open(complete_file, 'w') as f:
        json.dump(scenarios, f, indent=2)

    print(f"\n✅ Complete results saved to: {complete_file}")
    print(f"   Scenarios: {len(scenarios)}")
    print(f"   File size: {complete_file.stat().st_size / 1024:.1f} KB")

    # Save individual scenario files
    for scenario_name, scenario_data in scenarios.items():
        scenario_file = output_path / f'{scenario_name}_results.json'
        with open(scenario_file, 'w') as f:
            json.dump(scenario_data, f, indent=2)
        print(f"   ✅ {scenario_file.name} ({scenario_file.stat().st_size / 1024:.1f} KB)")

    return str(complete_file)


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("GENERATING EXTENSION 2 RESULTS (Wireless Loss Differentiation)")
    print("Running REAL SIMULATIONS - This will take several minutes...")
    print("=" * 80 + "\n")

    print("📊 Running simulations...")
    scenarios = generate_all_scenarios()

    print(f"\n✅ Generated {len(scenarios)} scenarios")
    for scenario_name in scenarios.keys():
        print(f"   • {scenario_name}")

    print("\n💾 Saving results...")
    complete_file = save_results(scenarios)

    print("\n" + "=" * 80)
    print("✅ EXTENSION 2 RESULTS GENERATED SUCCESSFULLY")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
