"""
Main entry point for PCC Vivace Extensions
"""
import argparse
import logging
import sys
from pathlib import Path
import json
import numpy as np

from .config import Config, get_bulk_transfer_config, get_streaming_config, get_realtime_config, get_wireless_config
from .network_simulator import NetworkSimulator
from .baseline_vivace import BaselineVivace
from .adaptive_vivace import AdaptiveVivace


def setup_logging(log_level='INFO', log_file=None):
    """Setup logging configuration"""
    import colorlog
    
    # Create formatter
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s: %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    
    handlers = [console]
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers
    )


def run_baseline(config: Config) -> dict:
    """Run baseline Vivace"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Running Baseline PCC Vivace")
    logger.info("=" * 60)
    
    # Create network
    network = NetworkSimulator(config.network)
    
    # Create Vivace instance
    vivace = BaselineVivace(network, config)
    
    # Run
    results = vivace.run(config.experiment.duration)
    
    return results


def run_adaptive(config: Config) -> dict:
    """Run adaptive Vivace with extensions"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Running Adaptive PCC Vivace with Extensions")
    logger.info("=" * 60)
    
    # Create network
    network = NetworkSimulator(config.network)
    
    # Create Vivace instance
    vivace = AdaptiveVivace(network, config)
    
    # Run
    results = vivace.run(config.experiment.duration)
    
    # Get extended results
    extended_results = vivace.get_extended_results()
    
    return extended_results


def run_comparison(config: Config) -> dict:
    """Run both baseline and adaptive for comparison"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Running Comparison: Baseline vs Adaptive")
    logger.info("=" * 60)
    
    # Run baseline
    logger.info("\n--- Baseline Vivace ---")
    baseline_results = run_baseline(config)
    
    # Reset and run adaptive
    logger.info("\n--- Adaptive Vivace ---")
    adaptive_results = run_adaptive(config)
    
    # Compute improvements
    improvements = {
        'throughput': (adaptive_results['avg_throughput'] - baseline_results['avg_throughput']) / baseline_results['avg_throughput'] * 100,
        'latency': (baseline_results['avg_latency'] - adaptive_results['avg_latency']) / baseline_results['avg_latency'] * 100,
        'utility': (adaptive_results['avg_utility'] - baseline_results['avg_utility']) / abs(baseline_results['avg_utility']) * 100,
    }
    
    return {
        'baseline': baseline_results,
        'adaptive': adaptive_results,
        'improvements': improvements
    }


def print_results(results: dict, mode: str):
    """Print formatted results"""
    logger = logging.getLogger(__name__)
    
    if mode == 'compare':
        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON RESULTS")
        logger.info("=" * 60)
        
        # Baseline
        logger.info("\nBaseline Vivace:")
        logger.info(f"  Throughput: {results['baseline']['avg_throughput']:.2f} ± {results['baseline']['std_throughput']:.2f} Mbps")
        logger.info(f"  Latency: {results['baseline']['avg_latency']:.2f} ms (p95={results['baseline']['p95_latency']:.2f}, p99={results['baseline']['p99_latency']:.2f})")
        logger.info(f"  Loss Rate: {results['baseline']['avg_loss']:.4f}")
        logger.info(f"  Utility: {results['baseline']['avg_utility']:.3f}")
        
        # Adaptive
        logger.info("\nAdaptive Vivace:")
        logger.info(f"  Throughput: {results['adaptive']['avg_throughput']:.2f} ± {results['adaptive']['std_throughput']:.2f} Mbps")
        logger.info(f"  Latency: {results['adaptive']['avg_latency']:.2f} ms (p95={results['adaptive']['p95_latency']:.2f}, p99={results['adaptive']['p99_latency']:.2f})")
        logger.info(f"  Loss Rate: {results['adaptive']['avg_loss']:.4f}")
        logger.info(f"  Utility: {results['adaptive']['avg_utility']:.3f}")
        logger.info(f"  Traffic Type: {results['adaptive']['traffic_type']} (confidence={results['adaptive']['classification_confidence']:.2%})")
        
        # Improvements
        logger.info("\nImprovements:")
        logger.info(f"  Throughput: {results['improvements']['throughput']:+.1f}%")
        logger.info(f"  Latency: {results['improvements']['latency']:+.1f}%")
        logger.info(f"  Utility: {results['improvements']['utility']:+.1f}%")
        
    else:
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS")
        logger.info("=" * 60)
        
        logger.info(f"\nAverage Throughput: {results['avg_throughput']:.2f} ± {results['std_throughput']:.2f} Mbps")
        logger.info(f"Average Latency: {results['avg_latency']:.2f} ms")
        logger.info(f"  95th percentile: {results['p95_latency']:.2f} ms")
        logger.info(f"  99th percentile: {results['p99_latency']:.2f} ms")
        logger.info(f"Loss Rate: {results['avg_loss']:.4f}")
        logger.info(f"Average Utility: {results['avg_utility']:.3f}")
        logger.info(f"Final Rate: {results['final_rate']:.2f} Mbps")
        
        if 'traffic_type' in results:
            logger.info(f"\nTraffic Type: {results['traffic_type']}")
            logger.info(f"Classification Confidence: {results['classification_confidence']:.2%}")


def save_results(results: dict, output_dir: str, mode: str):
    """Save results to file"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_file = output_path / f'results_{mode}.json'
    with open(json_file, 'w') as f:
        # Convert numpy types to native Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        json.dump(convert(results), f, indent=2)
    
    logger = logging.getLogger(__name__)
    logger.info(f"\nResults saved to {json_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='PCC Vivace Extensions - Application-Aware Congestion Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline Vivace
  python src/main.py --mode baseline --duration 60
  
  # Run adaptive Vivace for streaming
  python src/main.py --mode adaptive --traffic-type streaming --duration 60
  
  # Compare baseline vs adaptive
  python src/main.py --mode compare --duration 60
  
  # Use custom config
  python src/main.py --config configs/custom.yaml
  
  # Enable debug logging
  python src/main.py --mode adaptive --debug --log-file debug.log
        """
    )
    
    # Mode
    parser.add_argument('--mode', type=str, default='compare',
                       choices=['baseline', 'adaptive', 'compare'],
                       help='Execution mode')
    
    # Configuration
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--traffic-type', type=str, default='default',
                       choices=['default', 'bulk', 'streaming', 'realtime', 'wireless'],
                       help='Traffic type preset')
    
    # Network parameters
    parser.add_argument('--bandwidth', type=float, default=10.0,
                       help='Bandwidth in Mbps')
    parser.add_argument('--delay', type=float, default=50.0,
                       help='One-way delay in ms')
    parser.add_argument('--queue-size', type=int, default=100,
                       help='Queue size in packets')
    parser.add_argument('--loss-rate', type=float, default=0.0,
                       help='Random loss rate (0-1)')
    
    # Experiment parameters
    parser.add_argument('--duration', type=float, default=60.0,
                       help='Duration in seconds')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='results/latest',
                       help='Output directory')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results')
    
    # Logging
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.debug else args.log_level
    setup_logging(log_level, args.log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("PCC Vivace Extensions - Starting")
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")
    
    # Load or create configuration
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = Config.from_file(args.config)
    else:
        # Create config based on traffic type
        if args.traffic_type == 'bulk':
            config = get_bulk_transfer_config()
        elif args.traffic_type == 'streaming':
            config = get_streaming_config()
        elif args.traffic_type == 'realtime':
            config = get_realtime_config()
        elif args.traffic_type == 'wireless':
            config = get_wireless_config()
        else:
            config = Config()
        
        # Override with command line arguments
        config.network.bandwidth_mbps = args.bandwidth
        config.network.delay_ms = args.delay
        config.network.queue_size = args.queue_size
        config.network.loss_rate = args.loss_rate
        config.experiment.duration = args.duration
        config.experiment.output_dir = args.output_dir
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Network: {config.network.bandwidth_mbps} Mbps, {config.network.delay_ms} ms delay")
    logger.info(f"  Queue: {config.network.queue_size} packets")
    logger.info(f"  Loss: {config.network.loss_rate:.4f}")
    logger.info(f"  Duration: {config.experiment.duration} seconds")
    logger.info(f"  Mode: {args.mode}")
    
    # Run experiment
    try:
        if args.mode == 'baseline':
            results = run_baseline(config)
        elif args.mode == 'adaptive':
            results = run_adaptive(config)
        else:  # compare
            results = run_comparison(config)
        
        # Print results
        print_results(results, args.mode)
        
        # Save results
        if not args.no_save:
            save_results(results, args.output_dir, args.mode)
        
        logger.info("\nExperiment completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
