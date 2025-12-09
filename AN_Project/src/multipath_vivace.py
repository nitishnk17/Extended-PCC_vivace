"""
Multipath Rate Allocation Extension
Split traffic across multiple paths with different characteristics
"""
import numpy as np
from typing import List, Dict, Tuple
import logging

from .adaptive_vivace import AdaptiveVivace
from .network_simulator import NetworkSimulator

logger = logging.getLogger(__name__)


class MultipathNetwork:
    """
    Network with multiple paths
    Each path has different characteristics (bandwidth, delay, loss)
    """
    
    def __init__(self, paths: List[Dict]):
        """
        Initialize multipath network
        
        Args:
            paths: List of path configurations
                   Each dict should have: bandwidth_mbps, delay_ms, loss_rate, name
        """
        self.paths = []
        self.path_names = []
        
        for i, path_config in enumerate(paths):
            from src.config import NetworkConfig
            
            config = NetworkConfig(
                bandwidth_mbps=path_config.get('bandwidth_mbps', 10.0),
                delay_ms=path_config.get('delay_ms', 50.0),
                queue_size=path_config.get('queue_size', 100),
                loss_rate=path_config.get('loss_rate', 0.0)
            )
            
            network = NetworkSimulator(config)
            self.paths.append(network)
            self.path_names.append(path_config.get('name', f'Path {i}'))
        
        logger.info(f"Multipath network initialized with {len(self.paths)} paths")
        for i, name in enumerate(self.path_names):
            path = self.paths[i]
            logger.info(f"  {name}: {path.bandwidth_mbps} Mbps, {path.delay_ms} ms, loss={path.loss_rate}")
    
    def get_num_paths(self) -> int:
        """Get number of paths"""
        return len(self.paths)
    
    def get_path(self, path_id: int) -> NetworkSimulator:
        """Get specific path"""
        return self.paths[path_id]
    
    def reset_all(self):
        """Reset all paths"""
        for path in self.paths:
            path.reset()


class MultipathVivace(AdaptiveVivace):
    """
    Vivace with multipath support
    
    Features:
    - Per-path utility tracking
    - Softmax rate allocation across paths
    - Path correlation learning
    - Multi-armed bandit exploration
    """
    
    def __init__(self, multipath_network: MultipathNetwork, config):
        """
        Initialize Multipath Vivace
        
        Args:
            multipath_network: MultipathNetwork instance
            config: Configuration object
        """
        # Initialize with first path for compatibility
        super().__init__(multipath_network.get_path(0), config)
        
        self.multipath_network = multipath_network
        self.num_paths = multipath_network.get_num_paths()
        
        # Per-path state
        self.path_rates = [1.0] * self.num_paths  # Rate allocated to each path
        self.path_utilities = [0.0] * self.num_paths  # Recent utility for each path
        self.path_metrics_history = [[] for _ in range(self.num_paths)]  # Metrics history
        
        # Allocation parameters
        self.temperature = config.vivace.exploration_factor * 10  # For softmax
        self.min_path_rate = 0.1  # Minimum rate per path
        
        # Exploration (epsilon-greedy + softmax)
        self.epsilon = 0.1  # Probability of random exploration
        
        # Path correlation matrix (learned over time)
        self.path_correlations = np.eye(self.num_paths)  # Start with no correlation
        
        logger.info(f"Multipath Vivace initialized with {self.num_paths} paths")
    
    def compute_path_utilities(self, current_time: float) -> List[float]:
        """
        Compute utility for each path independently
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of per-path utilities
        """
        mi_duration = self.monitor_interval_ms / 1000.0
        path_duration = mi_duration / self.num_paths
        
        utilities = []
        
        for path_id in range(self.num_paths):
            path = self.multipath_network.get_path(path_id)
            rate = self.path_rates[path_id]
            
            # Temporarily switch network for this path
            original_network = self.network
            self.network = path
            
            # Send at allocated rate
            metrics = self._send_at_rate_adaptive(rate, path_duration, current_time)
            
            # Compute utility
            utility_func = self.utility_bank.get_utility_function(self.traffic_type)
            utility = utility_func(
                metrics['throughput'],
                metrics['latency'],
                metrics['loss']
            )
            
            utilities.append(utility)
            self.path_metrics_history[path_id].append(metrics)
            
            # Keep only recent history
            if len(self.path_metrics_history[path_id]) > 20:
                self.path_metrics_history[path_id].pop(0)
            
            # Restore network
            self.network = original_network
        
        self.path_utilities = utilities
        return utilities
    
    def allocate_rates_softmax(self, total_rate: float) -> List[float]:
        """
        Allocate total rate across paths using softmax
        
        Args:
            total_rate: Total sending rate to allocate
            
        Returns:
            List of per-path rates
        """
        # Epsilon-greedy: sometimes explore randomly
        if np.random.random() < self.epsilon:
            # Random allocation
            random_weights = np.random.dirichlet([1.0] * self.num_paths)
            allocated_rates = random_weights * total_rate
            return allocated_rates.tolist()
        
        # Softmax based on utilities
        utilities = np.array(self.path_utilities)
        
        # Add small noise to break ties
        utilities = utilities + np.random.normal(0, 0.01, self.num_paths)
        
        # Softmax allocation
        exp_utilities = np.exp(utilities / self.temperature)
        probabilities = exp_utilities / np.sum(exp_utilities)
        
        # Allocate rates proportionally
        allocated_rates = probabilities * total_rate

        # Ensure minimum rate per path
        for i in range(self.num_paths):
            allocated_rates[i] = max(allocated_rates[i], self.min_path_rate)

        # Normalize to maintain total rate
        # If enforcing minimums pushed us over budget, scale down only the rates above minimum
        rate_sum = np.sum(allocated_rates)
        if rate_sum > total_rate:
            # Calculate excess and which rates can be reduced
            excess = rate_sum - total_rate
            above_min_indices = [i for i in range(self.num_paths) if allocated_rates[i] > self.min_path_rate]

            if above_min_indices:
                # Distribute the excess reduction proportionally among rates above minimum
                above_min_sum = sum(allocated_rates[i] - self.min_path_rate for i in above_min_indices)

                if above_min_sum > excess:
                    # We can reduce to meet the target
                    for i in above_min_indices:
                        reduction_share = (allocated_rates[i] - self.min_path_rate) / above_min_sum
                        allocated_rates[i] -= excess * reduction_share
                else:
                    # Can't meet target while maintaining minimums
                    # Set all above-min rates to minimum
                    for i in above_min_indices:
                        allocated_rates[i] = self.min_path_rate

        return allocated_rates.tolist()
    
    def learn_path_correlations(self):
        """
        Learn correlation structure between paths
        Paths sharing bottleneck will have correlated throughput
        """
        if any(len(h) < 5 for h in self.path_metrics_history):
            return  # Not enough data
        
        # Extract recent throughput for each path
        throughputs = []
        for path_history in self.path_metrics_history:
            recent = [m['throughput'] for m in path_history[-5:]]
            throughputs.append(recent)
        
        # Compute correlation matrix
        throughputs_array = np.array(throughputs)

        if throughputs_array.shape[1] > 1:
            # Suppress warnings for correlation calculation
            with np.errstate(divide='ignore', invalid='ignore'):
                correlation = np.corrcoef(throughputs_array)

            # Exponential smoothing
            alpha = 0.1
            self.path_correlations = (
                alpha * correlation +
                (1 - alpha) * self.path_correlations
            )
    
    def account_for_correlation(self, utilities: List[float]) -> List[float]:
        """
        Adjust utilities to account for path correlation
        
        Args:
            utilities: Raw per-path utilities
            
        Returns:
            Adjusted utilities
        """
        # If paths are highly correlated (share bottleneck),
        # reduce utility of less efficient path
        
        adjusted = utilities.copy()
        
        for i in range(self.num_paths):
            for j in range(i + 1, self.num_paths):
                correlation = self.path_correlations[i][j]
                
                # If highly correlated
                if abs(correlation) > 0.7:
                    # Penalize the lower utility path
                    if utilities[i] < utilities[j]:
                        adjusted[i] *= (1 - abs(correlation) * 0.5)
                    else:
                        adjusted[j] *= (1 - abs(correlation) * 0.5)
        
        return adjusted
    
    def run_monitor_interval(self, current_time: float) -> Dict:
        """
        Execute one monitor interval with multipath allocation
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Dictionary with MI results
        """
        # Update classification
        if self.packet_count >= self.classification_interval and not self.classification_stable:
            self._update_classification()
        
        # Compute per-path utilities
        path_utilities = self.compute_path_utilities(current_time)
        
        # Learn path correlations
        self.learn_path_correlations()
        
        # Adjust for correlation
        adjusted_utilities = self.account_for_correlation(path_utilities)
        
        # Store adjusted utilities for allocation
        self.path_utilities = adjusted_utilities
        
        # Update total rate using standard Vivace logic
        # For simplicity, use aggregate utility
        aggregate_utility = np.mean(adjusted_utilities)
        
        # Simplified rate update (just for total rate)
        old_total_rate = sum(self.path_rates)
        
        # Gradient based on aggregate utility change
        if len(self.utility_history) > 0:
            utility_change = aggregate_utility - self.utility_history[-1]
            gradient = utility_change / self.learning_rate
        else:
            gradient = 0.0
        
        # Update total rate
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradient
        new_total_rate = old_total_rate + self.velocity
        new_total_rate = np.clip(new_total_rate, self.rate_min * self.num_paths, self.rate_max)
        
        # Allocate across paths
        self.path_rates = self.allocate_rates_softmax(new_total_rate)
        
        # Compute aggregate metrics
        aggregate_metrics = self._compute_aggregate_metrics()
        
        # Record history
        self.current_rate = new_total_rate
        self.rate_history.append(new_total_rate)
        self.throughput_history.append(aggregate_metrics['throughput'])
        self.latency_history.append(aggregate_metrics['latency'])
        self.loss_history.append(aggregate_metrics['loss'])
        self.utility_history.append(aggregate_utility)
        self.time_history.append(current_time)
        
        self.iteration += 1
        
        logger.debug(f"Multipath MI {self.iteration}: "
                    f"total_rate={new_total_rate:.2f}, "
                    f"path_rates={[f'{r:.2f}' for r in self.path_rates]}, "
                    f"utilities={[f'{u:.2f}' for u in path_utilities]}")
        
        return {
            'total_rate': new_total_rate,
            'path_rates': self.path_rates,
            'path_utilities': path_utilities,
            'aggregate_metrics': aggregate_metrics,
            'path_correlations': self.path_correlations.tolist()
        }
    
    def _compute_aggregate_metrics(self) -> Dict:
        """
        Compute aggregate metrics across all paths
        
        Returns:
            Aggregated metrics
        """
        if not any(self.path_metrics_history):
            return {'throughput': 0, 'latency': 0, 'loss': 0}
        
        # Get most recent metrics from each path
        recent_metrics = []
        for path_history in self.path_metrics_history:
            if path_history:
                recent_metrics.append(path_history[-1])
        
        if not recent_metrics:
            return {'throughput': 0, 'latency': 0, 'loss': 0}
        
        # Aggregate throughput: sum across paths
        aggregate_throughput = sum(m['throughput'] for m in recent_metrics)
        
        # Aggregate latency: weighted average by rate
        total_rate = sum(self.path_rates)
        if total_rate > 0:
            aggregate_latency = sum(
                m['latency'] * (self.path_rates[i] / total_rate)
                for i, m in enumerate(recent_metrics)
            )
        else:
            aggregate_latency = np.mean([m['latency'] for m in recent_metrics])
        
        # Aggregate loss: weighted average
        if total_rate > 0:
            aggregate_loss = sum(
                m['loss'] * (self.path_rates[i] / total_rate)
                for i, m in enumerate(recent_metrics)
            )
        else:
            aggregate_loss = np.mean([m['loss'] for m in recent_metrics])
        
        return {
            'throughput': aggregate_throughput,
            'latency': aggregate_latency,
            'loss': aggregate_loss
        }
    
    def run(self, duration: float) -> Dict:
        """
        Run multipath Vivace for specified duration
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Results including per-path statistics
        """
        logger.info(f"Starting Multipath Vivace (duration={duration}s)")
        
        current_time = 0.0
        mi_duration = self.monitor_interval_ms / 1000.0
        
        while current_time < duration:
            self.run_monitor_interval(current_time)
            current_time += mi_duration
        
        # Compute results
        results = {
            'avg_throughput': np.mean(self.throughput_history),
            'avg_latency': np.mean(self.latency_history),
            'avg_loss': np.mean(self.loss_history),
            'avg_utility': np.mean(self.utility_history),
            'final_total_rate': self.current_rate,
            'final_path_rates': self.path_rates,
            'path_correlations': self.path_correlations.tolist(),
            'history': {
                'time': self.time_history,
                'rate': self.rate_history,
                'throughput': self.throughput_history,
                'latency': self.latency_history,
            }
        }
        
        # Per-path statistics
        results['per_path'] = []
        for i in range(self.num_paths):
            if self.path_metrics_history[i]:
                path_throughputs = [m['throughput'] for m in self.path_metrics_history[i]]
                path_latencies = [m['latency'] for m in self.path_metrics_history[i]]
                
                path_stats = {
                    'path_id': i,
                    'path_name': self.multipath_network.path_names[i],
                    'avg_throughput': np.mean(path_throughputs),
                    'avg_latency': np.mean(path_latencies),
                    'final_rate': self.path_rates[i],
                    'avg_utility': self.path_utilities[i]
                }
                results['per_path'].append(path_stats)
        
        logger.info(f"Multipath Vivace completed:")
        logger.info(f"  Aggregate Throughput: {results['avg_throughput']:.2f} Mbps")
        logger.info(f"  Average Latency: {results['avg_latency']:.2f} ms")
        
        for path_stats in results['per_path']:
            logger.info(f"  {path_stats['path_name']}: "
                       f"{path_stats['avg_throughput']:.2f} Mbps at rate {path_stats['final_rate']:.2f}")
        
        return results


def create_multipath_scenario(scenario_type: str) -> MultipathNetwork:
    """
    Create predefined multipath scenarios
    
    Args:
        scenario_type: 'heterogeneous', 'symmetric', 'cellular_wifi'
        
    Returns:
        MultipathNetwork instance
    """
    if scenario_type == 'heterogeneous':
        # Different bandwidth and delay
        paths = [
            {'bandwidth_mbps': 10.0, 'delay_ms': 50.0, 'loss_rate': 0.0, 'name': 'Fast Path'},
            {'bandwidth_mbps': 5.0, 'delay_ms': 100.0, 'loss_rate': 0.0, 'name': 'Slow Path'}
        ]
    
    elif scenario_type == 'symmetric':
        # Same characteristics
        paths = [
            {'bandwidth_mbps': 10.0, 'delay_ms': 50.0, 'loss_rate': 0.0, 'name': 'Path 1'},
            {'bandwidth_mbps': 10.0, 'delay_ms': 50.0, 'loss_rate': 0.0, 'name': 'Path 2'}
        ]
    
    elif scenario_type == 'cellular_wifi':
        # Cellular and WiFi
        paths = [
            {'bandwidth_mbps': 20.0, 'delay_ms': 80.0, 'loss_rate': 0.01, 'name': 'Cellular'},
            {'bandwidth_mbps': 50.0, 'delay_ms': 20.0, 'loss_rate': 0.005, 'name': 'WiFi'}
        ]
    
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")
    
    return MultipathNetwork(paths)
