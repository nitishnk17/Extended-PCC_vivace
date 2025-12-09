"""
Baseline PCC Vivace Implementation
Original algorithm without extensions
"""
import numpy as np
from typing import Dict, List, Tuple
import logging
import time

logger = logging.getLogger(__name__)


class BaselineVivace:
    """
    Baseline PCC Vivace congestion control algorithm
    
    Core algorithm:
    1. Divide time into Monitor Intervals (MIs)
    2. Test multiple sending rates per MI
    3. Measure utility for each rate
    4. Update rate using gradient ascent
    """
    
    def __init__(self, network, config):
        """
        Initialize Baseline Vivace
        
        Args:
            network: NetworkSimulator instance
            config: Config object with vivace and utilities settings
        """
        self.network = network
        self.config = config
        
        # Vivace parameters
        self.monitor_interval_ms = config.vivace.monitor_interval_ms
        self.learning_rate = config.vivace.learning_rate
        self.rate_min = config.vivace.rate_min
        self.rate_max = config.vivace.rate_max
        self.exploration_factor = config.vivace.exploration_factor
        self.momentum = config.vivace.momentum
        self.rate_steps = config.vivace.rate_steps
        
        # State
        self.current_rate = 1.0  # Mbps
        self.velocity = 0.0  # For momentum
        self.iteration = 0

        # Metrics history (bounded to prevent memory leaks)
        self.max_history_length = 10000  # Keep last 10000 samples
        self.rate_history = []
        self.throughput_history = []
        self.latency_history = []
        self.loss_history = []
        self.utility_history = []
        self.time_history = []
        
        # Utility function (default)
        from .utility_bank import UtilityFunctionBank
        self.utility_bank = UtilityFunctionBank(config.utilities)
        self.utility_func = self.utility_bank.utility_default
        
        logger.info(f"Baseline Vivace initialized (MI={self.monitor_interval_ms}ms, "
                   f"lr={self.learning_rate})")
    
    def run_monitor_interval(self, current_time: float) -> Dict:
        """
        Execute one monitor interval
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Dictionary with MI results
        """
        mi_duration = self.monitor_interval_ms / 1000.0  # Convert to seconds
        
        # Generate rates to test
        rates_to_test = self._generate_test_rates()
        
        # Test each rate
        metrics_list = []
        for rate in rates_to_test:
            metrics = self._send_at_rate(rate, mi_duration / len(rates_to_test), current_time)
            metrics_list.append(metrics)
        
        # Compute utilities
        utilities = []
        for metrics in metrics_list:
            utility = self.utility_func(
                metrics['throughput'],
                metrics['latency'],
                metrics['loss']
            )
            utilities.append(utility)
        
        # Compute gradient
        gradient = self._compute_gradient(rates_to_test, utilities)
        
        # Update rate with momentum
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradient
        new_rate = self.current_rate + self.velocity
        
        # Clip to bounds
        new_rate = np.clip(new_rate, self.rate_min, self.rate_max)
        
        # Record metrics (with bounds to prevent memory leaks)
        avg_metrics = self._average_metrics(metrics_list)
        self._append_bounded(self.rate_history, self.current_rate)
        self._append_bounded(self.throughput_history, avg_metrics['throughput'])
        self._append_bounded(self.latency_history, avg_metrics['latency'])
        self._append_bounded(self.loss_history, avg_metrics['loss'])
        self._append_bounded(self.utility_history, np.mean(utilities))
        self._append_bounded(self.time_history, current_time)
        
        # Update state
        old_rate = self.current_rate
        self.current_rate = new_rate
        self.iteration += 1
        
        logger.debug(f"MI {self.iteration}: rate {old_rate:.2f}->{new_rate:.2f}, "
                    f"gradient={gradient:.4f}, utility={np.mean(utilities):.3f}")
        
        return {
            'rate': new_rate,
            'metrics': avg_metrics,
            'utility': np.mean(utilities),
            'gradient': gradient
        }
    
    def _generate_test_rates(self) -> List[float]:
        """
        Generate rates to test in this MI
        
        Returns:
            List of rates (Mbps)
        """
        if self.rate_steps == 3:
            # Test current, slightly lower, slightly higher
            delta = self.current_rate * self.exploration_factor
            rates = [
                max(self.rate_min, self.current_rate - delta),
                self.current_rate,
                min(self.rate_max, self.current_rate + delta)
            ]
        else:
            # Generate multiple test rates
            delta = self.current_rate * self.exploration_factor
            rates = np.linspace(
                max(self.rate_min, self.current_rate - delta),
                min(self.rate_max, self.current_rate + delta),
                self.rate_steps
            ).tolist()
        
        return rates
    
    def _send_at_rate(self, rate: float, duration: float, current_time: float) -> Dict:
        """
        Send data at specified rate for given duration
        
        Args:
            rate: Sending rate in Mbps
            duration: Duration in seconds
            current_time: Current time
            
        Returns:
            Dictionary with metrics
        """
        packet_size = 1500  # bytes
        bits_per_packet = packet_size * 8
        rate_bps = rate * 1e6
        
        # Calculate packets to send
        packets_to_send = int((rate_bps * duration) / bits_per_packet)
        
        # Calculate inter-packet time
        if packets_to_send > 0:
            inter_packet_time = duration / packets_to_send
        else:
            inter_packet_time = duration
            packets_to_send = 1
        
        # Send packets
        packets_sent = 0
        packets_dropped = 0
        rtt_samples = []
        
        t = 0.0
        while t < duration and packets_sent < packets_to_send:
            packet = self.network.send_packet(packet_size, current_time + t)
            
            if packet.dropped:
                packets_dropped += 1
            else:
                packets_sent += 1
                # Estimate RTT
                rtt = self.network.get_rtt()
                rtt_samples.append(rtt)
            
            t += inter_packet_time
        
        # Process network queue
        completed = self.network.process_queue(duration)
        
        # Calculate metrics
        network_metrics = self.network.get_metrics(duration)
        
        # Use actual achieved throughput
        throughput = network_metrics['throughput']
        
        # Average latency
        if rtt_samples:
            latency = np.mean(rtt_samples)
        else:
            latency = self.network.delay_ms * 2
        
        # Loss rate
        total = packets_sent + packets_dropped
        loss = packets_dropped / total if total > 0 else 0.0
        
        return {
            'throughput': throughput,
            'latency': latency,
            'loss': loss,
            'packets_sent': packets_sent,
            'packets_dropped': packets_dropped
        }
    
    def _compute_gradient(self, rates: List[float], utilities: List[float]) -> float:
        """
        Compute utility gradient using finite differences
        
        Args:
            rates: List of tested rates
            utilities: List of corresponding utilities
            
        Returns:
            Estimated gradient
        """
        if len(rates) < 2 or len(utilities) < 2:
            return 0.0
        
        # Use highest and lowest rate
        max_idx = np.argmax(rates)
        min_idx = np.argmin(rates)
        
        rate_diff = rates[max_idx] - rates[min_idx]
        utility_diff = utilities[max_idx] - utilities[min_idx]
        
        if abs(rate_diff) > 1e-6:
            gradient = utility_diff / rate_diff
        else:
            gradient = 0.0
        
        return gradient
    
    def _average_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Average metrics across multiple measurements"""
        if not metrics_list:
            return {'throughput': 0, 'latency': 0, 'loss': 0}
        
        avg = {
            'throughput': np.mean([m['throughput'] for m in metrics_list]),
            'latency': np.mean([m['latency'] for m in metrics_list]),
            'loss': np.mean([m['loss'] for m in metrics_list])
        }
        return avg
    
    def run(self, duration: float) -> Dict:
        """
        Run Vivace for specified duration
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Dictionary with results
        """
        logger.info(f"Starting Baseline Vivace run (duration={duration}s)")
        
        start_time = time.time()
        current_time = 0.0
        mi_duration = self.monitor_interval_ms / 1000.0
        
        while current_time < duration:
            self.run_monitor_interval(current_time)
            current_time += mi_duration
        
        elapsed = time.time() - start_time
        
        # Compute summary statistics
        results = {
            'avg_throughput': np.mean(self.throughput_history),
            'std_throughput': np.std(self.throughput_history),
            'avg_latency': np.mean(self.latency_history),
            'p95_latency': np.percentile(self.latency_history, 95),
            'p99_latency': np.percentile(self.latency_history, 99),
            'avg_loss': np.mean(self.loss_history),
            'avg_utility': np.mean(self.utility_history),
            'final_rate': self.current_rate,
            'iterations': self.iteration,
            'elapsed_time': elapsed,
            'history': {
                'time': self.time_history,
                'rate': self.rate_history,
                'throughput': self.throughput_history,
                'latency': self.latency_history,
                'loss': self.loss_history,
                'utility': self.utility_history
            }
        }
        
        logger.info(f"Baseline Vivace completed: throughput={results['avg_throughput']:.2f} Mbps, "
                   f"latency={results['avg_latency']:.2f} ms, loss={results['avg_loss']:.4f}")
        
        return results
    
    def reset(self):
        """Reset algorithm state"""
        self.current_rate = 1.0
        self.velocity = 0.0
        self.iteration = 0
        
        self.rate_history.clear()
        self.throughput_history.clear()
        self.latency_history.clear()
        self.loss_history.clear()
        self.utility_history.clear()
        self.time_history.clear()
        
        self.network.reset()

        logger.info("Baseline Vivace reset")

    def _append_bounded(self, history_list: List, value):
        """Append to history list with maximum length bound to prevent memory leaks"""
        history_list.append(value)
        if len(history_list) > self.max_history_length:
            history_list.pop(0)  # Remove oldest element
