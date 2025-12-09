"""
Multi-Flow Coordination Extension
Multiple Vivace flows competing over shared bottleneck with implicit coordination
"""
import numpy as np
from typing import List, Dict, Optional
import logging

from .adaptive_vivace import AdaptiveVivace

logger = logging.getLogger(__name__)


class CoordinatedVivace(AdaptiveVivace):
    """
    Vivace with multi-flow coordination capabilities
    
    Features:
    - Contention detection through gradient oscillations
    - Cooperative exploration (alternating probes)
    - Fairness-aware utility function
    - Virtual queue estimation
    """
    
    def __init__(self, network, config, flow_id: int = 0, coordinator=None):
        """
        Initialize Coordinated Vivace
        
        Args:
            network: SHARED NetworkSimulator instance
            config: Configuration object
            flow_id: Unique flow identifier
            coordinator: FlowCoordinator instance (optional)
        """
        super().__init__(network, config)
        
        self.flow_id = flow_id
        self.coordinator = coordinator
        
        # Coordination parameters
        self.contention_threshold = 0.3  # Gradient oscillation threshold
        self.fairness_weight = 0.5
        self.exploration_slot = 0  # For alternating exploration
        
        # State tracking
        self.gradient_history = []
        self.gradient_sign_changes = 0
        self.contention_detected = False
        self.estimated_fair_share = None
        self.virtual_queue = 0.0
        
        logger.info(f"Coordinated Vivace initialized for flow {flow_id}")
    
    def detect_contention(self) -> bool:
        """
        Detect if flow is experiencing contention based on gradient oscillations
        
        Returns:
            True if contention detected
        """
        if len(self.gradient_history) < 10:
            return False
        
        # Count sign changes in recent gradients
        recent_gradients = self.gradient_history[-10:]
        sign_changes = 0
        
        for i in range(len(recent_gradients) - 1):
            if np.sign(recent_gradients[i]) != np.sign(recent_gradients[i + 1]):
                sign_changes += 1
        
        # High sign change rate indicates contention
        oscillation_rate = sign_changes / len(recent_gradients)
        
        self.contention_detected = oscillation_rate > self.contention_threshold
        
        if self.contention_detected and not hasattr(self, '_contention_logged'):
            logger.info(f"Flow {self.flow_id}: Contention detected (oscillation={oscillation_rate:.2f})")
            self._contention_logged = True
        
        return self.contention_detected
    
    def estimate_fair_share(self, total_capacity: float, num_flows: int) -> float:
        """
        Estimate fair share of bandwidth
        
        Args:
            total_capacity: Total link capacity
            num_flows: Number of competing flows
            
        Returns:
            Estimated fair share rate
        """
        if num_flows == 0:
            return total_capacity
        
        # Simple max-min fairness
        fair_share = total_capacity / num_flows
        
        self.estimated_fair_share = fair_share
        return fair_share
    
    def should_explore(self) -> bool:
        """
        Decide if this flow should explore in current MI (alternating exploration)
        
        Returns:
            True if flow should explore
        """
        if not self.coordinator:
            return True  # No coordination, always explore
        
        # Get exploration schedule from coordinator
        num_flows = self.coordinator.get_num_flows()
        
        # Alternate exploration: flow explores if iteration % num_flows == flow_id
        should_explore = (self.iteration % num_flows) == self.flow_id
        
        return should_explore
    
    def compute_fairness_penalty(self) -> float:
        """
        Compute penalty for deviating from fair share
        
        Returns:
            Fairness penalty value
        """
        if self.estimated_fair_share is None:
            return 0.0
        
        # Penalty proportional to deviation from fair share
        deviation = abs(self.current_rate - self.estimated_fair_share)
        penalty = self.fairness_weight * deviation
        
        return penalty
    
    def update_virtual_queue(self, rtt: float, baseline_rtt: float):
        """
        Estimate virtual queue occupancy from RTT inflation
        
        Args:
            rtt: Current RTT
            baseline_rtt: Baseline RTT (no queue)
        """
        # Queue delay = RTT - baseline_RTT
        queue_delay_ms = max(0, rtt - baseline_rtt)
        
        # Convert to queue occupancy (packets)
        bandwidth_bps = self.network.bandwidth_bps
        queue_delay_s = queue_delay_ms / 1000.0
        
        # Bytes in queue = bandwidth * delay
        bytes_in_queue = bandwidth_bps * queue_delay_s / 8
        packets_in_queue = bytes_in_queue / 1500
        
        # Exponential smoothing
        alpha = 0.3
        self.virtual_queue = alpha * packets_in_queue + (1 - alpha) * self.virtual_queue
    
    def _send_at_rate_adaptive(self, rate: float, duration: float, current_time: float) -> Dict:
        """
        Send data at specified rate with packet tracking for classification
        NOW PROPERLY USES flow_id FOR SHARED NETWORK
        
        Args:
            rate: Sending rate in Mbps
            duration: Duration in seconds
            current_time: Current time
            
        Returns:
            Dictionary with metrics
        """
        packet_size = self._get_packet_size()
        bits_per_packet = packet_size * 8
        rate_bps = rate * 1e6
        
        # Calculate packets to send
        packets_to_send = int((rate_bps * duration) / bits_per_packet)
        
        if packets_to_send > 0:
            inter_packet_time = duration / packets_to_send
        else:
            inter_packet_time = duration
            packets_to_send = 1
        
        # Send packets with flow_id
        packets_sent = 0
        packets_dropped = 0
        rtt_samples = []
        
        t = 0.0
        while t < duration and packets_sent + packets_dropped < packets_to_send:
            # Add packet to classifier
            self.classifier.add_packet(packet_size, current_time + t)
            self.packet_count += 1
            
            # Send with flow_id - CRITICAL FIX
            packet = self.network.send_packet(packet_size, current_time + t, flow_id=self.flow_id)
            
            if packet.dropped:
                packets_dropped += 1
            else:
                packets_sent += 1
                rtt = self.network.get_rtt()
                rtt_samples.append(rtt)
            
            t += inter_packet_time
        
        # Process network queue (shared by all flows)
        completed = self.network.process_queue(duration)
        
        # Get per-flow metrics - CRITICAL FIX
        network_metrics = self.network.get_metrics(duration, flow_id=self.flow_id)
        
        throughput = network_metrics['throughput']
        latency = np.mean(rtt_samples) if rtt_samples else self.network.delay_ms * 2
        
        total = packets_sent + packets_dropped
        loss = packets_dropped / total if total > 0 else 0.0
        
        # Track for loss classification
        self.loss_classifier.add_observation(loss, latency)
        
        return {
            'throughput': throughput,
            'latency': latency,
            'loss': loss,
            'packets_sent': packets_sent,
            'packets_dropped': packets_dropped
        }
    
    def run_monitor_interval(self, current_time: float) -> Dict:
        """
        Execute one monitor interval with coordination
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Dictionary with MI results
        """
        mi_duration = self.monitor_interval_ms / 1000.0
        
        # Update classification if needed
        if self.packet_count >= self.classification_interval and not self.classification_stable:
            self._update_classification()
        
        # Detect contention
        self.detect_contention()
        
        # Get fair share estimate from coordinator
        if self.coordinator:
            total_bw = self.network.bandwidth_mbps
            num_flows = self.coordinator.get_num_flows()
            self.estimate_fair_share(total_bw, num_flows)
            
            # Register our current rate
            self.coordinator.register_flow_rate(self.flow_id, self.current_rate)
        
        # Decide exploration strategy
        if self.contention_detected and self.coordinator:
            # Coordinated exploration
            if self.should_explore():
                # This flow explores
                rates_to_test = self._generate_test_rates()
            else:
                # This flow maintains rate
                rates_to_test = [self.current_rate]
        else:
            # Normal exploration
            rates_to_test = self._generate_test_rates()
        
        # Test rates and collect metrics
        metrics_list = []
        for rate in rates_to_test:
            metrics = self._send_at_rate_adaptive(rate, mi_duration / len(rates_to_test), current_time)
            metrics_list.append(metrics)
            
            # Update virtual queue estimate
            baseline_rtt = self.network.delay_ms * 2
            self.update_virtual_queue(metrics['latency'], baseline_rtt)
        
        # Get utility function
        utility_func = self.utility_bank.get_utility_function(self.traffic_type)
        
        # Apply loss differentiation
        if self.loss_classifier.enabled:
            metrics_list = self._adjust_for_wireless_loss(metrics_list)
        
        # Compute utilities with fairness penalty
        utilities = []
        for metrics in metrics_list:
            base_utility = utility_func(
                metrics['throughput'],
                metrics['latency'],
                metrics['loss']
            )
            
            # Add fairness penalty if contention detected
            if self.contention_detected:
                fairness_penalty = self.compute_fairness_penalty()
                utility = base_utility - fairness_penalty
            else:
                utility = base_utility
            
            utilities.append(utility)
        
        # Compute gradient
        gradient = self._compute_gradient(rates_to_test, utilities)
        self.gradient_history.append(gradient)
        
        # Limit gradient history
        if len(self.gradient_history) > 20:
            self.gradient_history.pop(0)
        
        # Update rate with momentum
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradient
        new_rate = self.current_rate + self.velocity
        
        # Clip to bounds
        new_rate = np.clip(new_rate, self.rate_min, self.rate_max)
        
        # If fair share is known, don't exceed it significantly (during contention)
        if self.contention_detected and self.estimated_fair_share:
            max_rate = self.estimated_fair_share * 1.1  # Allow 10% overage
            new_rate = min(new_rate, max_rate)
        
        # Record metrics
        avg_metrics = self._average_metrics(metrics_list)
        self.rate_history.append(self.current_rate)
        self.throughput_history.append(avg_metrics['throughput'])
        self.latency_history.append(avg_metrics['latency'])
        self.loss_history.append(avg_metrics['loss'])
        self.utility_history.append(np.mean(utilities))
        self.time_history.append(current_time)
        
        # Update state
        old_rate = self.current_rate
        self.current_rate = new_rate
        self.iteration += 1
        
        logger.debug(f"Flow {self.flow_id} MI {self.iteration}: "
                    f"rate {old_rate:.2f}->{new_rate:.2f}, "
                    f"contention={self.contention_detected}")
        
        return {
            'rate': new_rate,
            'metrics': avg_metrics,
            'utility': np.mean(utilities),
            'gradient': gradient,
            'traffic_type': self.traffic_type,
            'confidence': self.classification_confidence,
            'contention': self.contention_detected,
            'fair_share': self.estimated_fair_share,
            'virtual_queue': self.virtual_queue
        }


class FlowCoordinator:
    """
    Coordinator for multiple Vivace flows
    Provides implicit coordination through shared state
    """
    
    def __init__(self):
        """Initialize flow coordinator"""
        self.flows = {}  # flow_id -> CoordinatedVivace
        self.flow_rates = {}  # flow_id -> current_rate
        self.num_flows = 0
        
        logger.info("Flow coordinator initialized")
    
    def register_flow(self, flow_id: int, flow):
        """
        Register a flow with coordinator
        
        Args:
            flow_id: Flow identifier
            flow: CoordinatedVivace instance
        """
        self.flows[flow_id] = flow
        self.flow_rates[flow_id] = 0.0
        self.num_flows = len(self.flows)
        
        logger.info(f"Flow {flow_id} registered with coordinator (total flows: {self.num_flows})")
    
    def register_flow_rate(self, flow_id: int, rate: float):
        """
        Update flow's current rate
        
        Args:
            flow_id: Flow identifier
            rate: Current sending rate
        """
        self.flow_rates[flow_id] = rate
    
    def get_num_flows(self) -> int:
        """Get number of active flows"""
        return self.num_flows
    
    def get_aggregate_rate(self) -> float:
        """Get total sending rate of all flows"""
        return sum(self.flow_rates.values())
    
    def get_flow_rates(self) -> Dict[int, float]:
        """Get rates of all flows"""
        return self.flow_rates.copy()


def run_multiflow_experiment(config, num_flows: int = 4, duration: float = 60.0) -> Dict:
    """
    Run experiment with multiple coordinated Vivace flows
    ALL FLOWS SHARE THE SAME NETWORK - CRITICAL FIX
    
    Args:
        config: Configuration object
        num_flows: Number of flows to create
        duration: Experiment duration
        
    Returns:
        Results dictionary
    """
    from src.network_simulator import NetworkSimulator
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Multi-Flow Experiment: {num_flows} flows")
    logger.info(f"{'='*60}")
    
    # Create SINGLE shared network - CRITICAL FIX
    network = NetworkSimulator(config.network)
    
    # Create coordinator
    coordinator = FlowCoordinator()
    
    # Create flows - ALL USING SAME NETWORK
    flows = []
    for i in range(num_flows):
        flow = CoordinatedVivace(network, config, flow_id=i, coordinator=coordinator)
        coordinator.register_flow(i, flow)
        flows.append(flow)
    
    # Run simulation
    mi_duration = config.vivace.monitor_interval_ms / 1000.0
    current_time = 0.0
    
    while current_time < duration:
        # Each flow runs its MI
        for flow in flows:
            flow.run_monitor_interval(current_time)
        
        current_time += mi_duration
    
    # Collect results
    results = {
        'num_flows': num_flows,
        'flows': []
    }
    
    for i, flow in enumerate(flows):
        flow_results = {
            'flow_id': i,
            'avg_throughput': np.mean(flow.throughput_history),
            'avg_latency': np.mean(flow.latency_history),
            'avg_loss': np.mean(flow.loss_history),
            'final_rate': flow.current_rate,
            'contention_detected': flow.contention_detected,
            'history': {
                'time': flow.time_history,
                'rate': flow.rate_history,
                'throughput': flow.throughput_history,
                'latency': flow.latency_history,
            }
        }
        results['flows'].append(flow_results)
    
    # Compute fairness
    throughputs = [f['avg_throughput'] for f in results['flows']]
    results['fairness_index'] = compute_jain_fairness(throughputs)
    
    # Aggregate metrics
    results['aggregate_throughput'] = sum(throughputs)
    results['avg_latency'] = np.mean([f['avg_latency'] for f in results['flows']])
    
    logger.info(f"\nMulti-Flow Results:")
    logger.info(f"  Aggregate Throughput: {results['aggregate_throughput']:.2f} Mbps")
    logger.info(f"  Fairness Index: {results['fairness_index']:.4f}")
    logger.info(f"  Average Latency: {results['avg_latency']:.2f} ms")
    
    for i, flow_result in enumerate(results['flows']):
        logger.info(f"  Flow {i}: {flow_result['avg_throughput']:.2f} Mbps")
    
    return results


def compute_jain_fairness(values: List[float]) -> float:
    """
    Compute Jain's fairness index
    
    Args:
        values: List of values (e.g., throughputs)
        
    Returns:
        Fairness index (0 to 1, 1 is perfectly fair)
    """
    if len(values) == 0:
        return 1.0
    
    n = len(values)
    sum_values = sum(values)
    sum_squares = sum(v**2 for v in values)
    
    if sum_squares == 0:
        return 1.0
    
    fairness = (sum_values ** 2) / (n * sum_squares)
    return fairness
