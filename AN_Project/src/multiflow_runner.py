"""
Multi-Flow Test Runner for PCC Vivace

Enables multiple PCC flows to compete for shared bandwidth on a single
bottleneck network. Essential for evaluating:
- Extension 3: Distributed Fairness (convergence time, Jain's fairness)
- Extension 4: Any multi-flow scenarios

Architecture:
- Single shared NetworkSimulator (bottleneck link)
- Multiple PCC flow instances (each with unique flow_id)
- Interleaved execution (round-robin scheduling)
- Per-flow and aggregate metrics tracking
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from .network_simulator import NetworkSimulator
from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class FlowMetrics:
    """Metrics for a single flow"""
    flow_id: int
    algorithm: str

    # Rate metrics
    sending_rate: float  # Mbps
    throughput: float  # Mbps

    # Performance metrics
    avg_latency: float  # ms
    loss_rate: float
    utility: float

    # Counters
    packets_sent: int
    packets_acked: int
    packets_lost: int


def jains_fairness_index(values: List[float]) -> float:
    """
    Calculate Jain's Fairness Index

    JFI = (Σx_i)² / (n·Σx_i²)

    Perfect fairness = 1.0
    Poor fairness → 0.0

    Args:
        values: List of throughputs or rates

    Returns:
        Fairness index [0, 1] - clamped to valid range
    """
    if not values or len(values) == 0:
        return 0.0

    n = len(values)
    sum_x = sum(values)
    sum_x_sq = sum(x**2 for x in values)

    if sum_x_sq == 0:
        return 0.0

    jfi = (sum_x ** 2) / (n * sum_x_sq)
    # Clamp to [0, 1] to handle floating-point precision errors
    # In theory JFI ≤ 1.0 always, but numerical computation may exceed 1.0 slightly
    return max(0.0, min(1.0, jfi))


class MultiFlowRunner:
    """
    Run multiple PCC flows competing on shared network

    Key Features:
    - Single shared NetworkSimulator (true bandwidth competition)
    - Round-robin flow execution (fair CPU scheduling)
    - Per-flow metrics tracking
    - Aggregate fairness metrics (Jain's index)
    - Convergence detection

    Usage:
        runner = MultiFlowRunner(network, config)
        flows = runner.create_flows(PccVivaceExtension3, num_flows=3)
        results = runner.run(flows, duration=30.0)
    """

    def __init__(self, network: NetworkSimulator, config: Config):
        """
        Initialize multi-flow runner

        Args:
            network: Shared NetworkSimulator instance
            config: Configuration (shared by all flows)
        """
        self.network = network
        self.config = config

        # Metrics history
        self.throughput_history = []  # List of (time, [flow0_tput, flow1_tput, ...])
        self.jfi_history = []  # List of (time, jfi)
        self.rate_history = []  # List of (time, [flow0_rate, flow1_rate, ...])

        # Timing
        self.measurement_interval = 1.0  # Measure metrics every 1 second

    def create_flows(self, flow_class, num_flows: int, **kwargs) -> List:
        """
        Create multiple flow instances

        Args:
            flow_class: PCC flow class (e.g., PccVivaceExtension3)
            num_flows: Number of flows to create
            **kwargs: Additional arguments for flow constructor

        Returns:
            List of flow instances
        """
        flows = []
        for flow_id in range(num_flows):
            # Each flow gets unique flow_id but shares network
            # multiflow_mode=True enables network stats-based feedback
            flow = flow_class(
                network=self.network,
                config=self.config,
                flow_id=flow_id,
                multiflow_mode=True,  # Enable multiflow mode
                **kwargs
            )
            flows.append(flow)

        logger.info(f"Created {num_flows} flows of type {flow_class.__name__}")
        return flows

    def run(self, flows: List, duration: float,
            convergence_threshold: float = 0.95) -> Dict:
        """
        Run multiple flows concurrently on shared network

        Execution model:
        - Time-sliced round-robin (each flow gets equal CPU time)
        - Flows share single NetworkSimulator queue (true bandwidth competition)
        - Metrics collected periodically

        Args:
            flows: List of PCC flow instances
            duration: Total simulation duration (seconds)
            convergence_threshold: JFI threshold for convergence (default 0.95)

        Returns:
            Dictionary with results and metrics
        """
        num_flows = len(flows)
        logger.info(f"Starting multi-flow simulation: {num_flows} flows, {duration}s duration")

        # Reset network
        self.network.reset()

        # Initialize tracking
        self.throughput_history = []
        self.jfi_history = []
        self.rate_history = []

        convergence_time = None

        # Time stepping
        time_step = 0.1  # 100ms per step
        num_steps = int(duration / time_step)

        # Per-flow state
        flow_states = []
        for flow in flows:
            flow_states.append({
                'mode': 'STARTING',
                'last_mi_time': 0.0,
                'mi_duration': flow.monitor_interval_duration,
                'packets_to_send': 0,
                'inter_packet_time': 0.0,
                'next_packet_time': 0.0
            })

        for step in range(num_steps):
            current_time = step * time_step

            # Each flow sends packets for this time step
            for flow_idx, flow in enumerate(flows):
                state = flow_states[flow_idx]

                # Check if flow needs to start new monitor interval
                if current_time - state['last_mi_time'] >= state['mi_duration']:
                    # Run monitor interval logic
                    mi_results = flow.run_monitor_interval(current_time)
                    state['last_mi_time'] = current_time
                    state['mi_duration'] = mi_results['duration']

                # Send packets at flow's current rate
                packets_per_step = max(1, int(
                    flow.sending_rate * 1e6 * time_step / (1500 * 8)
                ))

                step_time = time_step / packets_per_step if packets_per_step > 0 else time_step

                for pkt_idx in range(packets_per_step):
                    packet_time = current_time + pkt_idx * step_time
                    packet = self.network.send_packet(
                        size=1500,
                        current_time=packet_time,
                        flow_id=flow.flow_id
                    )

                    # Track packet in flow (for RTT measurement)
                    if hasattr(flow, 'pending_packets'):
                        flow.pending_packets.append(packet)

            # Process network queue (shared bottleneck)
            completed_packets = self.network.process_queue(time_step)

            # Deliver completed packets to respective flows
            for packet in completed_packets:
                if not packet.dropped:
                    flow_idx = packet.flow_id
                    if flow_idx < len(flows):
                        flow = flows[flow_idx]
                        # Flows can process ACKs if they track packets
                        if hasattr(flow, 'process_ack'):
                            flow.process_ack(packet)

            # Collect metrics every measurement_interval
            if step % int(self.measurement_interval / time_step) == 0:
                # Get per-flow throughputs
                throughputs = []
                rates = []

                for flow in flows:
                    # Throughput from sending_rate (Mbps)
                    tput = flow.sending_rate * 8 / 1e6
                    throughputs.append(tput)
                    rates.append(flow.sending_rate)

                # Calculate JFI
                jfi = jains_fairness_index(throughputs)

                # Store history
                self.throughput_history.append((current_time, throughputs.copy()))
                self.rate_history.append((current_time, rates.copy()))
                self.jfi_history.append((current_time, jfi))

                # Check convergence (JFI > threshold for 2 consecutive seconds)
                if convergence_time is None and len(self.jfi_history) >= 2:
                    recent_jfis = [j for (t, j) in self.jfi_history[-2:]]
                    if all(j >= convergence_threshold for j in recent_jfis):
                        convergence_time = current_time
                        logger.info(f"Convergence detected at t={convergence_time:.1f}s, JFI={jfi:.4f}")

        # Compute final results
        results = self._compute_results(flows, duration, convergence_time)

        logger.info(f"Simulation complete: final JFI={results['final_jfi']:.4f}, "
                   f"convergence={results['convergence_time']:.1f}s")

        return results

    def _compute_results(self, flows: List, duration: float,
                        convergence_time: Optional[float]) -> Dict:
        """
        Compute final results from simulation

        Args:
            flows: List of flow instances
            duration: Simulation duration
            convergence_time: Time when convergence occurred

        Returns:
            Dictionary with comprehensive results
        """
        num_flows = len(flows)

        # Per-flow final metrics
        flow_metrics = []
        final_throughputs = []
        final_rates = []

        for flow in flows:
            # Get flow statistics
            flow_stats = self.network.get_flow_stats(flow.flow_id)

            # Calculate metrics
            tput = (flow_stats['bytes_sent'] * 8) / (duration * 1e6)  # Mbps
            loss_rate = flow_stats['packets_dropped'] / max(1,
                flow_stats['packets_sent'] + flow_stats['packets_dropped'])

            metrics = FlowMetrics(
                flow_id=flow.flow_id,
                algorithm=flow.__class__.__name__,
                sending_rate=flow.sending_rate,
                throughput=tput,
                avg_latency=self.network.get_rtt(),
                loss_rate=loss_rate,
                utility=flow.latest_utility if hasattr(flow, 'latest_utility') else 0.0,
                packets_sent=flow_stats['packets_sent'],
                packets_acked=flow_stats['packets_sent'] - flow_stats['packets_dropped'],
                packets_lost=flow_stats['packets_dropped']
            )

            flow_metrics.append(metrics)
            final_throughputs.append(tput)
            final_rates.append(flow.sending_rate)

        # Aggregate metrics
        final_jfi = jains_fairness_index(final_throughputs)
        avg_jfi = np.mean([jfi for (t, jfi) in self.jfi_history]) if self.jfi_history else 0.0

        # Fairness metrics
        fair_share = self.config.network.bandwidth_mbps / num_flows
        throughput_std = np.std(final_throughputs)
        max_deviation = max(abs(t - fair_share) for t in final_throughputs)

        return {
            'num_flows': num_flows,
            'duration': duration,
            'convergence_time': convergence_time if convergence_time else duration,
            'converged': convergence_time is not None,

            # Fairness metrics
            'final_jfi': final_jfi,
            'avg_jfi': avg_jfi,
            'fair_share': fair_share,
            'throughput_std': throughput_std,
            'max_deviation': max_deviation,

            # Per-flow results
            'flow_metrics': flow_metrics,
            'final_throughputs': final_throughputs,
            'final_rates': final_rates,

            # History
            'throughput_history': self.throughput_history,
            'jfi_history': self.jfi_history,
            'rate_history': self.rate_history,

            # Network stats
            'network_stats': {
                'total_packets_sent': self.network.packets_sent,
                'total_packets_dropped': self.network.packets_dropped,
                'final_queue_length': self.network.get_current_queue_length()
            }
        }

    def print_results(self, results: Dict):
        """
        Print human-readable results summary

        Args:
            results: Results dictionary from run()
        """
        print()
        print("="*80)
        print("Multi-Flow Simulation Results")
        print("="*80)
        print(f"Flows: {results['num_flows']}")
        print(f"Duration: {results['duration']:.1f}s")
        print(f"Fair Share: {results['fair_share']:.2f} Mbps per flow")
        print()

        print("Fairness Metrics:")
        print(f"  Final Jain's Index: {results['final_jfi']:.4f}")
        print(f"  Average JFI: {results['avg_jfi']:.4f}")
        print(f"  Convergence Time: {results['convergence_time']:.1f}s")
        print(f"  Converged: {'Yes' if results['converged'] else 'No'}")
        print(f"  Throughput Std Dev: {results['throughput_std']:.2f} Mbps")
        print(f"  Max Deviation: {results['max_deviation']:.2f} Mbps")
        print()

        print("Per-Flow Results:")
        for metrics in results['flow_metrics']:
            print(f"  Flow {metrics.flow_id}:")
            print(f"    Throughput: {metrics.throughput:.2f} Mbps "
                  f"({metrics.throughput/results['fair_share']*100:.1f}% of fair share)")
            print(f"    Sending Rate: {metrics.sending_rate:.2f} Mbps")
            print(f"    Loss Rate: {metrics.loss_rate:.4f}")
            print(f"    Packets: {metrics.packets_acked}/{metrics.packets_sent} "
                  f"({metrics.packets_lost} lost)")

        print()
        print("Network Statistics:")
        print(f"  Total Packets Sent: {results['network_stats']['total_packets_sent']}")
        print(f"  Total Packets Dropped: {results['network_stats']['total_packets_dropped']}")
        print(f"  Final Queue Length: {results['network_stats']['final_queue_length']}")
        print("="*80)
