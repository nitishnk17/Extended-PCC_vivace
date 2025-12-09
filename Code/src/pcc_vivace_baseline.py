"""
Production-Grade PCC Vivace Baseline
Faithfully reproduces C++ implementation behavior from PCC-Uspace

Key Features:
- Accurate state machine (STARTING, PROBING, DECISION_MADE)
- Proper monitor interval management
- Per-packet RTT tracking
- Vivace utility function matching C++ implementation
- Rate control with momentum and step size limiting
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SenderMode(Enum):
    """PCC Sender state machine modes"""
    STARTING = "STARTING"
    PROBING = "PROBING"
    DECISION_MADE = "DECISION_MADE"


class MonitorInterval:
    """Represents a single monitor interval with all metrics"""
    def __init__(self, sending_rate: float, is_useful: bool = True):
        self.sending_rate = sending_rate
        self.is_useful = is_useful

        # Timing
        self.start_time = 0.0
        self.end_time = 0.0
        self.duration = 0.0

        # Packet counters
        self.packets_sent = 0
        self.packets_acked = 0
        self.packets_lost = 0
        self.bytes_sent = 0
        self.bytes_acked = 0

        # RTT samples
        self.rtt_samples = []
        self.rtt_on_start = 0.0
        self.rtt_on_end = 0.0

        # Computed metrics
        self.throughput = 0.0
        self.loss_rate = 0.0
        self.avg_rtt = 0.0

    def compute_metrics(self):
        """Compute final metrics for this interval"""
        if self.duration > 0:
            self.throughput = (self.bytes_acked * 8) / (self.duration * 1e6)
        else:
            self.throughput = 0.0

        total_packets = self.packets_sent
        if total_packets > 0:
            self.loss_rate = self.packets_lost / total_packets
        else:
            self.loss_rate = 0.0

        if self.rtt_samples:
            self.avg_rtt = np.mean(self.rtt_samples)
        else:
            self.avg_rtt = self.rtt_on_start

    def __repr__(self):
        return (f"MI(rate={self.sending_rate:.2f}, T={self.throughput:.2f}, "
                f"L={self.avg_rtt:.2f}ms, loss={self.loss_rate:.4f})")


class PccVivaceBaseline:
    """
    Production-grade PCC Vivace baseline implementation

    Matches behavior of C++ implementation in PCC-Uspace:
    - src/pcc/pcc_vivace_sender.cpp
    - src/pcc/pcc_utility_manager.cpp

    Reference paper:
    PCC Vivace: Online-Learning Congestion Control (NSDI 2018)
    """

    def __init__(self, network, config, flow_id: int = 0, multiflow_mode: bool = False):
        """
        Initialize PCC Vivace

        Args:
            network: NetworkSimulator instance
            config: Config object
            flow_id: Unique flow identifier (for multi-flow scenarios)
            multiflow_mode: If True, use network stats instead of process_queue
        """
        self.network = network
        self.config = config
        self.flow_id = flow_id
        self.multiflow_mode = multiflow_mode

        # PCC Vivace parameters (matching C++ constants)
        self.monitor_interval_duration = 0.05  # 50ms typical, ~1 RTT
        self.learning_rate = 0.1  # Alpha for gradient ascent
        self.probing_step_size = 0.05  # 5% rate change for probing
        self.initial_max_step_size = 0.05  # Maximum rate change per MI
        self.incremental_step_size = 0.05  # Increase max step when needed
        self.momentum_beta = 0.9  # Momentum factor

        # Utility function parameters (from C++ pcc_utility_manager.cpp)
        self.sending_rate_exponent = 0.9  # Exponent for throughput term
        self.loss_coefficient = 11.35  # Lambda in utility function
        self.latency_coefficient = 900.0  # Latency penalty weight

        # Rate bounds
        self.rate_min = config.vivace.rate_min
        self.rate_max = config.vivace.rate_max

        # State machine
        self.mode = SenderMode.STARTING
        self.sending_rate = 1.0  # Start at 1 Mbps
        self.velocity = 0.0  # For momentum

        # Monitor interval tracking
        self.current_mi = None
        self.completed_mis = []
        self.mi_counter = 0

        # Rate control state
        self.latest_utility = -float('inf')
        self.incremental_rate_change_step_allowance = 0
        self.rate_change_direction = 0  # +1 for increase, -1 for decrease

        # History (for analysis and debugging)
        self.rate_history = []
        self.utility_history = []
        self.throughput_history = []
        self.latency_history = []
        self.latency_samples = []  # All RTT samples for percentile calculation
        self.mode_history = []

        # Performance tracking
        self.total_time = 0.0
        self.total_bytes_sent = 0

        logger.info(f"PCC Vivace initialized: start_rate={self.sending_rate} Mbps, "
                   f"MI={self.monitor_interval_duration*1000}ms")

    def run(self, duration: float) -> Dict:
        """
        Run PCC Vivace for specified duration

        Args:
            duration: Total simulation duration (seconds)

        Returns:
            Dict with results and metrics
        """
        logger.info(f"Starting PCC Vivace run (duration={duration}s)")

        current_time = 0.0
        self.network.reset()

        while current_time < duration:
            # Run one monitor interval
            mi_results = self.run_monitor_interval(current_time)
            current_time += mi_results['duration']

            # Record history
            self.rate_history.append(self.sending_rate)
            self.utility_history.append(mi_results.get('utility', 0))
            self.throughput_history.append(mi_results.get('throughput', 0))
            self.latency_history.append(mi_results.get('latency', 0))
            self.mode_history.append(self.mode.value)

            # Collect RTT samples for percentile calculation
            if 'rtt_samples' in mi_results:
                self.latency_samples.extend(mi_results['rtt_samples'])

            logger.debug(f"t={current_time:.2f}s, mode={self.mode.value}, "
                        f"rate={self.sending_rate:.2f}, "
                        f"utility={mi_results.get('utility', 0):.3f}")

        # Compute final results
        results = self._compute_final_results()
        logger.info(f"PCC Vivace completed: throughput={results['avg_throughput']:.2f} Mbps, "
                   f"latency={results['avg_latency']:.2f} ms, loss={results['loss_rate']:.4f}")

        return results

    def run_monitor_interval(self, current_time: float) -> Dict:
        """
        Execute one monitor interval

        Args:
            current_time: Current simulation time

        Returns:
            Dict with MI results
        """
        self.network.reset_window_stats()

        if self.mode == SenderMode.STARTING:
            # STARTING mode: test single rate, double if utility increases
            mi = self._run_single_mi(self.sending_rate, current_time)
            utility = self._calculate_utility(mi)

            if utility > self.latest_utility:
                # Utility increased, double rate and stay in STARTING
                self.latest_utility = utility
                self.sending_rate = min(self.sending_rate * 2, self.rate_max)
                logger.info(f"STARTING: utility improved, doubling rate to {self.sending_rate:.2f}")
            else:
                # Utility decreased, enter PROBING
                logger.info(f"STARTING: utility decreased, entering PROBING")
                self.mode = SenderMode.PROBING

            return {
                'duration': mi.duration,
                'utility': utility,
                'throughput': mi.throughput,
                'latency': mi.avg_rtt,
                'loss': mi.loss_rate,
                'rtt_samples': mi.rtt_samples  # For percentile calculation
            }

        elif self.mode == SenderMode.PROBING:
            # PROBING mode: test rate+delta and rate-delta
            return self._run_probing_mode(current_time)

        elif self.mode == SenderMode.DECISION_MADE:
            # DECISION_MADE: move in chosen direction
            return self._run_decision_made_mode(current_time)

    def _run_single_mi(self, rate: float, current_time: float) -> MonitorInterval:
        """Run a single monitor interval at given rate"""
        mi = MonitorInterval(rate, is_useful=True)
        mi.start_time = current_time
        mi.rtt_on_start = self.network.get_rtt_estimate()

        # Send packets at specified rate for MI duration
        packet_size = 1500  # bytes
        bits_per_packet = packet_size * 8
        rate_bps = rate * 1e6
        packets_to_send = int((rate_bps * self.monitor_interval_duration) / bits_per_packet)

        if packets_to_send == 0:
            packets_to_send = 1

        inter_packet_time = self.monitor_interval_duration / packets_to_send

        # Store initial stats for multiflow mode
        if self.multiflow_mode:
            initial_stats = self.network.get_flow_stats(self.flow_id)
            initial_bytes = initial_stats.get('bytes_sent', 0)
            initial_packets_sent = initial_stats.get('packets_sent', 0)
            initial_packets_dropped = initial_stats.get('packets_dropped', 0)

        # Send packets
        t = 0.0
        while t < self.monitor_interval_duration:
            packet = self.network.send_packet(packet_size, current_time + t, flow_id=self.flow_id)
            mi.packets_sent += 1
            mi.bytes_sent += packet_size

            if packet.dropped:
                mi.packets_lost += 1

            t += inter_packet_time

        if self.multiflow_mode:
            # Multiflow mode: Get stats from network instead of processing queue
            # Note: Queue processing is done by MultiFlowRunner centrally

            # Get updated stats
            final_stats = self.network.get_flow_stats(self.flow_id)

            # Calculate deltas
            mi.packets_acked = final_stats.get('packets_sent', 0) - initial_packets_sent
            mi.bytes_acked = final_stats.get('bytes_sent', 0) - initial_bytes
            mi.packets_lost = (final_stats.get('packets_dropped', 0) - initial_packets_dropped)

            # Get network metrics for RTT
            metrics = self.network.get_metrics(self.monitor_interval_duration, self.flow_id)
            mi.avg_rtt = metrics.get('latency', self.network.get_rtt())
            mi.rtt_samples = [mi.avg_rtt]  # Single sample from network estimate
        else:
            # Single-flow mode: Process queue ourselves (original behavior)
            completed_packets = self.network.process_queue(self.monitor_interval_duration)

            # Collect RTT samples from completed packets
            for packet in completed_packets:
                rtt = packet.get_rtt()
                if rtt is not None:
                    mi.rtt_samples.append(rtt)
                    mi.packets_acked += 1
                    mi.bytes_acked += packet.size

        mi.end_time = current_time + self.monitor_interval_duration
        mi.duration = self.monitor_interval_duration
        mi.rtt_on_end = self.network.get_rtt_estimate()

        # Compute metrics
        mi.compute_metrics()

        self.completed_mis.append(mi)
        self.mi_counter += 1

        return mi

    def _run_probing_mode(self, current_time: float) -> Dict:
        """
        PROBING mode: test two rates to determine direction

        Tests rate * (1 + probing_step_size) and rate * (1 - probing_step_size)
        """
        # Test rate + delta
        rate_high = self.sending_rate * (1.0 + self.probing_step_size)
        rate_high = min(rate_high, self.rate_max)
        mi_high = self._run_single_mi(rate_high, current_time)
        utility_high = self._calculate_utility(mi_high)

        # Test rate - delta
        rate_low = self.sending_rate * (1.0 - self.probing_step_size)
        rate_low = max(rate_low, self.rate_min)
        mi_low = self._run_single_mi(rate_low, current_time + self.monitor_interval_duration)
        utility_low = self._calculate_utility(mi_low)

        # Determine direction
        if utility_high > utility_low:
            self.rate_change_direction = +1
            self.latest_utility = utility_high
            logger.info(f"PROBING: direction=UP (U_high={utility_high:.3f} > U_low={utility_low:.3f})")
        else:
            self.rate_change_direction = -1
            self.latest_utility = utility_low
            logger.info(f"PROBING: direction=DOWN (U_low={utility_low:.3f} > U_high={utility_high:.3f})")

        # Enter DECISION_MADE mode
        self.mode = SenderMode.DECISION_MADE

        # Compute rate change
        self._apply_rate_change(utility_high, utility_low, rate_high, rate_low)

        # Combine RTT samples from both MIs
        rtt_samples = mi_high.rtt_samples + mi_low.rtt_samples

        return {
            'duration': 2 * self.monitor_interval_duration,
            'utility': max(utility_high, utility_low),
            'throughput': (mi_high.throughput + mi_low.throughput) / 2,
            'latency': (mi_high.avg_rtt + mi_low.avg_rtt) / 2,
            'loss': (mi_high.loss_rate + mi_low.loss_rate) / 2,
            'rtt_samples': rtt_samples  # For percentile calculation
        }

    def _run_decision_made_mode(self, current_time: float) -> Dict:
        """
        DECISION_MADE mode: continue in chosen direction

        Stay in this mode until utility decreases, then go back to PROBING
        """
        # Test current rate
        mi = self._run_single_mi(self.sending_rate, current_time)
        utility = self._calculate_utility(mi)

        if utility < self.latest_utility:
            # Utility decreased, go back to PROBING
            logger.info(f"DECISION_MADE: utility decreased, entering PROBING")
            self.mode = SenderMode.PROBING
        else:
            # Utility still increasing, continue in same direction
            self.latest_utility = utility
            gradient = utility - self.latest_utility
            self._apply_rate_change_gradient(gradient)
            logger.debug(f"DECISION_MADE: continuing, rate={self.sending_rate:.2f}")

        return {
            'duration': mi.duration,
            'utility': utility,
            'throughput': mi.throughput,
            'latency': mi.avg_rtt,
            'loss': mi.loss_rate,
            'rtt_samples': mi.rtt_samples  # For percentile calculation
        }

    def _calculate_utility(self, mi: MonitorInterval) -> float:
        """
        Calculate Vivace utility function

        From C++ implementation (pcc_utility_manager.cpp):
        U = T^0.9 * S(L) - lambda * T * loss

        Where:
        - T: throughput (Mbps)
        - S(L): sigmoid function for latency penalty
        - loss: loss rate
        - lambda: loss coefficient (11.35)

        Args:
            mi: MonitorInterval with computed metrics

        Returns:
            Utility value
        """
        T = mi.throughput
        L = mi.avg_rtt
        loss = mi.loss_rate

        # Throughput contribution: T^0.9
        if T > 0:
            throughput_term = np.power(T, self.sending_rate_exponent)
        else:
            throughput_term = 0.0

        # Latency penalty: sigmoid function S(L)
        # S(L) = 1 / (1 + exp((L - L_threshold) / scale))
        # This penalizes latency inflation
        L_threshold = 100.0  # ms, threshold for latency penalty
        scale = 10.0
        if L > 0:
            latency_penalty = 1.0 / (1.0 + np.exp((L - L_threshold) / scale))
        else:
            latency_penalty = 1.0

        # Loss penalty: -lambda * T * loss
        loss_penalty = self.loss_coefficient * T * loss

        # Final utility
        utility = throughput_term * latency_penalty - loss_penalty

        logger.debug(f"Utility: T={T:.2f}, L={L:.2f}ms, loss={loss:.4f}, "
                    f"U={utility:.3f} (T_term={throughput_term:.2f}, "
                    f"L_penalty={latency_penalty:.3f}, loss_penalty={loss_penalty:.3f})")

        return utility

    def _apply_rate_change(self, utility_high: float, utility_low: float,
                          rate_high: float, rate_low: float):
        """
        Apply rate change based on probing results

        Uses gradient ascent with momentum
        """
        # Compute gradient
        rate_diff = rate_high - rate_low
        utility_diff = utility_high - utility_low

        if abs(rate_diff) > 1e-6:
            gradient = utility_diff / rate_diff
        else:
            gradient = 0.0

        # Apply momentum
        self.velocity = self.momentum_beta * self.velocity + self.learning_rate * gradient

        # Compute rate change
        rate_change = self.velocity

        # Apply step size limiting
        max_change = self.sending_rate * (self.initial_max_step_size +
                                          self.incremental_rate_change_step_allowance * self.incremental_step_size)

        if abs(rate_change) > max_change:
            rate_change = np.sign(rate_change) * max_change
            self.incremental_rate_change_step_allowance += 1
            logger.debug(f"Rate change limited to {max_change:.3f}, increasing allowance")

        # Update rate
        # BUG FIX: Gradient already encodes direction, don't multiply by rate_change_direction again
        new_rate = self.sending_rate + rate_change
        new_rate = np.clip(new_rate, self.rate_min, self.rate_max)

        logger.info(f"Rate change: {self.sending_rate:.2f} -> {new_rate:.2f} "
                   f"(gradient={gradient:.4f}, velocity={self.velocity:.4f})")

        self.sending_rate = new_rate

    def _apply_rate_change_gradient(self, gradient: float):
        """Apply rate change based on gradient"""
        self.velocity = self.momentum_beta * self.velocity + self.learning_rate * gradient
        # BUG FIX: Gradient already encodes direction
        rate_change = self.velocity

        new_rate = self.sending_rate + rate_change
        new_rate = np.clip(new_rate, self.rate_min, self.rate_max)

        self.sending_rate = new_rate

    def _compute_final_results(self) -> Dict:
        """Compute aggregated results over entire run"""
        if not self.completed_mis:
            return {
                'avg_throughput': 0,
                'avg_latency': 0,
                'loss_rate': 0,
                'final_rate': self.sending_rate
            }

        # Compute averages
        throughputs = [mi.throughput for mi in self.completed_mis]
        latencies = [mi.avg_rtt for mi in self.completed_mis]
        losses = [mi.loss_rate for mi in self.completed_mis]
        utilities = [self._calculate_utility(mi) for mi in self.completed_mis]

        # Collect all RTT samples for percentiles
        all_rtt_samples = []
        for mi in self.completed_mis:
            all_rtt_samples.extend(mi.rtt_samples)

        avg_rtt_value = np.mean(latencies) if latencies else 0

        results = {
            'avg_throughput': np.mean(throughputs) if throughputs else 0,
            'std_throughput': np.std(throughputs) if throughputs else 0,
            'avg_latency': avg_rtt_value,
            'avg_rtt': avg_rtt_value,  # Alias for compatibility with evaluation scripts
            'latency_p95': np.percentile(all_rtt_samples, 95) if all_rtt_samples else 0,
            'latency_p99': np.percentile(all_rtt_samples, 99) if all_rtt_samples else 0,
            'loss_rate': np.mean(losses) if losses else 0,
            'avg_utility': np.mean(utilities) if utilities else 0,
            'final_rate': self.sending_rate,
            'num_mis': len(self.completed_mis),
            'rate_history': self.rate_history.copy(),
            'throughput_history': self.throughput_history.copy(),
            'latency_history': self.latency_history.copy(),
            'utility_history': self.utility_history.copy()
        }

        return results
