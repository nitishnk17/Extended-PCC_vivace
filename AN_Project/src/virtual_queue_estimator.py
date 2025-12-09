"""
Virtual Queue Estimator for Extension 3: Distributed Fairness

Estimates bottleneck queue occupancy from RTT measurements and throughput
observations without requiring explicit network feedback.

Key Insight:
- RTT = Propagation delay + Queueing delay
- Baseline RTT ≈ Propagation delay (minimum observed RTT)
- Queue delay = RTT_current - RTT_baseline
- Queue bytes = Queue delay × Bandwidth

Algorithm:
1. Track RTT samples in sliding window
2. Estimate baseline RTT (5th percentile - minimum propagation delay)
3. Estimate bandwidth (95th percentile of throughput)
4. Calculate queue occupancy: Q = (RTT - RTT_base) × BW
5. Smooth estimates with exponential moving average

Features:
- Percentile-based baseline (robust to noise)
- Adaptive bandwidth estimation
- Confidence-weighted estimates
- Queue trend detection (filling vs draining)
- Memory-bounded data structures

Usage:
    estimator = VirtualQueueEstimator(config)

    # Each monitor interval
    estimator.add_rtt_sample(current_rtt_ms)
    estimator.add_throughput_sample(throughput_mbps)

    # Get queue estimate
    queue_info = estimator.estimate_queue(current_rtt_ms)
    print(f"Queue: {queue_info['queue_packets']} packets, "
          f"{queue_info['queue_delay_ms']:.1f}ms")
"""
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


@dataclass
class RTTSample:
    """
    Single RTT measurement

    Attributes:
        timestamp: Time of measurement (seconds)
        rtt_ms: Round-trip time (milliseconds)
        throughput_mbps: Throughput at this time (Mbps)
    """
    timestamp: float
    rtt_ms: float
    throughput_mbps: float = 0.0

    def __repr__(self):
        return f"RTTSample(t={self.timestamp:.2f}s, RTT={self.rtt_ms:.1f}ms)"


class QueueTrend(Enum):
    """
    Queue occupancy trend

    FILLING: Queue is increasing (congestion building)
    STABLE: Queue is relatively constant
    DRAINING: Queue is decreasing (congestion relieving)
    """
    FILLING = "filling"
    STABLE = "stable"
    DRAINING = "draining"

    def __str__(self):
        return self.value


class VirtualQueueEstimator:
    """
    Estimate bottleneck queue state from RTT and throughput observations

    The estimator tracks RTT measurements to establish a baseline (minimum RTT,
    representing propagation delay) and uses bandwidth estimates to convert
    queue delay into queue occupancy in bytes/packets.

    Key Metrics:
    1. Baseline RTT: 5th percentile of recent RTT samples (propagation delay)
    2. Queue Delay: Current RTT - Baseline RTT (queueing delay)
    3. Bandwidth: 95th percentile of recent throughput (link capacity)
    4. Queue Bytes: Queue delay × Bandwidth (BDP calculation)

    Smoothing: Exponential moving average for stable estimates
    """

    def __init__(self, config):
        """
        Initialize virtual queue estimator

        Args:
            config: VirtualQueueEstimatorConfig with parameters
        """
        # Configuration
        self.baseline_window = config.baseline_window  # RTT samples
        self.throughput_window = config.throughput_window  # Throughput samples
        self.smoothing_factor = config.smoothing_factor  # EMA alpha
        self.min_samples_baseline = config.min_samples_baseline
        self.min_samples_bandwidth = config.min_samples_bandwidth

        # RTT tracking (bounded queue for memory safety)
        self.rtt_history = deque(maxlen=self.baseline_window)

        # Throughput tracking
        self.throughput_history = deque(maxlen=self.throughput_window)

        # State
        self.baseline_rtt = None  # Minimum RTT (propagation delay)
        self.estimated_bandwidth = None  # Link capacity (Mbps)
        self.current_queue_ms = 0.0  # Smoothed queue delay
        self.current_queue_bytes = 0  # Smoothed queue size
        self.current_queue_packets = 0  # Smoothed queue packets

        # Trend tracking
        self.queue_history = deque(maxlen=10)  # Recent queue estimates
        self.current_trend = QueueTrend.STABLE

        # Statistics
        self.total_samples = 0
        self.estimate_count = 0

        logger.info(f"VirtualQueueEstimator initialized: "
                   f"baseline_window={self.baseline_window}, "
                   f"smoothing={self.smoothing_factor:.2f}")

    def add_rtt_sample(self, timestamp: float, rtt_ms: float,
                      throughput_mbps: float = 0.0):
        """
        Add RTT measurement

        Args:
            timestamp: Current time (seconds)
            rtt_ms: Round-trip time (milliseconds)
            throughput_mbps: Current throughput (Mbps)
        """
        if rtt_ms < 0:
            logger.warning(f"Invalid RTT: {rtt_ms}ms (negative), ignoring")
            return

        sample = RTTSample(timestamp, rtt_ms, throughput_mbps)
        self.rtt_history.append(sample)
        self.total_samples += 1

        # Update baseline RTT (5th percentile of recent samples)
        if len(self.rtt_history) >= self.min_samples_baseline:
            rtt_values = [s.rtt_ms for s in self.rtt_history]
            self.baseline_rtt = np.percentile(rtt_values, 5)

        logger.debug(f"Added RTT sample: {rtt_ms:.1f}ms, "
                    f"baseline={self.baseline_rtt:.1f}ms"
                    if self.baseline_rtt else "baseline=None")

    def add_throughput_sample(self, throughput_mbps: float):
        """
        Add throughput measurement for bandwidth estimation

        Args:
            throughput_mbps: Throughput measurement (Mbps)
        """
        if throughput_mbps < 0:
            logger.warning(f"Invalid throughput: {throughput_mbps}Mbps (negative), ignoring")
            return

        self.throughput_history.append(throughput_mbps)

        # Update bandwidth estimate (95th percentile of recent throughput)
        if len(self.throughput_history) >= self.min_samples_bandwidth:
            self.estimated_bandwidth = np.percentile(
                list(self.throughput_history), 95
            )

        logger.debug(f"Added throughput: {throughput_mbps:.1f}Mbps, "
                    f"bandwidth={self.estimated_bandwidth:.1f}Mbps"
                    if self.estimated_bandwidth else "bandwidth=None")

    def estimate_queue(self, current_rtt_ms: float) -> Dict:
        """
        Estimate current queue state

        Args:
            current_rtt_ms: Current RTT measurement (milliseconds)

        Returns:
            Dictionary with queue estimate:
            {
                'queue_delay_ms': float,      # Queue delay (ms)
                'queue_bytes': int,           # Queue size (bytes)
                'queue_packets': int,         # Queue size (packets)
                'utilization': float,         # Queue utilization [0, 1]
                'confidence': float,          # Estimate confidence [0, 1]
                'trend': QueueTrend,          # Filling/stable/draining
                'baseline_rtt': float,        # Baseline RTT (ms)
                'bandwidth_mbps': float       # Estimated bandwidth (Mbps)
            }

        Example:
            queue = estimator.estimate_queue(current_rtt_ms=55.0)
            if queue['confidence'] > 0.7 and queue['trend'] == QueueTrend.FILLING:
                print("High confidence: queue is filling rapidly")
        """
        self.estimate_count += 1

        # Need sufficient data for estimation
        if self.baseline_rtt is None or self.estimated_bandwidth is None:
            logger.debug("Insufficient data for queue estimation")
            return self._default_estimate()

        # Calculate instantaneous queue delay
        queue_delay_ms = max(0.0, current_rtt_ms - self.baseline_rtt)

        # Convert to bytes using bandwidth-delay product
        # Queue [bytes] = Delay [seconds] × Bandwidth [bits/sec] / 8
        queue_delay_s = queue_delay_ms / 1000.0
        bandwidth_bps = self.estimated_bandwidth * 1e6
        queue_bytes = int(queue_delay_s * bandwidth_bps / 8)

        # Estimate packets (assume 1500 byte MTU)
        MTU = 1500
        queue_packets = queue_bytes // MTU

        # Smooth estimates with exponential moving average
        # smoothed = α × new + (1-α) × old
        alpha = self.smoothing_factor
        self.current_queue_ms = alpha * queue_delay_ms + (1 - alpha) * self.current_queue_ms
        self.current_queue_bytes = int(alpha * queue_bytes + (1 - alpha) * self.current_queue_bytes)
        self.current_queue_packets = self.current_queue_bytes // MTU

        # Track trend
        self.queue_history.append(self.current_queue_packets)
        trend = self._detect_trend()
        self.current_trend = trend

        # Estimate utilization (assume max queue = 1000 packets ≈ 1.5MB)
        MAX_QUEUE_PACKETS = 1000
        utilization = min(1.0, self.current_queue_packets / MAX_QUEUE_PACKETS)

        # Calculate confidence
        confidence = self._calculate_confidence()

        result = {
            'queue_delay_ms': self.current_queue_ms,
            'queue_bytes': self.current_queue_bytes,
            'queue_packets': self.current_queue_packets,
            'utilization': utilization,
            'confidence': confidence,
            'trend': trend,
            'baseline_rtt': self.baseline_rtt,
            'bandwidth_mbps': self.estimated_bandwidth
        }

        logger.debug(f"Queue estimate: {self.current_queue_packets} packets "
                    f"({self.current_queue_ms:.1f}ms, {trend.value}), "
                    f"confidence={confidence:.2f}")

        return result

    def _detect_trend(self) -> QueueTrend:
        """
        Detect queue occupancy trend from recent history

        Returns:
            QueueTrend (FILLING, STABLE, DRAINING)
        """
        if len(self.queue_history) < 5:
            return QueueTrend.STABLE

        # Calculate linear trend (least squares fit)
        recent = list(self.queue_history)[-5:]
        x = np.arange(len(recent))
        y = np.array(recent)

        # Slope of best-fit line
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0.0

        # Classify trend based on slope
        # Thresholds: ±5 packets/sample
        if slope > 5.0:
            return QueueTrend.FILLING
        elif slope < -5.0:
            return QueueTrend.DRAINING
        else:
            return QueueTrend.STABLE

    def _calculate_confidence(self) -> float:
        """
        Calculate confidence in queue estimate [0, 1]

        Higher confidence when:
        - More RTT samples available (closer to full window)
        - More throughput samples available
        - Stable baseline and bandwidth estimates

        Returns:
            Confidence score [0, 1]
        """
        # RTT data confidence
        rtt_confidence = min(1.0, len(self.rtt_history) / self.baseline_window)

        # Throughput data confidence
        throughput_confidence = min(1.0, len(self.throughput_history) / self.throughput_window)

        # Baseline stability confidence (lower variance → higher confidence)
        if len(self.rtt_history) >= 10:
            rtt_values = [s.rtt_ms for s in self.rtt_history]
            rtt_std = np.std(rtt_values)
            rtt_mean = np.mean(rtt_values)
            # Coefficient of variation
            cv = rtt_std / rtt_mean if rtt_mean > 0 else 1.0
            # High stability (cv < 0.1) → conf=1.0, Low stability (cv > 0.5) → conf=0.0
            stability_confidence = max(0.0, min(1.0, (0.5 - cv) / 0.4))
        else:
            stability_confidence = 0.0

        # Combined confidence (weighted average)
        confidence = (0.3 * rtt_confidence +
                     0.3 * throughput_confidence +
                     0.4 * stability_confidence)

        return float(confidence)

    def _default_estimate(self) -> Dict:
        """
        Default estimate when insufficient data

        Returns:
            Dictionary with zero estimates and low confidence
        """
        return {
            'queue_delay_ms': 0.0,
            'queue_bytes': 0,
            'queue_packets': 0,
            'utilization': 0.0,
            'confidence': 0.0,
            'trend': QueueTrend.STABLE,
            'baseline_rtt': self.baseline_rtt if self.baseline_rtt else 0.0,
            'bandwidth_mbps': self.estimated_bandwidth if self.estimated_bandwidth else 0.0
        }

    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics

        Returns:
            Dictionary with estimator statistics
        """
        if len(self.rtt_history) > 0:
            rtt_values = [s.rtt_ms for s in self.rtt_history]
            rtt_stats = {
                'count': len(rtt_values),
                'mean': np.mean(rtt_values),
                'std': np.std(rtt_values),
                'min': np.min(rtt_values),
                'max': np.max(rtt_values),
                'p5': np.percentile(rtt_values, 5),
                'p50': np.percentile(rtt_values, 50),
                'p95': np.percentile(rtt_values, 95)
            }
        else:
            rtt_stats = {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'p5': 0.0,
                'p50': 0.0,
                'p95': 0.0
            }

        if len(self.throughput_history) > 0:
            throughput_stats = {
                'count': len(self.throughput_history),
                'mean': np.mean(self.throughput_history),
                'std': np.std(self.throughput_history),
                'min': np.min(self.throughput_history),
                'max': np.max(self.throughput_history),
                'p95': np.percentile(self.throughput_history, 95)
            }
        else:
            throughput_stats = {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'p95': 0.0
            }

        return {
            'current': {
                'queue_delay_ms': self.current_queue_ms,
                'queue_bytes': self.current_queue_bytes,
                'queue_packets': self.current_queue_packets,
                'trend': self.current_trend.value
            },
            'baseline': {
                'rtt_ms': self.baseline_rtt if self.baseline_rtt else 0.0,
                'bandwidth_mbps': self.estimated_bandwidth if self.estimated_bandwidth else 0.0
            },
            'rtt': rtt_stats,
            'throughput': throughput_stats,
            'history': {
                'total_samples': self.total_samples,
                'estimate_count': self.estimate_count
            }
        }

    def get_summary(self) -> str:
        """
        Get human-readable summary

        Returns:
            Formatted summary string
        """
        stats = self.get_statistics()

        summary = "Virtual Queue Estimator Summary:\n"
        summary += f"  Current Queue: {stats['current']['queue_packets']} packets "
        summary += f"({stats['current']['queue_delay_ms']:.1f}ms)\n"
        summary += f"  Trend: {stats['current']['trend']}\n"
        summary += f"  Baseline RTT: {stats['baseline']['rtt_ms']:.1f}ms\n"
        summary += f"  Estimated Bandwidth: {stats['baseline']['bandwidth_mbps']:.1f}Mbps\n"
        summary += f"  Total Samples: {stats['history']['total_samples']}\n"
        summary += f"  Estimates: {stats['history']['estimate_count']}\n"

        if stats['rtt']['count'] > 0:
            summary += f"  RTT Stats:\n"
            summary += f"    Range: [{stats['rtt']['min']:.1f}, {stats['rtt']['max']:.1f}]ms\n"
            summary += f"    Mean: {stats['rtt']['mean']:.1f} ± {stats['rtt']['std']:.1f}ms\n"
            summary += f"    Percentiles: P5={stats['rtt']['p5']:.1f}, "
            summary += f"P50={stats['rtt']['p50']:.1f}, P95={stats['rtt']['p95']:.1f}\n"

        if stats['throughput']['count'] > 0:
            summary += f"  Throughput Stats:\n"
            summary += f"    Mean: {stats['throughput']['mean']:.1f} ± {stats['throughput']['std']:.1f}Mbps\n"
            summary += f"    P95: {stats['throughput']['p95']:.1f}Mbps\n"

        return summary

    def reset(self):
        """Reset estimator state"""
        self.rtt_history.clear()
        self.throughput_history.clear()
        self.queue_history.clear()

        self.baseline_rtt = None
        self.estimated_bandwidth = None
        self.current_queue_ms = 0.0
        self.current_queue_bytes = 0
        self.current_queue_packets = 0
        self.current_trend = QueueTrend.STABLE

        self.total_samples = 0
        self.estimate_count = 0

        logger.info("VirtualQueueEstimator reset")
