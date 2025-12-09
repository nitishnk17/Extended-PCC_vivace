"""
Extension 4: Multipath Rate Allocation - Phase 2
PathMonitor: Real-time per-path metrics collection and path degradation detection

This module collects and maintains real-time performance metrics for each path,
providing historical data for utility calculation and rate allocation decisions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import threading
import time
import numpy as np


@dataclass
class PathMetrics:
    """Real-time metrics for a single path

    Attributes:
        path_id: Unique path identifier
        timestamp: Time when metrics were collected

        throughput: Achieved throughput in Mbps
        goodput: Application-layer throughput in Mbps

        rtt_min: Minimum RTT in milliseconds
        rtt_avg: Average RTT in milliseconds
        rtt_95p: 95th percentile RTT in milliseconds
        rtt_samples: Raw RTT samples for percentile calculation

        packets_sent: Number of packets sent
        packets_acked: Number of packets acknowledged
        packets_lost: Number of packets lost
        loss_rate: Packet loss rate [0, 1]

        sending_rate: Current sending rate in Mbps
        utilization: Path utilization (sending_rate / estimated_bandwidth)
    """
    path_id: int
    timestamp: float

    # Throughput
    throughput: float = 0.0  # Mbps
    goodput: float = 0.0  # Mbps

    # Latency
    rtt_min: float = 0.0  # ms
    rtt_avg: float = 0.0  # ms
    rtt_95p: float = 0.0  # ms
    rtt_samples: List[float] = field(default_factory=list)

    # Loss
    packets_sent: int = 0
    packets_acked: int = 0
    packets_lost: int = 0
    loss_rate: float = 0.0

    # Utilization
    sending_rate: float = 0.0  # Mbps
    utilization: float = 0.0  # [0, 1]

    def __repr__(self) -> str:
        """String representation for debugging"""
        return (f"PathMetrics(path={self.path_id}, t={self.timestamp:.2f}, "
                f"tput={self.throughput:.2f}Mbps, rtt={self.rtt_avg:.1f}ms, "
                f"loss={self.loss_rate:.3f}, util={self.utilization:.2f})")


@dataclass
class PathMonitorConfig:
    """Configuration for PathMonitor

    Attributes:
        history_window: Number of historical metrics to keep per path
        smoothing_factor: EMA smoothing factor (alpha) - higher = more reactive
        degradation_threshold: Throughput drop threshold to consider path degraded
        degradation_window: Number of recent samples for degradation detection
        enabled: Whether monitoring is enabled
    """
    history_window: int = 100  # Keep last 100 metrics
    smoothing_factor: float = 0.2  # EMA alpha
    degradation_threshold: float = 0.3  # 30% throughput drop
    degradation_window: int = 10  # Check last 10 samples
    enabled: bool = True


class PathMonitor:
    """Real-time per-path metrics collection and monitoring

    Maintains historical metrics for each path, applies exponential moving
    average smoothing, and detects path quality degradation.

    Thread-safe for concurrent metric updates and queries.
    """

    def __init__(self, config: Optional[PathMonitorConfig] = None):
        """Initialize PathMonitor

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or PathMonitorConfig()

        # Per-path metrics storage: path_id -> List[PathMetrics]
        self._metrics_history: Dict[int, List[PathMetrics]] = {}

        # Per-path smoothed metrics (EMA): path_id -> PathMetrics
        self._smoothed_metrics: Dict[int, PathMetrics] = {}

        # Baseline throughput for degradation detection: path_id -> float
        self._baseline_throughput: Dict[int, float] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self.total_updates = 0
        self.degradation_events = 0

    def update_metrics(self, path_id: int, mi) -> None:
        """Update metrics from a monitor interval

        Args:
            path_id: Path identifier
            mi: MonitorInterval object from PCC algorithm
        """
        if not self.config.enabled:
            return

        with self._lock:
            # Create PathMetrics from MonitorInterval
            metrics = self._create_metrics_from_mi(path_id, mi)

            # Initialize history if needed
            if path_id not in self._metrics_history:
                self._metrics_history[path_id] = []
                self._baseline_throughput[path_id] = metrics.throughput

            # Add to history
            self._metrics_history[path_id].append(metrics)

            # Trim history to window size
            if len(self._metrics_history[path_id]) > self.config.history_window:
                self._metrics_history[path_id].pop(0)

            # Update smoothed metrics using EMA
            self._update_smoothed_metrics(path_id, metrics)

            # Update baseline for degradation detection
            self._update_baseline_throughput(path_id, metrics)

            self.total_updates += 1

    def _create_metrics_from_mi(self, path_id: int, mi) -> PathMetrics:
        """Create PathMetrics from MonitorInterval

        Args:
            path_id: Path identifier
            mi: MonitorInterval object

        Returns:
            PathMetrics object
        """
        # Calculate throughput (Mbps)
        duration = mi.duration if mi.duration > 0 else 1.0
        throughput = (mi.bytes_acked * 8.0 / 1_000_000) / duration
        goodput = throughput  # Assume same as throughput for now

        # Calculate RTT statistics
        rtt_samples = mi.rtt_samples if hasattr(mi, 'rtt_samples') and mi.rtt_samples else []
        if rtt_samples:
            rtt_min = min(rtt_samples) * 1000  # Convert to ms
            rtt_avg = np.mean(rtt_samples) * 1000
            rtt_95p = np.percentile(rtt_samples, 95) * 1000
        else:
            rtt_min = rtt_avg = rtt_95p = 0.0

        # Calculate loss rate
        packets_sent = mi.packets_sent
        packets_acked = mi.packets_acked
        packets_lost = packets_sent - packets_acked if packets_sent >= packets_acked else 0
        loss_rate = packets_lost / packets_sent if packets_sent > 0 else 0.0

        # Calculate sending rate and utilization
        sending_rate = mi.sending_rate  # Already in Mbps
        # Utilization requires estimated bandwidth - will be set by caller if available
        utilization = 0.0

        return PathMetrics(
            path_id=path_id,
            timestamp=mi.end_time,
            throughput=throughput,
            goodput=goodput,
            rtt_min=rtt_min,
            rtt_avg=rtt_avg,
            rtt_95p=rtt_95p,
            rtt_samples=rtt_samples,
            packets_sent=packets_sent,
            packets_acked=packets_acked,
            packets_lost=packets_lost,
            loss_rate=loss_rate,
            sending_rate=sending_rate,
            utilization=utilization
        )

    def _update_smoothed_metrics(self, path_id: int, new_metrics: PathMetrics) -> None:
        """Update smoothed metrics using exponential moving average

        Args:
            path_id: Path identifier
            new_metrics: New metrics to incorporate
        """
        alpha = self.config.smoothing_factor

        if path_id not in self._smoothed_metrics:
            # First update - use raw metrics
            self._smoothed_metrics[path_id] = new_metrics
            return

        # Apply EMA: smoothed = alpha * new + (1 - alpha) * old
        old = self._smoothed_metrics[path_id]

        smoothed = PathMetrics(
            path_id=path_id,
            timestamp=new_metrics.timestamp,
            throughput=alpha * new_metrics.throughput + (1 - alpha) * old.throughput,
            goodput=alpha * new_metrics.goodput + (1 - alpha) * old.goodput,
            rtt_min=min(new_metrics.rtt_min, old.rtt_min) if new_metrics.rtt_min > 0 else old.rtt_min,
            rtt_avg=alpha * new_metrics.rtt_avg + (1 - alpha) * old.rtt_avg,
            rtt_95p=alpha * new_metrics.rtt_95p + (1 - alpha) * old.rtt_95p,
            rtt_samples=new_metrics.rtt_samples,  # Keep most recent samples
            packets_sent=new_metrics.packets_sent,
            packets_acked=new_metrics.packets_acked,
            packets_lost=new_metrics.packets_lost,
            loss_rate=alpha * new_metrics.loss_rate + (1 - alpha) * old.loss_rate,
            sending_rate=alpha * new_metrics.sending_rate + (1 - alpha) * old.sending_rate,
            utilization=alpha * new_metrics.utilization + (1 - alpha) * old.utilization
        )

        self._smoothed_metrics[path_id] = smoothed

    def _update_baseline_throughput(self, path_id: int, metrics: PathMetrics) -> None:
        """Update baseline throughput for degradation detection

        Uses the 90th percentile of historical throughput (excluding most recent samples)
        as baseline. Only updates baseline upward to avoid baseline decay during degradation.

        Args:
            path_id: Path identifier
            metrics: Current metrics
        """
        if path_id not in self._metrics_history:
            return

        history = self._metrics_history[path_id]

        # Need sufficient history for stable baseline
        if len(history) < self.config.degradation_window * 2:
            # Not enough history - use all available
            throughputs = [m.throughput for m in history if m.throughput > 0]
            if throughputs:
                self._baseline_throughput[path_id] = max(
                    self._baseline_throughput.get(path_id, 0),
                    np.percentile(throughputs, 90)
                )
            return

        # Use historical window (excluding most recent degradation_window samples)
        # to calculate baseline
        historical_end = len(history) - self.config.degradation_window
        historical_metrics = history[:historical_end]

        if len(historical_metrics) >= self.config.degradation_window:
            historical_throughputs = [m.throughput for m in historical_metrics if m.throughput > 0]

            if historical_throughputs:
                new_baseline = np.percentile(historical_throughputs, 90)
                # Only update baseline upward (avoid baseline decay during degradation)
                current_baseline = self._baseline_throughput.get(path_id, 0)
                self._baseline_throughput[path_id] = max(current_baseline, new_baseline)

    def get_path_metrics(self, path_id: int, smoothed: bool = True) -> Optional[PathMetrics]:
        """Get current metrics for a path

        Args:
            path_id: Path identifier
            smoothed: If True, return EMA-smoothed metrics; if False, return most recent raw

        Returns:
            PathMetrics if available, None otherwise
        """
        with self._lock:
            if smoothed:
                return self._smoothed_metrics.get(path_id)
            else:
                history = self._metrics_history.get(path_id)
                return history[-1] if history else None

    def get_path_history(self, path_id: int, duration: Optional[float] = None) -> List[PathMetrics]:
        """Get historical metrics for a path

        Args:
            path_id: Path identifier
            duration: If specified, return only metrics from last N seconds

        Returns:
            List of PathMetrics (oldest to newest)
        """
        with self._lock:
            history = self._metrics_history.get(path_id, [])

            if duration is None:
                return list(history)

            # Filter by duration
            if not history:
                return []

            current_time = history[-1].timestamp
            cutoff_time = current_time - duration

            return [m for m in history if m.timestamp >= cutoff_time]

    def is_path_degraded(self, path_id: int) -> bool:
        """Detect if path quality has degraded significantly

        Considers a path degraded if recent throughput has dropped more than
        the configured threshold relative to baseline.

        Args:
            path_id: Path identifier

        Returns:
            True if path is degraded, False otherwise
        """
        with self._lock:
            if path_id not in self._metrics_history:
                return False

            history = self._metrics_history[path_id]
            if len(history) < self.config.degradation_window:
                return False

            # Get recent throughput
            recent = history[-self.config.degradation_window:]
            recent_throughputs = [m.throughput for m in recent if m.throughput > 0]

            if not recent_throughputs:
                return False

            recent_avg = np.mean(recent_throughputs)
            baseline = self._baseline_throughput.get(path_id, recent_avg)

            # Check if throughput dropped significantly
            if baseline > 0:
                drop_ratio = (baseline - recent_avg) / baseline
                if drop_ratio >= self.config.degradation_threshold:
                    self.degradation_events += 1
                    return True

            return False

    def get_all_monitored_paths(self) -> List[int]:
        """Get list of all paths being monitored

        Returns:
            List of path IDs
        """
        with self._lock:
            return list(self._metrics_history.keys())

    def get_statistics(self) -> Dict:
        """Get monitoring statistics

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                'total_updates': self.total_updates,
                'degradation_events': self.degradation_events,
                'monitored_paths': len(self._metrics_history),
                'total_samples': sum(len(h) for h in self._metrics_history.values())
            }

    def clear_history(self, path_id: Optional[int] = None) -> None:
        """Clear metrics history

        Args:
            path_id: If specified, clear only this path; otherwise clear all
        """
        with self._lock:
            if path_id is not None:
                self._metrics_history.pop(path_id, None)
                self._smoothed_metrics.pop(path_id, None)
                self._baseline_throughput.pop(path_id, None)
            else:
                self._metrics_history.clear()
                self._smoothed_metrics.clear()
                self._baseline_throughput.clear()
                self.total_updates = 0
                self.degradation_events = 0

    def set_path_utilization(self, path_id: int, estimated_bandwidth: float) -> None:
        """Set path utilization based on estimated bandwidth

        Updates utilization for both smoothed and recent metrics.

        Args:
            path_id: Path identifier
            estimated_bandwidth: Estimated available bandwidth in Mbps
        """
        with self._lock:
            # Update smoothed metrics
            if path_id in self._smoothed_metrics and estimated_bandwidth > 0:
                metrics = self._smoothed_metrics[path_id]
                metrics.utilization = metrics.sending_rate / estimated_bandwidth

            # Update recent history
            if path_id in self._metrics_history:
                for metrics in self._metrics_history[path_id][-self.config.degradation_window:]:
                    if estimated_bandwidth > 0:
                        metrics.utilization = metrics.sending_rate / estimated_bandwidth

    def __repr__(self) -> str:
        """String representation"""
        with self._lock:
            return (f"PathMonitor(paths={len(self._metrics_history)}, "
                    f"updates={self.total_updates}, "
                    f"degradations={self.degradation_events})")
