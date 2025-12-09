"""
Extension 4: Multipath Rate Allocation - Phase 5
CorrelationDetector: Shared bottleneck detection between paths (Simplified MVP)

This module detects correlations between paths to identify shared bottlenecks.
MVP version implements cross-correlation analysis without active probing.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import threading

from src.path_monitor import PathMonitor


@dataclass
class PathCorrelation:
    """Correlation metrics between two paths

    Attributes:
        path1_id: First path identifier
        path2_id: Second path identifier
        throughput_correlation: Throughput correlation coefficient [-1, 1]
        rtt_correlation: RTT correlation coefficient [-1, 1]
        loss_correlation: Loss rate correlation coefficient [-1, 1]
        shared_bottleneck_prob: Estimated probability of shared bottleneck [0, 1]
        confidence: Confidence in the estimate [0, 1]
        sample_count: Number of samples used
        last_updated: Timestamp of last update
    """
    path1_id: int
    path2_id: int

    throughput_correlation: float = 0.0
    rtt_correlation: float = 0.0
    loss_correlation: float = 0.0

    shared_bottleneck_prob: float = 0.0
    confidence: float = 0.0

    sample_count: int = 0
    last_updated: float = 0.0

    def __repr__(self) -> str:
        """String representation"""
        return (f"PathCorrelation(p{self.path1_id}-p{self.path2_id}: "
                f"shared={self.shared_bottleneck_prob:.2f}, conf={self.confidence:.2f})")


@dataclass
class CorrelationDetectorConfig:
    """Configuration for CorrelationDetector

    Attributes:
        min_samples: Minimum samples needed for correlation
        history_window: Time window for correlation analysis (seconds)
        shared_threshold: Correlation threshold for shared bottleneck detection
        confidence_threshold: Minimum confidence for reliable detection
        enabled: Whether correlation detection is enabled
    """
    min_samples: int = 10
    history_window: float = 5.0  # 5 seconds
    shared_threshold: float = 0.6  # Correlation > 0.6 indicates sharing
    confidence_threshold: float = 0.7
    enabled: bool = True


class CorrelationDetector:
    """Simplified correlation detector for shared bottleneck detection

    Detects correlations between paths using throughput and RTT cross-correlation
    analysis. MVP version without active probing.

    Thread-safe for concurrent correlation detection.
    """

    def __init__(
        self,
        path_monitor: PathMonitor,
        config: Optional[CorrelationDetectorConfig] = None
    ):
        """Initialize CorrelationDetector

        Args:
            path_monitor: PathMonitor instance for metrics
            config: Configuration (uses defaults if None)
        """
        self.path_monitor = path_monitor
        self.config = config or CorrelationDetectorConfig()

        # Correlation cache: (path1_id, path2_id) -> PathCorrelation
        self._correlations: Dict[Tuple[int, int], PathCorrelation] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self.total_detections = 0

    def detect_correlations(self, path_ids: Optional[List[int]] = None) -> List[PathCorrelation]:
        """Detect all pairwise path correlations

        Args:
            path_ids: List of path IDs to analyze (if None, use all monitored paths)

        Returns:
            List of PathCorrelation objects for all pairs
        """
        if not self.config.enabled:
            return []

        with self._lock:
            if path_ids is None:
                path_ids = self.path_monitor.get_all_monitored_paths()

            if len(path_ids) < 2:
                return []

            correlations = []

            # Compute pairwise correlations
            for i, path1_id in enumerate(path_ids):
                for path2_id in path_ids[i+1:]:
                    corr = self._compute_correlation(path1_id, path2_id)
                    if corr:
                        correlations.append(corr)

                        # Cache the result
                        key = self._get_correlation_key(path1_id, path2_id)
                        self._correlations[key] = corr

            self.total_detections += 1

            return correlations

    def _compute_correlation(self, path1_id: int, path2_id: int) -> Optional[PathCorrelation]:
        """Compute correlation between two paths

        Args:
            path1_id: First path ID
            path2_id: Second path ID

        Returns:
            PathCorrelation object or None if insufficient data
        """
        # Get historical metrics
        history1 = self.path_monitor.get_path_history(path1_id, duration=self.config.history_window)
        history2 = self.path_monitor.get_path_history(path2_id, duration=self.config.history_window)

        if len(history1) < self.config.min_samples or len(history2) < self.config.min_samples:
            return None

        # Align time series (use common timestamps)
        aligned_metrics = self._align_metrics(history1, history2)

        if len(aligned_metrics) < self.config.min_samples:
            return None

        # Extract metric arrays
        throughput1 = np.array([m1.throughput for m1, _ in aligned_metrics])
        throughput2 = np.array([m2.throughput for _, m2 in aligned_metrics])

        rtt1 = np.array([m1.rtt_avg for m1, _ in aligned_metrics])
        rtt2 = np.array([m2.rtt_avg for _, m2 in aligned_metrics])

        loss1 = np.array([m1.loss_rate for m1, _ in aligned_metrics])
        loss2 = np.array([m2.loss_rate for _, m2 in aligned_metrics])

        # Compute correlations
        tput_corr = self._pearson_correlation(throughput1, throughput2)
        rtt_corr = self._pearson_correlation(rtt1, rtt2)
        loss_corr = self._pearson_correlation(loss1, loss2)

        # Estimate shared bottleneck probability
        shared_prob, confidence = self._estimate_shared_bottleneck(
            tput_corr, rtt_corr, loss_corr, len(aligned_metrics)
        )

        import time
        return PathCorrelation(
            path1_id=path1_id,
            path2_id=path2_id,
            throughput_correlation=tput_corr,
            rtt_correlation=rtt_corr,
            loss_correlation=loss_corr,
            shared_bottleneck_prob=shared_prob,
            confidence=confidence,
            sample_count=len(aligned_metrics),
            last_updated=time.time()
        )

    def _align_metrics(self, history1: List, history2: List) -> List[Tuple]:
        """Align two metric histories by timestamp

        Args:
            history1: Metrics for path 1
            history2: Metrics for path 2

        Returns:
            List of (metrics1, metrics2) tuples with matched timestamps
        """
        aligned = []

        # Create timestamp -> metrics mapping for path 2
        metrics2_by_time = {m.timestamp: m for m in history2}

        # Find matching timestamps (with small tolerance)
        tolerance = 0.01  # 10ms tolerance

        for m1 in history1:
            # Find closest timestamp in history2
            closest_time = min(metrics2_by_time.keys(),
                             key=lambda t: abs(t - m1.timestamp),
                             default=None)

            if closest_time and abs(closest_time - m1.timestamp) < tolerance:
                aligned.append((m1, metrics2_by_time[closest_time]))

        return aligned

    def _pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Pearson correlation coefficient

        Args:
            x: First array
            y: Second array

        Returns:
            Correlation coefficient [-1, 1]
        """
        if len(x) < 2 or len(y) < 2:
            return 0.0

        # Handle constant arrays
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0

        try:
            # Suppress runtime warnings for invalid division
            with np.errstate(divide='ignore', invalid='ignore'):
                corr = np.corrcoef(x, y)[0, 1]
            if np.isnan(corr):
                return 0.0
            return float(corr)
        except:
            return 0.0

    def _estimate_shared_bottleneck(
        self,
        tput_corr: float,
        rtt_corr: float,
        loss_corr: float,
        sample_count: int
    ) -> Tuple[float, float]:
        """Estimate shared bottleneck probability from correlations

        High positive RTT correlation + negative throughput correlation = likely shared

        Args:
            tput_corr: Throughput correlation
            rtt_corr: RTT correlation
            loss_corr: Loss correlation
            sample_count: Number of samples

        Returns:
            Tuple of (shared_probability, confidence)
        """
        # Evidence for shared bottleneck:
        # 1. Positive RTT correlation (queues fill together)
        # 2. Negative throughput correlation (competing for capacity)
        # 3. Positive loss correlation (both see congestion)

        evidence = 0.0
        weight_sum = 0.0

        # RTT correlation (weight 0.4)
        if rtt_corr > 0:
            evidence += 0.4 * rtt_corr
        weight_sum += 0.4

        # Throughput anti-correlation (weight 0.35)
        if tput_corr < 0:
            evidence += 0.35 * abs(tput_corr)
        weight_sum += 0.35

        # Loss correlation (weight 0.25)
        if loss_corr > 0:
            evidence += 0.25 * loss_corr
        weight_sum += 0.25

        # Normalize
        if weight_sum > 0:
            probability = evidence / weight_sum
        else:
            probability = 0.0

        # Compute confidence based on sample count and correlation strength
        confidence = min(1.0, sample_count / (self.config.min_samples * 3))

        # Adjust confidence based on correlation strength
        max_corr = max(abs(tput_corr), abs(rtt_corr), abs(loss_corr))
        confidence *= min(1.0, max_corr / 0.5)  # Scale by correlation strength

        return probability, confidence

    def estimate_shared_bottleneck(self, path1_id: int, path2_id: int) -> float:
        """Estimate probability of shared bottleneck between two paths

        Args:
            path1_id: First path ID
            path2_id: Second path ID

        Returns:
            Probability [0, 1]
        """
        with self._lock:
            # Check cache
            key = self._get_correlation_key(path1_id, path2_id)
            if key in self._correlations:
                return self._correlations[key].shared_bottleneck_prob

            # Compute fresh correlation
            corr = self._compute_correlation(path1_id, path2_id)
            if corr:
                self._correlations[key] = corr
                return corr.shared_bottleneck_prob

            return 0.0

    def get_correlation_matrix(self, path_ids: Optional[List[int]] = None) -> np.ndarray:
        """Get full correlation matrix

        Args:
            path_ids: List of path IDs (if None, use all monitored)

        Returns:
            NxN matrix of shared bottleneck probabilities
        """
        if path_ids is None:
            path_ids = sorted(self.path_monitor.get_all_monitored_paths())

        n = len(path_ids)
        matrix = np.zeros((n, n))

        for i, path1_id in enumerate(path_ids):
            for j, path2_id in enumerate(path_ids):
                if i == j:
                    matrix[i, j] = 1.0  # Path always correlates with itself
                elif i < j:
                    prob = self.estimate_shared_bottleneck(path1_id, path2_id)
                    matrix[i, j] = prob
                    matrix[j, i] = prob  # Symmetric

        return matrix

    def _get_correlation_key(self, path1_id: int, path2_id: int) -> Tuple[int, int]:
        """Get normalized correlation key (smaller ID first)

        Args:
            path1_id: First path ID
            path2_id: Second path ID

        Returns:
            Tuple with smaller ID first
        """
        return (min(path1_id, path2_id), max(path1_id, path2_id))

    def clear_cache(self) -> None:
        """Clear correlation cache"""
        with self._lock:
            self._correlations.clear()

    def get_statistics(self) -> Dict:
        """Get detector statistics

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                'total_detections': self.total_detections,
                'cached_correlations': len(self._correlations)
            }

    def __repr__(self) -> str:
        """String representation"""
        with self._lock:
            return (f"CorrelationDetector(detections={self.total_detections}, "
                    f"cached={len(self._correlations)})")
