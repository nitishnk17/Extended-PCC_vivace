"""
Extension 4: Multipath Rate Allocation - Phase 3
PathUtilityCalculator: Per-path utility computation for rate allocation

This module calculates utility for each path independently, combining throughput,
latency, loss, and stability metrics into a single utility value used by the
multipath scheduler for rate allocation decisions.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
import threading

from src.path_monitor import PathMonitor, PathMetrics
from src.path_manager import PathManager, Path


@dataclass
class PathUtilityConfig:
    """Configuration for PathUtilityCalculator

    Attributes:
        throughput_weight: Weight for throughput term (T^exponent)
        throughput_exponent: Exponent for throughput (0.9 in PCC Vivace)
        latency_weight: Weight for latency penalty term
        loss_weight: Weight for loss penalty term (λ)
        stability_weight: Weight for stability bonus (β)

        latency_threshold: RTT threshold for sigmoid penalty (ms)
        latency_scale: Scale parameter for sigmoid function

        primary_path_bonus: Bonus utility for primary path
        cellular_penalty: Penalty for cellular paths (cost consideration)

        enabled: Whether utility calculation is enabled
    """
    # Base utility weights (matching PCC Vivace)
    throughput_weight: float = 1.0
    throughput_exponent: float = 0.9
    latency_weight: float = 900.0
    loss_weight: float = 11.35  # λ parameter
    stability_weight: float = 0.5  # β parameter

    # Latency penalty parameters
    latency_threshold: float = 100.0  # ms
    latency_scale: float = 10.0

    # Path-specific adjustments
    primary_path_bonus: float = 0.1  # 10% bonus for primary path
    cellular_penalty: float = 0.05  # 5% penalty for cellular

    # Control
    enabled: bool = True


class PathUtilityCalculator:
    """Per-path utility computation for multipath rate allocation

    Calculates utility for each path using the formula:
    Up = Tp · S(Lp) - λp · Tp · lp + βp · Stability(Tp)

    Where:
    - Tp = throughput on path p
    - S(Lp) = sigmoid latency penalty for path p
    - λp = adaptive loss coefficient
    - lp = loss rate on path p
    - βp = stability bonus
    - Stability(Tp) = -Var(Tp) (penalize rate variance)

    Thread-safe for concurrent utility calculations.
    """

    def __init__(
        self,
        path_monitor: PathMonitor,
        path_manager: PathManager,
        config: Optional[PathUtilityConfig] = None
    ):
        """Initialize PathUtilityCalculator

        Args:
            path_monitor: PathMonitor instance for metrics
            path_manager: PathManager instance for path info
            config: Configuration (uses defaults if None)
        """
        self.path_monitor = path_monitor
        self.path_manager = path_manager
        self.config = config or PathUtilityConfig()

        # Per-path utility cache: path_id -> utility
        self._utility_cache: Dict[int, float] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self.total_calculations = 0

    def calculate_path_utility(
        self,
        path_id: int,
        metrics: Optional[PathMetrics] = None,
        use_cache: bool = False
    ) -> float:
        """Calculate utility for a single path

        Args:
            path_id: Path identifier
            metrics: PathMetrics (if None, fetches from monitor)
            use_cache: If True, return cached value if available

        Returns:
            Utility value (can be negative)
        """
        if not self.config.enabled:
            return 0.0

        with self._lock:
            # Check cache
            if use_cache and path_id in self._utility_cache:
                return self._utility_cache[path_id]

            # Get metrics
            if metrics is None:
                metrics = self.path_monitor.get_path_metrics(path_id, smoothed=True)
                if metrics is None:
                    return 0.0

            # Get path info
            path = self.path_manager.get_path(path_id)
            if path is None:
                return 0.0

            # Calculate utility components
            throughput = metrics.throughput  # Mbps
            latency = metrics.rtt_avg  # ms
            loss_rate = metrics.loss_rate  # [0, 1]

            # Throughput term: T^exponent
            if throughput > 0:
                throughput_term = np.power(throughput, self.config.throughput_exponent)
            else:
                throughput_term = 0.0

            # Latency penalty: sigmoid function S(L)
            latency_penalty = self._sigmoid_latency_penalty(latency)

            # Loss penalty: λ * T * loss_rate
            loss_penalty = self.config.loss_weight * throughput * loss_rate

            # Stability bonus: -Var(T)
            stability_bonus = self._calculate_stability_bonus(path_id)

            # Base utility
            utility = (
                self.config.throughput_weight * throughput_term * latency_penalty
                - loss_penalty
                + self.config.stability_weight * stability_bonus
            )

            # Apply path-specific adjustments
            utility = self._apply_path_adjustments(path, utility)

            # Cache result
            self._utility_cache[path_id] = utility
            self.total_calculations += 1

            return utility

    def _sigmoid_latency_penalty(self, latency: float) -> float:
        """Calculate sigmoid latency penalty

        S(L) = 1 / (1 + exp((L - threshold) / scale))

        Returns value in [0, 1] where:
        - 1.0 = low latency (good)
        - 0.5 = at threshold
        - 0.0 = high latency (bad)

        Args:
            latency: RTT in milliseconds

        Returns:
            Penalty factor [0, 1]
        """
        if latency <= 0:
            return 1.0

        exponent = (latency - self.config.latency_threshold) / self.config.latency_scale

        # Clip exponent to avoid overflow
        exponent = np.clip(exponent, -50, 50)

        sigmoid = 1.0 / (1.0 + np.exp(exponent))
        return sigmoid

    def _calculate_stability_bonus(self, path_id: int) -> float:
        """Calculate stability bonus from throughput variance

        Stability = -Var(T)

        Lower variance (more stable) gives higher (less negative) bonus.

        Args:
            path_id: Path identifier

        Returns:
            Stability bonus (typically negative)
        """
        # Get recent throughput history
        history = self.path_monitor.get_path_history(path_id, duration=5.0)

        if len(history) < 2:
            return 0.0

        throughputs = [m.throughput for m in history if m.throughput > 0]

        if len(throughputs) < 2:
            return 0.0

        # Calculate variance (lower is better)
        variance = np.var(throughputs)

        # Return negative variance as bonus
        return -variance

    def _apply_path_adjustments(self, path: Path, base_utility: float) -> float:
        """Apply path-specific bonuses and penalties

        Args:
            path: Path object
            base_utility: Base utility before adjustments

        Returns:
            Adjusted utility
        """
        adjusted = base_utility
        is_cellular = 'cellular' in path.interface.lower() or 'lte' in path.interface.lower()

        # Check if this is the primary path
        primary_path = self.path_manager.select_primary_path()
        is_primary = primary_path and path.path_id == primary_path.path_id

        # Apply adjustments:
        # - Primary bonus only applies to non-cellular paths (avoid bonus for expensive paths)
        # - Cellular penalty always applies
        if is_primary and not is_cellular:
            adjusted *= (1.0 + self.config.primary_path_bonus)
        elif is_cellular:
            adjusted *= (1.0 - self.config.cellular_penalty)

        return adjusted

    def calculate_all_utilities(self, use_cache: bool = False) -> Dict[int, float]:
        """Calculate utilities for all monitored paths

        Args:
            use_cache: If True, use cached values when available

        Returns:
            Dictionary mapping path_id to utility
        """
        utilities = {}

        # Get all monitored paths
        path_ids = self.path_monitor.get_all_monitored_paths()

        for path_id in path_ids:
            utility = self.calculate_path_utility(path_id, use_cache=use_cache)
            utilities[path_id] = utility

        return utilities

    def get_normalized_utilities(
        self,
        utilities: Optional[Dict[int, float]] = None
    ) -> Dict[int, float]:
        """Get utilities normalized to [0, 1]

        Useful for visualization and analysis. Uses min-max normalization.

        Args:
            utilities: Dictionary of utilities (if None, calculates all)

        Returns:
            Dictionary mapping path_id to normalized utility [0, 1]
        """
        if utilities is None:
            utilities = self.calculate_all_utilities()

        if not utilities:
            return {}

        # Handle single path case
        if len(utilities) == 1:
            path_id = list(utilities.keys())[0]
            return {path_id: 1.0}

        # Min-max normalization
        min_utility = min(utilities.values())
        max_utility = max(utilities.values())

        if max_utility == min_utility:
            # All utilities equal - return 0.5 for all
            return {pid: 0.5 for pid in utilities.keys()}

        normalized = {}
        for path_id, utility in utilities.items():
            normalized[path_id] = (utility - min_utility) / (max_utility - min_utility)

        return normalized

    def clear_cache(self) -> None:
        """Clear utility cache"""
        with self._lock:
            self._utility_cache.clear()

    def get_statistics(self) -> Dict:
        """Get calculation statistics

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                'total_calculations': self.total_calculations,
                'cached_utilities': len(self._utility_cache)
            }

    def get_utility_breakdown(self, path_id: int) -> Dict:
        """Get detailed utility breakdown for debugging

        Args:
            path_id: Path identifier

        Returns:
            Dictionary with utility components
        """
        metrics = self.path_monitor.get_path_metrics(path_id, smoothed=True)
        if metrics is None:
            return {}

        path = self.path_manager.get_path(path_id)
        if path is None:
            return {}

        # Calculate components
        throughput = metrics.throughput
        latency = metrics.rtt_avg
        loss_rate = metrics.loss_rate

        throughput_term = (
            np.power(throughput, self.config.throughput_exponent)
            if throughput > 0 else 0.0
        )
        latency_penalty = self._sigmoid_latency_penalty(latency)
        loss_penalty = self.config.loss_weight * throughput * loss_rate
        stability_bonus = self._calculate_stability_bonus(path_id)

        base_utility = (
            self.config.throughput_weight * throughput_term * latency_penalty
            - loss_penalty
            + self.config.stability_weight * stability_bonus
        )

        final_utility = self._apply_path_adjustments(path, base_utility)

        return {
            'throughput_mbps': throughput,
            'latency_ms': latency,
            'loss_rate': loss_rate,
            'throughput_term': throughput_term,
            'latency_penalty': latency_penalty,
            'loss_penalty': loss_penalty,
            'stability_bonus': stability_bonus,
            'base_utility': base_utility,
            'final_utility': final_utility
        }

    def __repr__(self) -> str:
        """String representation"""
        with self._lock:
            return (f"PathUtilityCalculator(calcs={self.total_calculations}, "
                    f"cached={len(self._utility_cache)})")
