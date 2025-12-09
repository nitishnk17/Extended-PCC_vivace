"""
Extension 4: Multipath Rate Allocation - Phase 4
MultipathScheduler: Softmax-based rate allocation across multiple paths

This module implements the multi-armed bandit approach to multipath rate allocation,
using softmax policy to distribute aggregate sending rate across available paths
based on their utility values.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import threading
import time

from src.path_manager import Path, PathState


@dataclass
class MultipathSchedulerConfig:
    """Configuration for MultipathScheduler

    Attributes:
        temperature: Base temperature parameter τ for softmax (exploration-exploitation)
        temperature_schedule: Temperature mode ('aggressive', 'balanced', 'cautious')
        min_path_rate: Minimum sending rate per active path (Mbps)
        max_path_utilization: Maximum fraction of estimated capacity to use
        rebalance_interval: Minimum time between rebalancing (seconds)
        enabled: Whether multipath scheduling is enabled
    """
    temperature: float = 0.5  # τ parameter
    temperature_schedule: str = 'balanced'  # 'aggressive', 'balanced', 'cautious'
    min_path_rate: float = 0.5  # Mbps
    max_path_utilization: float = 0.95  # 95% of capacity
    rebalance_interval: float = 0.1  # 100ms
    enabled: bool = True


class MultipathScheduler:
    """Softmax-based multipath rate allocation scheduler

    Allocates aggregate sending rate across multiple paths using softmax policy:

    r_p = R_total · exp(U_p/τ) / Σ_p' exp(U_p'/τ)

    Where:
    - r_p = rate allocated to path p
    - R_total = total aggregate sending rate
    - U_p = utility of path p
    - τ = temperature parameter (controls exploration vs exploitation)

    Thread-safe for concurrent rate allocations and rebalancing.
    """

    def __init__(self, config: Optional[MultipathSchedulerConfig] = None):
        """Initialize MultipathScheduler

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or MultipathSchedulerConfig()

        # Current allocations: path_id -> allocated_rate
        self._allocations: Dict[int, float] = {}

        # Last rebalance time
        self._last_rebalance_time = 0.0

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self.total_allocations = 0
        self.total_rebalances = 0
        self.constraint_violations = 0

    def allocate_rates(
        self,
        total_rate: float,
        path_utilities: Dict[int, float],
    # FIXME: softmax can be unstable with very different path qualities
        paths: Optional[Dict[int, Path]] = None,
        force: bool = False
    ) -> Dict[int, float]:
        """Allocate total rate across paths using softmax policy

        Args:
            total_rate: Total aggregate sending rate to allocate (Mbps)
            path_utilities: Dictionary mapping path_id to utility value
            paths: Optional dictionary of Path objects for capacity constraints
            force: If True, ignore rebalance interval

        Returns:
            Dictionary mapping path_id to allocated rate (Mbps)
        """
        if not self.config.enabled:
            # If disabled, return empty allocations
            return {}

        with self._lock:
            # Check rebalance interval
            current_time = time.time()
            if not force and (current_time - self._last_rebalance_time) < self.config.rebalance_interval:
                # Return cached allocations
                return dict(self._allocations)

            # Filter out paths with non-positive utilities and inactive paths
            active_utilities = {}
            for path_id, utility in path_utilities.items():
                if paths and path_id in paths:
                    path = paths[path_id]
                    if path.state not in [PathState.ACTIVE, PathState.PROBING]:
                        continue
                active_utilities[path_id] = utility

            if not active_utilities:
                # No active paths
                self._allocations = {}
                return {}

            # Calculate softmax weights
            softmax_weights = self._get_softmax_weights(active_utilities)

            # Allocate rates based on softmax weights
            raw_allocations = {}
            for path_id, weight in softmax_weights.items():
                raw_allocations[path_id] = total_rate * weight

            # Adjust for constraints
            final_allocations = self._adjust_for_constraints(raw_allocations, paths)

            # Update cache
            self._allocations = final_allocations
            self._last_rebalance_time = current_time
            self.total_allocations += 1

            return dict(final_allocations)

    def _get_softmax_weights(self, utilities: Dict[int, float]) -> Dict[int, float]:
        """Calculate softmax weights from utilities

        weight_p = exp(U_p/τ) / Σ_p' exp(U_p'/τ)

        Args:
            utilities: Dictionary mapping path_id to utility

        Returns:
            Dictionary mapping path_id to weight [0, 1] (sum to 1.0)
        """
        if not utilities:
            return {}

        # Get temperature
        temperature = self._get_temperature()

        # Calculate exponentials with numerical stability
        # Subtract max utility to avoid overflow
        max_utility = max(utilities.values())

        exp_values = {}
        for path_id, utility in utilities.items():
            # exp((U - U_max) / τ) is numerically stable
            exp_values[path_id] = np.exp((utility - max_utility) / temperature)

        # Normalize to get weights
        total_exp = sum(exp_values.values())

        weights = {}
        if total_exp > 0:
            for path_id, exp_val in exp_values.items():
                weights[path_id] = exp_val / total_exp
        else:
            # Fallback: uniform distribution
            uniform_weight = 1.0 / len(utilities)
            weights = {path_id: uniform_weight for path_id in utilities.keys()}

        return weights

    def _get_temperature(self) -> float:
        """Get temperature parameter based on schedule

        Returns:
            Temperature value τ
        """
        mode = self.config.temperature_schedule.lower()

        if mode == 'aggressive':
            # Low temperature → exploitation (concentrate on best paths)
            return 0.1
        elif mode == 'balanced':
            # Medium temperature → balanced
            return 0.5
        elif mode == 'cautious':
            # High temperature → exploration (more uniform)
            return 1.0
        else:
            # Use configured default
            return self.config.temperature

    def _adjust_for_constraints(
        self,
        allocations: Dict[int, float],
        paths: Optional[Dict[int, Path]]
    ) -> Dict[int, float]:
        """Adjust allocations to respect path capacity and minimum rate constraints

        Args:
            allocations: Raw allocations from softmax
            paths: Dictionary of Path objects (optional)

        Returns:
            Adjusted allocations respecting constraints
        """
        if not allocations:
            return {}

        adjusted = dict(allocations)

        # Apply capacity constraints if paths available
        if paths:
            for path_id, rate in list(adjusted.items()):
                if path_id in paths:
                    path = paths[path_id]

                    # Maximum rate based on estimated bandwidth
                    max_rate = path.estimated_bandwidth * self.config.max_path_utilization

                    if rate > max_rate:
                        # Cap at maximum
                        adjusted[path_id] = max_rate
                        self.constraint_violations += 1

        # Apply minimum rate constraint
        total_rate = sum(adjusted.values())
        min_rate = self.config.min_path_rate

        # Remove paths below minimum (unless they're the only path)
        if len(adjusted) > 1:
            to_remove = []
            for path_id, rate in adjusted.items():
                if rate < min_rate:
                    to_remove.append(path_id)

            # Remove and redistribute
            if to_remove:
                removed_rate = sum(adjusted[pid] for pid in to_remove)
                for path_id in to_remove:
                    del adjusted[path_id]

                # Redistribute removed rate proportionally
                if adjusted:
                    remaining_total = sum(adjusted.values())
                    if remaining_total > 0:
                        for path_id in adjusted:
                            adjusted[path_id] += removed_rate * (adjusted[path_id] / remaining_total)

        return adjusted

    def rebalance_on_path_failure(
        self,
        failed_path_id: int,
        path_utilities: Dict[int, float],
        paths: Optional[Dict[int, Path]] = None
    ) -> Dict[int, float]:
        """Reallocate traffic when a path fails

        Args:
            failed_path_id: ID of failed path
            path_utilities: Current utilities for all paths
            paths: Dictionary of Path objects

        Returns:
            New allocations excluding failed path
        """
        with self._lock:
            # Get current total rate
            total_rate = sum(self._allocations.values())

            # Remove failed path from utilities
            active_utilities = {
                pid: util for pid, util in path_utilities.items()
                if pid != failed_path_id
            }

            # Reallocate using softmax
            new_allocations = self.allocate_rates(
                total_rate=total_rate,
                path_utilities=active_utilities,
                paths=paths,
                force=True  # Force immediate rebalance
            )

            self.total_rebalances += 1

            return new_allocations

    def get_current_allocations(self) -> Dict[int, float]:
        """Get current rate allocations

        Returns:
            Dictionary mapping path_id to allocated rate
        """
        with self._lock:
            return dict(self._allocations)

    def get_allocation_stats(self) -> Dict:
        """Get allocation statistics

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                'total_allocations': self.total_allocations,
                'total_rebalances': self.total_rebalances,
                'constraint_violations': self.constraint_violations,
                'active_paths': len(self._allocations),
                'total_rate': sum(self._allocations.values())
            }

    def reset(self) -> None:
        """Reset scheduler state"""
        with self._lock:
            self._allocations.clear()
            self._last_rebalance_time = 0.0
            self.total_allocations = 0
            self.total_rebalances = 0
            self.constraint_violations = 0

    def __repr__(self) -> str:
        """String representation"""
        with self._lock:
            return (f"MultipathScheduler(paths={len(self._allocations)}, "
                    f"total_rate={sum(self._allocations.values()):.2f}Mbps, "
                    f"allocations={self.total_allocations})")
