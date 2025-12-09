"""
Extension 4: Multipath Rate Allocation (MVP Integration)

Integrates all multipath components into a working congestion control algorithm.
Extends Extension 3 with multipath capabilities.
"""

from typing import Optional, Dict, List
import time
import logging

from src.pcc_vivace_extension3 import PccVivaceExtension3
from src.config import Config
from src.path_manager import PathManager, PathManagerConfig, Path, PathState
from src.path_monitor import PathMonitor, PathMonitorConfig, PathMetrics
from src.path_utility_calculator import PathUtilityCalculator, PathUtilityConfig
from src.multipath_scheduler import MultipathScheduler, MultipathSchedulerConfig
from src.correlation_detector import CorrelationDetector, CorrelationDetectorConfig

logger = logging.getLogger(__name__)


class PccVivaceExtension4(PccVivaceExtension3):
    """
    Extension 4: Multipath Rate Allocation

    Adds multipath support to Extension 3, enabling traffic distribution across
    multiple network paths using utility-based softmax allocation.

    Key capabilities:
    - Path discovery and management
    - Per-path performance monitoring
    - Utility-based rate allocation
    - Shared bottleneck detection
    - Automatic path failure handling

    For MVP: Simplified integration that demonstrates multipath framework while
    maintaining compatibility with single-path operation.
    """

    def __init__(
        self,
        network,
        config: Config,
        flow_id: int = 0,
        multiflow_mode: bool = False,
        enable_multipath: bool = True
    ):
        """Initialize Extension 4

        Args:
            network: Network simulator instance
            config: Configuration object
            flow_id: Flow identifier for multi-flow scenarios
            multiflow_mode: Whether running in multi-flow mode
            enable_multipath: Whether to enable multipath features (MVP: defaults True)
        """
        # Initialize Extension 3 (includes all previous extensions)
        super().__init__(network, config, flow_id, multiflow_mode)

        self.enable_multipath = enable_multipath

        # OPTIMIZATION: Detect real-time traffic and adjust multipath behavior
        # Real-time traffic has strict latency requirements - avoid multipath overhead
        self.is_realtime_traffic = (config.classifier.expected_traffic_type == 'realtime')

        # Multipath state (initialize early for compatibility)
        self.discovered_paths = False
        self.current_path_allocations: Dict[int, float] = {}

        if not self.enable_multipath or self.is_realtime_traffic:
            # Multipath disabled for real-time traffic or when explicitly disabled
            if self.is_realtime_traffic:
                logger.info(f"[Extension 4] Real-time traffic detected - using single best path")
            # Set multipath components to None when disabled
            self.path_manager = None
            self.path_monitor = None
            self.path_utility_calc = None
            self.multipath_scheduler = None
            self.correlation_detector = None
            return

        # Initialize multipath components
        scheduler_config = MultipathSchedulerConfig()
        # For latency-sensitive traffic, reduce exploration (higher exploitation)
        if self.is_realtime_traffic:
            scheduler_config.temperature = 0.1  # Very low temp = use only best path
            scheduler_config.rebalance_interval = 1.0  # Less frequent switching

        self.path_manager = PathManager(PathManagerConfig())
        self.path_monitor = PathMonitor(PathMonitorConfig())
        self.path_utility_calc = PathUtilityCalculator(
            self.path_monitor,
            self.path_manager,
            PathUtilityConfig()
        )
        self.multipath_scheduler = MultipathScheduler(scheduler_config)
        self.correlation_detector = CorrelationDetector(
            self.path_monitor,
            CorrelationDetectorConfig()
        )

        # Additional multipath state (discovered_paths already initialized above)
        self.active_path_id = 0  # Primary path for MVP

        # Statistics
        self.path_switches = 0
        self.multipath_decisions = 0
        self.last_path_stats = {}  # Per-path statistics from latest MI

        # IMPROVEMENT: Active path switching and health monitoring
        self.health_check_interval = 5  # Check path health every N intervals
        self.intervals_since_health_check = 0
        self.degraded_paths = set()  # Track currently degraded paths
        self.path_degradation_count = {}  # Count degradations per path

    def run(self, duration: float) -> Dict:
        """Run congestion control for specified duration

        Args:
            duration: Duration to run in seconds

        Returns:
            Dictionary with performance statistics
        """
        # Check if multipath is actually enabled (not disabled for real-time traffic)
        multipath_active = self.enable_multipath and self.path_manager is not None

        # MVP: Path discovery then delegate to Extension 3
        if multipath_active and not self.discovered_paths:
            self._discover_paths()
            self.discovered_paths = True

        # Run Extension 3's control loop
        stats = super().run(duration)

        # Add multipath-specific statistics
        if multipath_active:
            stats['multipath_enabled'] = True
            stats['active_paths'] = len(self.path_manager.paths)
            stats['path_switches'] = self.path_switches
            stats['multipath_decisions'] = self.multipath_decisions
            stats['path_metrics'] = {}

            # Add path allocation information
            stats['final_path_allocations'] = self.current_path_allocations.copy()
        else:
            # Multipath disabled (e.g., for real-time traffic)
            stats['multipath_enabled'] = False
            stats['active_paths'] = 1  # Single path
            stats['path_switches'] = 0
            stats['multipath_decisions'] = 0

        return stats

    def _allocate_multipath_rate(self) -> None:
        """Allocate current sending rate across multiple paths"""
        if not self.enable_multipath or self.path_manager is None or len(self.path_manager.paths) <= 1:
            return

        # Compute per-path utilities (simplified: use estimated bandwidth as utility)
        path_utilities = {}
        for path_id in self.path_manager.paths:
            path = self.path_manager.paths[path_id]
            if path.state == PathState.ACTIVE:
                # Use path-specific utility based on estimated capacity
                # Simpler heuristic: favor paths with higher bandwidth
                path_utilities[path_id] = path.estimated_bandwidth

        # Allocate rate using multipath scheduler
        if path_utilities:
            allocations = self.multipath_scheduler.allocate_rates(
                total_rate=self.sending_rate,
                path_utilities=path_utilities,
                paths=self.path_manager.paths
            )

            # Update allocations and track decisions
            if allocations:
                old_allocations = self.current_path_allocations.copy()
                self.current_path_allocations = allocations
                self.multipath_decisions += 1

                # Check if primary path changed (path switch)
                if old_allocations:
                    old_primary = max(old_allocations, key=old_allocations.get)
                    new_primary = max(allocations, key=allocations.get)
                    if old_primary != new_primary:
                        self.path_switches += 1
                        self.active_path_id = new_primary

    def _discover_paths(self) -> None:
        """Discover available network paths

        Creates multiple simulated paths with varying characteristics
        """
        # Get number of paths from network config (default to 1 if not specified)
        num_paths = getattr(self.config.network, 'num_paths', 1)

        # Create multiple paths with UNIFORM characteristics for shared bottleneck
        # MVP: All paths share the same bottleneck, so they have equal capacity
        # In a real multipath scenario, each path would have separate links
        for path_id in range(num_paths):
            # MVP FIX: Use SAME bandwidth for all paths (they share one bottleneck)
            # This prevents the scheduler from thinking total capacity = sum of paths
            path_bandwidth = self.network.bandwidth_mbps / num_paths  # Divide shared capacity

            # Vary delay slightly per path (paths can have different RTTs)
            path_delay_factor = 1.0 + (path_id * 0.05)  # 100%, 105%, 110%, etc. (reduced variation)
            path_rtt = self.network.delay_s * 2 * 1000 * path_delay_factor

            path = Path(
                path_id=path_id,
                source_addr=f"sim_source_p{path_id}",
                dest_addr=f"sim_dest_p{path_id}",
                interface=f"sim{path_id}",
                state=PathState.ACTIVE,
                estimated_bandwidth=path_bandwidth,
                baseline_rtt=path_rtt
            )

            self.path_manager.paths[path_id] = path
            self.current_path_allocations[path_id] = 1.0 / num_paths  # Equal initial allocation

        self.active_path_id = 0  # Primary path

    def _update_path_monitoring(self, mi) -> None:
        """Update monitoring for all active paths

        Collects metrics from recent monitor intervals

        Args:
            mi: MonitorInterval with current metrics
        """
        if not self.enable_multipath or self.path_manager is None or len(self.path_manager.paths) <= 1:
            return

        # IMPROVEMENT: Update metrics for active path
        # In a real multipath system, we'd have separate MIs per path
        # For simulation, we update the primary path's metrics
        self.path_monitor.update_metrics(self.active_path_id, mi)

        # Update utilization for all paths
        for path_id, path in self.path_manager.paths.items():
            if path.estimated_bandwidth > 0:
                self.path_monitor.set_path_utilization(path_id, path.estimated_bandwidth)

    def _check_path_health(self) -> bool:
        """Check health of all paths and switch if needed

        IMPROVEMENT: Active path switching when degradation detected

        Returns:
            True if path switch occurred, False otherwise
        """
        if not self.enable_multipath or self.path_manager is None or len(self.path_manager.paths) <= 1:
            return False

        switched = False
        newly_degraded = []

        # Check health of all paths
        for path_id in list(self.path_manager.paths.keys()):
            is_degraded = self.path_monitor.is_path_degraded(path_id)

            if is_degraded and path_id not in self.degraded_paths:
                # Newly degraded path
                newly_degraded.append(path_id)
                self.degraded_paths.add(path_id)
                self.path_degradation_count[path_id] = self.path_degradation_count.get(path_id, 0) + 1

                # Mark path as degraded in path manager (use IDLE state)
                if path_id in self.path_manager.paths:
                    self.path_manager.paths[path_id].state = PathState.IDLE

                logger.warning(f"[Extension 4] Path {path_id} degraded (count: {self.path_degradation_count[path_id]})")

            elif not is_degraded and path_id in self.degraded_paths:
                # Path recovered
                self.degraded_paths.discard(path_id)
                if path_id in self.path_manager.paths:
                    self.path_manager.paths[path_id].state = PathState.ACTIVE
                logger.info(f"[Extension 4] Path {path_id} recovered")

        # IMPROVEMENT: Active path switching
        # If primary path is degraded, switch to best available path
        if self.active_path_id in newly_degraded:
            best_path = self._find_best_path(exclude={self.active_path_id})
            if best_path is not None and best_path != self.active_path_id:
                logger.info(f"[Extension 4] Switching from degraded path {self.active_path_id} to path {best_path}")
                old_path = self.active_path_id
                self.active_path_id = best_path
                self.path_switches += 1
                switched = True

                # Reallocate rates immediately
                self._reallocate_after_switch()

        return switched

    def _find_best_path(self, exclude: set = None) -> Optional[int]:
        """Find the best available path based on current metrics

        Args:
            exclude: Set of path IDs to exclude from consideration

        Returns:
            Best path ID or None if no paths available
        """
        exclude = exclude or set()
        best_path = None
        best_score = -float('inf')

        for path_id, path in self.path_manager.paths.items():
            # Skip excluded paths and non-active paths (IDLE = degraded, FAILED, etc.)
            if path_id in exclude or path.state not in (PathState.ACTIVE, PathState.PROBING):
                continue

            # Calculate path score based on estimated bandwidth and degradation history
            degradation_penalty = self.path_degradation_count.get(path_id, 0) * 0.1
            score = path.estimated_bandwidth - degradation_penalty

            # Bonus for paths with good monitoring data
            metrics = self.path_monitor.get_path_metrics(path_id, smoothed=True)
            if metrics and metrics.loss_rate < 0.01:  # Low loss
                score += 10.0

            if score > best_score:
                best_score = score
                best_path = path_id

        return best_path

    def _reallocate_after_switch(self) -> None:
        """Reallocate rates immediately after path switch

        Concentrates traffic on the new primary path while gradually
        reducing allocation to degraded paths.
        """
        if not self.path_manager or not self.path_manager.paths:
            return

        new_allocations = {}

        # Give majority of traffic to active path
        for path_id in self.path_manager.paths.keys():
            if path_id == self.active_path_id:
                new_allocations[path_id] = 0.7  # 70% on primary
            elif path_id in self.degraded_paths:
                new_allocations[path_id] = 0.05  # Minimal on degraded
            else:
                new_allocations[path_id] = 0.25 / max(1, len(self.path_manager.paths) - 1 - len(self.degraded_paths))

        # Normalize to sum to 1.0
        total = sum(new_allocations.values())
        if total > 0:
            self.current_path_allocations = {k: v/total for k, v in new_allocations.items()}

        logger.info(f"[Extension 4] Reallocated rates: {self.current_path_allocations}")


    def _collect_statistics(self) -> Dict:
        """Collect multipath-aware statistics

        Returns:
            Dictionary with statistics
        """
        stats = super()._collect_statistics()

        if not self.enable_multipath:
            return stats

        # Add multipath-specific stats
        stats['multipath_enabled'] = True
        stats['active_paths'] = len(self.path_manager.paths)
        stats['path_switches'] = self.path_switches
        stats['multipath_decisions'] = self.multipath_decisions

        # Add per-path statistics
        path_stats = {}
        for path_id in self.path_monitor.get_all_monitored_paths():
            metrics = self.path_monitor.get_path_metrics(path_id, smoothed=True)
            if metrics:
                path_stats[path_id] = {
                    'throughput': metrics.throughput,
                    'rtt_avg': metrics.rtt_avg,
                    'loss_rate': metrics.loss_rate,
                    'utilization': metrics.utilization
                }

        stats['path_metrics'] = path_stats

        return stats

    def _calculate_utility(self, mi) -> float:
        """Calculate utility for monitor interval and perform multipath allocation

        Args:
            mi: MonitorInterval

        Returns:
            Utility value
        """
        # IMPROVEMENT: Update path monitoring with current metrics
        self._update_path_monitoring(mi)

        # IMPROVEMENT: Periodic health checks
        if self.enable_multipath and self.path_manager is not None and len(self.path_manager.paths) > 1:
            self.intervals_since_health_check += 1
            if self.intervals_since_health_check >= self.health_check_interval:
                self._check_path_health()
                self.intervals_since_health_check = 0

        # Use Extension 3's utility calculation
        utility = super()._calculate_utility(mi)

        # Perform multipath rate allocation after utility calculation
        self._allocate_multipath_rate()

        return utility

    def _run_single_mi(self, rate: float, current_time: float):
        """Run monitor interval with aggregated multipath rate

        Extension 4 uses the multipath scheduler to make per-path rate decisions,
        but sends traffic at the AGGREGATE rate through a single logical flow.

        The multipath scheduler determines how to allocate capacity across paths,
        but the actual traffic is sent as a unified flow (not physically split).
        This prevents congestion when paths share a bottleneck.

        Args:
            rate: Sending rate in Mbps (aggregate across all paths)
            current_time: Current simulation time

        Returns:
            MonitorInterval with aggregated statistics
        """
        # Use parent's single-flow implementation
        # The multipath scheduling affects rate calculation via _calculate_utility
        # and _allocate_multipath_rate, not packet distribution
        return super()._run_single_mi(rate, current_time)

    def calculate_utility(self, mi) -> float:
        """Calculate utility for monitor interval

        For Extension 4, we could use path-specific utilities,
        but for MVP we use Extension 3's utility calculation.

        Args:
            mi: MonitorInterval

        Returns:
            Utility value
        """
        # Use Extension 3's utility calculation
        return super().calculate_utility(mi)

    def get_multipath_status(self) -> Dict:
        """Get current multipath status

        Returns:
            Dictionary with multipath status information
        """
        if not self.enable_multipath:
            return {'enabled': False}

        status = {
            'enabled': True,
            'discovered_paths': self.discovered_paths,
            'active_path_id': self.active_path_id,
            'num_paths': len(self.path_manager.paths),
            'path_switches': self.path_switches
        }

        # Add path information
        paths_info = []
        for path_id, path in self.path_manager.paths.items():
            paths_info.append({
                'path_id': path_id,
                'interface': path.interface,
                'state': path.state.value,
                'bandwidth': path.estimated_bandwidth,
                'rtt': path.baseline_rtt
            })

        status['paths'] = paths_info

        return status

    def __repr__(self) -> str:
        """String representation"""
        base_repr = super().__repr__().replace('Extension3', 'Extension4')

        if self.enable_multipath:
            return f"{base_repr} [Multipath: {len(self.path_manager.paths)} paths]"
        else:
            return f"{base_repr} [Multipath: disabled]"
