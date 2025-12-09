"""
Path manager for Extension 4: Multipath Rate Allocation

Manages multiple network paths for multipath. Handles path
discovery, tracking, state management, and primary path selection.

Key Features:
- Automatic path discovery from network interfaces
- Path state machine (ACTIVE, IDLE, FAILED, PROBING)
- Primary path selection based on quality metrics
- Path failure detection and recovery
- Thread-safe operations for concurrent access

Architecture:
    PathManager
    ├── Path discovery (enumerate network interfaces)
    ├── Path tracking (maintain path state)
    ├── Path validation (liveness checks)
    └── Primary selection (choose best path)

# TODO: path discovery only works on Linux, need platform-specific code for others
Usage:
    manager = PathManager(config.path_manager)
    paths = manager.discover_paths()
    active_paths = manager.get_active_paths()
    primary = manager.select_primary_path()
"""
import time
import logging
import threading
from enum import Enum
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class PathState(Enum):
    """Path states in the state machine"""
    ACTIVE = "active"        # Path is active and being used
    IDLE = "idle"            # Path is available but not currently used
    FAILED = "failed"        # Path has failed (timeouts, errors)
    PROBING = "probing"      # Path is being validated/probed
    DISABLED = "disabled"    # Path manually disabled


@dataclass
class Path:
    """
    Represents a network path

    A path is uniquely identified by (source_addr, dest_addr, interface).
    Tracks path characteristics, quality metrics, and state.
    """
    # Identity
    path_id: int
    source_addr: str
    dest_addr: str
    interface: str  # Network interface (e.g., "eth0", "wlan0", "lte0")

    # State
    state: PathState = PathState.PROBING

    # Path characteristics (estimated)
    estimated_bandwidth: float = 0.0  # Mbps
    baseline_rtt: float = 0.0  # ms
    loss_rate: float = 0.0

    # Path quality metrics (current)
    available_bandwidth: float = 0.0  # Current available BW (Mbps)
    current_rtt: float = 0.0  # Current RTT (ms)
    congestion_level: float = 0.0  # 0-1 scale
    stability: float = 1.0  # 1.0 = stable, 0.0 = highly variable

    # Quality of Service (if known)
    cost: float = 0.0  # Monetary cost per MB (e.g., cellular data)
    energy_cost: float = 0.0  # Energy cost (WiFi < Cellular)
    preference: float = 0.5  # User/policy preference [0, 1]

    # Statistics
    packets_sent: int = 0
    packets_acked: int = 0
    packets_lost: int = 0
    bytes_sent: int = 0
    bytes_acked: int = 0

    # Timing
    created_at: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    last_probe: float = 0.0

    # Failure tracking
    consecutive_failures: int = 0
    total_failures: int = 0

    def __post_init__(self):
        """Initialize computed fields"""
        if self.available_bandwidth == 0.0:
            self.available_bandwidth = self.estimated_bandwidth
        if self.current_rtt == 0.0:
            self.current_rtt = self.baseline_rtt

    def update_statistics(self, packets_sent: int, packets_acked: int,
                         bytes_sent: int, bytes_acked: int):
        """
        Update path statistics

        Args:
            packets_sent: New packets sent
            packets_acked: New packets acknowledged
            bytes_sent: New bytes sent
            bytes_acked: New bytes acknowledged
        """
        self.packets_sent += packets_sent
        self.packets_acked += packets_acked
        self.packets_lost += (packets_sent - packets_acked)
        self.bytes_sent += bytes_sent
        self.bytes_acked += bytes_acked

        # Update loss rate
        if self.packets_sent > 0:
            self.loss_rate = self.packets_lost / self.packets_sent

        self.last_update = time.time()

    def mark_used(self):
        """Mark path as recently used"""
        self.last_used = time.time()
        self.consecutive_failures = 0  # Reset failure counter

    def mark_failure(self):
        """Mark path as failed"""
        self.consecutive_failures += 1
        self.total_failures += 1
        self.last_update = time.time()

        if self.state != PathState.DISABLED:
            self.state = PathState.FAILED

    def mark_recovered(self):
        """Mark path as recovered from failure"""
        self.consecutive_failures = 0
        self.state = PathState.ACTIVE
        self.last_update = time.time()

    def is_healthy(self, timeout: float = 10.0) -> bool:
        """
        Check if path is healthy

        Args:
            timeout: Maximum time since last update (seconds)

        Returns:
            True if path is healthy
        """
        if self.state in [PathState.FAILED, PathState.DISABLED]:
            return False

        # Check timeout
        if time.time() - self.last_update > timeout:
            return False

        # Check failure rate
        if self.consecutive_failures >= 3:
            return False

        return True

    def get_quality_score(self) -> float:
        """
        Calculate overall path quality score [0, 1]

        Combines throughput, latency, loss, and stability into single metric.
        Higher is better.

        Returns:
            Quality score [0, 1]
        """
        if not self.is_healthy():
            return 0.0

        # Normalize components [0, 1]
        # Bandwidth: assume max 1000 Mbps
        bw_score = min(self.available_bandwidth / 1000.0, 1.0)

        # Latency: 0ms = 1.0, 500ms+ = 0.0
        latency_score = max(0.0, 1.0 - self.current_rtt / 500.0)

        # Loss: 0% = 1.0, 10%+ = 0.0
        loss_score = max(0.0, 1.0 - self.loss_rate * 10.0)

        # Stability: already [0, 1]
        stability_score = self.stability

        # Weighted combination
        quality = (0.4 * bw_score +
                  0.3 * latency_score +
                  0.2 * loss_score +
                  0.1 * stability_score)

        # Apply preference
        quality = quality * (0.5 + 0.5 * self.preference)

        return quality

    def __repr__(self):
        return (f"Path(id={self.path_id}, interface={self.interface}, "
                f"state={self.state.value}, bw={self.available_bandwidth:.1f}Mbps, "
                f"rtt={self.current_rtt:.1f}ms, loss={self.loss_rate:.4f})")


@dataclass
class PathManagerConfig:
    """Configuration for PathManager"""
    # Path limits
    max_paths: int = 8  # Maximum simultaneous paths
    min_paths: int = 1  # Minimum paths to maintain

    # Timeouts
    path_timeout: float = 10.0  # Path failure detection (seconds)
    idle_timeout: float = 60.0  # Move to IDLE after this long unused
    probe_interval: float = 5.0  # Path probing frequency

    # Failure handling
    max_consecutive_failures: int = 3  # Mark FAILED after this many
    recovery_probe_interval: float = 10.0  # Try to recover failed paths

    # Path discovery
    auto_discovery: bool = True  # Automatically discover new paths
    discovery_interval: float = 30.0  # Discovery frequency (seconds)

    # Primary path selection
    primary_path_preference: str = 'quality'  # 'quality', 'bandwidth', 'latency'
    primary_path_stickiness: float = 0.1  # Hysteresis for switching (0-1)

    enabled: bool = True


class PathManager:
    """
    Production-grade path manager for multipath scenarios

    Manages discovery, tracking, and state of multiple network paths.
    Thread-safe for concurrent access from multiple flows.

    Key Responsibilities:
    - Path discovery from network interfaces
    - Path state tracking (ACTIVE, IDLE, FAILED, PROBING)
    - Path validation and liveness checks
    - Primary path selection
    - Failure detection and recovery

    Usage:
        manager = PathManager(config)
        manager.discover_paths()
        paths = manager.get_active_paths()
        primary = manager.select_primary_path()
    """

    def __init__(self, config: PathManagerConfig):
        """
        Initialize path manager

        Args:
            config: PathManagerConfig instance
        """
        self.config = config

        # Path storage
        self.paths: Dict[int, Path] = {}  # path_id -> Path
        self.path_counter = 0

        # Primary path tracking
        self.primary_path_id: Optional[int] = None
        self.primary_path_switch_count = 0

        # Discovery tracking
        self.last_discovery = 0.0
        self.discovered_interfaces: Set[str] = set()

        # Thread safety
        self._lock = threading.RLock()

        # Performance tracking
        self.path_history = deque(maxlen=1000)  # Historical path states

        logger.info("="*60)
        logger.info("PathManager Initialized")
        logger.info("="*60)
        logger.info(f"  Max Paths: {config.max_paths}")
        logger.info(f"  Path Timeout: {config.path_timeout}s")
        logger.info(f"  Auto Discovery: {config.auto_discovery}")
        logger.info("="*60)

    def discover_paths(self) -> List[Path]:
        """
        Discover available network paths

        In production, this would enumerate network interfaces and create
        paths for each viable interface. For simulation, we create paths
        based on configuration.

        Returns:
            List of newly discovered paths
        """
        with self._lock:
            current_time = time.time()

            # Check if discovery needed
            if (current_time - self.last_discovery < self.config.discovery_interval):
                return []

            self.last_discovery = current_time

            # Simulate interface discovery
            # In production: use netifaces or similar to enumerate interfaces
            discovered_interfaces = self._simulate_interface_discovery()

            new_paths = []
            for interface in discovered_interfaces:
                if interface not in self.discovered_interfaces:
                    # Create new path
                    path = self._create_path_for_interface(interface)
                    if path:
                        new_paths.append(path)
                        self.discovered_interfaces.add(interface)
                        logger.info(f"Discovered new path: {path}")

            return new_paths

    def _simulate_interface_discovery(self) -> List[str]:
        """
        Simulate network interface discovery

        In production, this would use netifaces, psutil, or similar to
        enumerate actual network interfaces.

        Returns:
            List of interface names
        """
        # Default simulation: 1-2 interfaces
        # Can be extended based on config or environment
        interfaces = ["eth0"]  # Always have primary

        # Simulate WiFi
        if hasattr(self.config, 'simulate_wifi') and self.config.simulate_wifi:
            interfaces.append("wlan0")

        # Simulate cellular
        if hasattr(self.config, 'simulate_cellular') and self.config.simulate_cellular:
            interfaces.append("lte0")

        return interfaces

    def _create_path_for_interface(self, interface: str) -> Optional[Path]:
        """
        Create path for network interface

        Args:
            interface: Interface name (e.g., "eth0", "wlan0")

        Returns:
            Path object or None if creation fails
        """
        with self._lock:
            # Check path limit
            if len(self.paths) >= self.config.max_paths:
                logger.warning(f"Cannot create path for {interface}: max paths reached")
                return None

            # Generate path ID
            path_id = self.path_counter
            self.path_counter += 1

            # Estimate characteristics based on interface type
            estimated_bw, baseline_rtt, preference, cost = self._estimate_path_characteristics(interface)

            # Create path
            path = Path(
                path_id=path_id,
                source_addr="0.0.0.0",  # Will be filled in production
                dest_addr="0.0.0.0",    # Will be filled in production
                interface=interface,
                state=PathState.PROBING,
                estimated_bandwidth=estimated_bw,
                baseline_rtt=baseline_rtt,
                available_bandwidth=estimated_bw,
                current_rtt=baseline_rtt,
                preference=preference,
                cost=cost
            )

            # Add to paths
            self.paths[path_id] = path

            return path

    def _estimate_path_characteristics(self, interface: str) -> tuple:
        """
        Estimate path characteristics from interface type

        Args:
            interface: Interface name

        Returns:
            (bandwidth, rtt, preference, cost) tuple
        """
        # Default values
        bandwidth = 100.0  # Mbps
        rtt = 50.0  # ms
        preference = 0.5
        cost = 0.0

        # Ethernet/wired
        if interface.startswith('eth'):
            bandwidth = 1000.0
            rtt = 10.0
            preference = 0.9
            cost = 0.0

        # WiFi
        elif interface.startswith('wlan') or interface.startswith('wifi'):
            bandwidth = 100.0
            rtt = 20.0
            preference = 0.7
            cost = 0.0

        # Cellular/LTE/5G
        elif interface.startswith('lte') or interface.startswith('cell') or interface.startswith('5g'):
            bandwidth = 50.0
            rtt = 60.0
            preference = 0.5
            cost = 0.01  # $0.01 per MB

        return bandwidth, rtt, preference, cost

    def add_path(self, path: Path) -> bool:
        """
        Add path to manager

        Args:
            path: Path object to add

        Returns:
            True if added successfully
        """
        with self._lock:
            # Check limit
            if len(self.paths) >= self.config.max_paths:
                logger.warning(f"Cannot add path {path.path_id}: max paths reached")
                return False

            # Check for duplicate
            if path.path_id in self.paths:
                logger.warning(f"Path {path.path_id} already exists")
                return False

            # Add path
            self.paths[path.path_id] = path
            logger.info(f"Added path: {path}")

            return True

    def remove_path(self, path_id: int):
        """
        Remove path from manager

        Args:
            path_id: ID of path to remove
        """
        with self._lock:
            if path_id in self.paths:
                path = self.paths[path_id]
                del self.paths[path_id]

                # If this was primary, select new primary
                if path_id == self.primary_path_id:
                    self.primary_path_id = None
                    self.select_primary_path()

                logger.info(f"Removed path: {path}")
            else:
                logger.warning(f"Cannot remove non-existent path {path_id}")

    def get_path(self, path_id: int) -> Optional[Path]:
        """
        Get path by ID

        Args:
            path_id: Path ID

        Returns:
            Path object or None if not found
        """
        with self._lock:
            return self.paths.get(path_id)

    def get_active_paths(self) -> List[Path]:
        """
        Get all currently active paths

        Returns:
            List of active paths
        """
        with self._lock:
            return [p for p in self.paths.values()
                   if p.state == PathState.ACTIVE and p.is_healthy(self.config.path_timeout)]

    def get_all_paths(self) -> List[Path]:
        """
        Get all paths regardless of state

        Returns:
            List of all paths
        """
        with self._lock:
            return list(self.paths.values())

    def select_primary_path(self) -> Optional[Path]:
        """
        Select primary path based on configured strategy

        Uses hysteresis to avoid frequent switching (path stickiness).

        Returns:
            Primary path or None if no paths available
        """
        with self._lock:
            active_paths = self.get_active_paths()

            if not active_paths:
                self.primary_path_id = None
                return None

            # Get current primary
            current_primary = self.paths.get(self.primary_path_id) if self.primary_path_id is not None else None

            # Select based on strategy
            if self.config.primary_path_preference == 'quality':
                candidate = max(active_paths, key=lambda p: p.get_quality_score())
            elif self.config.primary_path_preference == 'bandwidth':
                candidate = max(active_paths, key=lambda p: p.available_bandwidth)
            elif self.config.primary_path_preference == 'latency':
                candidate = min(active_paths, key=lambda p: p.current_rtt)
            else:
                candidate = active_paths[0]

            # Apply hysteresis (path stickiness)
            if current_primary and current_primary.is_healthy():
                current_score = current_primary.get_quality_score()
                candidate_score = candidate.get_quality_score()
                threshold = current_score * (1 + self.config.primary_path_stickiness)

                # Only switch if candidate is significantly better
                logger.debug(f"Hysteresis check: current_id={current_primary.path_id} ({current_score:.4f}), candidate_id={candidate.path_id} ({candidate_score:.4f}), threshold={threshold:.4f}, should_switch={candidate_score > threshold}")
                if candidate_score > threshold:
                    # Switch
                    self.primary_path_id = candidate.path_id
                    self.primary_path_switch_count += 1
                    logger.info(f"Primary path switched: {current_primary.path_id} → {candidate.path_id}")
                else:
                    # Keep current
                    candidate = current_primary
                    self.primary_path_id = current_primary.path_id
            else:
                # No current primary or unhealthy, use candidate
                logger.debug(f"No current primary or unhealthy: current_primary={current_primary}, is_healthy={current_primary.is_healthy() if current_primary else None}")
                self.primary_path_id = candidate.path_id
                logger.info(f"Primary path selected: {candidate}")

            return candidate

    def update_path_quality(self, path_id: int, bandwidth: float, rtt: float,
                           loss_rate: float, stability: float):
        """
        Update path quality metrics

        Args:
            path_id: Path ID
            bandwidth: Available bandwidth (Mbps)
            rtt: Current RTT (ms)
            loss_rate: Current loss rate [0, 1]
            stability: Rate stability [0, 1]
        """
        with self._lock:
            path = self.paths.get(path_id)
            if path:
                path.available_bandwidth = bandwidth
                path.current_rtt = rtt
                path.loss_rate = loss_rate
                path.stability = stability
                path.last_update = time.time()

                # Only reset failure count if the path hasn't had too many failures
                # This prevents flaky paths from recovering too quickly
                if path.consecutive_failures < 3:
                    path.consecutive_failures = 0

                    # Update state based on quality
                    # If path was in FAILED state and now has good metrics, recover it
                    if path.state == PathState.FAILED:
                        path.mark_recovered()
                        logger.info(f"Path {path_id} recovered")
                else:
                    # Path has too many failures, mark as failed
                    path.mark_failure()
                    logger.warning(f"Path {path_id} failed (too many consecutive failures)")

    def get_statistics(self) -> Dict:
        """
        Get path manager statistics

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            active = len(self.get_active_paths())
            total = len(self.paths)

            states = {}
            for state in PathState:
                states[state.value] = len([p for p in self.paths.values() if p.state == state])

            return {
                'total_paths': total,
                'active_paths': active,
                'primary_path_id': self.primary_path_id,
                'path_switches': self.primary_path_switch_count,
                'states': states,
                'discovered_interfaces': list(self.discovered_interfaces)
            }

    def get_summary(self) -> str:
        """
        Get human-readable summary

        Returns:
            Formatted summary string
        """
        stats = self.get_statistics()

        summary = "Path Manager Summary:\n"
        summary += f"  Total Paths: {stats['total_paths']}\n"
        summary += f"  Active Paths: {stats['active_paths']}\n"
        summary += f"  Primary Path: {stats['primary_path_id']}\n"
        summary += f"  Path Switches: {stats['path_switches']}\n"
        summary += f"  States: {stats['states']}\n"
        summary += f"  Interfaces: {stats['discovered_interfaces']}\n"

        return summary
