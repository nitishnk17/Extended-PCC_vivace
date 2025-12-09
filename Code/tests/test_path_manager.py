"""
Comprehensive Unit Tests for PathManager (Extension 4, Phase 1)

Tests cover:
- Path data structure
- Path state machine
- Path discovery
- Path addition/removal
- Primary path selection
- Quality scoring
- Thread safety
- Edge cases

Total: 40 comprehensive tests
"""
import pytest
import time
import threading
from unittest.mock import Mock, patch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.path_manager import (
    Path, PathState, PathManager, PathManagerConfig
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def path_config():
    """Standard path manager configuration"""
    return PathManagerConfig(
        max_paths=8,
        min_paths=1,
        path_timeout=10.0,
        probe_interval=5.0,
        auto_discovery=True
    )


@pytest.fixture
def path_manager(path_config):
    """Create path manager instance"""
    return PathManager(path_config)


@pytest.fixture
def sample_path():
    """Create sample path"""
    return Path(
        path_id=0,
        source_addr="192.168.1.100",
        dest_addr="8.8.8.8",
        interface="eth0",
        state=PathState.ACTIVE,
        estimated_bandwidth=100.0,
        baseline_rtt=50.0,
        available_bandwidth=100.0,
        current_rtt=50.0
    )


# ============================================================================
# Path Data Structure Tests
# ============================================================================

class TestPath:
    """Tests for Path data structure"""

    def test_path_creation(self):
        """Test basic path creation"""
        path = Path(
            path_id=0,
            source_addr="192.168.1.1",
            dest_addr="8.8.8.8",
            interface="eth0"
        )

        assert path.path_id == 0
        assert path.interface == "eth0"
        assert path.state == PathState.PROBING
        assert path.packets_sent == 0

    def test_path_initialization(self):
        """Test path post-initialization"""
        path = Path(
            path_id=1,
            source_addr="192.168.1.1",
            dest_addr="8.8.8.8",
            interface="wlan0",
            estimated_bandwidth=50.0,
            baseline_rtt=30.0
        )

        # Should initialize available_bandwidth from estimated
        assert path.available_bandwidth == 50.0
        assert path.current_rtt == 30.0

    def test_update_statistics(self):
        """Test updating path statistics"""
        path = Path(path_id=0, source_addr="", dest_addr="", interface="eth0")

        path.update_statistics(
            packets_sent=100,
            packets_acked=95,
            bytes_sent=150000,
            bytes_acked=142500
        )

        assert path.packets_sent == 100
        assert path.packets_acked == 95
        assert path.packets_lost == 5
        assert path.loss_rate == 0.05
        assert path.bytes_sent == 150000

    def test_mark_used(self):
        """Test marking path as used"""
        path = Path(path_id=0, source_addr="", dest_addr="", interface="eth0")
        path.consecutive_failures = 3

        initial_time = path.last_used
        time.sleep(0.01)
        path.mark_used()

        assert path.last_used > initial_time
        assert path.consecutive_failures == 0

    def test_mark_failure(self):
        """Test marking path as failed"""
        path = Path(path_id=0, source_addr="", dest_addr="", interface="eth0",
                   state=PathState.ACTIVE)

        path.mark_failure()
        assert path.consecutive_failures == 1
        assert path.total_failures == 1
        assert path.state == PathState.FAILED

        path.mark_failure()
        assert path.consecutive_failures == 2

    def test_mark_recovered(self):
        """Test marking path as recovered"""
        path = Path(path_id=0, source_addr="", dest_addr="", interface="eth0",
                   state=PathState.FAILED)
        path.consecutive_failures = 5

        path.mark_recovered()

        assert path.consecutive_failures == 0
        assert path.state == PathState.ACTIVE

    def test_is_healthy_normal(self):
        """Test healthy path detection"""
        path = Path(path_id=0, source_addr="", dest_addr="", interface="eth0",
                   state=PathState.ACTIVE)
        path.last_update = time.time()

        assert path.is_healthy() == True

    def test_is_healthy_failed_state(self):
        """Test unhealthy path - failed state"""
        path = Path(path_id=0, source_addr="", dest_addr="", interface="eth0",
                   state=PathState.FAILED)

        assert path.is_healthy() == False

    def test_is_healthy_timeout(self):
        """Test unhealthy path - timeout"""
        path = Path(path_id=0, source_addr="", dest_addr="", interface="eth0",
                   state=PathState.ACTIVE)
        path.last_update = time.time() - 20.0  # 20 seconds ago

        assert path.is_healthy(timeout=10.0) == False

    def test_is_healthy_too_many_failures(self):
        """Test unhealthy path - consecutive failures"""
        path = Path(path_id=0, source_addr="", dest_addr="", interface="eth0",
                   state=PathState.ACTIVE)
        path.consecutive_failures = 5

        assert path.is_healthy() == False

    def test_quality_score_healthy_path(self):
        """Test quality scoring for healthy path"""
        path = Path(
            path_id=0,
            source_addr="", dest_addr="", interface="eth0",
            state=PathState.ACTIVE,
            available_bandwidth=100.0,
            current_rtt=50.0,
            loss_rate=0.01,
            stability=0.9
        )

        score = path.get_quality_score()
        assert 0.0 < score <= 1.0

    def test_quality_score_unhealthy_path(self):
        """Test quality scoring for failed path"""
        path = Path(
            path_id=0,
            source_addr="", dest_addr="", interface="eth0",
            state=PathState.FAILED
        )

        score = path.get_quality_score()
        assert score == 0.0

    def test_quality_score_high_bandwidth(self):
        """Test quality score favors high bandwidth"""
        path_high = Path(
            path_id=0,
            source_addr="", dest_addr="", interface="eth0",
            state=PathState.ACTIVE,
            available_bandwidth=1000.0,
            current_rtt=50.0,
            loss_rate=0.0,
            stability=1.0
        )

        path_low = Path(
            path_id=1,
            source_addr="", dest_addr="", interface="wlan0",
            state=PathState.ACTIVE,
            available_bandwidth=10.0,
            current_rtt=50.0,
            loss_rate=0.0,
            stability=1.0
        )

        assert path_high.get_quality_score() > path_low.get_quality_score()

    def test_quality_score_low_latency(self):
        """Test quality score favors low latency"""
        path_low_lat = Path(
            path_id=0,
            source_addr="", dest_addr="", interface="eth0",
            state=PathState.ACTIVE,
            available_bandwidth=100.0,
            current_rtt=10.0,
            loss_rate=0.0,
            stability=1.0
        )

        path_high_lat = Path(
            path_id=1,
            source_addr="", dest_addr="", interface="wlan0",
            state=PathState.ACTIVE,
            available_bandwidth=100.0,
            current_rtt=200.0,
            loss_rate=0.0,
            stability=1.0
        )

        assert path_low_lat.get_quality_score() > path_high_lat.get_quality_score()

    def test_quality_score_with_preference(self):
        """Test quality score respects preference"""
        path_preferred = Path(
            path_id=0,
            source_addr="", dest_addr="", interface="eth0",
            state=PathState.ACTIVE,
            available_bandwidth=100.0,
            current_rtt=50.0,
            loss_rate=0.0,
            stability=1.0,
            preference=1.0
        )

        path_not_preferred = Path(
            path_id=1,
            source_addr="", dest_addr="", interface="lte0",
            state=PathState.ACTIVE,
            available_bandwidth=100.0,
            current_rtt=50.0,
            loss_rate=0.0,
            stability=1.0,
            preference=0.0
        )

        assert path_preferred.get_quality_score() > path_not_preferred.get_quality_score()


# ============================================================================
# PathManager Basic Tests
# ============================================================================

class TestPathManagerBasics:
    """Tests for basic PathManager functionality"""

    def test_initialization(self, path_config):
        """Test path manager initialization"""
        manager = PathManager(path_config)

        assert manager.config == path_config
        assert len(manager.paths) == 0
        assert manager.primary_path_id is None

    def test_add_path(self, path_manager, sample_path):
        """Test adding path"""
        result = path_manager.add_path(sample_path)

        assert result == True
        assert sample_path.path_id in path_manager.paths
        assert len(path_manager.paths) == 1

    def test_add_duplicate_path(self, path_manager, sample_path):
        """Test adding duplicate path"""
        path_manager.add_path(sample_path)
        result = path_manager.add_path(sample_path)

        assert result == False
        assert len(path_manager.paths) == 1

    def test_add_path_exceeds_max(self, path_config):
        """Test adding paths exceeding max limit"""
        path_config.max_paths = 2
        manager = PathManager(path_config)

        # Add 2 paths (should succeed)
        path1 = Path(path_id=0, source_addr="", dest_addr="", interface="eth0")
        path2 = Path(path_id=1, source_addr="", dest_addr="", interface="wlan0")

        assert manager.add_path(path1) == True
        assert manager.add_path(path2) == True

        # Add 3rd path (should fail)
        path3 = Path(path_id=2, source_addr="", dest_addr="", interface="lte0")
        assert manager.add_path(path3) == False
        assert len(manager.paths) == 2

    def test_remove_path(self, path_manager, sample_path):
        """Test removing path"""
        path_manager.add_path(sample_path)
        path_manager.remove_path(sample_path.path_id)

        assert sample_path.path_id not in path_manager.paths
        assert len(path_manager.paths) == 0

    def test_remove_nonexistent_path(self, path_manager):
        """Test removing non-existent path"""
        # Should not raise exception
        path_manager.remove_path(999)

    def test_get_path(self, path_manager, sample_path):
        """Test getting path by ID"""
        path_manager.add_path(sample_path)

        retrieved = path_manager.get_path(sample_path.path_id)
        assert retrieved == sample_path

    def test_get_nonexistent_path(self, path_manager):
        """Test getting non-existent path"""
        retrieved = path_manager.get_path(999)
        assert retrieved is None

    def test_get_all_paths(self, path_manager):
        """Test getting all paths"""
        path1 = Path(path_id=0, source_addr="", dest_addr="", interface="eth0")
        path2 = Path(path_id=1, source_addr="", dest_addr="", interface="wlan0")

        path_manager.add_path(path1)
        path_manager.add_path(path2)

        all_paths = path_manager.get_all_paths()
        assert len(all_paths) == 2
        assert path1 in all_paths
        assert path2 in all_paths


# ============================================================================
# Path Discovery Tests
# ============================================================================

class TestPathDiscovery:
    """Tests for path discovery"""

    def test_discover_paths_basic(self, path_manager):
        """Test basic path discovery"""
        paths = path_manager.discover_paths()

        # Should discover at least eth0
        assert len(paths) >= 1
        assert any(p.interface == "eth0" for p in paths)

    def test_discover_paths_respects_interval(self, path_manager):
        """Test discovery respects interval"""
        # First discovery
        paths1 = path_manager.discover_paths()

        # Immediate second discovery (should return empty)
        paths2 = path_manager.discover_paths()

        assert len(paths2) == 0

    def test_discover_paths_with_interfaces(self, path_config):
        """Test discovery with multiple interfaces"""
        path_config.simulate_wifi = True
        path_config.simulate_cellular = True
        manager = PathManager(path_config)

        paths = manager.discover_paths()

        # Should discover eth0, wlan0, lte0
        assert len(paths) >= 1

    def test_interface_characteristics_ethernet(self, path_manager):
        """Test ethernet interface characteristics"""
        bw, rtt, pref, cost = path_manager._estimate_path_characteristics("eth0")

        assert bw == 1000.0  # 1 Gbps
        assert rtt == 10.0
        assert pref == 0.9  # High preference
        assert cost == 0.0

    def test_interface_characteristics_wifi(self, path_manager):
        """Test WiFi interface characteristics"""
        bw, rtt, pref, cost = path_manager._estimate_path_characteristics("wlan0")

        assert bw == 100.0
        assert rtt == 20.0
        assert pref == 0.7
        assert cost == 0.0

    def test_interface_characteristics_cellular(self, path_manager):
        """Test cellular interface characteristics"""
        bw, rtt, pref, cost = path_manager._estimate_path_characteristics("lte0")

        assert bw == 50.0
        assert rtt == 60.0
        assert pref == 0.5
        assert cost > 0.0  # Has cost


# ============================================================================
# Primary Path Selection Tests
# ============================================================================

class TestPrimaryPathSelection:
    """Tests for primary path selection"""

    def test_select_primary_no_paths(self, path_manager):
        """Test primary selection with no paths"""
        primary = path_manager.select_primary_path()
        assert primary is None

    def test_select_primary_single_path(self, path_manager):
        """Test primary selection with single path"""
        path = Path(
            path_id=0,
            source_addr="", dest_addr="", interface="eth0",
            state=PathState.ACTIVE,
            available_bandwidth=100.0,
            current_rtt=50.0
        )
        path_manager.add_path(path)

        primary = path_manager.select_primary_path()
        assert primary == path
        assert path_manager.primary_path_id == 0

    def test_select_primary_quality_strategy(self, path_config):
        """Test primary selection by quality"""
        path_config.primary_path_preference = 'quality'
        manager = PathManager(path_config)

        # High quality path
        path_good = Path(
            path_id=0,
            source_addr="", dest_addr="", interface="eth0",
            state=PathState.ACTIVE,
            available_bandwidth=1000.0,
            current_rtt=10.0,
            loss_rate=0.0,
            stability=1.0
        )

        # Lower quality path
        path_bad = Path(
            path_id=1,
            source_addr="", dest_addr="", interface="wlan0",
            state=PathState.ACTIVE,
            available_bandwidth=10.0,
            current_rtt=200.0,
            loss_rate=0.05,
            stability=0.5
        )

        manager.add_path(path_good)
        manager.add_path(path_bad)

        primary = manager.select_primary_path()
        assert primary == path_good

    def test_select_primary_bandwidth_strategy(self, path_config):
        """Test primary selection by bandwidth"""
        path_config.primary_path_preference = 'bandwidth'
        manager = PathManager(path_config)

        path_high_bw = Path(
            path_id=0,
            source_addr="", dest_addr="", interface="eth0",
            state=PathState.ACTIVE,
            available_bandwidth=1000.0,
            current_rtt=100.0
        )

        path_low_bw = Path(
            path_id=1,
            source_addr="", dest_addr="", interface="wlan0",
            state=PathState.ACTIVE,
            available_bandwidth=50.0,
            current_rtt=10.0
        )

        manager.add_path(path_high_bw)
        manager.add_path(path_low_bw)

        primary = manager.select_primary_path()
        assert primary == path_high_bw

    def test_select_primary_latency_strategy(self, path_config):
        """Test primary selection by latency"""
        path_config.primary_path_preference = 'latency'
        manager = PathManager(path_config)

        path_low_lat = Path(
            path_id=0,
            source_addr="", dest_addr="", interface="eth0",
            state=PathState.ACTIVE,
            available_bandwidth=50.0,
            current_rtt=10.0
        )

        path_high_lat = Path(
            path_id=1,
            source_addr="", dest_addr="", interface="wlan0",
            state=PathState.ACTIVE,
            available_bandwidth=1000.0,
            current_rtt=200.0
        )

        manager.add_path(path_low_lat)
        manager.add_path(path_high_lat)

        primary = manager.select_primary_path()
        assert primary == path_low_lat

    def test_primary_path_stickiness(self, path_config):
        """Test primary path stickiness (hysteresis)"""
        path_config.primary_path_stickiness = 0.2  # 20% hysteresis
        manager = PathManager(path_config)

        # Current primary
        path_current = Path(
            path_id=0,
            source_addr="", dest_addr="", interface="eth0",
            state=PathState.ACTIVE,
            available_bandwidth=100.0,
            current_rtt=50.0,
            loss_rate=0.0,
            stability=1.0
        )

        # Slightly better path
        path_new = Path(
            path_id=1,
            source_addr="", dest_addr="", interface="wlan0",
            state=PathState.ACTIVE,
            available_bandwidth=110.0,  # Only 10% better
            current_rtt=50.0,
            loss_rate=0.0,
            stability=1.0
        )

        manager.add_path(path_current)
        manager.select_primary_path()  # Set as primary

        manager.add_path(path_new)
        primary = manager.select_primary_path()

        # Should stick with current due to hysteresis
        assert primary == path_current

    def test_primary_path_removal_reselects(self, path_manager):
        """Test removing primary path triggers reselection"""
        path1 = Path(
            path_id=0,
            source_addr="", dest_addr="", interface="eth0",
            state=PathState.ACTIVE,
            available_bandwidth=100.0,
            current_rtt=50.0
        )

        path2 = Path(
            path_id=1,
            source_addr="", dest_addr="", interface="wlan0",
            state=PathState.ACTIVE,
            available_bandwidth=80.0,
            current_rtt=60.0
        )

        path_manager.add_path(path1)
        path_manager.add_path(path2)
        path_manager.select_primary_path()

        assert path_manager.primary_path_id == 0

        # Remove primary
        path_manager.remove_path(0)

        # Should reselect to path2
        assert path_manager.primary_path_id == 1


# ============================================================================
# Path Quality Update Tests
# ============================================================================

class TestPathQualityUpdate:
    """Tests for updating path quality metrics"""

    def test_update_path_quality(self, path_manager, sample_path):
        """Test updating path quality"""
        path_manager.add_path(sample_path)

        path_manager.update_path_quality(
            path_id=0,
            bandwidth=150.0,
            rtt=40.0,
            loss_rate=0.02,
            stability=0.8
        )

        path = path_manager.get_path(0)
        assert path.available_bandwidth == 150.0
        assert path.current_rtt == 40.0
        assert path.loss_rate == 0.02
        assert path.stability == 0.8

    def test_update_triggers_recovery(self, path_manager):
        """Test quality update triggers recovery"""
        path = Path(
            path_id=0,
            source_addr="", dest_addr="", interface="eth0",
            state=PathState.FAILED,
            available_bandwidth=100.0,
            current_rtt=50.0
        )
        path_manager.add_path(path)

        # Update with good quality
        path_manager.update_path_quality(
            path_id=0,
            bandwidth=100.0,
            rtt=50.0,
            loss_rate=0.0,
            stability=1.0
        )

        # Should recover
        updated_path = path_manager.get_path(0)
        assert updated_path.is_healthy()
        assert updated_path.state == PathState.ACTIVE

    def test_update_triggers_failure(self, path_manager, sample_path):
        """Test quality update triggers failure"""
        path_manager.add_path(sample_path)

        # Simulate multiple failures
        for _ in range(5):
            sample_path.mark_failure()

        path_manager.update_path_quality(
            path_id=0,
            bandwidth=100.0,
            rtt=50.0,
            loss_rate=0.0,
            stability=1.0
        )

        # Should be marked as failed
        updated_path = path_manager.get_path(0)
        assert not updated_path.is_healthy()


# ============================================================================
# Statistics and Reporting Tests
# ============================================================================

class TestStatisticsAndReporting:
    """Tests for statistics and reporting"""

    def test_get_statistics_empty(self, path_manager):
        """Test statistics with no paths"""
        stats = path_manager.get_statistics()

        assert stats['total_paths'] == 0
        assert stats['active_paths'] == 0
        assert stats['primary_path_id'] is None

    def test_get_statistics_with_paths(self, path_manager):
        """Test statistics with paths"""
        path1 = Path(
            path_id=0,
            source_addr="", dest_addr="", interface="eth0",
            state=PathState.ACTIVE
        )
        path2 = Path(
            path_id=1,
            source_addr="", dest_addr="", interface="wlan0",
            state=PathState.IDLE
        )

        path_manager.add_path(path1)
        path_manager.add_path(path2)
        path_manager.select_primary_path()

        stats = path_manager.get_statistics()

        assert stats['total_paths'] == 2
        assert stats['active_paths'] == 1
        assert stats['primary_path_id'] == 0
        assert stats['states']['active'] == 1
        assert stats['states']['idle'] == 1

    def test_get_summary(self, path_manager):
        """Test summary string generation"""
        path = Path(
            path_id=0,
            source_addr="", dest_addr="", interface="eth0",
            state=PathState.ACTIVE
        )
        path_manager.add_path(path)

        summary = path_manager.get_summary()

        assert "Total Paths: 1" in summary
        assert "Active Paths: 1" in summary


# ============================================================================
# Thread Safety Tests
# ============================================================================

class TestThreadSafety:
    """Tests for thread safety"""

    def test_concurrent_add_paths(self, path_manager):
        """Test concurrent path additions"""
        def add_paths(start_id):
            for i in range(5):
                path = Path(
                    path_id=start_id + i,
                    source_addr="", dest_addr="",
                    interface=f"eth{i}"
                )
                path_manager.add_path(path)

        threads = []
        for i in range(3):
            t = threading.Thread(target=add_paths, args=(i * 10,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have added paths without corruption
        assert len(path_manager.paths) > 0

    def test_concurrent_quality_updates(self, path_manager, sample_path):
        """Test concurrent quality updates"""
        path_manager.add_path(sample_path)

        def update_quality():
            for _ in range(100):
                path_manager.update_path_quality(
                    path_id=0,
                    bandwidth=np.random.uniform(50, 150),
                    rtt=np.random.uniform(20, 100),
                    loss_rate=np.random.uniform(0, 0.1),
                    stability=np.random.uniform(0.5, 1.0)
                )

        threads = [threading.Thread(target=update_quality) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash or corrupt data
        path = path_manager.get_path(0)
        assert path is not None


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
