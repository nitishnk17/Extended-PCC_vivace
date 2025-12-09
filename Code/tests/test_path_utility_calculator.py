"""
Unit tests for PathUtilityCalculator (Extension 4 Phase 3)

Tests per-path utility computation including throughput, latency, loss, and stability.
"""

import pytest
import threading
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.path_utility_calculator import PathUtilityCalculator, PathUtilityConfig
from src.path_monitor import PathMonitor, PathMetrics, PathMonitorConfig
from src.path_manager import PathManager, PathManagerConfig, Path, PathState


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def path_manager():
    """Create PathManager with test paths"""
    manager = PathManager(PathManagerConfig())

    # Add test paths
    path1 = Path(
        path_id=0,
        source_addr="192.168.1.1",
        dest_addr="10.0.0.1",
        interface="eth0",
        state=PathState.ACTIVE,
        estimated_bandwidth=100.0,
        baseline_rtt=50.0
    )

    path2 = Path(
        path_id=1,
        source_addr="192.168.1.2",
        dest_addr="10.0.0.2",
        interface="wlan0",
        state=PathState.ACTIVE,
        estimated_bandwidth=50.0,
        baseline_rtt=70.0
    )

    manager.paths[0] = path1
    manager.paths[1] = path2
    manager.primary_path_id = 0

    return manager


@pytest.fixture
def path_monitor():
    """Create PathMonitor with test metrics"""
    monitor = PathMonitor(PathMonitorConfig())
    return monitor


@pytest.fixture
def utility_calculator(path_monitor, path_manager):
    """Create PathUtilityCalculator"""
    return PathUtilityCalculator(path_monitor, path_manager)


def create_test_metrics(path_id=0, throughput=10.0, rtt_avg=50.0, loss_rate=0.01):
    """Helper to create test PathMetrics"""
    return PathMetrics(
        path_id=path_id,
        timestamp=1.0,
        throughput=throughput,
        goodput=throughput * 0.99,
        rtt_min=rtt_avg * 0.9,
        rtt_avg=rtt_avg,
        rtt_95p=rtt_avg * 1.1,
        loss_rate=loss_rate,
        sending_rate=throughput,
        utilization=0.8
    )


# ============================================================================
# PathUtilityConfig Tests
# ============================================================================

def test_config_defaults():
    """Test PathUtilityConfig default values"""
    config = PathUtilityConfig()

    assert config.throughput_weight == 1.0
    assert config.throughput_exponent == 0.9
    assert config.latency_weight == 900.0
    assert config.loss_weight == 11.35
    assert config.stability_weight == 0.5
    assert config.latency_threshold == 100.0
    assert config.latency_scale == 10.0
    assert config.primary_path_bonus == 0.1
    assert config.cellular_penalty == 0.05
    assert config.enabled is True


def test_config_custom():
    """Test PathUtilityConfig with custom values"""
    config = PathUtilityConfig(
        throughput_weight=2.0,
        loss_weight=20.0,
        primary_path_bonus=0.2,
        enabled=False
    )

    assert config.throughput_weight == 2.0
    assert config.loss_weight == 20.0
    assert config.primary_path_bonus == 0.2
    assert config.enabled is False


# ============================================================================
# PathUtilityCalculator Basic Tests
# ============================================================================

def test_calculator_creation(utility_calculator):
    """Test PathUtilityCalculator initialization"""
    assert utility_calculator.config.enabled is True
    assert utility_calculator.total_calculations == 0


def test_calculator_disabled():
    """Test that disabled calculator returns 0"""
    manager = PathManager(PathManagerConfig())
    monitor = PathMonitor()
    config = PathUtilityConfig(enabled=False)

    calc = PathUtilityCalculator(monitor, manager, config)

    metrics = create_test_metrics()
    utility = calc.calculate_path_utility(0, metrics)

    assert utility == 0.0


def test_calculator_repr(utility_calculator):
    """Test string representation"""
    repr_str = repr(utility_calculator)

    assert "PathUtilityCalculator" in repr_str
    assert "calcs=0" in repr_str


# ============================================================================
# Sigmoid Latency Penalty Tests
# ============================================================================

def test_sigmoid_low_latency(utility_calculator):
    """Test sigmoid penalty for low latency"""
    # 30ms << 100ms threshold
    penalty = utility_calculator._sigmoid_latency_penalty(30.0)

    # Should be close to 1.0 (minimal penalty)
    assert penalty > 0.95


def test_sigmoid_threshold_latency(utility_calculator):
    """Test sigmoid penalty at threshold"""
    # At threshold (100ms)
    penalty = utility_calculator._sigmoid_latency_penalty(100.0)

    # Should be 0.5 at threshold
    assert penalty == pytest.approx(0.5, abs=0.01)


def test_sigmoid_high_latency(utility_calculator):
    """Test sigmoid penalty for high latency"""
    # 200ms >> 100ms threshold
    penalty = utility_calculator._sigmoid_latency_penalty(200.0)

    # Should be low (high penalty)
    assert penalty < 0.1


def test_sigmoid_zero_latency(utility_calculator):
    """Test sigmoid penalty for zero latency"""
    penalty = utility_calculator._sigmoid_latency_penalty(0.0)

    # Zero latency should return 1.0
    assert penalty == 1.0


# ============================================================================
# Stability Bonus Tests
# ============================================================================

def test_stability_insufficient_history(utility_calculator, path_monitor):
    """Test stability bonus with insufficient history"""
    bonus = utility_calculator._calculate_stability_bonus(0)

    # No history should return 0
    assert bonus == 0.0


def test_stability_stable_throughput(utility_calculator, path_monitor):
    """Test stability bonus for stable throughput"""
    # Add stable throughput history
    for i in range(10):
        metrics = create_test_metrics(path_id=0, throughput=10.0)
        metrics.timestamp = float(i)
        path_monitor._metrics_history[0] = path_monitor._metrics_history.get(0, [])
        path_monitor._metrics_history[0].append(metrics)

    bonus = utility_calculator._calculate_stability_bonus(0)

    # Stable throughput should have low (close to 0) variance
    # Bonus is -Var(T), so should be close to 0
    assert abs(bonus) < 1.0


def test_stability_variable_throughput(utility_calculator, path_monitor):
    """Test stability bonus for variable throughput"""
    # Add variable throughput history
    throughputs = [5.0, 15.0, 8.0, 12.0, 6.0, 14.0, 7.0, 13.0, 9.0, 11.0]
    for i, tput in enumerate(throughputs):
        metrics = create_test_metrics(path_id=0, throughput=tput)
        metrics.timestamp = float(i)
        path_monitor._metrics_history[0] = path_monitor._metrics_history.get(0, [])
        path_monitor._metrics_history[0].append(metrics)

    bonus = utility_calculator._calculate_stability_bonus(0)

    # High variance should result in large negative bonus
    assert bonus < -5.0


# ============================================================================
# Path Adjustments Tests
# ============================================================================

def test_primary_path_bonus(utility_calculator, path_manager):
    """Test primary path receives bonus"""
    # Path 0 is primary (set in fixture)
    path = path_manager.get_path(0)
    base_utility = 10.0

    adjusted = utility_calculator._apply_path_adjustments(path, base_utility)

    # Should receive 10% bonus
    expected = base_utility * 1.1
    assert adjusted == pytest.approx(expected, rel=0.01)


def test_non_primary_path_no_bonus(utility_calculator, path_manager):
    """Test non-primary path receives no bonus"""
    # Path 1 is not primary
    path = path_manager.get_path(1)
    base_utility = 10.0

    adjusted = utility_calculator._apply_path_adjustments(path, base_utility)

    # Should have no primary bonus (but no penalty either)
    assert adjusted == pytest.approx(base_utility, rel=0.01)


def test_cellular_path_penalty():
    """Test cellular path receives penalty"""
    manager = PathManager(PathManagerConfig())
    monitor = PathMonitor()
    calc = PathUtilityCalculator(monitor, manager)

    # Create cellular path
    cellular_path = Path(
        path_id=0,
        source_addr="10.0.0.1",
        dest_addr="10.0.0.2",
        interface="cellular0",  # Contains 'cellular'
        state=PathState.ACTIVE,
        estimated_bandwidth=20.0,
        baseline_rtt=100.0
    )

    manager.paths[0] = cellular_path

    base_utility = 10.0
    adjusted = calc._apply_path_adjustments(cellular_path, base_utility)

    # Should receive 5% penalty
    expected = base_utility * 0.95
    assert adjusted == pytest.approx(expected, rel=0.01)


# ============================================================================
# Single Path Utility Calculation Tests
# ============================================================================

def test_calculate_path_utility_basic(utility_calculator, path_monitor, path_manager):
    """Test basic utility calculation"""
    # Add metrics
    metrics = create_test_metrics(path_id=0, throughput=10.0, rtt_avg=50.0, loss_rate=0.01)
    path_monitor._metrics_history[0] = [metrics]
    path_monitor._smoothed_metrics[0] = metrics

    utility = utility_calculator.calculate_path_utility(0)

    # Utility should be positive for good path
    assert utility > 0


def test_calculate_path_utility_high_throughput(utility_calculator, path_monitor, path_manager):
    """Test utility increases with throughput"""
    # Low throughput
    metrics_low = create_test_metrics(path_id=0, throughput=5.0, rtt_avg=50.0, loss_rate=0.01)
    path_monitor._smoothed_metrics[0] = metrics_low
    utility_low = utility_calculator.calculate_path_utility(0)

    # High throughput
    metrics_high = create_test_metrics(path_id=0, throughput=20.0, rtt_avg=50.0, loss_rate=0.01)
    path_monitor._smoothed_metrics[0] = metrics_high
    utility_calculator.clear_cache()
    utility_high = utility_calculator.calculate_path_utility(0)

    # Higher throughput should give higher utility
    assert utility_high > utility_low


def test_calculate_path_utility_high_latency(utility_calculator, path_monitor, path_manager):
    """Test utility decreases with latency"""
    # Low latency
    metrics_low = create_test_metrics(path_id=0, throughput=10.0, rtt_avg=30.0, loss_rate=0.01)
    path_monitor._smoothed_metrics[0] = metrics_low
    utility_low = utility_calculator.calculate_path_utility(0)

    # High latency
    metrics_high = create_test_metrics(path_id=0, throughput=10.0, rtt_avg=200.0, loss_rate=0.01)
    path_monitor._smoothed_metrics[0] = metrics_high
    utility_calculator.clear_cache()
    utility_high = utility_calculator.calculate_path_utility(0)

    # Higher latency should give lower utility
    assert utility_high < utility_low


def test_calculate_path_utility_high_loss(utility_calculator, path_monitor, path_manager):
    """Test utility decreases with loss rate"""
    # Low loss
    metrics_low = create_test_metrics(path_id=0, throughput=10.0, rtt_avg=50.0, loss_rate=0.01)
    path_monitor._smoothed_metrics[0] = metrics_low
    utility_low = utility_calculator.calculate_path_utility(0)

    # High loss
    metrics_high = create_test_metrics(path_id=0, throughput=10.0, rtt_avg=50.0, loss_rate=0.10)
    path_monitor._smoothed_metrics[0] = metrics_high
    utility_calculator.clear_cache()
    utility_high = utility_calculator.calculate_path_utility(0)

    # Higher loss should give lower utility
    assert utility_high < utility_low


def test_calculate_path_utility_nonexistent_path(utility_calculator):
    """Test utility calculation for non-existent path"""
    utility = utility_calculator.calculate_path_utility(999)

    assert utility == 0.0


# ============================================================================
# All Paths Utility Tests
# ============================================================================

def test_calculate_all_utilities(utility_calculator, path_monitor, path_manager):
    """Test calculating utilities for all paths"""
    # Add metrics for both paths
    metrics0 = create_test_metrics(path_id=0, throughput=10.0)
    metrics1 = create_test_metrics(path_id=1, throughput=8.0)

    path_monitor._metrics_history[0] = [metrics0]
    path_monitor._metrics_history[1] = [metrics1]
    path_monitor._smoothed_metrics[0] = metrics0
    path_monitor._smoothed_metrics[1] = metrics1

    utilities = utility_calculator.calculate_all_utilities()

    assert len(utilities) == 2
    assert 0 in utilities
    assert 1 in utilities


def test_calculate_all_utilities_empty(utility_calculator):
    """Test calculating utilities with no paths"""
    utilities = utility_calculator.calculate_all_utilities()

    assert utilities == {}


# ============================================================================
# Normalized Utilities Tests
# ============================================================================

def test_normalized_utilities(utility_calculator):
    """Test utility normalization"""
    # Mock utilities
    raw_utilities = {0: 10.0, 1: 5.0, 2: 15.0}

    normalized = utility_calculator.get_normalized_utilities(raw_utilities)

    # Check range [0, 1]
    assert all(0.0 <= v <= 1.0 for v in normalized.values())

    # Check min and max
    assert normalized[1] == 0.0  # min utility
    assert normalized[2] == 1.0  # max utility
    assert 0.0 < normalized[0] < 1.0


def test_normalized_utilities_single_path(utility_calculator):
    """Test normalization with single path"""
    raw_utilities = {0: 10.0}

    normalized = utility_calculator.get_normalized_utilities(raw_utilities)

    # Single path should get 1.0
    assert normalized[0] == 1.0


def test_normalized_utilities_equal(utility_calculator):
    """Test normalization with equal utilities"""
    raw_utilities = {0: 10.0, 1: 10.0, 2: 10.0}

    normalized = utility_calculator.get_normalized_utilities(raw_utilities)

    # All equal should get 0.5
    assert all(v == 0.5 for v in normalized.values())


def test_normalized_utilities_empty(utility_calculator):
    """Test normalization with empty utilities"""
    normalized = utility_calculator.get_normalized_utilities({})

    assert normalized == {}


# ============================================================================
# Utility Breakdown Tests
# ============================================================================

def test_utility_breakdown(utility_calculator, path_monitor, path_manager):
    """Test detailed utility breakdown"""
    metrics = create_test_metrics(path_id=0, throughput=10.0, rtt_avg=50.0, loss_rate=0.01)
    path_monitor._smoothed_metrics[0] = metrics
    path_monitor._metrics_history[0] = [metrics]

    breakdown = utility_calculator.get_utility_breakdown(0)

    assert 'throughput_mbps' in breakdown
    assert 'latency_ms' in breakdown
    assert 'loss_rate' in breakdown
    assert 'throughput_term' in breakdown
    assert 'latency_penalty' in breakdown
    assert 'loss_penalty' in breakdown
    assert 'stability_bonus' in breakdown
    assert 'base_utility' in breakdown
    assert 'final_utility' in breakdown


def test_utility_breakdown_nonexistent(utility_calculator):
    """Test breakdown for non-existent path"""
    breakdown = utility_calculator.get_utility_breakdown(999)

    assert breakdown == {}


# ============================================================================
# Caching Tests
# ============================================================================

def test_utility_caching(utility_calculator, path_monitor, path_manager):
    """Test utility caching"""
    metrics = create_test_metrics(path_id=0)
    path_monitor._smoothed_metrics[0] = metrics

    # First call - calculates
    utility1 = utility_calculator.calculate_path_utility(0, use_cache=False)
    calcs1 = utility_calculator.total_calculations

    # Second call with cache - should use cached value
    utility2 = utility_calculator.calculate_path_utility(0, use_cache=True)
    calcs2 = utility_calculator.total_calculations

    assert utility1 == utility2
    assert calcs2 == calcs1  # No new calculation


def test_clear_cache(utility_calculator, path_monitor, path_manager):
    """Test clearing utility cache"""
    metrics = create_test_metrics(path_id=0)
    path_monitor._smoothed_metrics[0] = metrics

    # Calculate and cache
    utility_calculator.calculate_path_utility(0, use_cache=False)

    assert len(utility_calculator._utility_cache) > 0

    # Clear cache
    utility_calculator.clear_cache()

    assert len(utility_calculator._utility_cache) == 0


# ============================================================================
# Statistics Tests
# ============================================================================

def test_statistics(utility_calculator, path_monitor, path_manager):
    """Test getting statistics"""
    metrics = create_test_metrics(path_id=0)
    path_monitor._smoothed_metrics[0] = metrics

    # Perform some calculations
    utility_calculator.calculate_path_utility(0)
    utility_calculator.calculate_path_utility(0)

    stats = utility_calculator.get_statistics()

    assert stats['total_calculations'] == 2
    assert stats['cached_utilities'] >= 1


# ============================================================================
# Thread Safety Tests
# ============================================================================

def test_concurrent_calculations(utility_calculator, path_monitor, path_manager):
    """Test concurrent utility calculations"""
    # Add metrics
    for i in range(5):
        metrics = create_test_metrics(path_id=i, throughput=10.0 + i)
        path_monitor._smoothed_metrics[i] = metrics
        path_manager.paths[i] = Path(
            path_id=i,
            source_addr=f"192.168.1.{i}",
            dest_addr="10.0.0.1",
            interface=f"eth{i}",
            state=PathState.ACTIVE,
            estimated_bandwidth=100.0,
            baseline_rtt=50.0
        )

    errors = []

    def calculate_worker(path_id):
        try:
            for _ in range(20):
                utility_calculator.calculate_path_utility(path_id)
        except Exception as e:
            errors.append(e)

    threads = []
    for i in range(5):
        t = threading.Thread(target=calculate_worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # No errors should occur
    assert len(errors) == 0


# ============================================================================
# Edge Cases
# ============================================================================

def test_zero_throughput(utility_calculator, path_monitor, path_manager):
    """Test utility with zero throughput"""
    metrics = create_test_metrics(path_id=0, throughput=0.0)
    path_monitor._smoothed_metrics[0] = metrics

    utility = utility_calculator.calculate_path_utility(0)

    # Zero throughput should give low utility
    assert utility <= 0


def test_negative_utility(utility_calculator, path_monitor, path_manager):
    """Test that utility can be negative"""
    # High loss, low throughput, high latency
    metrics = create_test_metrics(path_id=0, throughput=1.0, rtt_avg=300.0, loss_rate=0.5)
    path_monitor._smoothed_metrics[0] = metrics

    utility = utility_calculator.calculate_path_utility(0)

    # Bad path should have negative utility
    assert utility < 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
