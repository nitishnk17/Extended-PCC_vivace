"""
Unit tests for PathMonitor (Extension 4 Phase 2)

Tests real-time metrics collection, EMA smoothing, and path degradation detection.
"""

import pytest
import time
import threading
import numpy as np
from unittest.mock import Mock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.path_monitor import PathMonitor, PathMetrics, PathMonitorConfig


class MockMonitorInterval:
    """Mock MonitorInterval for testing"""

    def __init__(self, sending_rate=10.0, duration=0.1):
        self.sending_rate = sending_rate
        self.is_useful = True

        # Timing
        self.start_time = 0.0
        self.end_time = duration
        self.duration = duration

        # Packet counters
        self.packets_sent = 100
        self.packets_acked = 95
        self.bytes_sent = 150000
        self.bytes_acked = 142500

        # RTT samples (in seconds)
        self.rtt_samples = [0.050, 0.052, 0.048, 0.051, 0.049]  # ~50ms


# ============================================================================
# PathMetrics Tests
# ============================================================================

def test_path_metrics_creation():
    """Test PathMetrics initialization"""
    metrics = PathMetrics(
        path_id=0,
        timestamp=1.0,
        throughput=10.5,
        rtt_avg=50.0,
        loss_rate=0.01
    )

    assert metrics.path_id == 0
    assert metrics.timestamp == 1.0
    assert metrics.throughput == 10.5
    assert metrics.rtt_avg == 50.0
    assert metrics.loss_rate == 0.01


def test_path_metrics_defaults():
    """Test PathMetrics default values"""
    metrics = PathMetrics(path_id=0, timestamp=1.0)

    assert metrics.throughput == 0.0
    assert metrics.goodput == 0.0
    assert metrics.rtt_min == 0.0
    assert metrics.rtt_avg == 0.0
    assert metrics.loss_rate == 0.0
    assert metrics.sending_rate == 0.0
    assert metrics.utilization == 0.0
    assert metrics.rtt_samples == []


def test_path_metrics_repr():
    """Test PathMetrics string representation"""
    metrics = PathMetrics(
        path_id=1,
        timestamp=5.5,
        throughput=20.0,
        rtt_avg=30.0,
        loss_rate=0.02,
        utilization=0.8
    )

    repr_str = repr(metrics)
    assert "path=1" in repr_str
    assert "tput=20.00" in repr_str
    assert "rtt=30.0" in repr_str
    assert "loss=0.020" in repr_str


# ============================================================================
# PathMonitorConfig Tests
# ============================================================================

def test_config_defaults():
    """Test PathMonitorConfig default values"""
    config = PathMonitorConfig()

    assert config.history_window == 100
    assert config.smoothing_factor == 0.2
    assert config.degradation_threshold == 0.3
    assert config.degradation_window == 10
    assert config.enabled is True


def test_config_custom():
    """Test PathMonitorConfig with custom values"""
    config = PathMonitorConfig(
        history_window=50,
        smoothing_factor=0.3,
        degradation_threshold=0.5,
        enabled=False
    )

    assert config.history_window == 50
    assert config.smoothing_factor == 0.3
    assert config.degradation_threshold == 0.5
    assert config.enabled is False


# ============================================================================
# PathMonitor Basic Tests
# ============================================================================

def test_path_monitor_creation():
    """Test PathMonitor initialization"""
    monitor = PathMonitor()

    assert monitor.config.enabled is True
    assert monitor.total_updates == 0
    assert monitor.degradation_events == 0
    assert monitor.get_all_monitored_paths() == []


def test_path_monitor_custom_config():
    """Test PathMonitor with custom config"""
    config = PathMonitorConfig(history_window=50, smoothing_factor=0.5)
    monitor = PathMonitor(config)

    assert monitor.config.history_window == 50
    assert monitor.config.smoothing_factor == 0.5


def test_path_monitor_repr():
    """Test PathMonitor string representation"""
    monitor = PathMonitor()
    repr_str = repr(monitor)

    assert "PathMonitor" in repr_str
    assert "paths=0" in repr_str
    assert "updates=0" in repr_str


# ============================================================================
# Metrics Update Tests
# ============================================================================

def test_update_metrics_basic():
    """Test basic metrics update"""
    monitor = PathMonitor()
    mi = MockMonitorInterval(sending_rate=10.0, duration=0.1)

    monitor.update_metrics(path_id=0, mi=mi)

    assert monitor.total_updates == 1
    assert 0 in monitor.get_all_monitored_paths()

    metrics = monitor.get_path_metrics(0)
    assert metrics is not None
    assert metrics.path_id == 0


def test_update_metrics_throughput():
    """Test throughput calculation from MonitorInterval"""
    monitor = PathMonitor()
    mi = MockMonitorInterval()
    mi.bytes_acked = 1_500_000  # 1.5 MB
    mi.duration = 1.0  # 1 second

    monitor.update_metrics(path_id=0, mi=mi)

    metrics = monitor.get_path_metrics(0, smoothed=False)
    # 1.5 MB * 8 bits/byte / 1_000_000 = 12 Mbps
    assert metrics.throughput == pytest.approx(12.0, rel=0.01)


def test_update_metrics_rtt():
    """Test RTT statistics calculation"""
    monitor = PathMonitor()
    mi = MockMonitorInterval()
    mi.rtt_samples = [0.030, 0.040, 0.050, 0.060, 0.070]  # 30-70ms

    monitor.update_metrics(path_id=0, mi=mi)

    metrics = monitor.get_path_metrics(0, smoothed=False)
    assert metrics.rtt_min == pytest.approx(30.0, rel=0.01)  # ms
    assert metrics.rtt_avg == pytest.approx(50.0, rel=0.01)  # ms
    assert metrics.rtt_95p >= 60.0  # 95th percentile


def test_update_metrics_loss_rate():
    """Test loss rate calculation"""
    monitor = PathMonitor()
    mi = MockMonitorInterval()
    mi.packets_sent = 100
    mi.packets_acked = 90

    monitor.update_metrics(path_id=0, mi=mi)

    metrics = monitor.get_path_metrics(0, smoothed=False)
    assert metrics.packets_sent == 100
    assert metrics.packets_acked == 90
    assert metrics.packets_lost == 10
    assert metrics.loss_rate == pytest.approx(0.1, rel=0.01)


def test_update_metrics_multiple_paths():
    """Test updating metrics for multiple paths"""
    monitor = PathMonitor()
    mi1 = MockMonitorInterval(sending_rate=10.0)
    mi2 = MockMonitorInterval(sending_rate=20.0)

    monitor.update_metrics(path_id=0, mi=mi1)
    monitor.update_metrics(path_id=1, mi=mi2)

    assert len(monitor.get_all_monitored_paths()) == 2
    assert 0 in monitor.get_all_monitored_paths()
    assert 1 in monitor.get_all_monitored_paths()


def test_update_metrics_disabled():
    """Test that updates are ignored when monitoring is disabled"""
    config = PathMonitorConfig(enabled=False)
    monitor = PathMonitor(config)
    mi = MockMonitorInterval()

    monitor.update_metrics(path_id=0, mi=mi)

    assert monitor.total_updates == 0
    assert monitor.get_path_metrics(0) is None


# ============================================================================
# EMA Smoothing Tests
# ============================================================================

def test_ema_smoothing():
    """Test exponential moving average smoothing"""
    config = PathMonitorConfig(smoothing_factor=0.5)  # 50% weight to new
    monitor = PathMonitor(config)

    # First update - should use raw value
    mi1 = MockMonitorInterval()
    mi1.bytes_acked = 1_000_000  # 10 Mbps for 1 second
    mi1.duration = 1.0
    monitor.update_metrics(path_id=0, mi=mi1)

    metrics1 = monitor.get_path_metrics(0, smoothed=True)
    assert metrics1.throughput == pytest.approx(8.0, rel=0.01)

    # Second update - should apply EMA
    mi2 = MockMonitorInterval()
    mi2.bytes_acked = 2_000_000  # 16 Mbps for 1 second
    mi2.duration = 1.0
    mi2.end_time = 1.0
    monitor.update_metrics(path_id=0, mi=mi2)

    metrics2 = monitor.get_path_metrics(0, smoothed=True)
    # EMA = 0.5 * 16 + 0.5 * 8 = 12
    expected = 0.5 * 16.0 + 0.5 * 8.0
    assert metrics2.throughput == pytest.approx(expected, rel=0.05)


def test_ema_multiple_updates():
    """Test EMA with multiple updates"""
    config = PathMonitorConfig(smoothing_factor=0.2)
    monitor = PathMonitor(config)

    throughputs = [10.0, 20.0, 15.0, 25.0]

    for i, tput in enumerate(throughputs):
        mi = MockMonitorInterval()
        mi.bytes_acked = int(tput * 1_000_000 / 8)  # Convert Mbps to bytes
        mi.duration = 1.0
        mi.end_time = float(i)
        monitor.update_metrics(path_id=0, mi=mi)

    metrics = monitor.get_path_metrics(0, smoothed=True)
    # Should be somewhere between min and max, closer to recent values
    assert 10.0 <= metrics.throughput <= 25.0


def test_smoothed_vs_raw_metrics():
    """Test difference between smoothed and raw metrics"""
    config = PathMonitorConfig(smoothing_factor=0.1)  # Heavy smoothing
    monitor = PathMonitor(config)

    # Stable throughput, then spike
    for i in range(10):
        mi = MockMonitorInterval()
        mi.bytes_acked = 1_000_000  # 8 Mbps
        mi.duration = 1.0
        mi.end_time = float(i)
        monitor.update_metrics(path_id=0, mi=mi)

    # Now spike
    mi_spike = MockMonitorInterval()
    mi_spike.bytes_acked = 5_000_000  # 40 Mbps
    mi_spike.duration = 1.0
    mi_spike.end_time = 10.0
    monitor.update_metrics(path_id=0, mi=mi_spike)

    raw = monitor.get_path_metrics(0, smoothed=False)
    smoothed = monitor.get_path_metrics(0, smoothed=True)

    # Raw should show spike, smoothed should be more stable
    assert raw.throughput > smoothed.throughput


# ============================================================================
# History Management Tests
# ============================================================================

def test_history_accumulation():
    """Test that history accumulates properly"""
    monitor = PathMonitor()

    for i in range(20):
        mi = MockMonitorInterval()
        mi.end_time = float(i)
        monitor.update_metrics(path_id=0, mi=mi)

    history = monitor.get_path_history(0)
    assert len(history) == 20


def test_history_window_limit():
    """Test that history is limited to window size"""
    config = PathMonitorConfig(history_window=10)
    monitor = PathMonitor(config)

    for i in range(50):
        mi = MockMonitorInterval()
        mi.end_time = float(i)
        monitor.update_metrics(path_id=0, mi=mi)

    history = monitor.get_path_history(0)
    assert len(history) == 10


def test_history_time_filtering():
    """Test filtering history by time duration"""
    monitor = PathMonitor()

    for i in range(20):
        mi = MockMonitorInterval()
        mi.end_time = float(i)
        monitor.update_metrics(path_id=0, mi=mi)

    # Get last 5 seconds
    recent = monitor.get_path_history(0, duration=5.0)

    # Should have metrics from time 14 to 19 (last 5 seconds)
    assert len(recent) >= 5
    assert all(m.timestamp >= 14.0 for m in recent)


def test_history_empty():
    """Test getting history for non-existent path"""
    monitor = PathMonitor()

    history = monitor.get_path_history(999)
    assert history == []


# ============================================================================
# Path Degradation Tests
# ============================================================================

def test_degradation_detection_stable():
    """Test that stable path is not marked as degraded"""
    config = PathMonitorConfig(degradation_window=5, degradation_threshold=0.3)
    monitor = PathMonitor(config)

    # Stable throughput
    for i in range(20):
        mi = MockMonitorInterval()
        mi.bytes_acked = 1_000_000  # ~8 Mbps
        mi.duration = 1.0
        mi.end_time = float(i)
        monitor.update_metrics(path_id=0, mi=mi)

    assert not monitor.is_path_degraded(0)


def test_degradation_detection_significant_drop():
    """Test degradation detection on significant throughput drop"""
    config = PathMonitorConfig(degradation_window=5, degradation_threshold=0.3)
    monitor = PathMonitor(config)

    # Good throughput
    for i in range(15):
        mi = MockMonitorInterval()
        mi.bytes_acked = 2_000_000  # ~16 Mbps
        mi.duration = 1.0
        mi.end_time = float(i)
        monitor.update_metrics(path_id=0, mi=mi)

    # Now drop by 50% (exceeds 30% threshold)
    for i in range(15, 25):
        mi = MockMonitorInterval()
        mi.bytes_acked = 1_000_000  # ~8 Mbps (50% drop)
        mi.duration = 1.0
        mi.end_time = float(i)
        monitor.update_metrics(path_id=0, mi=mi)

    # Should detect degradation
    assert monitor.is_path_degraded(0)
    assert monitor.degradation_events > 0


def test_degradation_not_triggered_on_small_drop():
    """Test that small drops don't trigger degradation"""
    config = PathMonitorConfig(degradation_window=5, degradation_threshold=0.3)
    monitor = PathMonitor(config)

    # Good throughput
    for i in range(15):
        mi = MockMonitorInterval()
        mi.bytes_acked = 2_000_000  # ~16 Mbps
        mi.duration = 1.0
        mi.end_time = float(i)
        monitor.update_metrics(path_id=0, mi=mi)

    # Small drop (20%, below 30% threshold)
    for i in range(15, 25):
        mi = MockMonitorInterval()
        mi.bytes_acked = 1_600_000  # ~12.8 Mbps (20% drop)
        mi.duration = 1.0
        mi.end_time = float(i)
        monitor.update_metrics(path_id=0, mi=mi)

    # Should not detect degradation
    assert not monitor.is_path_degraded(0)


def test_degradation_insufficient_history():
    """Test that degradation is not detected with insufficient history"""
    config = PathMonitorConfig(degradation_window=10)
    monitor = PathMonitor(config)

    # Only a few samples
    for i in range(5):
        mi = MockMonitorInterval()
        mi.bytes_acked = 100  # Very low throughput
        mi.duration = 1.0
        mi.end_time = float(i)
        monitor.update_metrics(path_id=0, mi=mi)

    # Should not detect degradation (not enough history)
    assert not monitor.is_path_degraded(0)


def test_degradation_nonexistent_path():
    """Test degradation check for non-existent path"""
    monitor = PathMonitor()

    assert not monitor.is_path_degraded(999)


# ============================================================================
# Utilization Tests
# ============================================================================

def test_set_path_utilization():
    """Test setting path utilization based on estimated bandwidth"""
    monitor = PathMonitor()

    mi = MockMonitorInterval()
    mi.sending_rate = 10.0  # Mbps
    monitor.update_metrics(path_id=0, mi=mi)

    # Set estimated bandwidth
    monitor.set_path_utilization(path_id=0, estimated_bandwidth=20.0)

    metrics = monitor.get_path_metrics(0, smoothed=True)
    # Utilization = 10 / 20 = 0.5
    assert metrics.utilization == pytest.approx(0.5, rel=0.01)


def test_set_utilization_zero_bandwidth():
    """Test utilization handling with zero bandwidth"""
    monitor = PathMonitor()

    mi = MockMonitorInterval()
    mi.sending_rate = 10.0
    monitor.update_metrics(path_id=0, mi=mi)

    monitor.set_path_utilization(path_id=0, estimated_bandwidth=0.0)

    # Should not crash, utilization remains unchanged
    metrics = monitor.get_path_metrics(0, smoothed=True)
    assert metrics is not None


# ============================================================================
# Statistics Tests
# ============================================================================

def test_statistics():
    """Test getting monitoring statistics"""
    monitor = PathMonitor()

    for i in range(10):
        mi = MockMonitorInterval()
        monitor.update_metrics(path_id=0, mi=mi)

    for i in range(5):
        mi = MockMonitorInterval()
        monitor.update_metrics(path_id=1, mi=mi)

    stats = monitor.get_statistics()

    assert stats['total_updates'] == 15
    assert stats['monitored_paths'] == 2
    assert stats['total_samples'] == 15


def test_clear_history_single_path():
    """Test clearing history for a single path"""
    monitor = PathMonitor()

    for i in range(10):
        mi = MockMonitorInterval()
        monitor.update_metrics(path_id=0, mi=mi)
        monitor.update_metrics(path_id=1, mi=mi)

    monitor.clear_history(path_id=0)

    assert monitor.get_path_metrics(0) is None
    assert monitor.get_path_metrics(1) is not None
    assert len(monitor.get_all_monitored_paths()) == 1


def test_clear_history_all():
    """Test clearing all history"""
    monitor = PathMonitor()

    for i in range(10):
        mi = MockMonitorInterval()
        monitor.update_metrics(path_id=0, mi=mi)
        monitor.update_metrics(path_id=1, mi=mi)

    monitor.clear_history()

    assert len(monitor.get_all_monitored_paths()) == 0
    assert monitor.total_updates == 0
    assert monitor.degradation_events == 0


# ============================================================================
# Thread Safety Tests
# ============================================================================

def test_concurrent_updates():
    """Test concurrent metric updates from multiple threads"""
    monitor = PathMonitor()
    num_threads = 10
    updates_per_thread = 20

    def update_worker(path_id):
        for i in range(updates_per_thread):
            mi = MockMonitorInterval()
            mi.end_time = float(i)
            monitor.update_metrics(path_id=path_id, mi=mi)

    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=update_worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # All updates should be recorded
    assert monitor.total_updates == num_threads * updates_per_thread
    assert len(monitor.get_all_monitored_paths()) == num_threads


def test_concurrent_reads_and_writes():
    """Test concurrent reads and writes"""
    monitor = PathMonitor()
    num_updates = 100
    errors = []

    def writer():
        try:
            for i in range(num_updates):
                mi = MockMonitorInterval()
                mi.end_time = float(i)
                monitor.update_metrics(path_id=0, mi=mi)
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            for _ in range(num_updates * 2):
                monitor.get_path_metrics(0)
                monitor.get_path_history(0)
                monitor.is_path_degraded(0)
                time.sleep(0.0005)
        except Exception as e:
            errors.append(e)

    writer_thread = threading.Thread(target=writer)
    reader_threads = [threading.Thread(target=reader) for _ in range(3)]

    writer_thread.start()
    for t in reader_threads:
        t.start()

    writer_thread.join()
    for t in reader_threads:
        t.join()

    # No errors should occur
    assert len(errors) == 0


# ============================================================================
# Edge Cases
# ============================================================================

def test_zero_duration_mi():
    """Test handling MonitorInterval with zero duration"""
    monitor = PathMonitor()

    mi = MockMonitorInterval()
    mi.duration = 0.0

    # Should not crash, uses default duration of 1.0
    monitor.update_metrics(path_id=0, mi=mi)

    metrics = monitor.get_path_metrics(0)
    assert metrics is not None


def test_empty_rtt_samples():
    """Test handling MonitorInterval with no RTT samples"""
    monitor = PathMonitor()

    mi = MockMonitorInterval()
    mi.rtt_samples = []

    monitor.update_metrics(path_id=0, mi=mi)

    metrics = monitor.get_path_metrics(0, smoothed=False)
    assert metrics.rtt_min == 0.0
    assert metrics.rtt_avg == 0.0
    assert metrics.rtt_95p == 0.0


def test_zero_packets_sent():
    """Test handling MonitorInterval with zero packets"""
    monitor = PathMonitor()

    mi = MockMonitorInterval()
    mi.packets_sent = 0
    mi.packets_acked = 0

    monitor.update_metrics(path_id=0, mi=mi)

    metrics = monitor.get_path_metrics(0, smoothed=False)
    assert metrics.loss_rate == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
