"""
Comprehensive unit tests for TrafficClassifier

Tests cover:
1. Basic functionality
2. Feature extraction
3. Classification accuracy
4. Edge cases
5. Performance
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.traffic_classifier import TrafficClassifier
from src.config import ClassifierConfig


class TestTrafficClassifierBasics:
    """Test basic traffic classifier functionality"""

    @pytest.fixture
    def config(self):
        """Create classifier configuration"""
        return ClassifierConfig(
            window_size=50,
            confidence_threshold=0.7,
            feature_set=['packet_size', 'inter_arrival', 'entropy', 'burst_ratio'],
            enabled=True
        )

    @pytest.fixture
    def classifier(self, config):
        """Create classifier instance"""
        return TrafficClassifier(config)

    def test_initialization(self, classifier, config):
        """Test classifier initialization"""
        assert classifier.window_size == config.window_size
        assert classifier.confidence_threshold == config.confidence_threshold
        assert classifier.enabled == config.enabled
        assert len(classifier.packet_sizes) == 0
        assert len(classifier.inter_arrivals) == 0

    def test_add_packet(self, classifier):
        """Test adding packets"""
        classifier.add_packet(1500, 0.001)
        assert len(classifier.packet_sizes) == 1
        assert len(classifier.timestamps) == 1

        classifier.add_packet(1400, 0.002)
        assert len(classifier.packet_sizes) == 2
        assert len(classifier.inter_arrivals) == 1
        assert classifier.inter_arrivals[0] == pytest.approx(0.001)

    def test_window_size_limit(self, classifier):
        """Test that packet history is limited to window_size"""
        window_size = classifier.window_size

        # Add more packets than window_size
        for i in range(window_size + 20):
            classifier.add_packet(1400, i * 0.001)

        assert len(classifier.packet_sizes) == window_size
        assert len(classifier.timestamps) == window_size
        assert len(classifier.inter_arrivals) <= window_size

    def test_reset(self, classifier):
        """Test reset functionality"""
        # Add some packets
        for i in range(10):
            classifier.add_packet(1400, i * 0.001)

        assert len(classifier.packet_sizes) > 0

        # Reset
        classifier.reset()

        assert len(classifier.packet_sizes) == 0
        assert len(classifier.inter_arrivals) == 0
        assert len(classifier.timestamps) == 0


class TestFeatureExtraction:
    """Test feature extraction"""

    @pytest.fixture
    def classifier(self):
        config = ClassifierConfig()
        return TrafficClassifier(config)

    def test_insufficient_packets(self, classifier):
        """Test that features are empty with too few packets"""
        # Add only 5 packets (need at least 10)
        for i in range(5):
            classifier.add_packet(1400, i * 0.001)

        features = classifier.extract_features()
        assert features == {}

    def test_packet_size_features(self, classifier):
        """Test packet size feature extraction"""
        # Add packets with known sizes (need at least 20 packets)
        sizes = [1400, 1500, 1450, 1480, 1420, 1460, 1440, 1490, 1470, 1430,
                1410, 1450, 1460, 1470, 1480, 1400, 1500, 1450, 1480, 1420]

        for i, size in enumerate(sizes):
            classifier.add_packet(size, i * 0.001)

        features = classifier.extract_features()

        assert 'avg_packet_size' in features
        assert 'std_packet_size' in features
        assert 'median_packet_size' in features
        assert 'max_packet_size' in features
        assert 'min_packet_size' in features

        assert features['avg_packet_size'] == pytest.approx(np.mean(sizes), abs=1)
        assert features['median_packet_size'] == pytest.approx(np.median(sizes), abs=1)
        assert features['max_packet_size'] == max(sizes)
        assert features['min_packet_size'] == min(sizes)

    def test_inter_arrival_features(self, classifier):
        """Test inter-arrival time feature extraction"""
        # Add packets with regular intervals
        for i in range(20):
            classifier.add_packet(1400, i * 0.01)  # 10ms apart

        features = classifier.extract_features()

        assert 'avg_inter_arrival' in features
        assert 'std_inter_arrival' in features
        assert 'cv_inter_arrival' in features

        assert features['avg_inter_arrival'] == pytest.approx(0.01, abs=0.001)
        # Regular intervals should have low CV
        assert features['cv_inter_arrival'] < 0.1

    def test_entropy_calculation(self, classifier):
        """Test entropy feature calculation"""
        # Uniform packet sizes -> low entropy
        for i in range(20):
            classifier.add_packet(1400, i * 0.001)

        features = classifier.extract_features()
        assert 'packet_size_entropy' in features
        low_entropy = features['packet_size_entropy']

        # Reset and add variable sizes -> higher entropy
        classifier.reset()
        sizes = [100, 500, 1000, 1500, 200, 600, 1100, 1400,
                150, 550, 1050, 1450, 250, 650, 1150, 1350,
                300, 700, 1200, 1300]
        for i, size in enumerate(sizes):
            classifier.add_packet(size, i * 0.001)

        features = classifier.extract_features()
        high_entropy = features['packet_size_entropy']

        assert high_entropy > low_entropy

    def test_burst_detection(self, classifier):
        """Test burst ratio calculation"""
        # Regular intervals -> low burst ratio
        for i in range(20):
            classifier.add_packet(1400, i * 0.01)

        features = classifier.extract_features()
        assert 'burst_ratio' in features
        low_burst = features['burst_ratio']

        # Reset and add bursty pattern -> high burst ratio
        classifier.reset()
        for i in range(10):
            # Burst: 5 packets close together
            for j in range(5):
                classifier.add_packet(1400, i * 0.1 + j * 0.001)
            # Gap
            classifier.add_packet(1400, i * 0.1 + 0.05)

        features = classifier.extract_features()
        high_burst = features['burst_ratio']

        assert high_burst > low_burst


class TestTrafficClassification:
    """Test traffic type classification"""

    @pytest.fixture
    def classifier(self):
        config = ClassifierConfig(
            window_size=50,
            confidence_threshold=0.6,
            enabled=True
        )
        return TrafficClassifier(config)

    def test_bulk_traffic_classification(self, classifier):
        """Test classification of bulk transfer traffic"""
        # Bulk: Large packets, uniform sizes, smooth sending
        for i in range(60):
            size = 1400 + np.random.randint(-20, 20)  # Small variance
            timestamp = i * 0.001  # Regular 1ms intervals
            classifier.add_packet(size, timestamp)

        traffic_type, confidence = classifier.classify()

        assert traffic_type == 'bulk'
        assert confidence >= 0.6

    def test_streaming_traffic_classification(self, classifier):
        """Test classification of streaming traffic"""
        # Streaming: Medium packets, moderate variance, regular intervals
        for i in range(60):
            size = 1000 + np.random.randint(-100, 100)  # Moderate variance
            timestamp = i * 0.005  # Regular 5ms intervals
            classifier.add_packet(size, timestamp)

        traffic_type, confidence = classifier.classify()

        # Should classify as either streaming or bulk
        assert traffic_type in ['streaming', 'bulk']

    def test_realtime_traffic_classification(self, classifier):
        """Test classification of real-time traffic"""
        # Real-time: Small packets, bursty pattern
        timestamp = 0.0
        for i in range(30):
            # Burst of 3-5 small packets
            burst_size = np.random.randint(3, 6)
            for j in range(burst_size):
                size = 200 + np.random.randint(-50, 50)
                classifier.add_packet(size, timestamp)
                timestamp += 0.0001  # Very close together

            # Gap before next burst
            timestamp += 0.02 + np.random.uniform(0, 0.01)

        traffic_type, confidence = classifier.classify()

        assert traffic_type == 'realtime'
        assert confidence >= 0.5

    def test_low_confidence_returns_default(self, classifier):
        """Test that low confidence returns default"""
        # Add ambiguous traffic that doesn't match any pattern well
        for i in range(30):
            size = np.random.randint(200, 1500)  # Very random sizes
            timestamp = i * np.random.uniform(0.001, 0.02)  # Random intervals
            classifier.add_packet(size, timestamp)

        traffic_type, confidence = classifier.classify()

        # With high variation, confidence might be low
        if confidence < classifier.confidence_threshold:
            assert traffic_type == 'default'

    def test_insufficient_packets_returns_default(self, classifier):
        """Test that insufficient packets returns default with zero confidence"""
        # Add only a few packets
        for i in range(5):
            classifier.add_packet(1400, i * 0.001)

        traffic_type, confidence = classifier.classify()

        assert traffic_type == 'default'
        assert confidence == 0.0

    def test_disabled_classifier_returns_default(self):
        """Test that disabled classifier always returns default"""
        config = ClassifierConfig(enabled=False)
        classifier = TrafficClassifier(config)

        # Add bulk-like traffic
        for i in range(60):
            classifier.add_packet(1400, i * 0.001)

        traffic_type, confidence = classifier.classify()

        assert traffic_type == 'default'
        assert confidence == 1.0


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def classifier(self):
        config = ClassifierConfig()
        return TrafficClassifier(config)

    def test_zero_packet_size(self, classifier):
        """Test handling of zero packet size"""
        classifier.add_packet(0, 0.001)
        classifier.add_packet(1400, 0.002)

        assert len(classifier.packet_sizes) == 2

        # Should not crash when extracting features
        features = classifier.extract_features()

    def test_identical_timestamps(self, classifier):
        """Test handling of identical timestamps"""
        # Add packets with same timestamp
        classifier.add_packet(1400, 0.001)
        classifier.add_packet(1400, 0.001)
        classifier.add_packet(1400, 0.001)

        # Should handle zero inter-arrival times
        features = classifier.extract_features()

    def test_decreasing_timestamps(self, classifier):
        """Test handling of non-monotonic timestamps"""
        classifier.add_packet(1400, 0.003)
        classifier.add_packet(1400, 0.002)  # Timestamp goes backwards
        classifier.add_packet(1400, 0.001)

        # Should still work (though IAT will be negative)
        features = classifier.extract_features()

    def test_very_large_packet_size(self, classifier):
        """Test handling of very large packet sizes"""
        # Add MTU-sized packets
        for i in range(20):
            classifier.add_packet(9000, i * 0.001)  # Jumbo frame

        features = classifier.extract_features()
        assert features['avg_packet_size'] == pytest.approx(9000, abs=1)

    def test_very_small_packet_size(self, classifier):
        """Test handling of very small packet sizes"""
        # Add tiny packets
        for i in range(20):
            classifier.add_packet(40, i * 0.001)  # Minimum TCP/IP

        features = classifier.extract_features()
        assert features['avg_packet_size'] == pytest.approx(40, abs=1)


class TestStatistics:
    """Test statistics and reporting"""

    @pytest.fixture
    def classifier(self):
        config = ClassifierConfig()
        return TrafficClassifier(config)

    def test_get_statistics(self, classifier):
        """Test get_statistics method"""
        # Add some traffic
        for i in range(30):
            classifier.add_packet(1400, i * 0.001)

        stats = classifier.get_statistics()

        assert 'traffic_type' in stats
        assert 'confidence' in stats
        assert 'features' in stats
        assert 'num_packets' in stats

        assert stats['num_packets'] == 30

    def test_statistics_with_no_packets(self, classifier):
        """Test statistics with no packets"""
        stats = classifier.get_statistics()

        assert stats['traffic_type'] == 'default'
        assert stats['confidence'] == 0.0
        assert stats['features'] == {}
        assert stats['num_packets'] == 0


class TestPerformance:
    """Test performance characteristics"""

    @pytest.fixture
    def classifier(self):
        config = ClassifierConfig()
        return TrafficClassifier(config)

    def test_classification_latency(self, classifier):
        """Test that classification is fast enough"""
        import time

        # Add packets
        for i in range(100):
            classifier.add_packet(1400, i * 0.001)

        # Measure classification time
        start = time.time()
        for _ in range(100):
            classifier.classify()
        end = time.time()

        avg_latency = (end - start) / 100
        # Should be well under 1ms
        assert avg_latency < 0.001

    def test_memory_usage(self, classifier):
        """Test that memory usage is bounded"""
        window_size = classifier.window_size

        # Add many packets
        for i in range(10000):
            classifier.add_packet(1400, i * 0.001)

        # Should not exceed window_size
        assert len(classifier.packet_sizes) <= window_size
        assert len(classifier.timestamps) <= window_size
        assert len(classifier.inter_arrivals) <= window_size


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
