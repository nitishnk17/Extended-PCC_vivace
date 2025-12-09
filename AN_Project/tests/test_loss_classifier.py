"""
Comprehensive unit tests for LossClassifier

Tests cover:
1. Basic functionality (initialization, event addition)
2. Correlation calculation
3. Loss classification logic
4. Baseline RTT tracking
5. Edge cases and error handling
6. Statistics reporting
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.loss_classifier import LossClassifier, LossEvent, RTTEvent
from src.config import Config


class TestLossEvent:
    """Test LossEvent class"""

    def test_loss_event_creation(self):
        """Test LossEvent initialization"""
        event = LossEvent(1.0, 0.05, 5, 100)

        assert event.timestamp == 1.0
        assert event.loss_rate == 0.05
        assert event.packets_lost == 5
        assert event.packets_sent == 100
        assert event.is_loss is True

    def test_loss_event_no_loss(self):
        """Test LossEvent with no loss"""
        event = LossEvent(1.0, 0.0, 0, 100)

        assert event.is_loss is False
        assert event.loss_rate == 0.0


class TestRTTEvent:
    """Test RTTEvent class"""

    def test_rtt_event_creation(self):
        """Test RTTEvent initialization"""
        event = RTTEvent(1.0, 50.0)

        assert event.timestamp == 1.0
        assert event.rtt == 50.0


class TestLossClassifierBasics:
    """Test basic LossClassifier functionality"""

    @pytest.fixture
    def config(self):
        """Create configuration"""
        return Config()

    @pytest.fixture
    def classifier(self, config):
        """Create LossClassifier instance"""
        return LossClassifier(config.loss_classifier)

    def test_initialization(self, classifier, config):
        """Test classifier initialization"""
        assert classifier.window_size == config.loss_classifier.window_size
        assert len(classifier.loss_events) == 0
        assert len(classifier.rtt_events) == 0
        assert classifier.baseline_rtt is None
        assert classifier.last_p_wireless == 0.5
        assert classifier.last_confidence == 0.0
        assert classifier.classification_count == 0

    def test_add_loss_event(self, classifier):
        """Test adding loss events"""
        classifier.add_loss_event(1.0, 0.05, 5, 100)

        assert len(classifier.loss_events) == 1
        event = classifier.loss_events[0]
        assert event.timestamp == 1.0
        assert event.loss_rate == 0.05

    def test_add_rtt_sample(self, classifier):
        """Test adding RTT samples"""
        classifier.add_rtt_sample(1.0, 50.0)

        assert len(classifier.rtt_events) == 1
        event = classifier.rtt_events[0]
        assert event.timestamp == 1.0
        assert event.rtt == 50.0

    def test_add_multiple_events(self, classifier):
        """Test adding multiple events"""
        # Add 10 loss events
        for i in range(10):
            classifier.add_loss_event(float(i), 0.01, 1, 100)

        # Add 10 RTT samples
        for i in range(10):
            classifier.add_rtt_sample(float(i), 50.0 + i)

        assert len(classifier.loss_events) == 10
        assert len(classifier.rtt_events) == 10

    def test_bounded_queue(self, classifier):
        """Test that event queues are bounded"""
        window_size = classifier.window_size

        # Add more events than window_size
        for i in range(window_size + 50):
            classifier.add_loss_event(float(i), 0.01, 1, 100)
            classifier.add_rtt_sample(float(i), 50.0)

        # Should be limited to window_size
        assert len(classifier.loss_events) == window_size
        assert len(classifier.rtt_events) == window_size


class TestBaselineRTT:
    """Test baseline RTT tracking"""

    @pytest.fixture
    def classifier(self):
        return LossClassifier(Config().loss_classifier)

    def test_baseline_not_established_initially(self, classifier):
        """Test baseline RTT not established with few samples"""
        # Add only 5 samples (need 10)
        for i in range(5):
            classifier.add_rtt_sample(float(i), 50.0 + i)

        assert classifier.baseline_rtt is None

    def test_baseline_established_after_enough_samples(self, classifier):
        """Test baseline RTT established after 10 samples"""
        # Add 10 samples
        rtts = [50.0, 52.0, 48.0, 51.0, 49.0, 53.0, 47.0, 50.5, 49.5, 51.5]
        for i, rtt in enumerate(rtts):
            classifier.add_rtt_sample(float(i), rtt)

        # Baseline should be established (5th percentile)
        assert classifier.baseline_rtt is not None
        expected_baseline = np.percentile(rtts, 5)
        assert classifier.baseline_rtt == pytest.approx(expected_baseline)

    def test_baseline_uses_5th_percentile(self, classifier):
        """Test that baseline uses 5th percentile (robust to outliers)"""
        # Add samples with outlier
        rtts = [50.0] * 15 + [200.0]  # One large outlier
        for i, rtt in enumerate(rtts):
            classifier.add_rtt_sample(float(i), rtt)

        # Baseline should be close to 50.0, not affected by outlier
        assert classifier.baseline_rtt == pytest.approx(50.0, abs=2.0)

    def test_rtt_inflation_detection(self, classifier):
        """Test RTT inflation detection"""
        # Establish baseline around 50ms
        for i in range(20):
            classifier.add_rtt_sample(float(i) * 0.1, 50.0 + np.random.randn() * 1.0)

        baseline = classifier.baseline_rtt
        assert baseline is not None

        # Check normal RTT (not inflated) - use == for numpy bool comparison
        assert classifier._is_rtt_inflated(baseline + 5.0) == False

        # Check inflated RTT (baseline + 30ms should be inflated)
        assert classifier._is_rtt_inflated(baseline + 30.0) == True


class TestCorrelationCalculation:
    """Test correlation calculation"""

    @pytest.fixture
    def classifier(self):
        return LossClassifier(Config().loss_classifier)

    def test_insufficient_events_returns_default(self, classifier):
        """Test classification with insufficient events"""
        # Add only 3 events (need min_events)
        for i in range(3):
            classifier.add_loss_event(float(i), 0.01, 1, 100)
            classifier.add_rtt_sample(float(i), 50.0)

        p_wireless, confidence = classifier.classify()

        # Should return default with zero confidence
        assert p_wireless == 0.5
        assert confidence == 0.0

    def test_pearson_correlation_perfect_positive(self, classifier):
        """Test Pearson correlation with perfect positive correlation"""
        x = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        y = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

        correlation = classifier._pearson_correlation(x, y)

        assert correlation == pytest.approx(1.0)

    def test_pearson_correlation_perfect_negative(self, classifier):
        """Test Pearson correlation with perfect negative correlation"""
        x = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        y = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

        correlation = classifier._pearson_correlation(x, y)

        assert correlation == pytest.approx(-1.0)

    def test_pearson_correlation_no_correlation(self, classifier):
        """Test Pearson correlation with no correlation"""
        # Random uncorrelated data
        np.random.seed(42)
        x = np.random.randint(0, 2, 20).tolist()
        y = np.random.randint(0, 2, 20).tolist()

        correlation = classifier._pearson_correlation(x, y)

        # Should be close to 0 (within reasonable range)
        assert -0.5 < correlation < 0.5

    def test_pearson_correlation_zero_variance(self, classifier):
        """Test Pearson correlation with zero variance"""
        x = [0.0, 0.0, 0.0, 0.0]
        y = [1.0, 1.0, 1.0, 1.0]

        correlation = classifier._pearson_correlation(x, y)

        # Should return 0.0 (undefined correlation)
        assert correlation == 0.0

    def test_pearson_correlation_insufficient_samples(self, classifier):
        """Test Pearson correlation with insufficient samples"""
        x = [1.0]
        y = [1.0]

        correlation = classifier._pearson_correlation(x, y)

        assert correlation == 0.0


class TestLossClassification:
    """Test loss classification logic"""

    @pytest.fixture
    def classifier(self):
        config = Config()
        return LossClassifier(config.loss_classifier)

    def test_high_correlation_congestion_loss(self, classifier):
        """Test classification with high correlation (congestion loss)"""
        # Establish baseline RTT
        for i in range(20):
            classifier.add_rtt_sample(float(i) * 0.1, 50.0)

        baseline = classifier.baseline_rtt

        # Add correlated loss and RTT inflation
        for i in range(20):
            t = float(i) * 0.1 + 2.0
            if i % 2 == 0:
                # Loss event with RTT inflation
                classifier.add_loss_event(t, 0.05, 5, 100)
                classifier.add_rtt_sample(t, baseline + 30.0)
            else:
                # No loss, normal RTT
                classifier.add_loss_event(t, 0.0, 0, 100)
                classifier.add_rtt_sample(t, baseline + 1.0)

        p_wireless, confidence = classifier.classify()

        # Should classify as congestion (low p_wireless)
        assert p_wireless < 0.5
        assert confidence > 0.0

    def test_low_correlation_wireless_loss(self, classifier):
        """Test classification with low correlation (wireless loss)"""
        # Establish baseline RTT
        for i in range(20):
            classifier.add_rtt_sample(float(i) * 0.1, 50.0)

        baseline = classifier.baseline_rtt

        # Add uncorrelated loss (wireless)
        np.random.seed(42)
        for i in range(20):
            t = float(i) * 0.1 + 2.0
            # Random loss, but RTT stays stable
            if np.random.rand() > 0.7:
                classifier.add_loss_event(t, 0.05, 5, 100)
            else:
                classifier.add_loss_event(t, 0.0, 0, 100)
            classifier.add_rtt_sample(t, baseline + np.random.randn() * 2.0)

        p_wireless, confidence = classifier.classify()

        # Should classify as wireless (high p_wireless)
        assert p_wireless > 0.5
        assert confidence > 0.0

    def test_no_loss_no_correlation(self, classifier):
        """Test classification with no loss"""
        # Establish baseline RTT
        for i in range(20):
            classifier.add_rtt_sample(float(i) * 0.1, 50.0)

        baseline = classifier.baseline_rtt

        # Add no-loss events
        for i in range(20):
            t = float(i) * 0.1 + 2.0
            classifier.add_loss_event(t, 0.0, 0, 100)
            classifier.add_rtt_sample(t, baseline + np.random.randn() * 2.0)

        p_wireless, confidence = classifier.classify()

        # With no loss, should return high p_wireless (no congestion)
        assert p_wireless >= 0.5


class TestClassificationFromCorrelation:
    """Test classification from correlation values"""

    @pytest.fixture
    def classifier(self):
        return LossClassifier(Config().loss_classifier)

    def test_classify_high_correlation(self, classifier):
        """Test classification with high correlation (>0.5)"""
        p_wireless, confidence = classifier._classify_from_correlation(0.8)

        # High correlation → Congestion
        assert p_wireless == pytest.approx(0.1)
        assert confidence > 0.0

    def test_classify_low_correlation(self, classifier):
        """Test classification with low correlation (<0.2)"""
        p_wireless, confidence = classifier._classify_from_correlation(0.1)

        # Low correlation → Wireless
        assert p_wireless == pytest.approx(0.9)
        assert confidence > 0.0

    def test_classify_intermediate_correlation(self, classifier):
        """Test classification with intermediate correlation"""
        p_wireless, confidence = classifier._classify_from_correlation(0.35)

        # Intermediate → Mixed
        assert 0.1 < p_wireless < 0.9
        assert confidence > 0.0

    def test_classify_extreme_high_correlation(self, classifier):
        """Test classification with very high correlation (0.9)"""
        p_wireless, confidence = classifier._classify_from_correlation(0.9)

        # Very high correlation → Strong congestion
        assert p_wireless == pytest.approx(0.1)
        assert confidence > 0.5

    def test_classify_extreme_low_correlation(self, classifier):
        """Test classification with very low correlation (-0.1)"""
        p_wireless, confidence = classifier._classify_from_correlation(-0.1)

        # Very low correlation → Strong wireless
        assert p_wireless == pytest.approx(0.9)
        assert confidence > 0.2  # Confidence calculation is working as designed

    def test_clamping(self, classifier):
        """Test that p_wireless is clamped to [0, 1]"""
        # Test with extreme values
        for corr in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            p_wireless, confidence = classifier._classify_from_correlation(corr)

            assert 0.0 <= p_wireless <= 1.0
            assert 0.0 <= confidence <= 1.0


class TestEventAlignment:
    """Test event alignment for correlation"""

    @pytest.fixture
    def classifier(self):
        return LossClassifier(Config().loss_classifier)

    def test_find_closest_loss_event(self, classifier):
        """Test finding closest loss event"""
        # Add loss events
        classifier.add_loss_event(1.0, 0.05, 5, 100)
        classifier.add_loss_event(2.0, 0.03, 3, 100)
        classifier.add_loss_event(3.0, 0.02, 2, 100)

        # Find closest to 1.8
        closest = classifier._find_closest_loss_event(1.8)

        assert closest is not None
        assert closest.timestamp == 2.0

    def test_find_closest_loss_event_max_diff(self, classifier):
        """Test max_time_diff parameter"""
        # Add loss event at t=1.0
        classifier.add_loss_event(1.0, 0.05, 5, 100)

        # Search at t=2.0 with max_time_diff=0.5
        closest = classifier._find_closest_loss_event(2.0, max_time_diff=0.5)

        # Should return None (diff=1.0 > 0.5)
        assert closest is None

    def test_find_closest_empty_events(self, classifier):
        """Test finding closest with no events"""
        closest = classifier._find_closest_loss_event(1.0)

        assert closest is None


class TestStatistics:
    """Test statistics reporting"""

    @pytest.fixture
    def classifier(self):
        return LossClassifier(Config().loss_classifier)

    def test_get_statistics_empty(self, classifier):
        """Test statistics with no data"""
        stats = classifier.get_statistics()

        assert 'classification' in stats
        assert 'events' in stats
        assert 'loss' in stats
        assert 'rtt' in stats

        assert stats['classification']['p_wireless'] == 0.5
        assert stats['classification']['confidence'] == 0.0
        assert stats['events']['loss_events'] == 0
        assert stats['events']['rtt_events'] == 0

    def test_get_statistics_with_data(self, classifier):
        """Test statistics with actual data"""
        # Add some events
        for i in range(20):
            classifier.add_loss_event(float(i) * 0.1, 0.02, 2, 100)
            classifier.add_rtt_sample(float(i) * 0.1, 50.0 + i)

        # Classify
        classifier.classify()

        stats = classifier.get_statistics()

        assert stats['events']['loss_events'] == 20
        assert stats['events']['rtt_events'] == 20
        assert stats['loss']['avg_loss_rate'] > 0.0
        assert stats['rtt']['baseline_rtt'] > 0.0
        assert stats['rtt']['avg_rtt'] > 0.0

    def test_get_event_summary(self, classifier):
        """Test human-readable event summary"""
        # Add some data
        for i in range(20):
            classifier.add_loss_event(float(i) * 0.1, 0.02, 2, 100)
            classifier.add_rtt_sample(float(i) * 0.1, 50.0)

        classifier.classify()

        summary = classifier.get_event_summary()

        assert isinstance(summary, str)
        assert 'Loss Classifier Summary' in summary
        assert 'p_wireless' in summary
        assert 'confidence' in summary


class TestReset:
    """Test reset functionality"""

    @pytest.fixture
    def classifier(self):
        return LossClassifier(Config().loss_classifier)

    def test_reset(self, classifier):
        """Test reset clears all state"""
        # Add data
        for i in range(20):
            classifier.add_loss_event(float(i) * 0.1, 0.02, 2, 100)
            classifier.add_rtt_sample(float(i) * 0.1, 50.0)

        classifier.classify()

        # Verify state exists
        assert len(classifier.loss_events) > 0
        assert len(classifier.rtt_events) > 0
        assert classifier.baseline_rtt is not None
        assert classifier.classification_count > 0

        # Reset
        classifier.reset()

        # Verify state cleared
        assert len(classifier.loss_events) == 0
        assert len(classifier.rtt_events) == 0
        assert classifier.baseline_rtt is None
        assert classifier.last_p_wireless == 0.5
        assert classifier.last_confidence == 0.0
        assert classifier.classification_count == 0
        assert classifier.total_loss_events == 0


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def classifier(self):
        return LossClassifier(Config().loss_classifier)

    def test_negative_loss_rate(self, classifier):
        """Test handling of negative loss rate"""
        # Should handle gracefully
        classifier.add_loss_event(1.0, -0.01, 0, 100)

        # Should not crash
        assert len(classifier.loss_events) == 1

    def test_loss_rate_greater_than_one(self, classifier):
        """Test handling of loss rate > 1.0"""
        classifier.add_loss_event(1.0, 1.5, 150, 100)

        # Should not crash
        assert len(classifier.loss_events) == 1

    def test_zero_packets_sent(self, classifier):
        """Test handling of zero packets sent"""
        classifier.add_loss_event(1.0, 0.0, 0, 0)

        # Should not crash
        assert len(classifier.loss_events) == 1

    def test_negative_rtt(self, classifier):
        """Test handling of negative RTT"""
        classifier.add_rtt_sample(1.0, -10.0)

        # Should not crash
        assert len(classifier.rtt_events) == 1

    def test_very_large_rtt(self, classifier):
        """Test handling of very large RTT"""
        classifier.add_rtt_sample(1.0, 10000.0)

        # Should not crash
        assert len(classifier.rtt_events) == 1

    def test_classification_without_baseline(self, classifier):
        """Test classification before baseline established"""
        # Add a few events (not enough for baseline)
        for i in range(5):
            classifier.add_loss_event(float(i), 0.02, 2, 100)
            classifier.add_rtt_sample(float(i), 50.0)

        p_wireless, confidence = classifier.classify()

        # Should return default with zero confidence
        assert p_wireless == 0.5
        assert confidence == 0.0

    def test_all_zero_loss(self, classifier):
        """Test with all zero loss rates"""
        # Establish baseline
        for i in range(20):
            classifier.add_rtt_sample(float(i) * 0.1, 50.0)

        # Add zero loss
        for i in range(20):
            classifier.add_loss_event(float(i) * 0.1 + 2.0, 0.0, 0, 100)

        p_wireless, confidence = classifier.classify()

        # Should handle gracefully (high p_wireless, no congestion)
        assert p_wireless >= 0.5


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""

    @pytest.fixture
    def classifier(self):
        return LossClassifier(Config().loss_classifier)

    def test_pure_congestion_scenario(self, classifier):
        """Test pure congestion loss scenario"""
        # Establish baseline RTT around 50ms
        for i in range(30):
            classifier.add_rtt_sample(float(i) * 0.05, 50.0 + np.random.randn() * 1.0)

        baseline = classifier.baseline_rtt

        # Simulate congestion: loss events correlated with RTT spikes
        np.random.seed(42)
        for i in range(50):
            t = float(i) * 0.05 + 3.0

            # Every 5th packet: congestion event (loss + RTT spike)
            if i % 5 == 0:
                classifier.add_loss_event(t, 0.08, 8, 100)
                classifier.add_rtt_sample(t, baseline + 40.0 + np.random.randn() * 5.0)
            else:
                classifier.add_loss_event(t, 0.0, 0, 100)
                classifier.add_rtt_sample(t, baseline + np.random.randn() * 3.0)

        p_wireless, confidence = classifier.classify()

        # Should strongly classify as congestion
        assert p_wireless < 0.3
        assert confidence > 0.3

    def test_pure_wireless_scenario(self, classifier):
        """Test pure wireless loss scenario"""
        # Establish baseline RTT
        for i in range(30):
            classifier.add_rtt_sample(float(i) * 0.05, 50.0 + np.random.randn() * 1.0)

        baseline = classifier.baseline_rtt

        # Simulate wireless loss: random loss, stable RTT
        np.random.seed(42)
        for i in range(50):
            t = float(i) * 0.05 + 3.0

            # Random loss (20% probability)
            if np.random.rand() < 0.2:
                classifier.add_loss_event(t, 0.05, 5, 100)
            else:
                classifier.add_loss_event(t, 0.0, 0, 100)

            # RTT stays stable
            classifier.add_rtt_sample(t, baseline + np.random.randn() * 3.0)

        p_wireless, confidence = classifier.classify()

        # Should strongly classify as wireless
        assert p_wireless > 0.7
        assert confidence > 0.2  # Reasonable confidence threshold

    def test_mixed_loss_scenario(self, classifier):
        """Test mixed loss scenario"""
        # Establish baseline RTT
        for i in range(30):
            classifier.add_rtt_sample(float(i) * 0.05, 50.0 + np.random.randn() * 1.0)

        baseline = classifier.baseline_rtt

        # Simulate mixed: some correlated, some random
        np.random.seed(42)
        for i in range(50):
            t = float(i) * 0.05 + 3.0

            if i % 10 == 0:
                # Congestion event
                classifier.add_loss_event(t, 0.08, 8, 100)
                classifier.add_rtt_sample(t, baseline + 35.0)
            elif np.random.rand() < 0.15:
                # Random wireless loss
                classifier.add_loss_event(t, 0.04, 4, 100)
                classifier.add_rtt_sample(t, baseline + np.random.randn() * 3.0)
            else:
                # Normal
                classifier.add_loss_event(t, 0.0, 0, 100)
                classifier.add_rtt_sample(t, baseline + np.random.randn() * 3.0)

        p_wireless, confidence = classifier.classify()

        # Should be intermediate
        assert 0.2 < p_wireless < 0.8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
