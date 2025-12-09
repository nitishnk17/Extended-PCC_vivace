"""
Comprehensive unit tests for ContentionDetector

Tests cover:
1. Basic functionality (initialization, gradient addition)
2. Sign change detection
3. Volatility calculation
4. Flow count estimation
5. Contention classification
6. Confidence calculation
7. Statistics and reporting
8. Edge cases and error handling
9. Realistic scenarios (solo, light, moderate, heavy contention)
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.contention_detector import (
    ContentionDetector, ContentionLevel, GradientEvent
)
from src.config import Config


class TestGradientEvent:
    """Test GradientEvent dataclass"""

    def test_gradient_event_creation(self):
        """Test GradientEvent initialization"""
        event = GradientEvent(1.0, 0.5, 10.0, 5.0)

        assert event.timestamp == 1.0
        assert event.gradient == 0.5
        assert event.utility == 10.0
        assert event.sending_rate == 5.0


class TestContentionLevelEnum:
    """Test ContentionLevel enum"""

    def test_contention_levels_exist(self):
        """Test all contention levels defined"""
        assert ContentionLevel.SOLO.value == "solo"
        assert ContentionLevel.LIGHT.value == "light"
        assert ContentionLevel.MODERATE.value == "moderate"
        assert ContentionLevel.HEAVY.value == "heavy"


class TestContentionDetectorBasics:
    """Test basic ContentionDetector functionality"""

    @pytest.fixture
    def config(self):
        """Create configuration"""
        return Config()

    @pytest.fixture
    def detector(self, config):
        """Create ContentionDetector instance"""
        return ContentionDetector(config.contention_detector)

    def test_initialization(self, detector, config):
        """Test detector initialization"""
        assert detector.window_size == config.contention_detector.window_size
        assert detector.sign_change_threshold == config.contention_detector.sign_change_threshold
        assert detector.magnitude_threshold == config.contention_detector.magnitude_threshold
        assert len(detector.gradient_history) == 0
        assert detector.current_contention == ContentionLevel.SOLO
        assert detector.estimated_flow_count == 1
        assert detector.detection_confidence == 0.0

    def test_add_gradient(self, detector):
        """Test adding gradient observations"""
        detector.add_gradient(1.0, 0.5, 10.0, 5.0)

        assert len(detector.gradient_history) == 1
        event = detector.gradient_history[0]
        assert event.timestamp == 1.0
        assert event.gradient == 0.5
        assert event.utility == 10.0
        assert event.sending_rate == 5.0

    def test_add_multiple_gradients(self, detector):
        """Test adding multiple gradients"""
        for i in range(10):
            detector.add_gradient(float(i), 0.5, 10.0 + i, 5.0 + i)

        assert len(detector.gradient_history) == 10

    def test_bounded_queue(self, detector):
        """Test that gradient history is bounded"""
        window_size = detector.window_size

        # Add more gradients than window_size
        for i in range(window_size + 20):
            detector.add_gradient(float(i), 0.5, 10.0, 5.0)

        # Should be limited to window_size
        assert len(detector.gradient_history) == window_size

        # Should keep most recent
        assert detector.gradient_history[-1].timestamp == float(window_size + 19)


class TestSignChangeDetection:
    """Test gradient sign change detection"""

    @pytest.fixture
    def detector(self):
        return ContentionDetector(Config().contention_detector)

    def test_no_sign_changes_positive(self, detector):
        """Test no sign changes with all positive gradients"""
        # All positive gradients
        for i in range(10):
            detector.add_gradient(float(i), 0.5 + i * 0.1, 10.0, 5.0)

        sign_changes = detector._count_sign_changes()
        assert sign_changes == 0

    def test_no_sign_changes_negative(self, detector):
        """Test no sign changes with all negative gradients"""
        # All negative gradients
        for i in range(10):
            detector.add_gradient(float(i), -0.5 - i * 0.1, 10.0, 5.0)

        sign_changes = detector._count_sign_changes()
        assert sign_changes == 0

    def test_single_sign_change(self, detector):
        """Test single sign change"""
        # Positive then negative
        for i in range(5):
            detector.add_gradient(float(i), 0.5, 10.0, 5.0)
        for i in range(5, 10):
            detector.add_gradient(float(i), -0.5, 10.0, 5.0)

        sign_changes = detector._count_sign_changes()
        assert sign_changes == 1

    def test_multiple_sign_changes(self, detector):
        """Test multiple sign changes (oscillating)"""
        # Alternating positive and negative
        for i in range(20):
            gradient = 0.5 if i % 2 == 0 else -0.5
            detector.add_gradient(float(i), gradient, 10.0, 5.0)

        sign_changes = detector._count_sign_changes()
        # Should be 19 changes (20 gradients = 19 transitions)
        assert sign_changes == 19

    def test_ignore_small_gradients(self, detector):
        """Test that small gradients are ignored"""
        # Small gradients (below magnitude threshold)
        for i in range(10):
            gradient = 0.1 if i % 2 == 0 else -0.1  # Below 0.3 threshold
            detector.add_gradient(float(i), gradient, 10.0, 5.0)

        sign_changes = detector._count_sign_changes()
        # Should be 0 because all gradients too small
        assert sign_changes == 0


class TestVolatilityCalculation:
    """Test gradient volatility calculation"""

    @pytest.fixture
    def detector(self):
        return ContentionDetector(Config().contention_detector)

    def test_zero_volatility_constant(self, detector):
        """Test zero volatility with constant gradients"""
        # All same gradient
        for i in range(10):
            detector.add_gradient(float(i), 0.5, 10.0, 5.0)

        volatility = detector._calculate_volatility()
        assert volatility == pytest.approx(0.0, abs=0.01)

    def test_low_volatility_stable(self, detector):
        """Test low volatility with stable gradients"""
        # Small variations
        for i in range(10):
            gradient = 0.5 + np.random.randn() * 0.05
            detector.add_gradient(float(i), gradient, 10.0, 5.0)

        volatility = detector._calculate_volatility()
        assert volatility < 0.5  # Low volatility

    def test_high_volatility_oscillating(self, detector):
        """Test high volatility with oscillating gradients"""
        # Large variations
        for i in range(10):
            gradient = 1.0 if i % 2 == 0 else -1.0
            detector.add_gradient(float(i), gradient, 10.0, 5.0)

        volatility = detector._calculate_volatility()
        assert volatility > 0.5  # High volatility

    def test_volatility_with_insufficient_data(self, detector):
        """Test volatility with single gradient"""
        detector.add_gradient(1.0, 0.5, 10.0, 5.0)

        volatility = detector._calculate_volatility()
        assert volatility == 0.0


class TestFlowCountEstimation:
    """Test flow count estimation"""

    @pytest.fixture
    def detector(self):
        return ContentionDetector(Config().contention_detector)

    def test_estimate_solo_flow(self, detector):
        """Test estimation with no contention (solo)"""
        # Low sign change ratio, low volatility → 1 flow
        flow_count = detector._estimate_flow_count(
            sign_change_ratio=0.1,
            volatility=0.5
        )
        assert flow_count == 1

    def test_estimate_light_contention(self, detector):
        """Test estimation with light contention"""
        # Moderate sign change ratio → 2 flows
        flow_count = detector._estimate_flow_count(
            sign_change_ratio=0.3,
            volatility=1.0
        )
        assert flow_count == 2

    def test_estimate_moderate_contention(self, detector):
        """Test estimation with moderate contention"""
        # Higher sign change ratio → moderate flows
        # With SCR=0.5, volatility=1.5: base=4, factor=1.5 → 6 flows
        flow_count = detector._estimate_flow_count(
            sign_change_ratio=0.5,
            volatility=1.5
        )
        assert 4 <= flow_count <= 8

    def test_estimate_heavy_contention(self, detector):
        """Test estimation with heavy contention"""
        # Very high sign change ratio + volatility → many flows
        flow_count = detector._estimate_flow_count(
            sign_change_ratio=0.7,
            volatility=3.0
        )
        assert flow_count >= 6

    def test_flow_count_clamped(self, detector):
        """Test flow count is clamped to reasonable range"""
        # Extreme values should be clamped
        flow_count = detector._estimate_flow_count(
            sign_change_ratio=1.0,
            volatility=10.0
        )
        assert 1 <= flow_count <= 32


class TestContentionClassification:
    """Test contention level classification"""

    @pytest.fixture
    def detector(self):
        return ContentionDetector(Config().contention_detector)

    def test_classify_solo(self, detector):
        """Test classification of solo flow"""
        level = detector._classify_contention(flow_count=1)
        assert level == ContentionLevel.SOLO

    def test_classify_light(self, detector):
        """Test classification of light contention"""
        assert detector._classify_contention(2) == ContentionLevel.LIGHT

    def test_classify_moderate(self, detector):
        """Test classification of moderate contention"""
        assert detector._classify_contention(3) == ContentionLevel.MODERATE
        assert detector._classify_contention(5) == ContentionLevel.MODERATE

    def test_classify_heavy(self, detector):
        """Test classification of heavy contention"""
        assert detector._classify_contention(6) == ContentionLevel.HEAVY
        assert detector._classify_contention(10) == ContentionLevel.HEAVY
        assert detector._classify_contention(32) == ContentionLevel.HEAVY


class TestConfidenceCalculation:
    """Test confidence calculation"""

    @pytest.fixture
    def detector(self):
        return ContentionDetector(Config().contention_detector)

    def test_low_confidence_insufficient_data(self, detector):
        """Test low confidence with insufficient data"""
        # Add few gradients
        for i in range(5):
            detector.add_gradient(float(i), 0.5, 10.0, 5.0)

        confidence = detector._calculate_confidence(
            sign_change_ratio=0.5,
            volatility=1.0
        )

        # Low confidence due to insufficient data
        assert confidence < 0.5

    def test_high_confidence_clear_pattern(self, detector):
        """Test high confidence with clear pattern"""
        # Full window, clear pattern
        for i in range(detector.window_size):
            detector.add_gradient(float(i), 0.5, 10.0, 5.0)

        # Clear solo pattern (no oscillation)
        confidence = detector._calculate_confidence(
            sign_change_ratio=0.0,
            volatility=0.1
        )

        assert confidence > 0.7

    def test_medium_confidence_ambiguous(self, detector):
        """Test medium confidence with ambiguous pattern"""
        # Full window but ambiguous pattern
        for i in range(detector.window_size):
            detector.add_gradient(float(i), 0.5, 10.0, 5.0)

        # Ambiguous pattern (mid-range sign changes)
        confidence = detector._calculate_confidence(
            sign_change_ratio=0.5,
            volatility=1.5
        )

        assert 0.3 < confidence < 0.7


class TestFullDetection:
    """Test complete detection pipeline"""

    @pytest.fixture
    def detector(self):
        return ContentionDetector(Config().contention_detector)

    def test_detect_insufficient_data(self, detector):
        """Test detection with insufficient data"""
        # Add only 2 gradients (need more for detection)
        detector.add_gradient(1.0, 0.5, 10.0, 5.0)
        detector.add_gradient(2.0, 0.6, 10.5, 5.1)

        level, confidence, flow_count = detector.detect_contention()

        # Should return solo with zero confidence
        assert level == ContentionLevel.SOLO
        assert confidence == 0.0
        assert flow_count == 1

    def test_detect_solo_flow(self, detector):
        """Test detection of solo flow"""
        # Monotonic gradients (all positive)
        for i in range(30):
            detector.add_gradient(float(i), 0.5 + i * 0.01, 10.0, 5.0)

        level, confidence, flow_count = detector.detect_contention()

        assert level == ContentionLevel.SOLO
        assert flow_count == 1
        assert confidence > 0.5

    def test_detect_light_contention(self, detector):
        """Test detection of light contention"""
        # Some oscillation (occasional sign changes)
        for i in range(30):
            if i % 5 == 0:
                gradient = -0.5
            else:
                gradient = 0.5
            detector.add_gradient(float(i), gradient, 10.0, 5.0)

        level, confidence, flow_count = detector.detect_contention()

        assert level in [ContentionLevel.LIGHT, ContentionLevel.MODERATE]
        assert flow_count >= 1

    def test_detect_heavy_contention(self, detector):
        """Test detection of heavy contention"""
        # Frequent oscillation (alternating signs)
        for i in range(30):
            gradient = 0.5 if i % 2 == 0 else -0.5
            detector.add_gradient(float(i), gradient, 10.0, 5.0)

        level, confidence, flow_count = detector.detect_contention()

        assert level in [ContentionLevel.MODERATE, ContentionLevel.HEAVY]
        assert flow_count >= 3


class TestStatistics:
    """Test statistics reporting"""

    @pytest.fixture
    def detector(self):
        return ContentionDetector(Config().contention_detector)

    def test_get_statistics_empty(self, detector):
        """Test statistics with no data"""
        stats = detector.get_statistics()

        assert 'current' in stats
        assert 'history' in stats
        assert 'recent' in stats

        assert stats['current']['level'] == 'solo'
        assert stats['current']['confidence'] == 0.0
        assert stats['current']['flow_count'] == 1
        assert stats['history']['total_detections'] == 0

    def test_get_statistics_with_data(self, detector):
        """Test statistics with actual data"""
        # Add gradients and run detections
        for i in range(50):
            gradient = 0.5 if i % 2 == 0 else -0.5
            detector.add_gradient(float(i), gradient, 10.0, 5.0)

            if i >= 10 and i % 5 == 0:
                detector.detect_contention()

        stats = detector.get_statistics()

        assert stats['history']['total_detections'] > 0
        assert stats['history']['samples_count'] > 0
        assert stats['recent']['avg_flow_count'] >= 1

    def test_get_summary(self, detector):
        """Test human-readable summary"""
        # Add some data
        for i in range(20):
            detector.add_gradient(float(i), 0.5, 10.0, 5.0)

        detector.detect_contention()

        summary = detector.get_summary()

        assert isinstance(summary, str)
        assert 'Contention Detector Summary' in summary
        assert 'Current State' in summary
        assert 'Confidence' in summary


class TestReset:
    """Test reset functionality"""

    @pytest.fixture
    def detector(self):
        return ContentionDetector(Config().contention_detector)

    def test_reset(self, detector):
        """Test reset clears all state"""
        # Add data
        for i in range(20):
            detector.add_gradient(float(i), 0.5, 10.0, 5.0)

        detector.detect_contention()

        # Verify state exists
        assert len(detector.gradient_history) > 0
        assert detector.total_detections > 0

        # Reset
        detector.reset()

        # Verify state cleared
        assert len(detector.gradient_history) == 0
        assert detector.current_contention == ContentionLevel.SOLO
        assert detector.estimated_flow_count == 1
        assert detector.detection_confidence == 0.0
        assert detector.total_detections == 0
        assert len(detector.detection_history) == 0


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def detector(self):
        return ContentionDetector(Config().contention_detector)

    def test_zero_gradients(self, detector):
        """Test handling of all zero gradients"""
        for i in range(20):
            detector.add_gradient(float(i), 0.0, 10.0, 5.0)

        level, confidence, flow_count = detector.detect_contention()

        # Should handle gracefully (treat as solo)
        assert level == ContentionLevel.SOLO
        assert flow_count == 1

    def test_very_large_gradients(self, detector):
        """Test handling of very large gradients"""
        for i in range(20):
            gradient = 100.0 if i % 2 == 0 else -100.0
            detector.add_gradient(float(i), gradient, 10.0, 5.0)

        level, confidence, flow_count = detector.detect_contention()

        # Should detect high contention
        assert level in [ContentionLevel.MODERATE, ContentionLevel.HEAVY]

    def test_negative_utilities(self, detector):
        """Test handling of negative utilities"""
        for i in range(20):
            detector.add_gradient(float(i), 0.5, -10.0, 5.0)

        level, confidence, flow_count = detector.detect_contention()

        # Should handle gracefully
        assert flow_count >= 1


class TestRealisticScenarios:
    """Test realistic contention scenarios"""

    @pytest.fixture
    def detector(self):
        return ContentionDetector(Config().contention_detector)

    def test_solo_flow_realistic(self, detector):
        """Test realistic solo flow scenario"""
        # Simulate solo flow: gradual increase then plateau
        np.random.seed(42)
        for i in range(40):
            # Gradually decreasing gradient (approaching optimal)
            base_gradient = max(0.1, 1.0 - i * 0.02)
            noise = np.random.randn() * 0.05
            gradient = base_gradient + noise

            detector.add_gradient(float(i), gradient, 10.0 + i, 5.0)

            if i >= 10 and i % 5 == 0:
                level, confidence, flow_count = detector.detect_contention()

        # Final detection should be solo
        level, confidence, flow_count = detector.detect_contention()
        assert level == ContentionLevel.SOLO
        assert flow_count == 1

    def test_two_flows_realistic(self, detector):
        """Test realistic two-flow scenario"""
        # Simulate two flows competing: oscillating gradients
        np.random.seed(42)
        for i in range(40):
            # Oscillating with noise
            if i % 4 < 2:
                base_gradient = 0.5
            else:
                base_gradient = -0.3

            noise = np.random.randn() * 0.1
            gradient = base_gradient + noise

            detector.add_gradient(float(i), gradient, 10.0, 5.0)

            if i >= 10 and i % 5 == 0:
                level, confidence, flow_count = detector.detect_contention()

        # Should detect light/moderate contention
        level, confidence, flow_count = detector.detect_contention()
        assert level in [ContentionLevel.LIGHT, ContentionLevel.MODERATE]
        assert flow_count >= 2

    def test_many_flows_realistic(self, detector):
        """Test realistic many-flow scenario"""
        # Simulate many flows: rapid oscillation with high volatility
        # Multiple flows cause frequent utility gradient reversals
        np.random.seed(42)
        for i in range(40):
            # Rapid alternating pattern (simulates constant competition)
            # Period of 3 samples with noise
            if i % 3 == 0:
                base_gradient = 1.2
            elif i % 3 == 1:
                base_gradient = -0.8
            else:
                base_gradient = -1.0

            noise = np.random.randn() * 0.3
            gradient = base_gradient + noise

            detector.add_gradient(float(i), gradient, 10.0, 5.0)

            if i >= 10 and i % 5 == 0:
                level, confidence, flow_count = detector.detect_contention()

        # Should detect moderate to heavy contention (frequent sign changes)
        level, confidence, flow_count = detector.detect_contention()
        assert level in [ContentionLevel.MODERATE, ContentionLevel.HEAVY]
        assert flow_count >= 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
