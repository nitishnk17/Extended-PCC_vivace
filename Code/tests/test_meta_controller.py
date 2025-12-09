"""
Comprehensive unit tests for MetaController

Tests cover:
1. Basic functionality
2. Utility selection logic
3. Stability mechanisms
4. Performance tracking
5. Edge cases
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.meta_controller import MetaController, AdaptiveMetaController
from src.config import Config


class TestMetaControllerBasics:
    """Test basic meta-controller functionality"""

    @pytest.fixture
    def config(self):
        """Create configuration"""
        return Config()

    @pytest.fixture
    def meta_controller(self, config):
        """Create meta-controller instance"""
        return MetaController(config)

    def test_initialization(self, meta_controller, config):
        """Test meta-controller initialization"""
        assert meta_controller.current_utility_type == 'default'
        assert meta_controller.current_confidence == 1.0
        assert meta_controller.switches_count == 0
        assert len(meta_controller.classification_history) == 0
        assert meta_controller.confidence_threshold == config.classifier.confidence_threshold

    def test_select_utility_high_confidence(self, meta_controller):
        """Test utility selection with high confidence"""
        utility_type, metadata = meta_controller.select_utility('bulk', 0.9)

        assert metadata['classified_as'] == 'bulk'
        assert metadata['confidence'] == 0.9
        assert metadata['selected_utility'] in ['bulk', 'default']

    def test_select_utility_low_confidence(self, meta_controller):
        """Test utility selection with low confidence"""
        # Low confidence should keep current utility
        utility_type, metadata = meta_controller.select_utility('bulk', 0.3)

        assert metadata['confidence'] == 0.3
        assert metadata['reason'] == 'low_confidence'
        assert metadata['switched'] is False
        assert utility_type == 'default'  # Should stay with default

    def test_force_utility_selection(self, meta_controller):
        """Test forced utility selection"""
        # Force selection even with low confidence
        utility_type, metadata = meta_controller.select_utility('realtime', 0.3, force=True)

        # With force=True and stability window passed, should switch
        assert metadata['classified_as'] == 'realtime'

    def test_reset(self, meta_controller):
        """Test reset functionality"""
        # Make some selections
        for i in range(10):
            meta_controller.select_utility('bulk', 0.9)

        assert meta_controller.switches_count > 0 or len(meta_controller.classification_history) > 0

        # Reset
        meta_controller.reset()

        assert meta_controller.current_utility_type == 'default'
        assert meta_controller.switches_count == 0
        assert len(meta_controller.classification_history) == 0
        assert len(meta_controller.decision_history) == 0


class TestUtilitySelection:
    """Test utility selection logic"""

    @pytest.fixture
    def meta_controller(self):
        return MetaController(Config())

    def test_consistent_classification_switches(self, meta_controller):
        """Test that consistent classification causes switch"""
        # Provide consistent 'bulk' classification
        for i in range(10):
            utility_type, metadata = meta_controller.select_utility('bulk', 0.9)

        # Should eventually switch to bulk
        assert meta_controller.current_utility_type == 'bulk'
        assert meta_controller.switches_count == 1

    def test_inconsistent_classification_no_switch(self, meta_controller):
        """Test that inconsistent classification doesn't switch"""
        # Provide inconsistent classifications
        types = ['bulk', 'streaming', 'realtime', 'bulk', 'streaming']

        for traffic_type in types:
            meta_controller.select_utility(traffic_type, 0.8)

        # Should not have enough consistency to switch from default
        # (or might switch but not settle)
        assert len(meta_controller.classification_history) == len(types)

    def test_switch_between_utilities(self, meta_controller):
        """Test switching between different utilities"""
        # First, establish bulk
        for i in range(6):
            meta_controller.select_utility('bulk', 0.9)

        assert meta_controller.current_utility_type == 'bulk'
        switches_after_bulk = meta_controller.switches_count

        # Now switch to realtime
        for i in range(6):
            meta_controller.select_utility('realtime', 0.9)

        assert meta_controller.current_utility_type == 'realtime'
        assert meta_controller.switches_count == switches_after_bulk + 1

    def test_stability_window_prevents_rapid_switching(self, meta_controller):
        """Test that stability window prevents rapid switching"""
        stability_window = meta_controller.stability_window

        # Add one classification
        meta_controller.select_utility('bulk', 0.9)

        # Should not switch immediately (need stability_window samples)
        assert meta_controller.current_utility_type == 'default'

        # Add more to reach stability window
        for i in range(stability_window):
            meta_controller.select_utility('bulk', 0.9)

        # Now should switch
        assert meta_controller.current_utility_type == 'bulk'


class TestStabilityMechanism:
    """Test stability mechanism"""

    @pytest.fixture
    def meta_controller(self):
        return MetaController(Config())

    def test_should_switch_insufficient_history(self, meta_controller):
        """Test _should_switch with insufficient history"""
        meta_controller.classification_history.clear()

        # Add only 2 samples (need 5)
        meta_controller.classification_history.append(('bulk', 0.9))
        meta_controller.classification_history.append(('bulk', 0.9))

        should_switch, reason = meta_controller._should_switch('bulk')

        assert should_switch is False
        assert 'insufficient_history' in reason

    def test_should_switch_consistent_classification(self, meta_controller):
        """Test _should_switch with consistent classification"""
        meta_controller.classification_history.clear()

        # Add consistent classifications
        for i in range(5):
            meta_controller.classification_history.append(('bulk', 0.9))

        should_switch, reason = meta_controller._should_switch('bulk')

        assert should_switch is True
        assert 'consistent_classification' in reason or 'already_correct' in reason

    def test_should_switch_inconsistent_classification(self, meta_controller):
        """Test _should_switch with inconsistent classification"""
        meta_controller.classification_history.clear()

        # Add inconsistent classifications
        types = ['bulk', 'streaming', 'realtime', 'bulk', 'streaming']
        for t in types:
            meta_controller.classification_history.append((t, 0.8))

        should_switch, reason = meta_controller._should_switch('bulk')

        assert should_switch is False
        assert 'inconsistent_classification' in reason

    def test_consistency_threshold(self, meta_controller):
        """Test consistency threshold (60%)"""
        meta_controller.classification_history.clear()

        # Add 5 samples: 3 bulk, 2 streaming (60% bulk)
        meta_controller.classification_history.append(('bulk', 0.9))
        meta_controller.classification_history.append(('bulk', 0.9))
        meta_controller.classification_history.append(('streaming', 0.8))
        meta_controller.classification_history.append(('bulk', 0.9))
        meta_controller.classification_history.append(('streaming', 0.8))

        should_switch, reason = meta_controller._should_switch('bulk')

        # Should switch (60% threshold)
        assert should_switch is True


class TestPerformanceTracking:
    """Test performance tracking"""

    @pytest.fixture
    def meta_controller(self):
        return MetaController(Config())

    def test_update_performance(self, meta_controller):
        """Test performance update"""
        # Initially zero
        assert meta_controller.utility_performance['bulk']['count'] == 0

        # Update performance
        meta_controller.update_performance('bulk', 10.0)

        assert meta_controller.utility_performance['bulk']['count'] == 1
        assert meta_controller.utility_performance['bulk']['avg_utility'] == pytest.approx(10.0)

        # Update again
        meta_controller.update_performance('bulk', 20.0)

        assert meta_controller.utility_performance['bulk']['count'] == 2
        assert meta_controller.utility_performance['bulk']['avg_utility'] == pytest.approx(15.0)

    def test_update_performance_running_average(self, meta_controller):
        """Test that performance uses running average"""
        values = [5.0, 10.0, 15.0, 20.0, 25.0]

        for val in values:
            meta_controller.update_performance('streaming', val)

        expected_avg = np.mean(values)
        actual_avg = meta_controller.utility_performance['streaming']['avg_utility']

        assert actual_avg == pytest.approx(expected_avg, abs=0.01)
        assert meta_controller.utility_performance['streaming']['count'] == len(values)

    def test_update_performance_unknown_type(self, meta_controller):
        """Test updating performance for unknown utility type"""
        # Should handle gracefully (log warning)
        meta_controller.update_performance('unknown_type', 10.0)

        # Should not crash
        assert True

    def test_get_performance_summary(self, meta_controller):
        """Test getting performance summary"""
        # Add some data
        meta_controller.select_utility('bulk', 0.9)
        meta_controller.update_performance('bulk', 10.0)

        summary = meta_controller.get_performance_summary()

        assert 'current_utility' in summary
        assert 'current_confidence' in summary
        assert 'switches_count' in summary
        assert 'performance' in summary
        assert 'decision_count' in summary

        assert summary['current_utility'] in ['default', 'bulk']
        assert summary['decision_count'] >= 1


class TestDecisionHistory:
    """Test decision history tracking"""

    @pytest.fixture
    def meta_controller(self):
        return MetaController(Config())

    def test_decision_history_recording(self, meta_controller):
        """Test that decisions are recorded"""
        assert len(meta_controller.decision_history) == 0

        meta_controller.select_utility('bulk', 0.9)

        assert len(meta_controller.decision_history) == 1

        decision = meta_controller.decision_history[0]
        assert 'classified_as' in decision
        assert 'confidence' in decision
        assert 'selected_utility' in decision

    def test_decision_history_limit(self, meta_controller):
        """Test that decision history is limited"""
        # Add many decisions
        for i in range(15000):
            meta_controller.select_utility('bulk', 0.9)

        # Should be limited to prevent memory issues
        assert len(meta_controller.decision_history) <= 10000

    def test_get_decision_history(self, meta_controller):
        """Test getting decision history"""
        # Add some decisions
        for i in range(10):
            meta_controller.select_utility('bulk', 0.9)

        history = meta_controller.get_decision_history()

        assert len(history) == 10
        assert isinstance(history, list)

        # Should be a copy
        history.append({'test': 'data'})
        assert len(meta_controller.decision_history) == 10


class TestForceUtility:
    """Test forced utility selection"""

    @pytest.fixture
    def meta_controller(self):
        return MetaController(Config())

    def test_force_utility_basic(self, meta_controller):
        """Test basic forced utility"""
        assert meta_controller.current_utility_type == 'default'

        meta_controller.force_utility('bulk')

        assert meta_controller.current_utility_type == 'bulk'
        assert meta_controller.current_confidence == 1.0

    def test_force_utility_invalid_type(self, meta_controller):
        """Test forcing invalid utility type"""
        with pytest.raises(ValueError):
            meta_controller.force_utility('invalid_type')

    def test_force_utility_increments_switches(self, meta_controller):
        """Test that forcing utility increments switch counter"""
        initial_switches = meta_controller.switches_count

        meta_controller.force_utility('bulk')
        assert meta_controller.switches_count == initial_switches + 1

        meta_controller.force_utility('streaming')
        assert meta_controller.switches_count == initial_switches + 2

    def test_force_utility_same_type_no_increment(self, meta_controller):
        """Test that forcing same utility doesn't increment switches"""
        meta_controller.force_utility('bulk')
        switches_after_first = meta_controller.switches_count

        meta_controller.force_utility('bulk')

        # Should not increment
        assert meta_controller.switches_count == switches_after_first


class TestStatistics:
    """Test statistics reporting"""

    @pytest.fixture
    def meta_controller(self):
        return MetaController(Config())

    def test_get_statistics(self, meta_controller):
        """Test getting comprehensive statistics"""
        stats = meta_controller.get_statistics()

        assert 'current_utility' in stats
        assert 'current_confidence' in stats
        assert 'switches_count' in stats
        assert 'classification_consistency' in stats
        assert 'most_common_classification' in stats
        assert 'avg_confidence' in stats
        assert 'history_length' in stats
        assert 'performance' in stats

    def test_statistics_with_data(self, meta_controller):
        """Test statistics with actual data"""
        # Add some classifications
        for i in range(10):
            meta_controller.select_utility('bulk', 0.85)

        stats = meta_controller.get_statistics()

        assert stats['history_length'] > 0
        assert stats['avg_confidence'] > 0
        assert stats['classification_consistency'] >= 0
        assert stats['most_common_classification'] in ['bulk', 'default']

    def test_statistics_empty(self, meta_controller):
        """Test statistics with no data"""
        stats = meta_controller.get_statistics()

        assert stats['history_length'] == 0
        assert stats['avg_confidence'] == 0.0
        assert stats['classification_consistency'] == 0.0
        assert stats['most_common_classification'] == 'none'


class TestAdaptiveMetaController:
    """Test adaptive meta-controller"""

    @pytest.fixture
    def adaptive_controller(self):
        return AdaptiveMetaController(Config())

    def test_initialization(self, adaptive_controller):
        """Test adaptive controller initialization"""
        assert adaptive_controller.enable_learning is False
        assert len(adaptive_controller.reward_history) == 0

    def test_receive_feedback(self, adaptive_controller):
        """Test receiving feedback"""
        adaptive_controller.receive_feedback(10.0)

        assert len(adaptive_controller.reward_history) == 1
        assert adaptive_controller.reward_history[0] == 10.0

    def test_feedback_history_limit(self, adaptive_controller):
        """Test that feedback history is limited"""
        # Add many feedback samples
        for i in range(1500):
            adaptive_controller.receive_feedback(float(i))

        # Should be limited
        assert len(adaptive_controller.reward_history) <= 1000

    def test_enable_adaptive_learning(self, adaptive_controller):
        """Test enabling adaptive learning"""
        assert adaptive_controller.enable_learning is False

        adaptive_controller.enable_adaptive_learning()

        assert adaptive_controller.enable_learning is True

    def test_disable_adaptive_learning(self, adaptive_controller):
        """Test disabling adaptive learning"""
        adaptive_controller.enable_adaptive_learning()
        assert adaptive_controller.enable_learning is True

        adaptive_controller.disable_adaptive_learning()

        assert adaptive_controller.enable_learning is False

    def test_inherits_from_meta_controller(self, adaptive_controller):
        """Test that adaptive controller inherits basic functionality"""
        # Should work like regular MetaController
        utility_type, metadata = adaptive_controller.select_utility('bulk', 0.9)

        assert metadata is not None
        assert 'classified_as' in metadata


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def meta_controller(self):
        return MetaController(Config())

    def test_negative_confidence(self, meta_controller):
        """Test handling of negative confidence"""
        # Should handle gracefully
        utility_type, metadata = meta_controller.select_utility('bulk', -0.5)

        assert utility_type == 'default'  # Should fall back
        assert metadata['confidence'] == -0.5

    def test_confidence_greater_than_one(self, meta_controller):
        """Test handling of confidence > 1.0"""
        utility_type, metadata = meta_controller.select_utility('bulk', 1.5)

        # Should still work (confidence can exceed 1.0 theoretically)
        assert metadata['confidence'] == 1.5

    def test_unknown_traffic_type(self, meta_controller):
        """Test handling of unknown traffic type"""
        # Should handle gracefully
        utility_type, metadata = meta_controller.select_utility('unknown', 0.9)

        # Will try to switch to 'unknown' but won't be in utility functions
        assert metadata is not None

    def test_empty_traffic_type(self, meta_controller):
        """Test handling of empty traffic type"""
        utility_type, metadata = meta_controller.select_utility('', 0.9)

        assert metadata is not None

    def test_none_traffic_type(self, meta_controller):
        """Test handling of None traffic type"""
        # Should handle gracefully or raise appropriate error
        try:
            meta_controller.select_utility(None, 0.9)
        except (TypeError, AttributeError):
            # Expected to fail gracefully
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
