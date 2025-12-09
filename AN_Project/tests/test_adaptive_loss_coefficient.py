"""
Comprehensive unit tests for AdaptiveLossCoefficient

Tests cover:
1. Basic functionality (initialization, updates)
2. Coefficient calculation logic
3. Smoothing and confidence weighting
4. Edge cases and error handling
5. Statistics and reporting
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.adaptive_loss_coefficient import AdaptiveLossCoefficient


class TestInitialization:
    """Test AdaptiveLossCoefficient initialization"""

    def test_default_initialization(self):
        """Test initialization with default parameters"""
        alc = AdaptiveLossCoefficient()

        assert alc.lambda_base == 10.0
        assert alc.lambda_wireless == 2.0
        assert alc.smoothing_factor == 0.1
        assert alc.confidence_threshold == 0.3
        assert alc.current_lambda == 10.0
        assert alc.update_count == 0

    def test_custom_initialization(self):
        """Test initialization with custom parameters"""
        alc = AdaptiveLossCoefficient(
            lambda_base=15.0,
            lambda_wireless=3.0,
            smoothing_factor=0.2,
            confidence_threshold=0.4
        )

        assert alc.lambda_base == 15.0
        assert alc.lambda_wireless == 3.0
        assert alc.smoothing_factor == 0.2
        assert alc.confidence_threshold == 0.4

    def test_invalid_lambda_base(self):
        """Test validation of lambda_base"""
        with pytest.raises(ValueError):
            AdaptiveLossCoefficient(lambda_base=-1.0)

        with pytest.raises(ValueError):
            AdaptiveLossCoefficient(lambda_base=0.0)

    def test_invalid_lambda_wireless(self):
        """Test validation of lambda_wireless"""
        with pytest.raises(ValueError):
            AdaptiveLossCoefficient(lambda_wireless=-1.0)

        with pytest.raises(ValueError):
            AdaptiveLossCoefficient(lambda_wireless=0.0)

    def test_invalid_smoothing_factor(self):
        """Test validation of smoothing_factor"""
        with pytest.raises(ValueError):
            AdaptiveLossCoefficient(smoothing_factor=-0.1)

        with pytest.raises(ValueError):
            AdaptiveLossCoefficient(smoothing_factor=1.5)

    def test_invalid_confidence_threshold(self):
        """Test validation of confidence_threshold"""
        with pytest.raises(ValueError):
            AdaptiveLossCoefficient(confidence_threshold=-0.1)

        with pytest.raises(ValueError):
            AdaptiveLossCoefficient(confidence_threshold=1.5)

    def test_warning_wireless_greater_than_base(self, caplog):
        """Test warning when lambda_wireless > lambda_base"""
        # This should work but log a warning
        alc = AdaptiveLossCoefficient(lambda_base=5.0, lambda_wireless=10.0)

        assert alc.lambda_base == 5.0
        assert alc.lambda_wireless == 10.0
        # Warning should be logged


class TestBasicUpdate:
    """Test basic update functionality"""

    @pytest.fixture
    def alc(self):
        """Create AdaptiveLossCoefficient instance"""
        return AdaptiveLossCoefficient(lambda_base=10.0, lambda_wireless=2.0)

    def test_update_high_confidence_congestion(self, alc):
        """Test update with high confidence congestion loss"""
        # p_wireless=0.0 (pure congestion) with high confidence
        lambda_effective = alc.update(p_wireless=0.0, confidence=0.9)

        # Should be close to lambda_base
        assert lambda_effective <= 10.0
        assert lambda_effective > 8.0  # With smoothing, not instant

    def test_update_high_confidence_wireless(self, alc):
        """Test update with high confidence wireless loss"""
        # Run multiple updates to reach steady state (due to smoothing)
        # Use confidence=1.0 for full effect
        for _ in range(50):  # More iterations for convergence with smoothing_factor=0.1
            lambda_effective = alc.update(p_wireless=1.0, confidence=1.0)

        # Should be close to lambda_wireless
        assert lambda_effective == pytest.approx(2.0, abs=0.1)

    def test_update_low_confidence(self, alc):
        """Test update with low confidence"""
        initial_lambda = alc.current_lambda

        # Low confidence should keep current value
        lambda_effective = alc.update(p_wireless=1.0, confidence=0.1)

        assert lambda_effective == initial_lambda

    def test_update_increments_count(self, alc):
        """Test that update increments counter"""
        assert alc.update_count == 0

        alc.update(p_wireless=0.5, confidence=0.8)
        assert alc.update_count == 1

        alc.update(p_wireless=0.5, confidence=0.8)
        assert alc.update_count == 2

    def test_update_records_history(self, alc):
        """Test that update records history"""
        assert len(alc.history_lambda) == 0

        alc.update(p_wireless=0.5, confidence=0.8)

        assert len(alc.history_lambda) == 1
        assert len(alc.history_p_wireless) == 1
        assert len(alc.history_confidence) == 1


class TestCoefficientCalculation:
    """Test coefficient calculation logic"""

    @pytest.fixture
    def alc(self):
        return AdaptiveLossCoefficient(lambda_base=10.0, lambda_wireless=2.0)

    def test_pure_congestion_loss(self, alc):
        """Test coefficient for pure congestion (p_wireless=0.0)"""
        lambda_eff = alc._calculate_lambda(0.0)

        assert lambda_eff == pytest.approx(10.0)

    def test_pure_wireless_loss(self, alc):
        """Test coefficient for pure wireless (p_wireless=1.0)"""
        lambda_eff = alc._calculate_lambda(1.0)

        assert lambda_eff == pytest.approx(2.0)

    def test_mixed_loss_50_50(self, alc):
        """Test coefficient for mixed loss (50/50)"""
        lambda_eff = alc._calculate_lambda(0.5)

        # Should be midpoint: (10 + 2) / 2 = 6
        assert lambda_eff == pytest.approx(6.0)

    def test_mixed_loss_75_wireless(self, alc):
        """Test coefficient for 75% wireless loss"""
        lambda_eff = alc._calculate_lambda(0.75)

        # λ = 10 * 0.25 + 2 * 0.75 = 2.5 + 1.5 = 4.0
        assert lambda_eff == pytest.approx(4.0)

    def test_linear_interpolation(self, alc):
        """Test that interpolation is linear"""
        values = [alc._calculate_lambda(p) for p in np.linspace(0, 1, 11)]

        # Check monotonicity (decreasing)
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

        # Check endpoints
        assert values[0] == pytest.approx(10.0)
        assert values[-1] == pytest.approx(2.0)


class TestSmoothing:
    """Test smoothing mechanism"""

    def test_smoothing_gradual_change(self):
        """Test that smoothing causes gradual changes"""
        alc = AdaptiveLossCoefficient(
            lambda_base=10.0,
            lambda_wireless=2.0,
            smoothing_factor=0.1  # Slow smoothing
        )

        # Start with congestion (λ=10)
        assert alc.current_lambda == pytest.approx(10.0)

        # Switch to wireless (target λ=2)
        lambda_1 = alc.update(p_wireless=1.0, confidence=1.0)

        # Should not reach target immediately
        assert lambda_1 < 10.0
        assert lambda_1 > 2.0

        # Continue updating
        lambda_2 = alc.update(p_wireless=1.0, confidence=1.0)

        # Should continue decreasing
        assert lambda_2 < lambda_1

    def test_smoothing_instant_with_factor_1(self):
        """Test that smoothing_factor=1.0 gives instant change"""
        alc = AdaptiveLossCoefficient(
            lambda_base=10.0,
            lambda_wireless=2.0,
            smoothing_factor=1.0  # Instant change
        )

        lambda_eff = alc.update(p_wireless=1.0, confidence=1.0)

        # Should reach target immediately (within small tolerance)
        assert lambda_eff == pytest.approx(2.0, abs=0.01)

    def test_smoothing_no_change_with_factor_0(self):
        """Test that smoothing_factor=0.0 prevents change"""
        alc = AdaptiveLossCoefficient(
            lambda_base=10.0,
            lambda_wireless=2.0,
            smoothing_factor=0.0  # No smoothing
        )

        lambda_eff = alc.update(p_wireless=1.0, confidence=1.0)

        # Should not change
        assert lambda_eff == pytest.approx(10.0)

    def test_convergence_to_target(self):
        """Test that smoothing eventually converges to target"""
        alc = AdaptiveLossCoefficient(
            lambda_base=10.0,
            lambda_wireless=2.0,
            smoothing_factor=0.2
        )

        # Run many updates
        for _ in range(50):
            lambda_eff = alc.update(p_wireless=1.0, confidence=1.0)

        # Should converge to target
        assert lambda_eff == pytest.approx(2.0, abs=0.01)


class TestConfidenceWeighting:
    """Test confidence weighting"""

    @pytest.fixture
    def alc(self):
        return AdaptiveLossCoefficient(
            lambda_base=10.0,
            lambda_wireless=2.0,
            smoothing_factor=1.0,  # Instant for easier testing
            confidence_threshold=0.3
        )

    def test_below_threshold_no_change(self, alc):
        """Test that confidence below threshold causes no change"""
        initial = alc.current_lambda

        alc.update(p_wireless=1.0, confidence=0.2)

        assert alc.current_lambda == initial

    def test_at_threshold_minimal_change(self, alc):
        """Test that confidence at threshold causes minimal change"""
        initial = alc.current_lambda

        lambda_eff = alc.update(p_wireless=1.0, confidence=0.3)

        # Should change slightly (confidence_weight = 0)
        # Actually at threshold, weight = 0, so no change expected
        assert lambda_eff == initial

    def test_above_threshold_proportional_change(self, alc):
        """Test that confidence above threshold causes proportional change"""
        # Confidence = 0.65 (midpoint between 0.3 and 1.0)
        # confidence_weight = (0.65 - 0.3) / (1.0 - 0.3) = 0.5
        lambda_eff = alc.update(p_wireless=1.0, confidence=0.65)

        # effective_p_wireless = 1.0 * 0.5 = 0.5
        # target = 10 * 0.5 + 2 * 0.5 = 6.0
        assert lambda_eff == pytest.approx(6.0, abs=0.1)

    def test_full_confidence_full_change(self, alc):
        """Test that full confidence uses full p_wireless"""
        lambda_eff = alc.update(p_wireless=1.0, confidence=1.0)

        # Should use full p_wireless (1.0)
        assert lambda_eff == pytest.approx(2.0, abs=0.1)


class TestGetters:
    """Test getter methods"""

    @pytest.fixture
    def alc(self):
        return AdaptiveLossCoefficient(lambda_base=10.0, lambda_wireless=2.0)

    def test_get_current(self, alc):
        """Test get_current method"""
        assert alc.get_current() == 10.0

        alc.current_lambda = 5.0
        assert alc.get_current() == 5.0

    def test_get_reduction_factor_no_reduction(self, alc):
        """Test reduction factor with no reduction"""
        # At base: 10.0 / 10.0 = 1.0
        assert alc.get_reduction_factor() == pytest.approx(1.0)

    def test_get_reduction_factor_50_percent(self, alc):
        """Test reduction factor at 50%"""
        alc.current_lambda = 5.0

        # 5.0 / 10.0 = 0.5
        assert alc.get_reduction_factor() == pytest.approx(0.5)

    def test_get_reduction_factor_80_percent(self, alc):
        """Test reduction factor at 80%"""
        alc.current_lambda = 2.0

        # 2.0 / 10.0 = 0.2
        assert alc.get_reduction_factor() == pytest.approx(0.2)


class TestForceAndReset:
    """Test force and reset functionality"""

    @pytest.fixture
    def alc(self):
        return AdaptiveLossCoefficient(lambda_base=10.0, lambda_wireless=2.0)

    def test_force_lambda(self, alc):
        """Test forcing lambda to specific value"""
        alc.force_lambda(5.0)

        assert alc.current_lambda == 5.0
        assert alc.target_lambda == 5.0

    def test_force_lambda_invalid(self, alc):
        """Test forcing invalid lambda"""
        with pytest.raises(ValueError):
            alc.force_lambda(0.0)

        with pytest.raises(ValueError):
            alc.force_lambda(-1.0)

    def test_reset(self, alc):
        """Test reset functionality"""
        # Make some updates
        for _ in range(10):
            alc.update(p_wireless=0.8, confidence=0.9)

        # Verify state changed
        assert alc.current_lambda != 10.0
        assert alc.update_count == 10
        assert len(alc.history_lambda) == 10

        # Reset
        alc.reset()

        # Verify reset to initial state
        assert alc.current_lambda == 10.0
        assert alc.target_lambda == 10.0
        assert alc.update_count == 0
        assert len(alc.history_lambda) == 0


class TestStatistics:
    """Test statistics reporting"""

    @pytest.fixture
    def alc(self):
        return AdaptiveLossCoefficient(lambda_base=10.0, lambda_wireless=2.0)

    def test_get_statistics_empty(self, alc):
        """Test statistics with no updates"""
        stats = alc.get_statistics()

        assert 'lambda' in stats
        assert 'p_wireless' in stats
        assert 'confidence' in stats
        assert 'update_count' in stats
        assert 'reduction_factor' in stats
        assert 'config' in stats

        assert stats['update_count'] == 0
        assert stats['lambda']['current'] == 10.0

    def test_get_statistics_with_data(self, alc):
        """Test statistics with updates"""
        # Add some updates
        for i in range(20):
            p = i / 20.0
            alc.update(p_wireless=p, confidence=0.8)

        stats = alc.get_statistics()

        assert stats['update_count'] == 20
        assert stats['lambda']['mean'] > 0
        assert stats['lambda']['std'] >= 0
        assert stats['p_wireless']['mean'] >= 0
        assert stats['confidence']['mean'] > 0

    def test_get_summary(self, alc):
        """Test human-readable summary"""
        alc.update(p_wireless=0.5, confidence=0.8)

        summary = alc.get_summary()

        assert isinstance(summary, str)
        assert 'Adaptive Loss Coefficient' in summary
        assert 'Current λ' in summary
        assert 'Reduction' in summary


class TestHistoryLimit:
    """Test history size limiting"""

    def test_history_limited_to_1000(self):
        """Test that history is limited to prevent memory issues"""
        alc = AdaptiveLossCoefficient()

        # Add many updates
        for i in range(1500):
            alc.update(p_wireless=0.5, confidence=0.8)

        # History should be limited
        assert len(alc.history_lambda) <= 1000
        assert len(alc.history_p_wireless) <= 1000
        assert len(alc.history_confidence) <= 1000

        # Should keep recent data
        assert alc.history_lambda[-1] == alc.current_lambda


class TestInputValidation:
    """Test input validation and clamping"""

    @pytest.fixture
    def alc(self):
        return AdaptiveLossCoefficient()

    def test_negative_p_wireless_clamped(self, alc):
        """Test that negative p_wireless is clamped to 0"""
        lambda_eff = alc.update(p_wireless=-0.5, confidence=0.8)

        # Should treat as 0.0
        assert alc.last_p_wireless == 0.0

    def test_p_wireless_greater_than_one_clamped(self, alc):
        """Test that p_wireless > 1.0 is clamped"""
        lambda_eff = alc.update(p_wireless=1.5, confidence=0.8)

        # Should treat as 1.0
        assert alc.last_p_wireless == 1.0

    def test_negative_confidence_clamped(self, alc):
        """Test that negative confidence is clamped"""
        lambda_eff = alc.update(p_wireless=0.5, confidence=-0.5)

        # Should treat as 0.0
        assert alc.last_confidence == 0.0

    def test_confidence_greater_than_one_clamped(self, alc):
        """Test that confidence > 1.0 is clamped"""
        lambda_eff = alc.update(p_wireless=0.5, confidence=1.5)

        # Should treat as 1.0
        assert alc.last_confidence == 1.0


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""

    def test_gradual_transition_congestion_to_wireless(self):
        """Test gradual transition from congestion to wireless"""
        alc = AdaptiveLossCoefficient(
            lambda_base=10.0,
            lambda_wireless=2.0,
            smoothing_factor=0.2
        )

        # Start with congestion
        for _ in range(10):
            alc.update(p_wireless=0.0, confidence=1.0)

        lambda_congestion = alc.current_lambda
        assert lambda_congestion > 8.0

        # Transition to wireless (use confidence=1.0 for full effect)
        for _ in range(30):
            alc.update(p_wireless=1.0, confidence=1.0)

        lambda_wireless = alc.current_lambda
        assert lambda_wireless < 3.0

        # Should be smooth transition
        assert lambda_congestion > lambda_wireless

    def test_fluctuating_classification(self):
        """Test with fluctuating classification"""
        alc = AdaptiveLossCoefficient(
            lambda_base=10.0,
            lambda_wireless=2.0,
            smoothing_factor=0.1
        )

        # Alternate between congestion and wireless
        for i in range(20):
            p_wireless = 0.0 if i % 2 == 0 else 1.0
            alc.update(p_wireless=p_wireless, confidence=0.8)

        # Lambda should be somewhere in between (smoothing helps)
        assert 2.0 < alc.current_lambda < 10.0

    def test_low_confidence_maintains_conservative(self):
        """Test that low confidence maintains conservative behavior"""
        alc = AdaptiveLossCoefficient(
            lambda_base=10.0,
            lambda_wireless=2.0,
            smoothing_factor=1.0
        )

        # Many updates with low confidence
        for _ in range(20):
            alc.update(p_wireless=1.0, confidence=0.1)

        # Should stay at base (conservative)
        assert alc.current_lambda == pytest.approx(10.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
