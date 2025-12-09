"""
Adaptive Loss Coefficient for Extension 2: Wireless Loss Differentiation

Adjusts loss penalty based on wireless loss fraction to avoid over-reacting
to random wireless losses while maintaining responsiveness to congestion.

Key Concept:
- Congestion loss (λ_congestion): Full penalty, triggers rate reduction
- Wireless loss (λ_wireless): Reduced penalty, maintain rate
- Adaptive coefficient: Blend between two extremes based on p_wireless

Algorithm:
λ_effective = λ_base × (1 - p_wireless) + λ_wireless × p_wireless

Where:
- λ_base: Base loss coefficient (for pure congestion loss)
- λ_wireless: Reduced coefficient (for pure wireless loss)
- p_wireless: Fraction of wireless loss (0.0 to 1.0)
- λ_effective: Resulting coefficient to use

Features:
- Smooth transitions (exponential moving average)
- Confidence-weighted updates
- Conservative fallback when confidence is low
- Bounds checking and validation
"""
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AdaptiveLossCoefficient:
    """
    Adaptive loss coefficient that adjusts based on wireless loss fraction

    Reduces loss penalty for wireless losses while maintaining full penalty
    for congestion losses. Uses confidence-weighted exponential moving average
    for smooth transitions.

    Usage:
        alc = AdaptiveLossCoefficient(lambda_base=10.0, lambda_wireless=2.0)

        # Update with classification
        lambda_effective = alc.update(p_wireless=0.7, confidence=0.8)

        # Use in utility calculation
        utility = throughput - lambda_effective * loss_rate
    """

    def __init__(self, lambda_base: float = 10.0,
                 lambda_wireless: float = 2.0,
                 smoothing_factor: float = 0.1,
                 confidence_threshold: float = 0.3):
        """
        Initialize adaptive loss coefficient

        Args:
            lambda_base: Base loss coefficient (for congestion loss)
            lambda_wireless: Reduced coefficient (for wireless loss)
            smoothing_factor: EMA smoothing (0=no smoothing, 1=instant change)
            confidence_threshold: Minimum confidence to apply adaptation
        """
        # Configuration
        self.lambda_base = lambda_base
        self.lambda_wireless = lambda_wireless
        self.smoothing_factor = smoothing_factor
        self.confidence_threshold = confidence_threshold

        # Validate parameters
        if lambda_base <= 0:
            raise ValueError(f"lambda_base must be > 0, got {lambda_base}")
        if lambda_wireless <= 0:
            raise ValueError(f"lambda_wireless must be > 0, got {lambda_wireless}")
        if lambda_wireless > lambda_base:
            logger.warning(f"lambda_wireless ({lambda_wireless}) > lambda_base ({lambda_base}), "
                          "this will increase penalty for wireless loss")
        if not 0 <= smoothing_factor <= 1:
            raise ValueError(f"smoothing_factor must be in [0, 1], got {smoothing_factor}")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError(f"confidence_threshold must be in [0, 1], got {confidence_threshold}")

        # State
        self.current_lambda = lambda_base  # Start with base (conservative)
        self.target_lambda = lambda_base
        self.last_p_wireless = 0.0
        self.last_confidence = 0.0
        self.update_count = 0

        # Statistics
        self.history_lambda = []
        self.history_p_wireless = []
        self.history_confidence = []

        logger.info(f"AdaptiveLossCoefficient initialized: "
                   f"λ_base={lambda_base:.2f}, λ_wireless={lambda_wireless:.2f}, "
                   f"smoothing={smoothing_factor:.2f}")

    def update(self, p_wireless: float, confidence: float) -> float:
        """
        Update loss coefficient based on wireless loss fraction

        Args:
            p_wireless: Fraction of wireless loss (0.0 to 1.0)
            confidence: Confidence in classification (0.0 to 1.0)

        Returns:
            Effective loss coefficient (lambda_effective)
        """
        # Validate inputs
        p_wireless = np.clip(p_wireless, 0.0, 1.0)
        confidence = np.clip(confidence, 0.0, 1.0)

        # Save for statistics
        self.last_p_wireless = p_wireless
        self.last_confidence = confidence
        self.update_count += 1

        # Check if confidence is sufficient
        if confidence < self.confidence_threshold:
            # Low confidence: use current value (no change)
            logger.debug(f"Low confidence ({confidence:.3f}), keeping λ={self.current_lambda:.2f}")
            return self.current_lambda

        # Calculate target coefficient with confidence weighting
        # High confidence → use p_wireless fully
        # Low confidence → blend toward base (conservative)
        confidence_weight = (confidence - self.confidence_threshold) / (1.0 - self.confidence_threshold)
        confidence_weight = np.clip(confidence_weight, 0.0, 1.0)

        # Weighted p_wireless (lower when confidence is low)
        effective_p_wireless = p_wireless * confidence_weight

        # Calculate target lambda
        target = self._calculate_lambda(effective_p_wireless)
        self.target_lambda = target

        # Smooth transition using exponential moving average
        # λ_new = λ_old + α * (λ_target - λ_old)
        self.current_lambda = self.current_lambda + self.smoothing_factor * (target - self.current_lambda)

        # Update history
        self.history_lambda.append(self.current_lambda)
        self.history_p_wireless.append(p_wireless)
        self.history_confidence.append(confidence)

        # Limit history size
        if len(self.history_lambda) > 1000:
            self.history_lambda = self.history_lambda[-500:]
            self.history_p_wireless = self.history_p_wireless[-500:]
            self.history_confidence = self.history_confidence[-500:]

        logger.debug(f"Update #{self.update_count}: p_wireless={p_wireless:.3f}, "
                    f"conf={confidence:.3f}, λ={self.current_lambda:.3f} "
                    f"(target={target:.3f})")

        return self.current_lambda

    def _calculate_lambda(self, p_wireless: float) -> float:
        """
        Calculate loss coefficient from wireless fraction

        Uses linear interpolation:
        λ = λ_base × (1 - p_wireless) + λ_wireless × p_wireless

        Args:
            p_wireless: Fraction of wireless loss (0.0 to 1.0)

        Returns:
            Loss coefficient
        """
        lambda_effective = self.lambda_base * (1.0 - p_wireless) + self.lambda_wireless * p_wireless

        # Ensure bounds
        lambda_effective = np.clip(lambda_effective,
                                   min(self.lambda_wireless, self.lambda_base),
                                   max(self.lambda_wireless, self.lambda_base))

        return lambda_effective

    def get_current(self) -> float:
        """
        Get current loss coefficient

        Returns:
            Current effective lambda
        """
        return self.current_lambda

    def get_reduction_factor(self) -> float:
        """
        Get current reduction factor relative to base

        Returns:
            Ratio of current to base (1.0 = no reduction, 0.2 = 80% reduction)
        """
        return self.current_lambda / self.lambda_base

    def force_lambda(self, lambda_value: float):
        """
        Force loss coefficient to specific value (for testing/debugging)

        Args:
            lambda_value: Loss coefficient to use
        """
        if lambda_value <= 0:
            raise ValueError(f"lambda_value must be > 0, got {lambda_value}")

        self.current_lambda = lambda_value
        self.target_lambda = lambda_value

        logger.info(f"Forced λ={lambda_value:.2f}")

    def reset(self):
        """Reset to initial state"""
        self.current_lambda = self.lambda_base
        self.target_lambda = self.lambda_base
        self.last_p_wireless = 0.0
        self.last_confidence = 0.0
        self.update_count = 0

        self.history_lambda.clear()
        self.history_p_wireless.clear()
        self.history_confidence.clear()

        logger.info("AdaptiveLossCoefficient reset")

    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics

        Returns:
            Dictionary with statistics
        """
        if len(self.history_lambda) > 0:
            stats_lambda = {
                'current': self.current_lambda,
                'target': self.target_lambda,
                'mean': np.mean(self.history_lambda),
                'std': np.std(self.history_lambda),
                'min': np.min(self.history_lambda),
                'max': np.max(self.history_lambda)
            }

            stats_p_wireless = {
                'last': self.last_p_wireless,
                'mean': np.mean(self.history_p_wireless),
                'std': np.std(self.history_p_wireless)
            }

            stats_confidence = {
                'last': self.last_confidence,
                'mean': np.mean(self.history_confidence),
                'std': np.std(self.history_confidence)
            }
        else:
            stats_lambda = {
                'current': self.current_lambda,
                'target': self.target_lambda,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }

            stats_p_wireless = {
                'last': 0.0,
                'mean': 0.0,
                'std': 0.0
            }

            stats_confidence = {
                'last': 0.0,
                'mean': 0.0,
                'std': 0.0
            }

        return {
            'lambda': stats_lambda,
            'p_wireless': stats_p_wireless,
            'confidence': stats_confidence,
            'update_count': self.update_count,
            'reduction_factor': self.get_reduction_factor(),
            'config': {
                'lambda_base': self.lambda_base,
                'lambda_wireless': self.lambda_wireless,
                'smoothing_factor': self.smoothing_factor,
                'confidence_threshold': self.confidence_threshold
            }
        }

    def get_summary(self) -> str:
        """
        Get human-readable summary

        Returns:
            Formatted string with summary
        """
        stats = self.get_statistics()

        summary = "Adaptive Loss Coefficient Summary:\n"
        summary += f"  Current λ: {stats['lambda']['current']:.3f} (target: {stats['lambda']['target']:.3f})\n"
        summary += f"  Reduction: {stats['reduction_factor']:.1%} of base\n"
        summary += f"  Wireless fraction: {stats['p_wireless']['last']:.3f}\n"
        summary += f"  Confidence: {stats['confidence']['last']:.3f}\n"
        summary += f"  Updates: {stats['update_count']}\n"

        if stats['update_count'] > 0:
            summary += f"  λ range: [{stats['lambda']['min']:.2f}, {stats['lambda']['max']:.2f}]\n"
            summary += f"  λ mean: {stats['lambda']['mean']:.2f} ± {stats['lambda']['std']:.2f}\n"

        return summary
