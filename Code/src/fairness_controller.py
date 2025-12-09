"""
Fairness controller - makes flows share bandwidth more equally

Implements fairness-augmented utility function to drive flows toward
equal bandwidth sharing without explicit coordination.

    controller = FairnessController(config)

    # Update fair share estimate
    controller.estimate_fair_share(

Basically adds a penalty when a flow uses more than its fair share.
Fair share is estimated as total_bandwidth / num_flows.
Penalty is proportional to how far off you are from fair share.

TODO: penalty weight (mu) should adapt based on how many flows are competing
        my_throughput=10.0,
        estimated_flow_count=3,
        aggregate_throughput=30.0
    )

    # Augment utility with fairness penalty
    utility_fair = controller.augment_utility(
        original_utility=50.0,
        current_rate=15.0,
        contention_level=ContentionLevel.MODERATE
    )
"""
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional
import logging

# Import ContentionLevel
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.contention_detector import ContentionLevel

logger = logging.getLogger(__name__)


@dataclass
class FairnessState:
    """
    Snapshot of fairness state at a point in time

    Attributes:
        timestamp: Time of snapshot (seconds)
        estimated_fair_share: Estimated fair share rate (Mbps)
        current_rate: Current sending rate (Mbps)
        deviation: Deviation from fair share (Mbps)
        penalty: Fairness penalty applied
        original_utility: Original utility value
        augmented_utility: Utility after fairness penalty
        mu_effective: Effective fairness weight used
        contention_level: Contention level at time
    """
    timestamp: float
    estimated_fair_share: float
    current_rate: float
    deviation: float
    penalty: float
    original_utility: float
    augmented_utility: float
    mu_effective: float
    contention_level: ContentionLevel

    def __repr__(self):
        return (f"FairnessState(t={self.timestamp:.2f}s, "
                f"rate={self.current_rate:.1f}, fair={self.estimated_fair_share:.1f}, "
                f"penalty={self.penalty:.2f}, μ={self.mu_effective:.2f})")


class FairnessController:
    """
    Augment utility with fairness penalty to drive toward equal sharing

    The controller estimates the fair share of bandwidth (total / n_flows)
    and penalizes deviations from this target. The penalty strength adapts
    based on contention level: no penalty when SOLO, strong penalty when HEAVY.

    Fair Share Estimation:
    - If aggregate throughput known: fair = aggregate / n
    - If aggregate unknown: fair ≈ my_throughput (assumes current is fair)
    - Smoothed with exponential moving average for stability

    Penalty Calculation:
    - penalty = μ · |current_rate - fair_share|
    - μ adapts: SOLO=0, LIGHT=0.5×base, MODERATE=base, HEAVY=1.5×base
    """

    def __init__(self, config):
        """
        Initialize fairness controller

        Args:
            config: FairnessControllerConfig with parameters
        """
        # Configuration
        self.mu = config.fairness_penalty_weight  # Base penalty weight
        self.adaptive_mu = config.adaptive_mu
        self.smoothing_factor = config.smoothing_factor

        # Fair share estimation
        self.estimated_fair_share = None  # Mbps
        self.fair_share_history = deque(maxlen=50)

        # State tracking
        self.current_rate = 0.0
        self.deviation_from_fair = 0.0
        self.last_penalty = 0.0

        # History (bounded for memory safety)
        self.state_history = []  # Will manually bound to 100 entries
        self.max_history_size = 100

        # Statistics
        self.total_augmentations = 0
        self.total_penalty_applied = 0.0

        logger.info(f"FairnessController initialized: μ={self.mu:.3f}, "
                   f"adaptive={self.adaptive_mu}")

    def estimate_fair_share(self, my_throughput: float,
                           estimated_flow_count: int,
                           aggregate_throughput: Optional[float] = None):
        """
        Estimate fair share of bandwidth

        Uses either aggregate throughput (if known) or heuristic estimation
        based on current throughput.

        Args:
            my_throughput: This flow's current throughput (Mbps)
            estimated_flow_count: Estimated number of competing flows
            aggregate_throughput: Total throughput of all flows (Mbps), if known

        Example:
            With 3 flows sharing 30 Mbps:
            fair_share = 30 / 3 = 10 Mbps

            If aggregate unknown and I'm getting 12 Mbps:
            Heuristic: assume 12 Mbps is approximately fair
        """
        if estimated_flow_count <= 0:
            logger.warning(f"Invalid flow count: {estimated_flow_count}, defaulting to 1")
            estimated_flow_count = 1

        # Calculate raw estimate
        if aggregate_throughput is not None and aggregate_throughput > 0:
            # Direct calculation: divide total by flow count
            raw_estimate = aggregate_throughput / estimated_flow_count
            logger.debug(f"Fair share from aggregate: {aggregate_throughput:.1f} / {estimated_flow_count} = {raw_estimate:.1f} Mbps")
        else:
            # Heuristic: assume current allocation is roughly fair
            # This is a conservative estimate that adapts over time
            raw_estimate = my_throughput
            logger.debug(f"Fair share heuristic: my_throughput={my_throughput:.1f} Mbps")

        # Smooth estimate with exponential moving average
        if self.estimated_fair_share is not None and len(self.fair_share_history) > 0:
            # EMA: new = α × raw + (1-α) × old
            alpha = self.smoothing_factor
            self.estimated_fair_share = (
                alpha * raw_estimate +
                (1 - alpha) * self.estimated_fair_share
            )
        else:
            # First estimate
            self.estimated_fair_share = raw_estimate

        # Add to history
        self.fair_share_history.append(self.estimated_fair_share)

        logger.debug(f"Estimated fair share: {self.estimated_fair_share:.2f} Mbps")

    def calculate_fairness_penalty(self, current_rate: float) -> float:
        """
        Calculate fairness penalty term

        Penalty is proportional to deviation from fair share:
        penalty = μ · |current_rate - fair_share|

        Args:
            current_rate: Current sending rate (Mbps)

        Returns:
            Penalty value (non-negative)

        Example:
            If fair_share=10 Mbps, current_rate=15 Mbps, μ=0.5:
            penalty = 0.5 × |15 - 10| = 2.5
        """
        if self.estimated_fair_share is None:
            # No fair share estimate yet
            return 0.0

        # Calculate deviation
        self.deviation_from_fair = abs(current_rate - self.estimated_fair_share)

        # Penalty = deviation (normalized)
        # Return raw deviation; μ is applied in augment_utility
        penalty = self.deviation_from_fair

        logger.debug(f"Fairness penalty: rate={current_rate:.2f}, "
                    f"fair={self.estimated_fair_share:.2f}, "
                    f"deviation={self.deviation_from_fair:.2f}")

        return penalty

    def augment_utility(self, original_utility: float,
                       current_rate: float,
                       contention_level: ContentionLevel,
                       timestamp: Optional[float] = None) -> float:
        """
        Add fairness penalty to utility

        Computes: U_fair = U_original − μ_eff · penalty

        The effective μ adapts based on contention level:
        - SOLO: μ = 0 (no fairness enforcement)
        - LIGHT: μ = 0.5 × base
        - MODERATE: μ = base
        - HEAVY: μ = 1.5 × base (strong enforcement)

        Args:
            original_utility: Original utility value
            current_rate: Current sending rate (Mbps)
            contention_level: Current contention level
            timestamp: Current time (seconds), optional

        Returns:
            Augmented utility with fairness penalty

        Example:
            original_utility = 50.0
            current_rate = 15.0 Mbps
            fair_share = 10.0 Mbps
            contention = MODERATE (μ_eff = 0.5)

            penalty = 0.5 × |15 - 10| = 2.5
            U_fair = 50.0 - 2.5 = 47.5
        """
        self.total_augmentations += 1
        self.current_rate = current_rate

        # Get effective μ based on contention
        if self.adaptive_mu:
            mu_effective = self._get_adaptive_mu(contention_level)
        else:
            mu_effective = self.mu

        # Calculate penalty
        raw_penalty = self.calculate_fairness_penalty(current_rate)
        weighted_penalty = mu_effective * raw_penalty

        # Augmented utility
        utility_fair = original_utility - weighted_penalty

        # Track statistics
        self.last_penalty = weighted_penalty
        self.total_penalty_applied += weighted_penalty

        # Record state
        if timestamp is not None:
            state = FairnessState(
                timestamp=timestamp,
                estimated_fair_share=self.estimated_fair_share if self.estimated_fair_share else 0.0,
                current_rate=current_rate,
                deviation=self.deviation_from_fair,
                penalty=weighted_penalty,
                original_utility=original_utility,
                augmented_utility=utility_fair,
                mu_effective=mu_effective,
                contention_level=contention_level
            )

            self.state_history.append(state)
            if len(self.state_history) > self.max_history_size:
                self.state_history.pop(0)

        logger.debug(f"Augmented utility: {original_utility:.2f} → {utility_fair:.2f} "
                    f"(penalty={weighted_penalty:.2f}, μ={mu_effective:.3f})")

        return utility_fair

    def _get_adaptive_mu(self, contention_level: ContentionLevel) -> float:
        """
        Adapt fairness penalty weight based on contention

        Higher contention → stronger fairness enforcement
        Lower contention → weaker/no fairness enforcement

        Args:
            contention_level: Current contention level

        Returns:
            Effective μ value

        Logic:
            SOLO: No fairness needed (μ = 0)
            LIGHT: Mild fairness (μ = 0.5 × base)
            MODERATE: Normal fairness (μ = base)
            HEAVY: Strong fairness (μ = 1.5 × base)
        """
        if contention_level == ContentionLevel.SOLO:
            return 0.0  # No penalty when alone
        elif contention_level == ContentionLevel.LIGHT:
            return self.mu * 0.5
        elif contention_level == ContentionLevel.MODERATE:
            return self.mu
        else:  # HEAVY
            return self.mu * 1.5  # Stronger enforcement

    def get_fairness_ratio(self) -> float:
        """
        Calculate current fairness ratio

        Returns:
            ratio = min(current_rate, fair_share) / max(current_rate, fair_share)
            Perfect fairness = 1.0, poor fairness → 0.0
        """
        if self.estimated_fair_share is None or self.estimated_fair_share == 0:
            return 1.0

        numerator = min(self.current_rate, self.estimated_fair_share)
        denominator = max(self.current_rate, self.estimated_fair_share)

        if denominator == 0:
            return 1.0

        return numerator / denominator

    def is_above_fair_share(self) -> bool:
        """
        Check if current rate is above fair share

        Returns:
            True if current rate > fair share
        """
        if self.estimated_fair_share is None:
            return False
        return self.current_rate > self.estimated_fair_share

    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics

        Returns:
            Dictionary with fairness controller statistics
        """
        # Current state
        fairness_ratio = self.get_fairness_ratio()
        above_fair = self.is_above_fair_share()

        # Penalty statistics
        avg_penalty = (self.total_penalty_applied / self.total_augmentations
                      if self.total_augmentations > 0 else 0.0)

        # History statistics
        if self.state_history:
            recent_states = self.state_history[-20:] if len(self.state_history) >= 20 else self.state_history
            recent_deviations = [s.deviation for s in recent_states]
            recent_penalties = [s.penalty for s in recent_states]
            recent_utilities = [s.augmented_utility for s in recent_states]

            avg_deviation = np.mean(recent_deviations)
            avg_recent_penalty = np.mean(recent_penalties)
            avg_utility = np.mean(recent_utilities)
        else:
            avg_deviation = 0.0
            avg_recent_penalty = 0.0
            avg_utility = 0.0

        return {
            'current': {
                'fair_share': self.estimated_fair_share if self.estimated_fair_share else 0.0,
                'current_rate': self.current_rate,
                'deviation': self.deviation_from_fair,
                'fairness_ratio': fairness_ratio,
                'above_fair_share': above_fair,
                'last_penalty': self.last_penalty
            },
            'statistics': {
                'total_augmentations': self.total_augmentations,
                'total_penalty': self.total_penalty_applied,
                'avg_penalty': avg_penalty
            },
            'recent': {
                'avg_deviation': avg_deviation,
                'avg_penalty': avg_recent_penalty,
                'avg_utility': avg_utility
            },
            'history': {
                'state_count': len(self.state_history),
                'fair_share_count': len(self.fair_share_history)
            }
        }

    def get_summary(self) -> str:
        """
        Get human-readable summary

        Returns:
            Formatted summary string
        """
        stats = self.get_statistics()

        summary = "Fairness Controller Summary:\n"
        summary += f"  Current State:\n"
        summary += f"    Fair Share: {stats['current']['fair_share']:.2f} Mbps\n"
        summary += f"    Current Rate: {stats['current']['current_rate']:.2f} Mbps\n"
        summary += f"    Deviation: {stats['current']['deviation']:.2f} Mbps\n"
        summary += f"    Fairness Ratio: {stats['current']['fairness_ratio']:.3f}\n"
        summary += f"    Above Fair: {'Yes' if stats['current']['above_fair_share'] else 'No'}\n"
        summary += f"    Last Penalty: {stats['current']['last_penalty']:.2f}\n"
        summary += f"  Statistics:\n"
        summary += f"    Total Augmentations: {stats['statistics']['total_augmentations']}\n"
        summary += f"    Avg Penalty: {stats['statistics']['avg_penalty']:.2f}\n"
        summary += f"  Recent (last 20):\n"
        summary += f"    Avg Deviation: {stats['recent']['avg_deviation']:.2f} Mbps\n"
        summary += f"    Avg Penalty: {stats['recent']['avg_penalty']:.2f}\n"
        summary += f"    Avg Utility: {stats['recent']['avg_utility']:.2f}\n"

        return summary

    def reset(self):
        """Reset controller state"""
        self.estimated_fair_share = None
        self.fair_share_history.clear()
        self.current_rate = 0.0
        self.deviation_from_fair = 0.0
        self.last_penalty = 0.0
        self.state_history.clear()
        self.total_augmentations = 0
        self.total_penalty_applied = 0.0

        logger.info("FairnessController reset")
