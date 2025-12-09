"""
Meta-Controller for PCC Vivace Extension 1

The meta-controller is responsible for selecting the appropriate utility function
based on traffic classification and managing the decision-making process.

Key Responsibilities:
1. Select utility function based on traffic type
2. Handle classification uncertainty
3. Implement stability mechanisms (avoid frequent switching)
4. Log all decisions for analysis
5. Provide fallback mechanisms
"""
import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import deque
import logging

logger = logging.getLogger(__name__)


class MetaController:
    """
    Meta-Controller for adaptive utility function selection

    Selects appropriate utility function based on:
    - Traffic classification results
    - Classification confidence
    - Historical performance
    - Stability considerations
    """

    def __init__(self, config):
        """
        Initialize meta-controller

        Args:
            config: Config object with classifier and utilities configuration
        """
        self.config = config
        self.confidence_threshold = config.classifier.confidence_threshold

        # Current state
        self.current_utility_type = 'default'
        self.current_confidence = 1.0
        self.switches_count = 0

        # Stability mechanism - avoid rapid switching
        self.stability_window = 5  # Number of MIs to wait before switching
        self.classification_history = deque(maxlen=self.stability_window)

        # Performance tracking
        self.utility_performance = {
            'bulk': {'count': 0, 'avg_utility': 0.0},
            'streaming': {'count': 0, 'avg_utility': 0.0},
            'realtime': {'count': 0, 'avg_utility': 0.0},
            'default': {'count': 0, 'avg_utility': 0.0}
        }

        # Decision history for analysis
        self.decision_history = []

        logger.info(f"MetaController initialized (confidence_threshold={self.confidence_threshold})")

    def select_utility(self,
                      traffic_type: str,
                      confidence: float,
                      force: bool = False) -> Tuple[str, Dict[str, any]]:
        """
        Select utility function based on traffic classification

        Args:
            traffic_type: Classified traffic type ('bulk', 'streaming', 'realtime', 'default')
            confidence: Classification confidence (0.0-1.0)
            force: Force selection even with low confidence (for testing)

        Returns:
            (selected_utility_type, decision_metadata) tuple
        """
        decision_metadata = {
            'classified_as': traffic_type,
            'confidence': confidence,
            'previous_utility': self.current_utility_type,
            'switched': False,
            'reason': ''
        }

        # Handle low confidence
        if confidence < self.confidence_threshold and not force:
            logger.debug(f"Low confidence ({confidence:.3f} < {self.confidence_threshold}), "
                        f"keeping current utility: {self.current_utility_type}")
            decision_metadata['reason'] = 'low_confidence'
            decision_metadata['selected_utility'] = self.current_utility_type
            self._record_decision(decision_metadata)
            return self.current_utility_type, decision_metadata

        # Add to classification history
        self.classification_history.append((traffic_type, confidence))

        # Check if we should switch based on stability
        should_switch, reason = self._should_switch(traffic_type)

        if should_switch or force:
            # Switch to new utility
            old_utility = self.current_utility_type
            self.current_utility_type = traffic_type
            self.current_confidence = confidence

            if old_utility != traffic_type:
                self.switches_count += 1
                decision_metadata['switched'] = True
                logger.info(f"Switching utility: {old_utility} -> {traffic_type} "
                          f"(confidence={confidence:.3f}, reason={reason})")

            decision_metadata['reason'] = reason
        else:
            # Keep current utility
            decision_metadata['reason'] = f'stability_check_failed: {reason}'
            logger.debug(f"Not switching: {reason}")

        decision_metadata['selected_utility'] = self.current_utility_type
        self._record_decision(decision_metadata)

        return self.current_utility_type, decision_metadata

    def _should_switch(self, new_traffic_type: str) -> Tuple[bool, str]:
        """
        Determine if utility function should be switched

        Implements stability mechanism to avoid rapid switching:
        - Requires consistent classification over stability_window
        - Requires sufficient confidence
        - Can override if performance is poor

        Args:
            new_traffic_type: Newly classified traffic type

        Returns:
            (should_switch, reason) tuple
        """
        # If not enough history, don't switch
        if len(self.classification_history) < self.stability_window:
            return False, f'insufficient_history ({len(self.classification_history)}/{self.stability_window})'

        # Check consistency: majority of recent classifications agree
        recent_types = [t for t, c in self.classification_history]
        type_counts = {}
        for t in recent_types:
            type_counts[t] = type_counts.get(t, 0) + 1

        most_common = max(type_counts, key=type_counts.get)
        consistency_ratio = type_counts[most_common] / len(recent_types)

        # Require 60% consistency
        if consistency_ratio < 0.6:
            return False, f'inconsistent_classification (consistency={consistency_ratio:.2f})'

        # If consistent and different from current, switch
        if most_common != self.current_utility_type:
            return True, f'consistent_classification (consistency={consistency_ratio:.2f})'

        # Already using the right utility
        return True, 'already_correct_utility'

    def update_performance(self, utility_type: str, utility_value: float):
        """
        Update performance statistics for utility function

        Args:
            utility_type: Type of utility function used
            utility_value: Utility value achieved
        """
        if utility_type not in self.utility_performance:
            logger.warning(f"Unknown utility type: {utility_type}")
            return

        stats = self.utility_performance[utility_type]
        count = stats['count']

        # Running average
        stats['avg_utility'] = (stats['avg_utility'] * count + utility_value) / (count + 1)
        stats['count'] = count + 1

    def get_performance_summary(self) -> Dict[str, any]:
        """
        Get performance summary for all utility functions

        Returns:
            Dictionary with performance statistics
        """
        return {
            'current_utility': self.current_utility_type,
            'current_confidence': self.current_confidence,
            'switches_count': self.switches_count,
            'performance': self.utility_performance.copy(),
            'decision_count': len(self.decision_history)
        }

    def _record_decision(self, decision_metadata: Dict[str, any]):
        """
        Record decision for analysis

        Args:
            decision_metadata: Decision metadata dictionary
        """
        self.decision_history.append(decision_metadata)

        # Limit history size to prevent memory issues
        if len(self.decision_history) > 10000:
            self.decision_history = self.decision_history[-5000:]

    def get_decision_history(self) -> List[Dict[str, any]]:
        """
        Get decision history

        Returns:
            List of decision metadata dictionaries
        """
        return self.decision_history.copy()

    def reset(self):
        """Reset meta-controller state"""
        self.current_utility_type = 'default'
        self.current_confidence = 1.0
        self.switches_count = 0
        self.classification_history.clear()
        self.decision_history.clear()

        # Reset performance stats
        for stats in self.utility_performance.values():
            stats['count'] = 0
            stats['avg_utility'] = 0.0

        logger.info("MetaController reset")

    def force_utility(self, utility_type: str):
        """
        Force selection of specific utility function (for testing)

        Args:
            utility_type: Utility type to force
        """
        if utility_type not in ['bulk', 'streaming', 'realtime', 'default']:
            raise ValueError(f"Invalid utility type: {utility_type}")

        old_utility = self.current_utility_type
        self.current_utility_type = utility_type
        self.current_confidence = 1.0

        if old_utility != utility_type:
            self.switches_count += 1

        logger.info(f"Forced utility switch: {old_utility} -> {utility_type}")

    def get_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive statistics

        Returns:
            Dictionary with all statistics
        """
        # Classification consistency
        if len(self.classification_history) > 0:
            recent_types = [t for t, c in self.classification_history]
            type_counts = {}
            for t in recent_types:
                type_counts[t] = type_counts.get(t, 0) + 1
            most_common = max(type_counts, key=type_counts.get)
            consistency = type_counts[most_common] / len(recent_types)
        else:
            consistency = 0.0
            most_common = 'none'

        # Average confidence
        if len(self.classification_history) > 0:
            avg_confidence = np.mean([c for t, c in self.classification_history])
        else:
            avg_confidence = 0.0

        return {
            'current_utility': self.current_utility_type,
            'current_confidence': self.current_confidence,
            'switches_count': self.switches_count,
            'classification_consistency': consistency,
            'most_common_classification': most_common,
            'avg_confidence': avg_confidence,
            'history_length': len(self.classification_history),
            'performance': self.utility_performance.copy()
        }


class AdaptiveMetaController(MetaController):
    """
    Advanced meta-controller with adaptive learning

    Future enhancement: Can use reinforcement learning to:
    - Learn optimal switching thresholds
    - Adapt to application feedback
    - Optimize utility selection based on historical performance

    For now, inherits from MetaController with future extensibility in mind.
    """

    def __init__(self, config):
        super().__init__(config)

        # Placeholders for future RL components
        self.enable_learning = False
        self.learning_rate = 0.01
        self.reward_history = []

        logger.info("AdaptiveMetaController initialized (learning disabled for now)")

    def receive_feedback(self, reward: float):
        """
        Receive application-level feedback

        Future: Use for RL-based utility selection

        Args:
            reward: Application-level reward signal
        """
        self.reward_history.append(reward)

        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-500:]

        # Future: Update learning model based on reward
        if self.enable_learning:
            logger.debug(f"Received feedback reward: {reward:.3f}")

    def enable_adaptive_learning(self):
        """Enable adaptive learning (placeholder for future)"""
        self.enable_learning = True
        logger.info("Adaptive learning enabled")

    def disable_adaptive_learning(self):
        """Disable adaptive learning"""
        self.enable_learning = False
        logger.info("Adaptive learning disabled")
