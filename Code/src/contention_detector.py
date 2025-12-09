"""
Contention Detector for Extension 3: Distributed Fairness

Detects presence and intensity of competing traffic through implicit signaling
by analyzing utility gradient oscillations. When multiple flows compete for
bandwidth, their rate adjustments cause oscillating gradients for each flow.

Key Insight:
- Solo flow: Monotonic gradient (consistently positive or negative)
- Competing flows: Oscillating gradient (frequent sign changes)
- More flows → Higher oscillation frequency and volatility

Algorithm:
1. Track utility gradient history over sliding window
2. Detect gradient sign changes and volatility
3. Estimate number of competing flows
4. Classify contention level (solo, light, moderate, heavy)

Features:
- Implicit signaling (no network modification)
- Confidence-weighted detection
- Robust to noise and transient oscillations
- Real-time contention tracking

Usage:
    detector = ContentionDetector(config)

    # Each monitor interval
    detector.add_gradient(timestamp, gradient, utility, rate)

    # Detect contention
    level, confidence, flow_count = detector.detect_contention()
"""
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Dict, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


@dataclass
class GradientEvent:
    """
    Single utility gradient observation

    Attributes:
        timestamp: Time of observation (seconds)
        gradient: Utility gradient (∇U)
        utility: Utility value at this point
        sending_rate: Sending rate (Mbps)
    """
    timestamp: float
    gradient: float
    utility: float
    sending_rate: float

    def __repr__(self):
        return (f"GradientEvent(t={self.timestamp:.2f}s, "
                f"∇U={self.gradient:.3f}, U={self.utility:.3f})")


class ContentionLevel(Enum):
    """
    Contention intensity classification

    SOLO: No competing traffic detected
    LIGHT: 1-2 competing flows
    MODERATE: 3-5 competing flows
    HEAVY: 6+ competing flows
    """
    SOLO = "solo"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"

    def __str__(self):
        return self.value


class ContentionDetector:
    """
    Detect competing traffic through gradient oscillation analysis

    The detector observes utility gradient behavior to infer the presence
    and number of competing flows. Key metrics:

    1. Sign Change Ratio: Fraction of gradient sign changes in window
       - Solo: <0.2 (monotonic behavior)
       - Light: 0.2-0.4 (occasional conflicts)
       - Moderate: 0.4-0.6 (frequent conflicts)
       - Heavy: >0.6 (constant conflicts)

    2. Volatility: Standard deviation of gradients
       - Higher volatility indicates more contention

    3. Flow Count Estimation: Heuristic based on oscillation patterns
    """
        # TODO: use more sophisticated flow count estimation (machine learning?)

    def __init__(self, config):
        """
        Initialize contention detector

        Args:
            config: ContentionDetectorConfig with parameters
        """
        # Configuration
        self.window_size = config.window_size  # Number of gradient samples
        self.sign_change_threshold = config.sign_change_threshold  # 0.6 default
        self.magnitude_threshold = config.magnitude_threshold  # Min gradient

        # Gradient history (bounded queue for memory safety)
        self.gradient_history = deque(maxlen=self.window_size)

        # Current state
        self.current_contention = ContentionLevel.SOLO
        self.estimated_flow_count = 1
        self.detection_confidence = 0.0

        # Statistics
        self.total_detections = 0
        self.detection_history = []  # Recent detection results

        logger.info(f"ContentionDetector initialized: window={self.window_size}, "
                   f"threshold={self.sign_change_threshold:.2f}")

    def add_gradient(self, timestamp: float, gradient: float,
                    utility: float, sending_rate: float):
        """
        Record gradient observation

        Args:
            timestamp: Current time (seconds)
            gradient: Utility gradient (∇U)
            utility: Utility value
            sending_rate: Current sending rate (Mbps)
        """
        event = GradientEvent(timestamp, gradient, utility, sending_rate)
        self.gradient_history.append(event)

        logger.debug(f"Added gradient: ∇U={gradient:.3f}, U={utility:.3f}")

    def detect_contention(self) -> Tuple[ContentionLevel, float, int]:
        """
        Detect contention level from gradient history

        Returns:
            Tuple of (contention_level, confidence, estimated_flow_count)

        Example:
            level, confidence, flows = detector.detect_contention()
            if level == ContentionLevel.HEAVY and confidence > 0.8:
                print(f"High contention detected: ~{flows} competing flows")
        """
        # Need sufficient data
        if len(self.gradient_history) < max(5, self.window_size // 4):
            return (ContentionLevel.SOLO, 0.0, 1)

        # Calculate sign change frequency
        sign_changes = self._count_sign_changes()
        total_transitions = len(self.gradient_history) - 1
        sign_change_ratio = sign_changes / total_transitions if total_transitions > 0 else 0.0

        # Calculate gradient volatility (normalized std deviation)
        volatility = self._calculate_volatility()

        # Estimate number of competing flows
        flow_count = self._estimate_flow_count(sign_change_ratio, volatility)

        # Classify contention level
        contention_level = self._classify_contention(flow_count)

        # Calculate confidence
        confidence = self._calculate_confidence(sign_change_ratio, volatility)

        # Update state
        self.current_contention = contention_level
        self.estimated_flow_count = flow_count
        self.detection_confidence = confidence
        self.total_detections += 1

        # Record detection
        self.detection_history.append({
            'timestamp': self.gradient_history[-1].timestamp if self.gradient_history else 0,
            'level': contention_level,
            'confidence': confidence,
            'flow_count': flow_count,
            'sign_change_ratio': sign_change_ratio,
            'volatility': volatility
        })

        # Limit history size
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-500:]

        logger.debug(f"Detection: {contention_level} (conf={confidence:.2f}, "
                    f"flows={flow_count}, scr={sign_change_ratio:.2f})")

        return (contention_level, confidence, flow_count)

    def _count_sign_changes(self) -> int:
        """
        Count gradient sign changes in window

        Only counts significant gradients (above magnitude threshold)
        to avoid noise from near-zero gradients.

        Returns:
            Number of sign changes
        """
        changes = 0

        for i in range(1, len(self.gradient_history)):
            prev_grad = self.gradient_history[i-1].gradient
            curr_grad = self.gradient_history[i].gradient

            # Only count significant gradients (filter noise)
            if (abs(prev_grad) > self.magnitude_threshold and
                abs(curr_grad) > self.magnitude_threshold):
                # Check for sign change (opposite signs)
                if prev_grad * curr_grad < 0:
                    changes += 1

        return changes

    def _calculate_volatility(self) -> float:
        """
        Calculate gradient volatility (normalized standard deviation)

        Returns:
            Volatility measure [0, ∞), typically [0, 5]
        """
        if len(self.gradient_history) < 2:
            return 0.0

        gradients = [e.gradient for e in self.gradient_history]

        # Standard deviation
        std = np.std(gradients)

        # Normalize by mean absolute gradient (avoid division by zero)
        mean_abs = np.mean(np.abs(gradients))
        if mean_abs > 1e-6:
            volatility = std / mean_abs
        else:
            volatility = std

        return float(volatility)

    def _estimate_flow_count(self, sign_change_ratio: float,
                            volatility: float) -> int:
        """
        Estimate number of competing flows

        Uses heuristic model based on sign change ratio and volatility.
        More flows → more oscillations → higher sign change ratio.

        IMPROVED: Raised SOLO threshold from 0.15 to 0.30 to reduce false positives.
        Single flows naturally have some oscillations from probing - don't confuse
        this with contention.

        Args:
            sign_change_ratio: Fraction of sign changes [0, 1]
            volatility: Gradient volatility measure

        Returns:
            Estimated flow count (1 to 32+)
        """
        # Base estimate from sign change ratio
        # IMPROVED: Higher threshold for SOLO (0.30 instead of 0.15)
        # Reduces false contention detection in single-flow scenarios
        if sign_change_ratio < 0.30:
            base_estimate = 1  # Solo
        elif sign_change_ratio < 0.45:
            base_estimate = 2  # Light contention
        elif sign_change_ratio < 0.65:
            base_estimate = 4  # Moderate contention
        else:
            base_estimate = 8  # Heavy contention

        # Refine with volatility (more volatile → more flows)
        # IMPROVED: Reduced volatility impact (1.5 max instead of 3.0)
        volatility_factor = max(1.0, min(1.5, volatility))

        # Combined estimate
        flow_count = int(base_estimate * volatility_factor)

        # Clamp to reasonable range [1, 32]
        flow_count = max(1, min(32, flow_count))

        return flow_count

    def _classify_contention(self, flow_count: int) -> ContentionLevel:
        """
        Classify contention level from estimated flow count

        Args:
            flow_count: Estimated number of flows

        Returns:
            ContentionLevel enum
        """
        if flow_count <= 1:
            return ContentionLevel.SOLO
        elif flow_count <= 2:
            return ContentionLevel.LIGHT
        elif flow_count <= 5:
            return ContentionLevel.MODERATE
        else:
            return ContentionLevel.HEAVY

    def _calculate_confidence(self, sign_change_ratio: float,
                             volatility: float) -> float:
        """
        Calculate detection confidence [0, 1]

        Higher confidence when:
        - More data available (closer to full window)
        - Clear patterns (high or low sign change ratio, not ambiguous)
        - Consistent volatility

        Args:
            sign_change_ratio: Fraction of sign changes
            volatility: Gradient volatility

        Returns:
            Confidence score [0, 1]
        """
        # Data confidence: more samples → higher confidence
        data_confidence = min(1.0, len(self.gradient_history) / self.window_size)

        # Pattern confidence: clear patterns → higher confidence
        # Distance from ambiguous center (0.5) indicates clarity
        # Far from 0.5 (either 0.0 or 1.0) → clear pattern → high confidence
        # Close to 0.5 → ambiguous → low confidence
        distance_from_ambiguous = abs(sign_change_ratio - 0.5)
        pattern_confidence = distance_from_ambiguous / 0.5
        pattern_confidence = min(1.0, pattern_confidence)

        # Volatility confidence: consistent volatility → higher confidence
        # Very low or very high volatility is clear, medium is ambiguous
        if volatility < 0.5 or volatility > 3.0:
            volatility_confidence = 1.0
        else:
            volatility_confidence = 0.5

        # Combined confidence (weighted average)
        confidence = (0.4 * data_confidence +
                     0.4 * pattern_confidence +
                     0.2 * volatility_confidence)

        return float(confidence)

    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics

        Returns:
            Dictionary with detection statistics
        """
        if len(self.detection_history) > 0:
            recent_detections = self.detection_history[-50:]  # Last 50

            stats = {
                'current': {
                    'level': self.current_contention.value,
                    'confidence': self.detection_confidence,
                    'flow_count': self.estimated_flow_count
                },
                'history': {
                    'total_detections': self.total_detections,
                    'window_size': self.window_size,
                    'samples_count': len(self.gradient_history)
                },
                'recent': {
                    'avg_flow_count': np.mean([d['flow_count'] for d in recent_detections]),
                    'avg_confidence': np.mean([d['confidence'] for d in recent_detections]),
                    'sign_change_ratio': np.mean([d['sign_change_ratio'] for d in recent_detections]),
                    'volatility': np.mean([d['volatility'] for d in recent_detections])
                }
            }
        else:
            stats = {
                'current': {
                    'level': 'solo',
                    'confidence': 0.0,
                    'flow_count': 1
                },
                'history': {
                    'total_detections': 0,
                    'window_size': self.window_size,
                    'samples_count': 0
                },
                'recent': {
                    'avg_flow_count': 0.0,
                    'avg_confidence': 0.0,
                    'sign_change_ratio': 0.0,
                    'volatility': 0.0
                }
            }

        return stats

    def get_summary(self) -> str:
        """
        Get human-readable summary

        Returns:
            Formatted summary string
        """
        stats = self.get_statistics()

        summary = "Contention Detector Summary:\n"
        summary += f"  Current State: {stats['current']['level']} "
        summary += f"(~{stats['current']['flow_count']} flows)\n"
        summary += f"  Confidence: {stats['current']['confidence']:.2f}\n"
        summary += f"  Total Detections: {stats['history']['total_detections']}\n"
        summary += f"  Window: {stats['history']['samples_count']}/{stats['history']['window_size']}\n"

        if stats['history']['total_detections'] > 0:
            summary += f"  Recent Average:\n"
            summary += f"    Flow Count: {stats['recent']['avg_flow_count']:.1f}\n"
            summary += f"    Confidence: {stats['recent']['avg_confidence']:.2f}\n"
            summary += f"    Sign Change Ratio: {stats['recent']['sign_change_ratio']:.2f}\n"
            summary += f"    Volatility: {stats['recent']['volatility']:.2f}\n"

        return summary

    def reset(self):
        """Reset detector state"""
        self.gradient_history.clear()
        self.current_contention = ContentionLevel.SOLO
        self.estimated_flow_count = 1
        self.detection_confidence = 0.0
        self.total_detections = 0
        self.detection_history.clear()

        logger.info("ContentionDetector reset")
