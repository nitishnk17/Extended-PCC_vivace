"""
Loss classifier - tries to figure out if packet loss is from congestion or wireless.

Basic idea: congestion loss happens when buffers fill up (RTT spikes), so there
should be correlation between loss and RTT. Wireless loss is random channel errors
so no correlation.

We track both over time and compute Pearson correlation:
- High correlation (>0.5) -> probably congestion
- Low correlation (<0.2) -> probably wireless
- In between -> mixed or uncertain

Returns p_wireless (0-1 probability) and confidence score.

TODO: thresholds might need tuning for different network types (cellular vs wifi)
FIXME: need more samples than we currently require for reliable correlation
"""
import numpy as np
from typing import Tuple, List, Dict, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class LossEvent:
    """Represents a loss event at a specific time"""
    def __init__(self, timestamp: float, loss_rate: float, packets_lost: int, packets_sent: int):
        self.timestamp = timestamp
        self.loss_rate = loss_rate
        self.packets_lost = packets_lost
        self.packets_sent = packets_sent
        self.is_loss = loss_rate > 0.0


class RTTEvent:
    """Represents an RTT measurement at a specific time"""
    def __init__(self, timestamp: float, rtt: float):
        self.timestamp = timestamp
        self.rtt = rtt


class LossClassifier:
    """
    Classifies loss as congestion vs wireless using correlation.

    Main challenge is having enough samples - need at least 20-30 events to get
    reliable correlation. Baseline RTT can drift so we track it with sliding window.
    """

    def __init__(self, config):
        # TODO: make thresholds configurable via config
        # Configuration
        self.window_size = config.window_size
        self.loss_threshold = config.loss_threshold
        self.rtt_inflation_margin = config.rtt_inflation_margin
        self.rtt_inflation_percent = config.rtt_inflation_percent
        self.correlation_threshold_high = config.correlation_threshold_high
        self.correlation_threshold_low = config.correlation_threshold_low
        self.min_events = config.min_events
        self.baseline_rtt_window = config.baseline_rtt_window

        # Event history (bounded queues)
        self.loss_events = deque(maxlen=self.window_size)
        self.rtt_events = deque(maxlen=self.window_size)

        # Baseline RTT tracking
        self.baseline_rtt = None
        self.baseline_rtt_samples = deque(maxlen=self.baseline_rtt_window)

        # Classification state
        self.last_p_wireless = 0.5  # Default: unknown
        self.last_confidence = 0.0
        self.last_correlation = 0.0
        self.classification_count = 0

        # Statistics
        self.total_loss_events = 0
        self.total_rtt_inflation_events = 0

        logger.info(f"LossClassifier initialized: window={self.window_size}, "
                   f"thresholds=[{self.correlation_threshold_low}, {self.correlation_threshold_high}]")

    def add_loss_event(self, timestamp: float, loss_rate: float,
                       packets_lost: int, packets_sent: int):
        """
        Add a loss event observation

        Args:
            timestamp: Time of observation
            loss_rate: Loss rate (0.0 to 1.0)
            packets_lost: Number of packets lost
            packets_sent: Number of packets sent
        """
        event = LossEvent(timestamp, loss_rate, packets_lost, packets_sent)
        self.loss_events.append(event)

        if event.is_loss and loss_rate >= self.loss_threshold:
            self.total_loss_events += 1
            logger.debug(f"Loss event: t={timestamp:.2f}, rate={loss_rate:.4f}")

    def add_rtt_sample(self, timestamp: float, rtt: float):
        """
        Add an RTT measurement

        Args:
            timestamp: Time of measurement
            rtt: RTT value in milliseconds
        """
        event = RTTEvent(timestamp, rtt)
        self.rtt_events.append(event)

        # Update baseline RTT tracking
        self.baseline_rtt_samples.append(rtt)

        # Update baseline RTT (minimum observed)
        if len(self.baseline_rtt_samples) >= 10:
            # Use 5th percentile as baseline (robust to outliers)
            self.baseline_rtt = np.percentile(list(self.baseline_rtt_samples), 5)

        # Check for inflation
        if self.baseline_rtt is not None:
            is_inflated = self._is_rtt_inflated(rtt)
            if is_inflated:
                self.total_rtt_inflation_events += 1
                logger.debug(f"RTT inflation: t={timestamp:.2f}, rtt={rtt:.2f}ms, "
                           f"baseline={self.baseline_rtt:.2f}ms")

    def classify(self) -> Tuple[float, float]:
        """
        Classify loss type based on correlation analysis

        Returns:
            Tuple of (p_wireless, confidence)
            - p_wireless: Estimated fraction of wireless loss (0.0 to 1.0)
            - confidence: Confidence in classification (0.0 to 1.0)
        """
        # Check if we have enough data
        if len(self.loss_events) < self.min_events or len(self.rtt_events) < self.min_events:
            logger.debug(f"Insufficient events for classification "
                        f"(loss={len(self.loss_events)}, rtt={len(self.rtt_events)})")
            return self.last_p_wireless, 0.0

        # Check if baseline RTT is established
        if self.baseline_rtt is None:
            logger.debug("Baseline RTT not established yet")
            return 0.5, 0.0  # Unknown

        # Calculate correlation
        correlation = self._calculate_correlation()
        self.last_correlation = correlation

        # Classify based on correlation
        p_wireless, confidence = self._classify_from_correlation(correlation)

        # Update state
        self.last_p_wireless = p_wireless
        self.last_confidence = confidence
        self.classification_count += 1

        logger.debug(f"Classification #{self.classification_count}: "
                    f"ρ={correlation:.3f}, p_wireless={p_wireless:.3f}, "
                    f"confidence={confidence:.3f}")

        return p_wireless, confidence

    def _calculate_correlation(self) -> float:
        """
        Calculate Pearson correlation between loss and RTT inflation

        Returns:
            Correlation coefficient (-1.0 to 1.0)
        """
        # Align events by timestamp (use closest matches)
        loss_indicators = []
        rtt_indicators = []

        # For each RTT event, find closest loss event
        for rtt_event in self.rtt_events:
            # Find closest loss event (within reasonable time window)
            closest_loss = self._find_closest_loss_event(rtt_event.timestamp)

            if closest_loss is not None:
                # Binary indicators
                loss_occurred = 1.0 if closest_loss.loss_rate >= self.loss_threshold else 0.0
                rtt_inflated = 1.0 if self._is_rtt_inflated(rtt_event.rtt) else 0.0

                loss_indicators.append(loss_occurred)
                rtt_indicators.append(rtt_inflated)

        # Need at least min_events aligned samples
        if len(loss_indicators) < self.min_events:
            logger.debug(f"Insufficient aligned events: {len(loss_indicators)}")
            return 0.0

        # Calculate Pearson correlation
        correlation = self._pearson_correlation(loss_indicators, rtt_indicators)

        return correlation

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """
        Calculate Pearson correlation coefficient

        Args:
            x: First variable
            y: Second variable

        Returns:
            Correlation coefficient (-1.0 to 1.0)
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        x_array = np.array(x)
        y_array = np.array(y)

        # Check for zero variance
        if np.std(x_array) == 0 or np.std(y_array) == 0:
            # No variance in one variable, correlation undefined
            # If both have no loss and no inflation, assume no correlation (wireless)
            if np.mean(x_array) == 0 and np.mean(y_array) == 0:
                return 0.0
            # If loss but no RTT inflation, assume wireless
            if np.mean(x_array) > 0 and np.mean(y_array) == 0:
                return 0.0
            # If RTT inflation but no loss, unusual but no correlation
            if np.mean(x_array) == 0 and np.mean(y_array) > 0:
                return 0.0
            return 0.0

        # Calculate correlation with warning suppression for division by zero
        try:
            # Suppress runtime warnings for invalid division (e.g., zero variance)
            with np.errstate(divide='ignore', invalid='ignore'):
                correlation = np.corrcoef(x_array, y_array)[0, 1]

            # Handle NaN (can occur with constant arrays)
            if np.isnan(correlation):
                return 0.0

            return correlation

        except Exception as e:
            logger.warning(f"Error calculating correlation: {e}")
            return 0.0

    def _classify_from_correlation(self, correlation: float) -> Tuple[float, float]:
        """
        Classify loss type from correlation value

        Args:
            correlation: Correlation coefficient

        Returns:
            Tuple of (p_wireless, confidence)
        """
        # Thresholds
        high_thresh = self.correlation_threshold_high
        low_thresh = self.correlation_threshold_low

        # Classification logic
        if correlation > high_thresh:
            # High correlation → Congestion loss
            p_wireless = 0.1
            confidence = min(1.0, (correlation - high_thresh) / (1.0 - high_thresh))
        elif correlation < low_thresh:
            # Low correlation → Wireless loss
            p_wireless = 0.9
            confidence = min(1.0, (low_thresh - correlation) / (low_thresh + 1.0))
        else:
            # Intermediate → Mixed loss
            # Linear interpolation between thresholds
            mid_point = (high_thresh + low_thresh) / 2.0

            if correlation > mid_point:
                # Closer to congestion
                p_wireless = 0.1 + 0.4 * (high_thresh - correlation) / (high_thresh - mid_point)
            else:
                # Closer to wireless
                p_wireless = 0.5 + 0.4 * (mid_point - correlation) / (mid_point - low_thresh)

            # Lower confidence for intermediate values
            confidence = 1.0 - abs(correlation - mid_point) / (high_thresh - low_thresh)

        # Clamp values
        p_wireless = np.clip(p_wireless, 0.0, 1.0)
        confidence = np.clip(confidence, 0.0, 1.0)

        return p_wireless, confidence

    def _is_rtt_inflated(self, rtt: float) -> bool:
        """
        Check if RTT is inflated compared to baseline

        Args:
            rtt: Current RTT value

        Returns:
            True if inflated, False otherwise
        """
        if self.baseline_rtt is None:
            return False

        # Use both absolute margin and percentage threshold
        absolute_margin = self.rtt_inflation_margin
        percentage_margin = self.baseline_rtt * self.rtt_inflation_percent

        # Inflated if exceeds either threshold
        margin = max(absolute_margin, percentage_margin)

        return rtt > (self.baseline_rtt + margin)

    def _find_closest_loss_event(self, timestamp: float,
                                 max_time_diff: float = 0.5) -> Optional[LossEvent]:
        """
        Find loss event closest to given timestamp

        Args:
            timestamp: Target timestamp
            max_time_diff: Maximum time difference to consider (seconds)

        Returns:
            Closest LossEvent or None if no event within max_time_diff
        """
        if not self.loss_events:
            return None

        closest_event = None
        min_diff = max_time_diff

        for event in self.loss_events:
            time_diff = abs(event.timestamp - timestamp)
            if time_diff < min_diff:
                min_diff = time_diff
                closest_event = event

        return closest_event

    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics

        Returns:
            Dictionary with statistics
        """
        # Loss event statistics
        if self.loss_events:
            loss_rates = [e.loss_rate for e in self.loss_events]
            avg_loss_rate = np.mean(loss_rates)
            max_loss_rate = np.max(loss_rates)
        else:
            avg_loss_rate = 0.0
            max_loss_rate = 0.0

        # RTT statistics
        if self.rtt_events:
            rtts = [e.rtt for e in self.rtt_events]
            avg_rtt = np.mean(rtts)
            min_rtt = np.min(rtts)
            max_rtt = np.max(rtts)
        else:
            avg_rtt = 0.0
            min_rtt = 0.0
            max_rtt = 0.0

        return {
            'classification': {
                'p_wireless': self.last_p_wireless,
                'confidence': self.last_confidence,
                'correlation': self.last_correlation,
                'count': self.classification_count
            },
            'events': {
                'loss_events': len(self.loss_events),
                'rtt_events': len(self.rtt_events),
                'total_loss_events': self.total_loss_events,
                'total_rtt_inflation_events': self.total_rtt_inflation_events
            },
            'loss': {
                'avg_loss_rate': avg_loss_rate,
                'max_loss_rate': max_loss_rate
            },
            'rtt': {
                'baseline_rtt': self.baseline_rtt if self.baseline_rtt else 0.0,
                'avg_rtt': avg_rtt,
                'min_rtt': min_rtt,
                'max_rtt': max_rtt
            }
        }

    def reset(self):
        """Reset classifier state"""
        self.loss_events.clear()
        self.rtt_events.clear()
        self.baseline_rtt_samples.clear()
        self.baseline_rtt = None
        self.last_p_wireless = 0.5
        self.last_confidence = 0.0
        self.last_correlation = 0.0
        self.classification_count = 0
        self.total_loss_events = 0
        self.total_rtt_inflation_events = 0

        logger.info("LossClassifier reset")

    def get_event_summary(self) -> str:
        """
        Get human-readable event summary

        Returns:
            Formatted string with event summary
        """
        stats = self.get_statistics()

        summary = f"Loss Classifier Summary:\n"
        summary += f"  Classification: p_wireless={stats['classification']['p_wireless']:.3f}, "
        summary += f"confidence={stats['classification']['confidence']:.3f}\n"
        summary += f"  Correlation: ρ={stats['classification']['correlation']:.3f}\n"
        summary += f"  Events: {stats['events']['loss_events']} loss, "
        summary += f"{stats['events']['rtt_events']} RTT\n"
        summary += f"  Baseline RTT: {stats['rtt']['baseline_rtt']:.2f}ms\n"

        return summary
