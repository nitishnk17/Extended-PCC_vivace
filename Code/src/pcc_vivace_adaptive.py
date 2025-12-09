"""
Production-Grade PCC Vivace with Extensions 1 & 2

Extension 1: Application-Aware Traffic Classification and Adaptive Utilities
Extension 2: Wireless Loss Differentiation

Extends baseline PCC Vivace with application-specific optimization
"""
import numpy as np
from typing import Dict, List, Optional
import logging

from .pcc_vivace_baseline import PccVivaceBaseline, MonitorInterval

logger = logging.getLogger(__name__)


class TrafficType(str):
    """Traffic types for classification"""
    BULK = "bulk"
    STREAMING = "streaming"
    REALTIME = "realtime"
    UNKNOWN = "unknown"


class PccVivaceAdaptive(PccVivaceBaseline):
    """
    PCC Vivace with Application-Aware and Loss-Aware Extensions

    Extension 1: Traffic Classification and Adaptive Utilities
    - Classifies traffic based on packet patterns
    - Selects utility function optimized for traffic type
    - Bulk: Maximize throughput
    - Streaming: Stable rate with minimal variance
    - Realtime: Minimize latency

    Extension 2: Wireless Loss Differentiation
    - Distinguishes random loss (wireless) from congestion loss
    - Adapts loss penalty coefficient based on loss type
    - Maintains responsiveness to congestion while tolerating wireless loss
    """

    def __init__(self, network, config):
        """
        Initialize adaptive PCC Vivace

        Args:
            network: NetworkSimulator instance
            config: Config object
        """
        super().__init__(network, config)

        # Extension 1: Traffic Classification
        # Check if traffic type is specified in config
        if hasattr(config, 'classifier') and config.classifier.expected_traffic_type:
            self.traffic_type = config.classifier.expected_traffic_type
            self.classification_confidence = 1.0
            logger.info(f"Using expected traffic type from config: {self.traffic_type}")
        else:
            self.traffic_type = TrafficType.UNKNOWN
            self.classification_confidence = 0.0

        self.classification_window = 50  # Classify after N packets
        self.packets_observed = 0

        # Packet features for classification
        self.packet_sizes = []
        self.inter_arrival_times = []
        self.last_packet_time = 0.0

        # Extension 1: Utility Function Parameters per Traffic Type
        self.utility_params = {
            TrafficType.BULK: {
                'alpha1': 1.0,  # Throughput weight
                'alpha2': 0.95,  # Loss penalty scaling (close to baseline for stability)
                'alpha3': 0.1,  # Latency penalty (low)
                'exponent': 0.9  # Same as baseline initially
            },
            TrafficType.STREAMING: {
                'gamma1': 1.0,  # Rate adequacy weight
                'gamma2': 5.0,  # Variance penalty (high)
                'gamma3': 0.8,  # Loss penalty scaling
                'min_rate': 3.0,  # Minimum acceptable rate (Mbps)
                'exponent': 1.0
            },
            TrafficType.REALTIME: {
                'beta1': 0.7,  # Throughput weight (LOWER - latency is priority)
                'beta2': 30.0,  # Latency penalty (VERY HIGH)
                'beta3': 1.0,  # Loss penalty scaling
                'latency_threshold': 30.0,  # Target latency (ms) - much lower than baseline 100ms
                'scale': 3.0,  # Sharper sigmoid transition
                'exponent': 0.85  # Slightly lower throughput exponent
            }
        }

        # Extension 2: Loss Differentiation
        self.loss_classification_enabled = True
        self.loss_window_size = 10  # MIs to consider
        self.loss_events = []  # (time, loss_occurred)
        self.rtt_inflation_events = []  # (time, rtt_inflated)
        self.wireless_loss_ratio = 0.0  # Estimated fraction of wireless loss

        # Adaptive loss coefficient
        self.base_loss_coefficient = 11.35  # Default Vivace value
        self.wireless_loss_coefficient = 5.0  # Reduced penalty for wireless loss

        # Throughput history for streaming variance calculation
        self.recent_throughputs = []
        self.throughput_history_size = 10

        logger.info(f"Adaptive PCC Vivace initialized with Extensions 1 & 2")
        logger.info(f"  Extension 1: Traffic classification enabled")
        logger.info(f"  Extension 2: Loss differentiation enabled")

    def run_monitor_interval(self, current_time: float) -> Dict:
        """
        Execute monitor interval with adaptive features

        Args:
            current_time: Current simulation time

        Returns:
            Dict with MI results
        """
        # Run base monitor interval
        results = super().run_monitor_interval(current_time)

        # Extension 1: Update classification if enough data
        if self.packets_observed >= self.classification_window and self.traffic_type == TrafficType.UNKNOWN:
            self._classify_traffic()

        # Extension 2: Update loss classification
        if self.loss_classification_enabled:
            self._update_loss_classification(results)

        # Track throughput for streaming variance
        if 'throughput' in results:
            self.recent_throughputs.append(results['throughput'])
            if len(self.recent_throughputs) > self.throughput_history_size:
                self.recent_throughputs.pop(0)

        return results

    def _calculate_utility(self, mi: MonitorInterval) -> float:
        """
        Calculate utility using traffic-aware function (Extension 1) with
        adaptive loss coefficient (Extension 2)

        Selects appropriate utility function based on classified traffic type
        and applies adaptive loss coefficient for wireless loss handling.

        Args:
            mi: MonitorInterval with metrics

        Returns:
            Utility value
        """
        # Extension 2: Get adaptive loss coefficient based on loss type
        effective_loss_coef = self._get_effective_loss_coefficient()

        # Extension 1: Select utility function based on traffic type
        if self.traffic_type == TrafficType.BULK:
            utility = self._utility_bulk(mi, effective_loss_coef)
            logger.debug(f"Using BULK utility: U={utility:.3f}")
        elif self.traffic_type == TrafficType.STREAMING:
            utility = self._utility_streaming(mi, effective_loss_coef)
            logger.debug(f"Using STREAMING utility: U={utility:.3f}")
        elif self.traffic_type == TrafficType.REALTIME:
            utility = self._utility_realtime(mi, effective_loss_coef)
            logger.debug(f"Using REALTIME utility: U={utility:.3f}")
        else:
            # Unknown traffic type, use baseline
            utility = self._utility_baseline(mi, effective_loss_coef)
            logger.debug(f"Using BASELINE utility: U={utility:.3f}")

        return utility

    def _utility_baseline(self, mi: MonitorInterval, loss_coef: float) -> float:
        """
        Baseline Vivace utility function with adaptive loss coefficient

        This is the standard PCC Vivace utility but with Extension 2's
        adaptive loss coefficient.
        """
        T = mi.throughput
        L = mi.avg_rtt
        loss = mi.loss_rate

        if T > 0:
            throughput_term = np.power(T, 0.9)
        else:
            throughput_term = 0.0

        # Latency penalty (baseline sigmoid)
        L_threshold = 100.0  # ms
        scale = 10.0
        latency_penalty = 1.0 / (1.0 + np.exp((L - L_threshold) / scale))

        # Loss penalty with adaptive coefficient
        loss_penalty = loss_coef * T * loss

        return throughput_term * latency_penalty - loss_penalty

    def _utility_bulk(self, mi: MonitorInterval, loss_coef: float) -> float:
        """
        Utility for bulk transfers: maximize throughput

        U_bulk = alpha1 * T^exp - alpha2 * T * loss - alpha3 * L

        Characteristics:
        - High throughput weight
        - Moderate loss penalty
        - Low latency penalty (can tolerate some buffering)
        """
        params = self.utility_params[TrafficType.BULK]

        T = mi.throughput
        L = mi.avg_rtt
        loss = mi.loss_rate

        if T > 0:
            throughput_term = params['alpha1'] * np.power(T, params['exponent'])
        else:
            throughput_term = 0.0

        loss_penalty = params['alpha2'] * loss_coef * T * loss
        latency_penalty = params['alpha3'] * (L / 100.0)  # Normalize by 100ms

        utility = throughput_term - loss_penalty - latency_penalty

        logger.debug(f"Bulk utility: T={T:.2f}, L={L:.2f}, loss={loss:.4f}, U={utility:.3f}")

        return utility

    def _utility_streaming(self, mi: MonitorInterval, loss_coef: float) -> float:
        """
        Utility for streaming: stable rate above threshold

        U_streaming = gamma1 * T * I(T > T_min) - gamma2 * Var(T)

        Characteristics:
        - Reward achieving minimum rate
        - Heavily penalize rate variance (causes rebuffering)
        - Moderate loss penalty
        """
        params = self.utility_params[TrafficType.STREAMING]

        T = mi.throughput
        loss = mi.loss_rate
        min_rate = params['min_rate']

        # Rate adequacy: reward if above minimum
        if T >= min_rate:
            rate_term = params['gamma1'] * T
        else:
            # Heavy penalty if below minimum rate
            rate_term = params['gamma1'] * T - 10.0 * (min_rate - T)

        # Variance penalty: penalize throughput fluctuations
        if len(self.recent_throughputs) >= 3:
            throughput_variance = np.var(self.recent_throughputs)
            variance_penalty = params['gamma2'] * throughput_variance
        else:
            variance_penalty = 0.0

        # Loss penalty (streaming is somewhat loss-tolerant)
        loss_penalty = params['gamma3'] * loss_coef * T * loss

        utility = rate_term - variance_penalty - loss_penalty

        logger.debug(f"Streaming utility: T={T:.2f}, var={variance_penalty:.2f}, U={utility:.3f}")

        return utility

    def _utility_realtime(self, mi: MonitorInterval, loss_coef: float) -> float:
        """
        Utility for realtime: minimize latency

        U_realtime = beta1 * T * S(L) - beta3 * T * loss

        Where S(L) is sigmoid heavily penalizing latency

        Characteristics:
        - Strong latency penalty
        - Moderate throughput reward
        - Moderate loss penalty
        """
        params = self.utility_params[TrafficType.REALTIME]

        T = mi.throughput
        L = mi.avg_rtt
        loss = mi.loss_rate

        # Throughput term (lower priority than baseline)
        if T > 0:
            throughput_term = params['beta1'] * np.power(T, params['exponent'])
        else:
            throughput_term = 0.0

        # AGGRESSIVE latency penalty with low threshold
        # Goal: Keep latency below 30ms (vs baseline's 100ms)
        latency_threshold = params['latency_threshold']
        scale = params['scale']  # Sharp transition
        # Sigmoid centered at threshold, steep penalty above it
        latency_penalty = params['beta2'] * (1.0 / (1.0 + np.exp(-(L - latency_threshold) / scale)))

        # Loss penalty
        loss_penalty = params['beta3'] * loss_coef * T * loss

        utility = throughput_term - latency_penalty - loss_penalty

        logger.debug(f"Realtime utility: T={T:.2f}, L={L:.2f} ms, "
                    f"lat_penalty={latency_penalty:.2f}, U={utility:.3f}")

        return utility

    def _classify_traffic(self):
        """
        Extension 1: Classify traffic based on observed patterns

        Features:
        - Packet size distribution (entropy, mean, variance)
        - Inter-arrival time patterns (mean, variance, burst ratio)

        Classification:
        - Bulk: Large packets (>1400 bytes), continuous flow, low IAT variance
        - Streaming: Medium packets (1000-1400), regular IAT, low IAT variance
        - Realtime: Small packets (<500), bursty, high IAT variance
        """
        if len(self.packet_sizes) < 10:
            return  # Need more samples

        # Compute features
        avg_size = np.mean(self.packet_sizes)
        size_variance = np.var(self.packet_sizes)

        if len(self.inter_arrival_times) > 1:
            avg_iat = np.mean(self.inter_arrival_times)
            iat_variance = np.var(self.inter_arrival_times)
            iat_cv = np.sqrt(iat_variance) / avg_iat if avg_iat > 0 else 0
        else:
            avg_iat = 0
            iat_cv = 0

        # Size entropy
        if len(self.packet_sizes) > 0:
            size_counts = np.bincount(np.array(self.packet_sizes, dtype=int))
            size_probs = size_counts[size_counts > 0] / len(self.packet_sizes)
            size_entropy = -np.sum(size_probs * np.log2(size_probs + 1e-10))
        else:
            size_entropy = 0

        # Classification logic
        confidence = 0.0

        if avg_size > 1400 and iat_cv < 0.3:
            # Large packets, regular timing → Bulk transfer
            self.traffic_type = TrafficType.BULK
            confidence = 0.85
            logger.info(f"Traffic classified as BULK (size={avg_size:.0f}, iat_cv={iat_cv:.2f})")

        elif 1000 <= avg_size <= 1400 and iat_cv < 0.5:
            # Medium packets, fairly regular → Streaming
            self.traffic_type = TrafficType.STREAMING
            confidence = 0.80
            logger.info(f"Traffic classified as STREAMING (size={avg_size:.0f}, iat_cv={iat_cv:.2f})")

        elif avg_size < 500 or iat_cv > 0.5:
            # Small packets or bursty → Realtime
            self.traffic_type = TrafficType.REALTIME
            confidence = 0.75
            logger.info(f"Traffic classified as REALTIME (size={avg_size:.0f}, iat_cv={iat_cv:.2f})")

        else:
            # Uncertain, use bulk as default
            self.traffic_type = TrafficType.BULK
            confidence = 0.50
            logger.info(f"Traffic classification uncertain, defaulting to BULK")

        self.classification_confidence = confidence

    def _update_loss_classification(self, mi_results: Dict):
        """
        Extension 2: Update loss type classification

        Distinguishes random loss (wireless) from congestion loss by analyzing
        correlation between loss events and RTT inflation.

        High correlation → congestion loss
        Low correlation → random (wireless) loss
        """
        if 'loss' not in mi_results or 'latency' not in mi_results:
            logger.debug("Loss classification skipped: missing keys")
            return

        loss = mi_results['loss']
        latency = mi_results['latency']

        # Detect loss event
        loss_occurred = loss > 0.01  # >1% loss

        # Detect RTT inflation
        base_rtt = self.network.delay_ms * 2  # Base RTT without queuing
        rtt_inflated = latency > base_rtt * 1.5  # >50% inflation

        logger.debug(f"Loss classification: loss={loss:.4f}, loss_occurred={loss_occurred}, "
                    f"latency={latency:.2f}, base_rtt={base_rtt:.2f}, rtt_inflated={rtt_inflated}")

        # Record events
        self.loss_events.append(loss_occurred)
        self.rtt_inflation_events.append(rtt_inflated)

        # Keep window size
        if len(self.loss_events) > self.loss_window_size:
            self.loss_events.pop(0)
            self.rtt_inflation_events.pop(0)

        # Compute correlation if enough samples
        if len(self.loss_events) >= 5:
            try:
                # Convert to numpy arrays
                loss_arr = np.array(self.loss_events, dtype=float)
                rtt_arr = np.array(self.rtt_inflation_events, dtype=float)

                # Check for variance
                loss_variance = np.std(loss_arr)
                rtt_variance = np.std(rtt_arr)

                # If loss occurs but RTT doesn't vary much, likely wireless loss
                if loss_variance > 0 and rtt_variance == 0:
                    # Loss without RTT inflation → wireless loss
                    self.wireless_loss_ratio = 0.9
                    logger.debug(f"Loss without RTT inflation → wireless_ratio={self.wireless_loss_ratio:.2f}")
                elif loss_variance > 0 and rtt_variance > 0:
                    # Both vary, compute correlation with warning suppression
                    with np.errstate(divide='ignore', invalid='ignore'):
                        correlation = np.corrcoef(loss_arr, rtt_arr)[0, 1]

                    if not np.isnan(correlation):
                        # High correlation → congestion loss
                        # Low correlation → wireless loss
                        if correlation > 0.5:
                            # Mostly congestion loss
                            self.wireless_loss_ratio = 0.1
                        elif correlation > 0.2:
                            # Mixed loss
                            self.wireless_loss_ratio = 0.5
                        else:
                            # Mostly wireless loss
                            self.wireless_loss_ratio = 0.9

                        logger.debug(f"Loss classification: correlation={correlation:.3f}, "
                                   f"wireless_ratio={self.wireless_loss_ratio:.2f}")
                    else:
                        # NaN correlation, assume some wireless loss
                        self.wireless_loss_ratio = 0.5
                else:
                    # No loss variance, can't classify
                    # Keep previous value or use conservative default
                    if len(self.loss_events) > 0 and np.mean(loss_arr) > 0:
                        # Consistent loss suggests wireless
                        self.wireless_loss_ratio = 0.7
                    else:
                        self.wireless_loss_ratio = 0.0

            except Exception as e:
                logger.warning(f"Loss classification error: {e}")
                self.wireless_loss_ratio = 0.0

    def _get_effective_loss_coefficient(self) -> float:
        """
        Extension 2: Get adaptive loss coefficient

        Adjusts loss penalty based on estimated wireless loss ratio:
        lambda_effective = lambda_base * (1 - 0.5 * p_wireless)

        This reduces loss penalty when loss is likely wireless, while
        maintaining responsiveness to congestion loss.
        """
        if not self.loss_classification_enabled:
            return self.base_loss_coefficient

        # Interpolate between base and wireless coefficients
        effective_coef = (self.base_loss_coefficient * (1 - self.wireless_loss_ratio) +
                         self.wireless_loss_coefficient * self.wireless_loss_ratio)

        return effective_coef

    def _run_single_mi(self, rate: float, current_time: float) -> MonitorInterval:
        """
        Override to collect packet features for classification

        Args:
            rate: Sending rate (Mbps)
            current_time: Current time

        Returns:
            Completed MonitorInterval
        """
        # Track packet timing for classification
        if self.last_packet_time > 0:
            iat = current_time - self.last_packet_time
            self.inter_arrival_times.append(iat)
            if len(self.inter_arrival_times) > 100:
                self.inter_arrival_times.pop(0)

        self.last_packet_time = current_time

        # Run base MI implementation
        mi = super()._run_single_mi(rate, current_time)

        # Collect packet size info
        if mi.packets_sent > 0:
            # Assuming fixed packet size for now, but could track actual sizes
            packet_size = mi.bytes_sent // mi.packets_sent if mi.packets_sent > 0 else 1500
            self.packet_sizes.append(packet_size)
            if len(self.packet_sizes) > 100:
                self.packet_sizes.pop(0)

            self.packets_observed += mi.packets_sent

        return mi

    def _compute_final_results(self) -> Dict:
        """Override to include extension-specific results"""
        results = super()._compute_final_results()

        # Add extension-specific info
        results['traffic_type'] = self.traffic_type
        results['classification_confidence'] = self.classification_confidence
        results['wireless_loss_ratio'] = self.wireless_loss_ratio
        results['extensions_active'] = {
            'traffic_classification': self.traffic_type != TrafficType.UNKNOWN,
            'loss_differentiation': self.loss_classification_enabled
        }

        return results
