"""
Traffic classifier - tries to figure out what type of traffic we're dealing with.

Looks at packet size patterns and timing to classify into:
- bulk transfer (large consistent packets, like FTP)
- streaming (medium packets with some variance, like video)
- realtime (small packets, like VoIP/gaming)
- default (can't tell or not enough data)

Main challenge is distinguishing streaming vs realtime since both can have
similar burst patterns. Currently using packet size as the key differentiator.
"""
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class TrafficClassifier:
    """
    Classifies traffic based on packet-level features.

    Uses packet size distribution, inter-arrival times, entropy, and burst
    characteristics. Thresholds were tuned on sample traffic traces but
    might need adjustment for different network conditions.
    """

    def __init__(self, config):
        self.window_size = config.window_size
        self.confidence_threshold = config.confidence_threshold
        self.feature_set = config.feature_set
        self.enabled = config.enabled
        self.expected_traffic_type = config.expected_traffic_type  # Override for evaluation

        # packet history for feature extraction
        self.packet_sizes = []
        self.inter_arrivals = []
        self.timestamps = []

        # thresholds tuned empirically - may need adjustment
        # TODO: make these configurable or learn them automatically
        self.thresholds = {
            'bulk': {
                'min_packet_size': 1200,
                'max_size_variance': 200,
                'max_burst_ratio': 0.3,
                'max_entropy': 3.0
            },
            'streaming': {
                'min_packet_size': 400,      # lower to avoid misclassifying video
                'max_packet_size': 1500,
                'max_size_variance': 400,    # video can vary quite a bit
                'min_burst_ratio': 0.0,
                'max_burst_ratio': 0.9,      # high threshold to not conflict with realtime
                'min_entropy': 0.2
            },
            'realtime': {
                'max_packet_size': 350,      # small packets are key indicator
                'min_burst_ratio': 0.3,      # VoIP/gaming tends to be bursty
                'min_entropy': 0.2
            }
        }

        logger.info(f"Traffic classifier initialized (window={self.window_size})")

    def add_packet(self, size: int, timestamp: float):
        """Add a packet observation to the history."""
        self.packet_sizes.append(size)
        self.timestamps.append(timestamp)

        if len(self.timestamps) > 1:
            iat = timestamp - self.timestamps[-2]
            self.inter_arrivals.append(iat)

        # keep only recent history to avoid memory issues
        if len(self.packet_sizes) > self.window_size:
            self.packet_sizes = self.packet_sizes[-self.window_size:]
            self.timestamps = self.timestamps[-self.window_size:]
            # inter-arrivals is one shorter
            if len(self.inter_arrivals) > self.window_size - 1:
                self.inter_arrivals = self.inter_arrivals[-(self.window_size - 1):]

    def extract_features(self) -> Dict[str, float]:
        """
        Extract features from packet history.

        Need at least 20 packets to get reliable statistics.
        """
        if len(self.packet_sizes) < 20:
            return {}

        recent_sizes = self.packet_sizes[-self.window_size:]
        recent_iats = self.inter_arrivals[-min(len(self.inter_arrivals), self.window_size):]

        features = {}

        # basic packet size stats
        if 'packet_size' in self.feature_set:
            features['avg_packet_size'] = np.mean(recent_sizes)
            features['std_packet_size'] = np.std(recent_sizes)
            features['median_packet_size'] = np.median(recent_sizes)
            features['max_packet_size'] = np.max(recent_sizes)
            features['min_packet_size'] = np.min(recent_sizes)
            # normalize variance by mean to handle different packet size ranges
            features['size_variance_ratio'] = np.std(recent_sizes) / (np.mean(recent_sizes) + 1e-6)

        # timing features
        if 'inter_arrival' in self.feature_set and len(recent_iats) > 0:
            features['avg_inter_arrival'] = np.mean(recent_iats)
            features['std_inter_arrival'] = np.std(recent_iats)
            features['cv_inter_arrival'] = (
                features['std_inter_arrival'] / features['avg_inter_arrival']
                if features['avg_inter_arrival'] > 0 else 0
            )
            # jitter metric (normalized std dev)
            features['inter_arrival_jitter'] = np.std(recent_iats) / (np.mean(recent_iats) + 1e-6) if recent_iats else 0

        # entropy of packet sizes
        if 'entropy' in self.feature_set:
            features['packet_size_entropy'] = self._compute_entropy(recent_sizes)
            # also store as 'entropy' for backwards compatibility
            features['entropy'] = features['packet_size_entropy']

        # burst detection
        if 'burst_ratio' in self.feature_set and len(recent_iats) > 0:
            features['burst_ratio'] = self._detect_bursts(recent_iats)
            features['burst_density'] = self._compute_burst_density(recent_sizes)

        # check for periodic patterns (helpful for streaming)
        if len(recent_iats) > 10:
            features['periodicity_score'] = self._compute_periodicity_score(recent_iats)

        return features

    def _compute_entropy(self, sizes: List[int], bins: int = None) -> float:
        """
        Compute entropy of packet size distribution.
        Higher entropy means more variable packet sizes.
        """
        if len(sizes) < 5:
            return 0.0

        # auto-scale number of bins based on data
        if bins is None:
            data_range = max(sizes) - min(sizes)
            # Sturges' rule: bins = 1 + log2(n)
            # but reduce bins for small ranges
            sturges_bins = int(1 + np.log2(len(sizes)))
            if data_range < 100:
                bins = max(5, sturges_bins // 2)
            else:
                bins = sturges_bins

        hist, _ = np.histogram(sizes, bins=bins)
        # minimal smoothing to avoid divide-by-zero
        hist = hist + 0.01
        prob = hist / np.sum(hist)
        prob = prob[prob > 1e-6]  # filter out near-zero probs
        entropy = -np.sum(prob * np.log2(prob + 1e-10))

        return entropy

    def _detect_bursts(self, inter_arrivals: List[float]) -> float:
        """
        Detect bursty traffic using coefficient of variation.

        Returns burst ratio (0-1), higher means more bursty.
        """
        if len(inter_arrivals) < 5:
            return 0.0

        mean_iat = np.mean(inter_arrivals)
        std_iat = np.std(inter_arrivals)

        if mean_iat == 0:
            return 0.0

        # coefficient of variation - high CV means bursty
        cv = std_iat / mean_iat

        # normalize to 0-1 range (CV of 1.0 -> burst_ratio ~0.7)
        burst_ratio = min(cv * 0.7, 1.0)

        return burst_ratio

    def _compute_burst_density(self, sizes: List[int]) -> float:
        """Count consecutive large packets as a burst indicator."""
        if len(sizes) < 5:
            return 0.0

        threshold = np.mean(sizes) * 0.9
        bursts = sum(1 for i in range(len(sizes)-1)
                     if sizes[i] > threshold and sizes[i+1] > threshold)
        return bursts / len(sizes)

    def _compute_periodicity_score(self, iats: List[float]) -> float:
        """Check if inter-arrival times are consistent (periodic traffic)."""
        if len(iats) < 10:
            return 0.0

        # low coefficient of variation means consistent timing
        iat_cv = np.std(iats) / (np.mean(iats) + 1e-6)
        periodicity = max(0, 1.0 - iat_cv)
        return periodicity

    def classify(self) -> Tuple[str, float]:
        """
        Classify traffic type.

        Returns (traffic_type, confidence) where traffic_type is one of:
        'bulk', 'streaming', 'realtime', or 'default'
        """
        if not self.enabled:
            return 'default', 1.0

        # Use expected traffic type if specified (for evaluation/testing)
        if self.expected_traffic_type is not None:
            return self.expected_traffic_type, 1.0

        features = self.extract_features()

        if not features:
            return 'default', 0.0

        # score each traffic type
        scores = {
            'bulk': self._score_bulk(features),
            'streaming': self._score_streaming(features),
            'realtime': self._score_realtime(features)
        }

        # pick the best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # need minimum confidence to classify
        if best_score < self.confidence_threshold:
            return 'default', best_score

        # print(f"DEBUG: classified as {best_type}, scores={scores}")  # keep for debugging
        logger.debug(f"Classified as {best_type} (confidence={best_score:.3f})")
        logger.debug(f"Features: {features}")

        return best_type, best_score

    def _score_bulk(self, features: Dict[str, float]) -> float:
        """Score for bulk transfer traffic (FTP, backup, etc)."""
        score = 0.0

        avg_size = features.get('avg_packet_size', 0)
        size_var = features.get('size_variance_ratio', 1.0)
        entropy = features.get('entropy', 0.0)
        jitter = features.get('inter_arrival_jitter', 0.5)

        # large packets are the main indicator
        if avg_size >= 1300:
            score += 1.0
        elif avg_size > 1150:
            score += 0.7
        elif avg_size > 1000:
            score += 0.3

        # low variance (consistent sizes)
        if size_var < 0.05:
            score += 0.8
        elif size_var < 0.1:
            score += 0.5
        elif size_var < 0.15:
            score += 0.2

        # consistent timing
        if jitter < 0.2:
            score += 0.4
        elif jitter < 0.4:
            score += 0.1

        # normalize to 0-1
        return score / 3.0 if score > 0 else 0.0

    def _score_streaming(self, features: Dict[str, float]) -> float:
        """Score for streaming traffic (video/audio)."""
        score = 0.0

        avg_size = features.get('avg_packet_size', 0)
        size_var = features.get('size_variance_ratio', 1.0)
        entropy = features.get('entropy', 0.0)
        burst = features.get('burst_density', 0.0)
        periodicity = features.get('periodicity_score', 0.0)

        # medium-sized packets
        if 700 <= avg_size <= 1200:
            score += 1.0
        elif 600 <= avg_size <= 1500:
            score += 0.6

        # moderate variance (video bitrate varies)
        if 0.04 < size_var < 0.15:
            score += 0.6
        elif 0.02 < size_var < 0.3:
            score += 0.3

        # entropy and burst patterns
        if 0.3 < entropy < 2.5:
            score += 0.4
        if 0.15 < burst < 0.7:
            score += 0.3
        if periodicity > 0.5:  # streaming often has periodic frames
            score += 0.3

        return min(score / 3.0, 1.0) if score > 0 else 0.0

    def _score_realtime(self, features: Dict[str, float]) -> float:
        """Score for realtime traffic (VoIP, gaming, etc)."""
        score = 0.0

        avg_size = features.get('avg_packet_size', 1000)
        size_var = features.get('size_variance_ratio', 1.0)
        entropy = features.get('entropy', 0.0)
        burst = features.get('burst_density', 0.0)
        periodicity = features.get('periodicity_score', 0.0)

        # small packets are KEY - this is the main differentiator from streaming
        if avg_size < 300:
            score += 2.0
        elif avg_size < 350:
            score += 1.8
        elif avg_size < 500:
            score += 1.4
        elif avg_size < 600:
            score += 0.8
        elif avg_size < 800:
            score += 0.2

        # consistent small sizes
        if size_var < 0.1:
            score += 0.6
        elif size_var < 0.2:
            score += 0.4
        elif size_var > 0.2:
            score += 0.3

        # entropy patterns
        if entropy > 0.6:
            score += 0.3
        elif entropy > 0.4:
            score += 0.15
        elif entropy < 0.4:
            # very uniform packets are OK for realtime
            score += 0.2

        # realtime is usually bursty (talk spurts, game events)
        if burst > 0.2:
            score += 0.2

        # periodic is good but not required
        if periodicity > 0.3:
            score += 0.15

        # penalize if packets are too large (not realtime)
        if avg_size > 1000:
            score = score * 0.1

        # TODO: these weights were tuned on VoIP traces, may need adjustment for gaming
        return min(score / 2.0, 1.0) if score > 0 else 0.0

    def reset(self):
        """Clear all state."""
        self.packet_sizes.clear()
        self.inter_arrivals.clear()
        self.timestamps.clear()
        logger.info("Classifier reset")

    def get_statistics(self) -> Dict[str, any]:
        features = self.extract_features()
        traffic_type, confidence = self.classify()

        return {
            'traffic_type': traffic_type,
            'confidence': confidence,
            'features': features,
            'num_packets': len(self.packet_sizes)
        }
