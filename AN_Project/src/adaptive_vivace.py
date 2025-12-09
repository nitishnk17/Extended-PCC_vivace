"""
Adaptive PCC Vivace with Extensions
- Traffic classification
- Application-aware utility functions
- Loss differentiation
"""
import numpy as np
from typing import Dict, Optional, List
import logging

from .baseline_vivace import BaselineVivace
from .traffic_classifier import TrafficClassifier
from .utility_bank import UtilityFunctionBank

logger = logging.getLogger(__name__)


class AdaptiveVivace(BaselineVivace):
    """
    Adaptive PCC Vivace with application-aware extensions
    
    Extensions:
    1. Traffic classification (bulk, streaming, realtime)
    2. Dynamic utility function selection
    3. Wireless loss differentiation
    """
    
    def __init__(self, network, config):
        """
        Initialize Adaptive Vivace
# TODO: integrate loss classification feedback into utility calculation
        
        Args:
            network: NetworkSimulator instance
            config: Config object
        """
        super().__init__(network, config)
        
        # Traffic classifier
        self.classifier = TrafficClassifier(config.classifier)
        self.traffic_type = 'default'
        self.classification_confidence = 0.0
        self.classification_stable = False
        
        # Utility function bank
        self.utility_bank = UtilityFunctionBank(config.utilities)
        
        # Loss classifier
        self.loss_classifier = LossClassifier(config.loss_classifier)
        
        # Classification history
        self.classification_history = []
        
        # Packet tracking for classification
        self.packet_count = 0
        self.classification_interval = 50  # Classify every N packets
        
        logger.info("Adaptive Vivace initialized with extensions")
    
    def run_monitor_interval(self, current_time: float) -> Dict:
        """
        Execute one monitor interval with adaptive features
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Dictionary with MI results
        """
        mi_duration = self.monitor_interval_ms / 1000.0
        
        # Update classification if enough packets
        if self.packet_count >= self.classification_interval and not self.classification_stable:
            self._update_classification()
        
        # Generate rates to test
        rates_to_test = self._generate_test_rates()
        
        # Test each rate and collect metrics
        metrics_list = []
        for rate in rates_to_test:
            metrics = self._send_at_rate_adaptive(rate, mi_duration / len(rates_to_test), current_time)
            metrics_list.append(metrics)
        
        # Get appropriate utility function
        utility_func = self.utility_bank.get_utility_function(self.traffic_type)
        
        # Apply loss differentiation if enabled
        if self.loss_classifier.enabled:
            metrics_list = self._adjust_for_wireless_loss(metrics_list)
        
        # Compute utilities
        utilities = []
        for metrics in metrics_list:
            utility = utility_func(
                metrics['throughput'],
                metrics['latency'],
                metrics['loss']
            )
            utilities.append(utility)
        
        # Compute gradient
        gradient = self._compute_gradient(rates_to_test, utilities)
        
        # Update rate with momentum
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradient
        new_rate = self.current_rate + self.velocity
        
        # Clip to bounds
        new_rate = np.clip(new_rate, self.rate_min, self.rate_max)
        
        # Record metrics
        avg_metrics = self._average_metrics(metrics_list)
        self.rate_history.append(self.current_rate)
        self.throughput_history.append(avg_metrics['throughput'])
        self.latency_history.append(avg_metrics['latency'])
        self.loss_history.append(avg_metrics['loss'])
        self.utility_history.append(np.mean(utilities))
        self.time_history.append(current_time)
        
        # Update state
        old_rate = self.current_rate
        self.current_rate = new_rate
        self.iteration += 1
        
        logger.debug(f"MI {self.iteration}: type={self.traffic_type}, "
                    f"rate {old_rate:.2f}->{new_rate:.2f}, utility={np.mean(utilities):.3f}")
        
        return {
            'rate': new_rate,
            'metrics': avg_metrics,
            'utility': np.mean(utilities),
            'gradient': gradient,
            'traffic_type': self.traffic_type,
            'confidence': self.classification_confidence
        }
    
    def _send_at_rate_adaptive(self, rate: float, duration: float, current_time: float) -> Dict:
        """
        Send data at specified rate with packet tracking for classification
        
        Args:
            rate: Sending rate in Mbps
            duration: Duration in seconds
            current_time: Current time
            
        Returns:
            Dictionary with metrics
        """
        packet_size = self._get_packet_size()  # Adaptive packet size
        bits_per_packet = packet_size * 8
        rate_bps = rate * 1e6
        
        # Calculate packets to send
        packets_to_send = int((rate_bps * duration) / bits_per_packet)
        
        if packets_to_send > 0:
            inter_packet_time = duration / packets_to_send
        else:
            inter_packet_time = duration
            packets_to_send = 1
        
        # Send packets and track for classification
        packets_sent = 0
        packets_dropped = 0
        rtt_samples = []
        
        t = 0.0
        while t < duration and packets_sent < packets_to_send:
            # Add packet to classifier
            self.classifier.add_packet(packet_size, current_time + t)
            self.packet_count += 1
            
            packet = self.network.send_packet(packet_size, current_time + t)
            
            if packet.dropped:
                packets_dropped += 1
            else:
                packets_sent += 1
                rtt = self.network.get_rtt()
                rtt_samples.append(rtt)
            
            t += inter_packet_time
        
        # Process network queue
        completed = self.network.process_queue(duration)
        
        # Calculate metrics
        network_metrics = self.network.get_metrics(duration)
        
        throughput = network_metrics['throughput']
        latency = np.mean(rtt_samples) if rtt_samples else self.network.delay_ms * 2
        
        total = packets_sent + packets_dropped
        loss = packets_dropped / total if total > 0 else 0.0
        
        # Track for loss classification
        self.loss_classifier.add_observation(loss, latency)
        
        return {
            'throughput': throughput,
            'latency': latency,
            'loss': loss,
            'packets_sent': packets_sent,
            'packets_dropped': packets_dropped
        }
    
    def _get_packet_size(self) -> int:
        """Get packet size based on traffic type"""
        if self.traffic_type == 'bulk':
            return 1500  # MTU
        elif self.traffic_type == 'streaming':
            return 1200  # Typical for video
        elif self.traffic_type == 'realtime':
            # Mix of small and occasional larger packets
            if np.random.random() < 0.9:
                return 160  # VoIP
            else:
                return 800  # Game state update
        else:
            return 1500
    
    def _update_classification(self):
        """Update traffic classification"""
        new_type, confidence = self.classifier.classify()
        
        # Check if classification is stable
        if new_type == self.traffic_type:
            self.classification_stable = True
        else:
            # Type changed
            if confidence >= self.classifier.confidence_threshold:
                logger.info(f"Traffic reclassified: {self.traffic_type} -> {new_type} "
                          f"(confidence={confidence:.3f})")
                self.traffic_type = new_type
                self.classification_confidence = confidence
                
                # Reset utility bank history for new type
                self.utility_bank.reset_history()
            else:
                logger.debug(f"Classification confidence too low: {confidence:.3f}")
        
        self.classification_history.append({
            'iteration': self.iteration,
            'type': new_type,
            'confidence': confidence
        })
    
    def _adjust_for_wireless_loss(self, metrics_list: List[Dict]) -> List[Dict]:
        """
        Adjust metrics for wireless loss differentiation
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Adjusted metrics list
        """
        adjusted = []
        
        for metrics in metrics_list:
            loss_type, confidence = self.loss_classifier.classify()
            
            if loss_type == 'wireless':
                # Reduce effective loss for utility calculation
                adjusted_loss = metrics['loss'] * (1 - self.loss_classifier.wireless_penalty_reduction)
                
                logger.debug(f"Wireless loss detected (confidence={confidence:.2f}): "
                           f"{metrics['loss']:.4f} -> {adjusted_loss:.4f}")
                
                metrics_adjusted = metrics.copy()
                metrics_adjusted['loss'] = adjusted_loss
                adjusted.append(metrics_adjusted)
            else:
                adjusted.append(metrics)
        
        return adjusted
    
    def run(self, duration: float) -> Dict:
        """
        Run Adaptive Vivace for specified duration

        Args:
            duration: Duration in seconds

        Returns:
            Dictionary with results including extension-specific metrics
        """
        # Run the base algorithm
        base_results = super().run(duration)

        # Add extension metrics
        base_results.update({
            'traffic_type': self.traffic_type,
            'classification_confidence': self.classification_confidence,
            'classification_history': self.classification_history,
            'classifier_stats': self.classifier.get_statistics(),
            'loss_classification': self.loss_classifier.get_statistics()
        })

        return base_results

    def get_extended_results(self) -> Dict:
        """Get results including extension-specific metrics (deprecated - use run())"""
        # This method is kept for backwards compatibility
        return {
            'traffic_type': self.traffic_type,
            'classification_confidence': self.classification_confidence,
            'classification_history': self.classification_history,
            'classifier_stats': self.classifier.get_statistics(),
            'loss_classification': self.loss_classifier.get_statistics()
        }
    
    def reset(self):
        """Reset algorithm and extensions"""
        super().reset()
        
        self.classifier.reset()
        self.loss_classifier.reset()
        self.utility_bank.reset_history()
        
        self.traffic_type = 'default'
        self.classification_confidence = 0.0
        self.classification_stable = False
        self.packet_count = 0
        self.classification_history.clear()
        
        logger.info("Adaptive Vivace reset")


class LossClassifier:
    """
    Classifies packet loss as congestion-induced or random (wireless)
    """
    
    def __init__(self, config):
        """
        Initialize loss classifier
        
        Args:
            config: LossClassifierConfig object
        """
        self.window_size = config.window_size
        self.correlation_threshold = config.correlation_threshold_low
        self.wireless_penalty_reduction = config.wireless_penalty_reduction
        self.enabled = config.enabled
        
        self.loss_history = []
        self.rtt_history = []
        
        logger.info("Loss classifier initialized")
    
    def add_observation(self, loss: float, rtt: float):
        """Add loss and RTT observation"""
        self.loss_history.append(loss)
        self.rtt_history.append(rtt)
        
        # Keep window
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
            self.rtt_history.pop(0)
    
    def classify(self) -> tuple:
        """
        Classify loss type
        
        Returns:
            (loss_type, confidence) where loss_type is 'congestion' or 'wireless'
        """
        if not self.enabled or len(self.loss_history) < self.window_size // 2:
            return 'congestion', 0.5
        
        # Compute correlation between loss and RTT
        if len(self.loss_history) < 3:
            return 'congestion', 0.5
        
        # Convert to binary loss events
        loss_events = [1 if l > 0 else 0 for l in self.loss_history]
        
        if sum(loss_events) == 0:
            return 'congestion', 1.0
        
        # Normalize RTT
        rtt_normalized = (np.array(self.rtt_history) - np.mean(self.rtt_history)) / (np.std(self.rtt_history) + 1e-6)

        # Correlation with warning suppression
        if len(loss_events) > 1:
            with np.errstate(divide='ignore', invalid='ignore'):
                correlation = np.corrcoef(loss_events, rtt_normalized)[0, 1]
        else:
            correlation = 0.0

        # Interpret correlation
        if np.isnan(correlation):
            correlation = 0.0
        
        if abs(correlation) > self.correlation_threshold:
            # High correlation: congestion loss
            return 'congestion', abs(correlation)
        else:
            # Low correlation: wireless loss
            confidence = 1.0 - abs(correlation)
            return 'wireless', confidence
    
    def reset(self):
        """Reset classifier state"""
        self.loss_history.clear()
        self.rtt_history.clear()
    
    def get_statistics(self) -> Dict:
        """Get classifier statistics"""
        loss_type, confidence = self.classify()
        return {
            'loss_type': loss_type,
            'confidence': confidence,
            'observations': len(self.loss_history)
        }
