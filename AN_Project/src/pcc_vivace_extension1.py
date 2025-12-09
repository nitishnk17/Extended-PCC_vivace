"""
Production-Grade PCC Vivace with Extension 1: Application-Aware Utilities

Implements hierarchical multi-objective learning framework:
- Traffic Classification using TrafficClassifier
- Specialized Utility Functions using UtilityFunctionBank
- Intelligent Selection using MetaController

This is the complete, production-ready implementation of Extension 1.
"""
import numpy as np
from typing import Dict, List, Optional
import logging

from .pcc_vivace_baseline import PccVivaceBaseline, MonitorInterval
from .traffic_classifier import TrafficClassifier
from .utility_bank import UtilityFunctionBank
from .meta_controller import MetaController

logger = logging.getLogger(__name__)


class PccVivaceExtension1(PccVivaceBaseline):
    """
    PCC Vivace with Extension 1: Application-Aware Utilities

    Extends baseline PCC Vivace with:
    1. TrafficClassifier - Identifies traffic type from packet patterns
    2. UtilityFunctionBank - Specialized utilities per traffic type
    3. MetaController - Intelligent utility selection

    Traffic Types:
    - bulk: Maximize throughput
    - streaming: Stable rate, minimize variance
    - realtime: Minimize latency
    - default: Baseline Vivace utility

    Features:
    - Automatic traffic classification
    - Per-traffic-type optimization
    - Stability mechanisms (avoid rapid switching)
    - Comprehensive logging and telemetry
    - Production-ready error handling
    """

    def __init__(self, network, config, flow_id: int = 0, multiflow_mode: bool = False):
        """
        Initialize PCC Vivace with Extension 1

        Args:
            network: NetworkSimulator instance
            config: Config object with classifier, utilities, vivace sections
            flow_id: Unique flow identifier (for multi-flow scenarios)
            multiflow_mode: If True, use network stats instead of process_queue
        """
        # Initialize baseline PCC Vivace
        super().__init__(network, config, flow_id, multiflow_mode)

        # Initialize Extension 1 components
        self.classifier = TrafficClassifier(config.classifier)
        self.utility_bank = UtilityFunctionBank(config.utilities)
        self.meta_controller = MetaController(config)

        # Current state
        self.current_traffic_type = 'default'
        self.current_confidence = 0.0
        self.classification_count = 0

        # Performance tracking
        self.utility_history_by_type = {
            'bulk': [],
            'streaming': [],
            'realtime': [],
            'default': []
        }

        # Packet counting for classification
        self.packets_sent_total = 0

        logger.info("="*60)
        logger.info("PCC Vivace Extension 1 Initialized")
        logger.info("="*60)
        logger.info(f"  TrafficClassifier: window={config.classifier.window_size}, "
                   f"threshold={config.classifier.confidence_threshold}")
        logger.info(f"  UtilityFunctionBank: 4 utility functions (bulk, streaming, realtime, default)")
        logger.info(f"  MetaController: stability_window={self.meta_controller.stability_window}")
        logger.info("="*60)

    def send_packet(self, size: int, current_time: float) -> Dict:
        """
        Send packet with traffic classification

        Args:
            size: Packet size in bytes
            current_time: Current simulation time

        Returns:
            Dict with packet send results
        """
        # Send packet through network
        result = self.network.send_packet(size, 0, current_time)  # flow_id=0

        # Add to classifier
        if self.classifier.enabled:
            self.classifier.add_packet(size, current_time)

        self.packets_sent_total += 1

        return result

    def _calculate_utility(self, mi: MonitorInterval) -> float:
        """
        Calculate utility using Extension 1 framework

        Process:
        1. Classify traffic (if enough data)
        2. MetaController selects utility function
        3. Calculate utility using selected function
        4. Update performance tracking

        Args:
            mi: MonitorInterval with metrics

        Returns:
            Utility value
        """
        # Step 1: Classify traffic
        traffic_type, confidence = self.classifier.classify()

        # Step 2: Meta-controller selects utility function
        selected_utility, decision_metadata = self.meta_controller.select_utility(
            traffic_type,
            confidence
        )

        # Update current state
        self.current_traffic_type = selected_utility
        self.current_confidence = confidence
        self.classification_count += 1

        # Log classification if switched
        if decision_metadata['switched']:
            logger.info(f"★ UTILITY SWITCHED: {decision_metadata['previous_utility']} → "
                       f"{selected_utility} (confidence={confidence:.3f})")

        # Step 3: Calculate utility using selected function
        utility = self._compute_utility(mi, selected_utility)

        # Step 4: Update performance tracking
        self.meta_controller.update_performance(selected_utility, utility)
        self.utility_history_by_type[selected_utility].append(utility)

        # Limit history size
        for history_list in self.utility_history_by_type.values():
            if len(history_list) > 1000:
                history_list[:] = history_list[-500:]

        logger.debug(f"Extension1: type={selected_utility}, conf={confidence:.2f}, U={utility:.2f}")

        return utility

    def _compute_utility(self, mi: MonitorInterval, utility_type: str) -> float:
        """
        Compute utility using UtilityFunctionBank

        Args:
            mi: MonitorInterval with metrics
            utility_type: Selected utility type

        Returns:
            Utility value
        """
        # Get utility function from bank
        utility_func = self.utility_bank.get_utility_function(utility_type)

        # Compute utility
        utility = utility_func(mi.throughput, mi.avg_rtt, mi.loss_rate)

        return utility

    def run(self, duration: float) -> Dict:
        """
        Run Extension 1 PCC Vivace for specified duration

        Args:
            duration: Duration in seconds

        Returns:
            Dict with comprehensive results including Extension 1 metrics
        """
        # Run baseline
        results = super().run(duration)

        # Add Extension 1 specific results
        extension1_results = self._get_extension1_results()
        results['extension1'] = extension1_results

        # Log summary
        self._log_extension1_summary(extension1_results)

        return results

    def _get_extension1_results(self) -> Dict:
        """
        Get Extension 1 specific results

        Returns:
            Dict with Extension 1 metrics
        """
        # Traffic classification stats
        classifier_stats = self.classifier.get_statistics()

        # Meta-controller stats
        meta_stats = self.meta_controller.get_statistics()

        # Utility performance by type
        utility_performance = {}
        for utility_type, history in self.utility_history_by_type.items():
            if len(history) > 0:
                utility_performance[utility_type] = {
                    'count': len(history),
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'min': np.min(history),
                    'max': np.max(history)
                }
            else:
                utility_performance[utility_type] = {
                    'count': 0,
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0
                }

        return {
            'traffic_classification': {
                'final_type': classifier_stats['traffic_type'],
                'final_confidence': classifier_stats['confidence'],
                'features': classifier_stats['features'],
                'num_packets': classifier_stats['num_packets']
            },
            'meta_controller': {
                'current_utility': meta_stats['current_utility'],
                'switches_count': meta_stats['switches_count'],
                'classification_consistency': meta_stats['classification_consistency'],
                'avg_confidence': meta_stats['avg_confidence']
            },
            'utility_performance': utility_performance,
            'packets_sent_total': self.packets_sent_total,
            'classification_count': self.classification_count
        }

    def _log_extension1_summary(self, ext1_results: Dict):
        """
        Log Extension 1 summary

        Args:
            ext1_results: Extension 1 results dict
        """
        logger.info("="*60)
        logger.info("Extension 1 Summary")
        logger.info("="*60)

        # Classification
        traffic_class = ext1_results['traffic_classification']
        logger.info(f"Traffic Classification:")
        logger.info(f"  Type: {traffic_class['final_type']}")
        logger.info(f"  Confidence: {traffic_class['final_confidence']:.3f}")
        logger.info(f"  Packets Observed: {traffic_class['num_packets']}")

        # Meta-controller
        meta = ext1_results['meta_controller']
        logger.info(f"Meta-Controller:")
        logger.info(f"  Final Utility: {meta['current_utility']}")
        logger.info(f"  Switches: {meta['switches_count']}")
        logger.info(f"  Consistency: {meta['classification_consistency']:.3f}")
        logger.info(f"  Avg Confidence: {meta['avg_confidence']:.3f}")

        # Utility Performance
        logger.info(f"Utility Performance:")
        for utility_type, perf in ext1_results['utility_performance'].items():
            if perf['count'] > 0:
                logger.info(f"  {utility_type}: mean={perf['mean']:.2f}, "
                           f"count={perf['count']}, std={perf['std']:.2f}")

        logger.info("="*60)

    def get_detailed_statistics(self) -> Dict:
        """
        Get comprehensive statistics for analysis

        Returns:
            Dict with all statistics
        """
        # Baseline stats
        base_stats = {
            'total_time': self.total_time,
            'total_bytes': self.total_bytes_sent,
            'avg_throughput': np.mean(self.throughput_history) if self.throughput_history else 0.0,
            'avg_latency': np.mean(self.latency_history) if self.latency_history else 0.0,
            'avg_utility': np.mean(self.utility_history) if self.utility_history else 0.0,
            'mode_switches': len([i for i in range(1, len(self.mode_history))
                                 if self.mode_history[i] != self.mode_history[i-1]])
        }

        # Extension 1 stats
        ext1_stats = self._get_extension1_results()

        # Classifier detailed stats
        classifier_stats = self.classifier.get_statistics()

        # Meta-controller decision history
        decision_history = self.meta_controller.get_decision_history()

        return {
            'baseline': base_stats,
            'extension1': ext1_stats,
            'classifier_detailed': classifier_stats,
            'decision_history': decision_history,
            'utility_history': {
                'values': self.utility_history,
                'by_type': self.utility_history_by_type
            }
        }

    def reset(self):
        """Reset Extension 1 state"""
        # Reset baseline
        super().reset() if hasattr(super(), 'reset') else None

        # Reset Extension 1 components
        self.classifier.reset()
        self.utility_bank.reset_history()
        self.meta_controller.reset()

        # Reset state
        self.current_traffic_type = 'default'
        self.current_confidence = 0.0
        self.classification_count = 0
        self.packets_sent_total = 0

        # Clear histories
        for history in self.utility_history_by_type.values():
            history.clear()

        logger.info("Extension 1 state reset")
