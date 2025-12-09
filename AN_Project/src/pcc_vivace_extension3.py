"""
Production-Grade PCC Vivace with Extension 3: Distributed Fairness

Extends Extension 2 (Wireless Loss Differentiation) with:
- ContentionDetector - Detect competing flows via gradient oscillation
- VirtualQueueEstimator - Estimate bottleneck queue from RTT
- CooperativeExplorer - Hash-based turn-taking for exploration
- FairnessController - Fairness-augmented utility function

Key Innovation:
Traditional PCC Vivace flows converge slowly and unfairly when competing.
Extension 3 enables distributed coordination without explicit communication:
- Implicit contention detection
- Coordinated exploration (reduced collisions)
- Fairness-driven rate adjustment

Expected Benefits:
- 3-5x faster convergence to fair allocation
- Jain's fairness index >0.98 (vs 0.85-0.90 baseline)
- Reduced probe collisions
- Maintained single-flow performance
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from .pcc_vivace_extension2 import PccVivaceExtension2, MonitorInterval
from .contention_detector import ContentionDetector, ContentionLevel
from .virtual_queue_estimator import VirtualQueueEstimator
from .cooperative_explorer import CooperativeExplorer
from .fairness_controller import FairnessController

logger = logging.getLogger(__name__)


class PccVivaceExtension3(PccVivaceExtension2):
    """
    PCC Vivace with Extension 3: Distributed Fairness
    
    Extends Extension 2 with multi-flow coordination capabilities.
    
    Architecture:
    1. Extension 1 (inherited): Traffic-aware utilities
    2. Extension 2 (inherited): Wireless loss differentiation
    3. Extension 3 (new):
       - ContentionDetector → detect competing flows
       - VirtualQueueEstimator → estimate queue occupancy
       - CooperativeExplorer → coordinate exploration
       - FairnessController → fairness-augmented utility
    
    Process Flow:
    1. Send packets → collect metrics
    2. ContentionDetector analyzes utility gradient oscillations
    3. VirtualQueueEstimator tracks RTT/queue state
    4. CooperativeExplorer decides whether to explore (turn-taking)
    5. FairnessController augments utility with fairness penalty
    6. Update sending rate based on augmented utility
    """
    
    def __init__(self, network, config, flow_id: int = 0, multiflow_mode: bool = False):
        """
        Initialize PCC Vivace with Extension 3

        Args:
            network: NetworkSimulator instance
            config: Config object with all configuration sections
            flow_id: Unique flow ID for coordination (default: 0)
            multiflow_mode: If True, use network stats instead of process_queue
        """
        # Initialize Extension 2 (includes Extension 1 + baseline)
        super().__init__(network, config, flow_id, multiflow_mode)

        # Flow identity already set by baseline, but keep for clarity
        self.flow_id = flow_id
        self.multiflow_mode = multiflow_mode

        # Initialize Extension 3 components
        self.contention_detector = ContentionDetector(config.contention_detector)
        self.queue_estimator = VirtualQueueEstimator(config.virtual_queue_estimator)
        self.cooperative_explorer = CooperativeExplorer(flow_id, config.cooperative_explorer)
        self.fairness_controller = FairnessController(config.fairness_controller)

        # Extension 3 state
        self.current_contention = ContentionLevel.SOLO
        self.estimated_flow_count = 1
        self.contention_confidence = 0.0

        # Utility gradient tracking for contention detection
        self.last_utility = None
        self.last_rate = None

        # Performance tracking
        self.contention_history = []
        self.fairness_history = []
        self.exploration_decisions = []

        # IMPROVEMENT: Adaptive optimization mode for single-flow scenarios
        # When consistently in SOLO mode, reduce overhead by skipping fairness components
        self.optimization_mode_enabled = True  # Enable optimization
        self.solo_streak = 0  # Consecutive SOLO detections
        self.solo_threshold = 5  # Enter optimization after N consecutive SOLO detections (was 20, now 5 for faster response)
        self.check_interval = 20  # Check for contention every N intervals when optimized (was 10, now 20 for less overhead)
        self.intervals_since_check = 0
        self.is_optimized = False  # Currently in optimized (lightweight) mode
        
        logger.info("="*60)
        logger.info("PCC Vivace Extension 3 Initialized")
        logger.info("="*60)
        logger.info(f"  Extension 1: Traffic-Aware Utilities ✓")
        logger.info(f"  Extension 2: Wireless Loss Differentiation ✓")
        logger.info(f"  Extension 3: Distributed Fairness ✓")
        logger.info(f"  Flow ID: {self.flow_id}")
        logger.info(f"  ContentionDetector: window={config.contention_detector.window_size}")
        logger.info(f"  VirtualQueueEstimator: baseline_window={config.virtual_queue_estimator.baseline_window}")
        logger.info(f"  CooperativeExplorer: cycle={config.cooperative_explorer.exploration_cycle_ms}ms")
        logger.info(f"  FairnessController: μ={config.fairness_controller.fairness_penalty_weight:.2f}")
        logger.info("="*60)
    
    def _monitor_interval(self, interval_duration: float) -> MonitorInterval:
        """
        Monitor interval with Extension 3 contention/queue tracking
        
        Args:
            interval_duration: Duration of monitoring interval
        
        Returns:
            MonitorInterval with metrics
        """
        # Get standard monitoring from Extension 2
        mi = super()._monitor_interval(interval_duration)
        
        # Extension 3: Add RTT and throughput to queue estimator
        if self.queue_estimator and mi.avg_rtt > 0:
            self.queue_estimator.add_rtt_sample(
                timestamp=self.total_time,
                rtt_ms=mi.avg_rtt,
                throughput_mbps=mi.throughput
            )
            
            # Also add throughput separately for bandwidth estimation
            if mi.throughput > 0:
                self.queue_estimator.add_throughput_sample(mi.throughput)
        
        return mi
    
    def _calculate_utility(self, mi: MonitorInterval) -> float:
        """
        Calculate utility with Extension 3 fairness augmentation

        Process:
        1. Extension 1: Traffic classification + utility selection
        2. Extension 2: Loss classification + adaptive coefficient
        3. Extension 3: Fairness augmentation (adaptive - skipped in SOLO mode)

        Args:
            mi: MonitorInterval with metrics

        Returns:
            Fairness-augmented utility value
        """
        # Get base utility from Extension 2 (includes Extension 1)
        original_utility = super()._calculate_utility(mi)

        # IMPROVEMENT: Check if in optimized mode (skip contention detection periodically)
        should_check_contention = True
        if self.optimization_mode_enabled and self.is_optimized:
            self.intervals_since_check += 1
            if self.intervals_since_check < self.check_interval:
                # Skip contention detection, use cached SOLO state
                should_check_contention = False
                # Return original utility without fairness overhead
                logger.debug(f"[Flow {self.flow_id}] Optimized mode: skipping fairness (interval {self.intervals_since_check}/{self.check_interval})")
                return original_utility
            else:
                # Time to check again
                self.intervals_since_check = 0
                logger.debug(f"[Flow {self.flow_id}] Optimized mode: periodic contention check")

        # Extension 3: Detect contention from utility gradient
        if self.last_utility is not None and self.last_rate is not None:
            # Calculate utility gradient
            rate_change = self.sending_rate - self.last_rate
            if abs(rate_change) > 0.001:  # Avoid division by zero
                utility_change = original_utility - self.last_utility
                gradient = utility_change / rate_change

                # Add gradient to contention detector
                self.contention_detector.add_gradient(
                    timestamp=self.total_time,
                    gradient=gradient,
                    utility=original_utility,
                    sending_rate=self.sending_rate
                )

        # Update last values for next gradient calculation
        self.last_utility = original_utility
        self.last_rate = self.sending_rate

        # Detect contention level
        contention_level, confidence, flow_count = self.contention_detector.detect_contention()
        self.current_contention = contention_level
        self.estimated_flow_count = flow_count
        self.contention_confidence = confidence

        # IMPROVEMENT: Update optimization mode based on contention
        # Require higher confidence (0.6) to detect contention and exit SOLO mode
        # This reduces false positives in single-flow scenarios
        if self.optimization_mode_enabled:
            if contention_level == ContentionLevel.SOLO and confidence > 0.5:
                self.solo_streak += 1
                if not self.is_optimized and self.solo_streak >= self.solo_threshold:
                    self.is_optimized = True
                    self.intervals_since_check = 0
                    logger.info(f"[Flow {self.flow_id}] Entering optimized mode (SOLO streak: {self.solo_streak})")
            elif contention_level != ContentionLevel.SOLO and confidence > 0.6:
                # IMPROVED: Only exit optimization if CONFIDENT about contention (>0.6)
                # Prevents false exits due to noise
                if self.is_optimized:
                    logger.info(f"[Flow {self.flow_id}] Exiting optimized mode: {contention_level.value} detected (conf={confidence:.2f})")
                self.solo_streak = 0
                self.is_optimized = False
            # Otherwise: ambiguous signal, maintain current mode

        # If SOLO with moderate confidence, skip fairness augmentation
        # IMPROVED: Lower confidence threshold (0.5 instead of 0.8) for faster SOLO detection
        if contention_level == ContentionLevel.SOLO and confidence > 0.5:
            logger.debug(f"[Flow {self.flow_id}] SOLO mode: skipping fairness augmentation")
            # Still track history for monitoring
            self.contention_history.append({
                'timestamp': self.total_time,
                'contention_level': contention_level.value,
                'flow_count': flow_count,
                'confidence': confidence
            })
            return original_utility

        # Estimate fair share for fairness controller
        # Get queue estimate for aggregate throughput estimation
        queue_info = self.queue_estimator.estimate_queue(mi.avg_rtt)

        # Estimate aggregate throughput (heuristic: my_rate × flow_count)
        estimated_aggregate = mi.throughput * flow_count if mi.throughput > 0 else None

        # Update fair share estimate
        self.fairness_controller.estimate_fair_share(
            my_throughput=mi.throughput,
            estimated_flow_count=flow_count,
            aggregate_throughput=estimated_aggregate
        )

        # Augment utility with fairness penalty
        utility_fair = self.fairness_controller.augment_utility(
            original_utility=original_utility,
            current_rate=self.sending_rate,
            contention_level=contention_level,
            timestamp=self.total_time
        )

        # Track history
        self.contention_history.append({
            'timestamp': self.total_time,
            'contention_level': contention_level.value,
            'flow_count': flow_count,
            'confidence': confidence
        })

        self.fairness_history.append({
            'timestamp': self.total_time,
            'original_utility': original_utility,
            'augmented_utility': utility_fair,
            'fair_share': self.fairness_controller.estimated_fair_share,
            'current_rate': self.sending_rate
        })

        logger.debug(f"[Flow {self.flow_id}] Utility: {original_utility:.2f} → {utility_fair:.2f}, "
                    f"Contention: {contention_level.value}, Flows: {flow_count}")

        return utility_fair
    
    def _should_explore(self) -> bool:
        """
        Decide whether to explore using cooperative coordination
        
        Extension 3 uses hash-based turn-taking to reduce collisions.
        
        Returns:
            True if should explore, False otherwise
        """
        # Use cooperative explorer for coordinated exploration
        should_explore = self.cooperative_explorer.should_explore(
            current_time=self.total_time,
            contention_level=self.current_contention,
            estimated_flow_count=self.estimated_flow_count
        )
        
        # Track decision
        self.exploration_decisions.append({
            'timestamp': self.total_time,
            'should_explore': should_explore,
            'contention_level': self.current_contention.value,
            'flow_count': self.estimated_flow_count
        })
        
        return should_explore
    
    def _get_exploration_magnitude(self) -> float:
        """
        Get exploration rate change magnitude
        
        Extension 3 adapts magnitude based on contention.
        
        Returns:
            Exploration magnitude (e.g., 0.1 = 10% increase)
        """
        return self.cooperative_explorer.get_exploration_magnitude(
            self.current_contention
        )
    
    def _record_exploration_outcome(self, utility_before: float, utility_after: float):
        """
        Record whether exploration was successful
        
        Args:
            utility_before: Utility before exploration
            utility_after: Utility after exploration
        """
        magnitude = self._get_exploration_magnitude()
        
        self.cooperative_explorer.record_exploration_outcome(
            timestamp=self.total_time,
            contention_level=self.current_contention,
            exploration_magnitude=magnitude,
            utility_before=utility_before,
            utility_after=utility_after
        )
    
    def get_extension3_statistics(self) -> Dict:
        """
        Get Extension 3 specific statistics

        Returns:
            Dictionary with Extension 3 metrics
        """
        contention_stats = self.contention_detector.get_statistics()
        queue_stats = self.queue_estimator.get_statistics()
        explorer_stats = self.cooperative_explorer.get_statistics()
        fairness_stats = self.fairness_controller.get_statistics()

        return {
            'flow_id': self.flow_id,
            'contention': contention_stats,
            'queue': queue_stats,
            'exploration': explorer_stats,
            'fairness': fairness_stats,
            'current_state': {
                'contention_level': self.current_contention.value,
                'estimated_flow_count': self.estimated_flow_count,
                'contention_confidence': self.contention_confidence,
                'is_optimized': self.is_optimized,
                'solo_streak': self.solo_streak
            }
        }
    
    def get_summary(self) -> str:
        """
        Get human-readable summary including all extensions
        
        Returns:
            Formatted summary string
        """
        # Get base summary from Extension 2
        base_summary = super().get_summary()
        
        # Add Extension 3 summary
        stats = self.get_extension3_statistics()
        
        ext3_summary = "\n" + "="*60 + "\n"
        ext3_summary += "Extension 3: Distributed Fairness\n"
        ext3_summary += "="*60 + "\n"
        ext3_summary += f"Flow ID: {self.flow_id}\n"
        ext3_summary += f"Current Contention: {stats['current_state']['contention_level']}\n"
        ext3_summary += f"Estimated Flows: {stats['current_state']['estimated_flow_count']}\n"
        ext3_summary += f"Exploration Efficiency: {stats['exploration']['efficiency']['efficiency']:.1%}\n"
        ext3_summary += f"Collision Rate: {stats['exploration']['exploration']['collision_rate']:.1%}\n"
        ext3_summary += f"Fairness Ratio: {stats['fairness']['current']['fairness_ratio']:.3f}\n"
        ext3_summary += f"Fair Share: {stats['fairness']['current']['fair_share']:.2f} Mbps\n"
        
        return base_summary + ext3_summary
