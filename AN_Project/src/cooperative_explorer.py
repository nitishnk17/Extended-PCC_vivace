"""
Cooperative rate probing - helps flows avoid stepping on each other

Coordinates exploration (rate probing) between competing flows to reduce
collisions through implicit hash-based turn-taking.

    explorer = CooperativeExplorer(flow_id=1, config)

    # Each decision interval
    if explorer.should_explore(current_time, contention_level, flow_count):

When multiple flows probe at the same time, everyone sees worse utility.
This tries to coordinate exploration so flows take turns (using flow_id hash).

Reduces probe collisions without explicit communication.

FIXME: doesnt work great when flow count estimate is wrong
        magnitude = explorer.get_exploration_magnitude(contention_level)
        new_rate = current_rate * (1 + magnitude)

    # After receiving feedback
    explorer.record_exploration_outcome(utility_increased=True)
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from enum import Enum
import logging

# Import ContentionLevel from contention_detector
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.contention_detector import ContentionLevel

logger = logging.getLogger(__name__)


@dataclass
class ExplorationEvent:
    """
    Record of a single exploration attempt

    Attributes:
        timestamp: Time of exploration (seconds)
        contention_level: Contention level at time of exploration
        exploration_magnitude: Rate increase magnitude used
        utility_before: Utility before exploration
        utility_after: Utility after exploration
        collision: Whether exploration resulted in collision (utility drop)
    """
    timestamp: float
    contention_level: ContentionLevel
    exploration_magnitude: float
    utility_before: float
    utility_after: float
    collision: bool

    def __repr__(self):
        outcome = "collision" if self.collision else "success"
        return (f"ExplorationEvent(t={self.timestamp:.2f}s, "
                f"{self.contention_level.value}, mag={self.exploration_magnitude:.2f}, "
                f"{outcome})")


class CooperativeExplorer:
    """
    Coordinate exploration across competing flows using hash-based turn-taking

    The explorer divides time into cycles and assigns each flow to a phase
    within the cycle based on their flow ID. Flows only explore during their
    assigned phase, reducing collisions.

    Algorithm Details:
    - Cycle duration: Configurable (default 500ms)
    - Phase assignment: flow_id % estimated_flow_count
    - Current phase: (time_in_cycle / cycle_duration) × flow_count
    - Exploration magnitude: Scaled by contention level

    When contention is SOLO, coordination is disabled for maximum agility.
    When contention is HEAVY, exploration is conservative to avoid disruption.
    """

    def __init__(self, flow_id: int, config):
        """
        Initialize cooperative explorer

        Args:
            flow_id: Unique identifier for this flow (used for phase assignment)
            config: CooperativeExplorerConfig with parameters
        """
        # Identity
        self.flow_id = flow_id

        # Configuration
        self.exploration_cycle_ms = config.exploration_cycle_ms
        self.base_exploration_rate = config.base_exploration_rate
        self.reduced_exploration_rate = config.reduced_exploration_rate
        self.solo_exploration_probability = config.solo_exploration_probability

        # State
        self.last_exploration_time = 0.0
        self.exploration_count = 0
        self.collision_count = 0
        self.successful_exploration_count = 0

        # History (bounded for memory safety)
        self.exploration_history = []  # Will manually bound to 100 entries
        self.max_history_size = 100

        # Statistics
        self.total_should_explore_calls = 0
        self.total_explore_granted = 0

        logger.info(f"CooperativeExplorer initialized: flow_id={flow_id}, "
                   f"cycle={self.exploration_cycle_ms}ms, "
                   f"base_rate={self.base_exploration_rate:.2%}")

    def should_explore(self, current_time: float,
                      contention_level: ContentionLevel,
                      estimated_flow_count: int) -> bool:
        """
        Decide if this flow should explore now

        Uses implicit coordination based on time-slicing within cycles:
        - Each cycle is divided into N phases (N = flow count)
        - Flow explores only during phase = flow_id % N
        - When SOLO, always explore (no coordination needed)

        Args:
            current_time: Current time (seconds)
            contention_level: Current contention level
            estimated_flow_count: Estimated number of competing flows

        Returns:
            True if this flow should explore, False otherwise

        Example:
            With 3 flows (IDs 0, 1, 2) and 500ms cycle:
            - Phase 0 (0-167ms): Flow 0 explores
            - Phase 1 (167-333ms): Flow 1 explores
            - Phase 2 (333-500ms): Flow 2 explores
        """
        self.total_should_explore_calls += 1

        # Special case: SOLO (no contention, no coordination needed)
        if contention_level == ContentionLevel.SOLO:
            # Explore with high probability (but not always, for stability)
            should_explore = np.random.random() < self.solo_exploration_probability
            if should_explore:
                self.total_explore_granted += 1
            return should_explore

        # Ensure valid flow count
        if estimated_flow_count <= 0:
            logger.warning(f"Invalid flow count: {estimated_flow_count}, defaulting to 1")
            estimated_flow_count = 1

        # Calculate cycle time (time within current cycle)
        cycle_duration_s = self.exploration_cycle_ms / 1000.0
        cycle_time = current_time % cycle_duration_s

        # Calculate current phase (which phase are we in?)
        # Divide cycle into N equal phases
        current_phase = int(cycle_time * estimated_flow_count / cycle_duration_s)

        # This flow's assigned phase (hash-based on flow ID)
        my_phase = self.flow_id % max(1, estimated_flow_count)

        # Check if it's my turn
        is_my_turn = (current_phase == my_phase)

        # Adjust based on contention level
        if contention_level == ContentionLevel.LIGHT:
            # Light contention: relax coordination slightly
            # Allow exploration if in my phase OR adjacent phase
            adjacent_phase = (my_phase + 1) % estimated_flow_count
            is_my_turn = is_my_turn or (current_phase == adjacent_phase)

        logger.debug(f"should_explore: flow={self.flow_id}, phase={current_phase}/{estimated_flow_count}, "
                    f"my_phase={my_phase}, is_my_turn={is_my_turn}, "
                    f"contention={contention_level.value}")

        if is_my_turn:
            self.total_explore_granted += 1

        return is_my_turn

    def get_exploration_magnitude(self, contention_level: ContentionLevel) -> float:
        """
        Get exploration rate increase magnitude

        Reduces magnitude when contention is high to avoid disruption.
        Higher contention → smaller probes → less oscillation.

        Args:
            contention_level: Current contention level

        Returns:
            Exploration magnitude (e.g., 0.1 = 10% rate increase)

        Example:
            SOLO: 10% increase (aggressive exploration)
            LIGHT: 8% increase
            MODERATE: 5% increase
            HEAVY: 3.5% increase (conservative)
        """
        if contention_level == ContentionLevel.SOLO:
            magnitude = self.base_exploration_rate
        elif contention_level == ContentionLevel.LIGHT:
            magnitude = self.base_exploration_rate * 0.8
        elif contention_level == ContentionLevel.MODERATE:
            magnitude = self.reduced_exploration_rate
        else:  # HEAVY
            magnitude = self.reduced_exploration_rate * 0.7

        logger.debug(f"get_exploration_magnitude: {contention_level.value} → {magnitude:.2%}")

        return magnitude

    def record_exploration_outcome(self, timestamp: float,
                                   contention_level: ContentionLevel,
                                   exploration_magnitude: float,
                                   utility_before: float,
                                   utility_after: float):
        """
        Record result of exploration

        Tracks whether exploration was successful (utility increased) or
        resulted in collision (utility decreased).

        Args:
            timestamp: Time of exploration
            contention_level: Contention level at time
            exploration_magnitude: Magnitude used
            utility_before: Utility before exploration
            utility_after: Utility after exploration
        """
        # Determine if collision occurred
        collision = utility_after < utility_before

        # Create event record
        event = ExplorationEvent(
            timestamp=timestamp,
            contention_level=contention_level,
            exploration_magnitude=exploration_magnitude,
            utility_before=utility_before,
            utility_after=utility_after,
            collision=collision
        )

        # Add to history (bounded)
        self.exploration_history.append(event)
        if len(self.exploration_history) > self.max_history_size:
            self.exploration_history.pop(0)

        # Update counters
        self.exploration_count += 1
        self.last_exploration_time = timestamp

        if collision:
            self.collision_count += 1
            logger.debug(f"Exploration collision: utility {utility_before:.2f} → {utility_after:.2f}")
        else:
            self.successful_exploration_count += 1
            logger.debug(f"Exploration success: utility {utility_before:.2f} → {utility_after:.2f}")

    def get_collision_rate(self) -> float:
        """
        Calculate exploration collision rate

        Returns:
            Collision rate [0, 1] (0 = no collisions, 1 = all collisions)
        """
        total = self.collision_count + self.successful_exploration_count
        if total == 0:
            return 0.0
        return self.collision_count / total

    def get_success_rate(self) -> float:
        """
        Calculate exploration success rate

        Returns:
            Success rate [0, 1] (0 = no successes, 1 = all successes)
        """
        return 1.0 - self.get_collision_rate()

    def get_exploration_efficiency(self) -> float:
        """
        Calculate exploration efficiency

        Efficiency = fraction of should_explore calls that resulted in exploration

        Returns:
            Efficiency [0, 1] (higher = more exploration opportunities)
        """
        if self.total_should_explore_calls == 0:
            return 0.0
        return self.total_explore_granted / self.total_should_explore_calls

    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics

        Returns:
            Dictionary with explorer statistics
        """
        # Collision metrics
        collision_rate = self.get_collision_rate()
        success_rate = self.get_success_rate()
        efficiency = self.get_exploration_efficiency()

        # Recent collision rate (last 20 explorations)
        recent_events = self.exploration_history[-20:] if len(self.exploration_history) >= 20 else self.exploration_history
        if recent_events:
            recent_collisions = sum(1 for e in recent_events if e.collision)
            recent_collision_rate = recent_collisions / len(recent_events)
        else:
            recent_collision_rate = 0.0

        # Utility statistics from history
        if self.exploration_history:
            utility_changes = [e.utility_after - e.utility_before for e in self.exploration_history]
            avg_utility_change = np.mean(utility_changes)
            std_utility_change = np.std(utility_changes)
        else:
            avg_utility_change = 0.0
            std_utility_change = 0.0

        return {
            'flow_id': self.flow_id,
            'exploration': {
                'total_count': self.exploration_count,
                'collision_count': self.collision_count,
                'success_count': self.successful_exploration_count,
                'collision_rate': collision_rate,
                'success_rate': success_rate,
                'recent_collision_rate': recent_collision_rate
            },
            'efficiency': {
                'total_should_explore_calls': self.total_should_explore_calls,
                'total_explore_granted': self.total_explore_granted,
                'efficiency': efficiency
            },
            'utility': {
                'avg_change': avg_utility_change,
                'std_change': std_utility_change
            },
            'history': {
                'event_count': len(self.exploration_history),
                'last_exploration_time': self.last_exploration_time
            }
        }

    def get_summary(self) -> str:
        """
        Get human-readable summary

        Returns:
            Formatted summary string
        """
        stats = self.get_statistics()

        summary = f"Cooperative Explorer Summary (Flow {self.flow_id}):\n"
        summary += f"  Total Explorations: {stats['exploration']['total_count']}\n"
        summary += f"  Success Rate: {stats['exploration']['success_rate']:.1%}\n"
        summary += f"  Collision Rate: {stats['exploration']['collision_rate']:.1%}\n"
        summary += f"  Recent Collision Rate: {stats['exploration']['recent_collision_rate']:.1%}\n"
        summary += f"  Exploration Efficiency: {stats['efficiency']['efficiency']:.1%}\n"
        summary += f"  Avg Utility Change: {stats['utility']['avg_change']:+.3f} ± {stats['utility']['std_change']:.3f}\n"
        summary += f"  Event History: {stats['history']['event_count']} events\n"

        return summary

    def reset(self):
        """Reset explorer state"""
        self.last_exploration_time = 0.0
        self.exploration_count = 0
        self.collision_count = 0
        self.successful_exploration_count = 0
        self.exploration_history.clear()
        self.total_should_explore_calls = 0
        self.total_explore_granted = 0

        logger.info(f"CooperativeExplorer reset (flow {self.flow_id})")
