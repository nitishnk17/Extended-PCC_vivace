"""
Comprehensive unit tests for CooperativeExplorer

Tests cover:
1. Basic functionality (initialization, exploration decisions)
2. Turn assignment and phase calculation
3. Exploration magnitude adjustment
4. Collision tracking and statistics
5. Multi-flow coordination simulation
6. Edge cases and error handling
7. Realistic scenarios
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cooperative_explorer import (
    CooperativeExplorer, ExplorationEvent
)
from src.contention_detector import ContentionLevel
from src.config import Config


class TestExplorationEvent:
    """Test ExplorationEvent dataclass"""

    def test_exploration_event_creation(self):
        """Test ExplorationEvent initialization"""
        event = ExplorationEvent(
            timestamp=1.0,
            contention_level=ContentionLevel.MODERATE,
            exploration_magnitude=0.05,
            utility_before=10.0,
            utility_after=9.5,
            collision=True
        )

        assert event.timestamp == 1.0
        assert event.contention_level == ContentionLevel.MODERATE
        assert event.exploration_magnitude == 0.05
        assert event.utility_before == 10.0
        assert event.utility_after == 9.5
        assert event.collision == True


class TestCooperativeExplorerBasics:
    """Test basic CooperativeExplorer functionality"""

    @pytest.fixture
    def config(self):
        """Create configuration"""
        return Config()

    @pytest.fixture
    def explorer(self, config):
        """Create CooperativeExplorer instance"""
        return CooperativeExplorer(flow_id=0, config=config.cooperative_explorer)

    def test_initialization(self, explorer, config):
        """Test explorer initialization"""
        assert explorer.flow_id == 0
        assert explorer.exploration_cycle_ms == config.cooperative_explorer.exploration_cycle_ms
        assert explorer.base_exploration_rate == config.cooperative_explorer.base_exploration_rate
        assert explorer.exploration_count == 0
        assert explorer.collision_count == 0
        assert len(explorer.exploration_history) == 0

    def test_different_flow_ids(self, config):
        """Test creating explorers with different flow IDs"""
        explorer0 = CooperativeExplorer(flow_id=0, config=config.cooperative_explorer)
        explorer1 = CooperativeExplorer(flow_id=1, config=config.cooperative_explorer)
        explorer2 = CooperativeExplorer(flow_id=2, config=config.cooperative_explorer)

        assert explorer0.flow_id == 0
        assert explorer1.flow_id == 1
        assert explorer2.flow_id == 2


class TestSoloExploration:
    """Test exploration when SOLO (no coordination)"""

    @pytest.fixture
    def explorer(self):
        config = Config()
        return CooperativeExplorer(flow_id=0, config=config.cooperative_explorer)

    def test_solo_always_explores(self, explorer):
        """Test SOLO flows explore frequently"""
        # SOLO flows should explore with high probability (90%)
        explore_count = 0
        trials = 100

        for i in range(trials):
            if explorer.should_explore(float(i) * 0.1, ContentionLevel.SOLO, 1):
                explore_count += 1

        # Should explore ~90 times out of 100
        assert 80 <= explore_count <= 100

    def test_solo_magnitude(self, explorer):
        """Test exploration magnitude when SOLO"""
        magnitude = explorer.get_exploration_magnitude(ContentionLevel.SOLO)

        # Should use base exploration rate (10%)
        assert magnitude == 0.1


class TestTurnAssignment:
    """Test turn assignment and phase calculation"""

    @pytest.fixture
    def config(self):
        return Config()

    def test_phase_assignment_3_flows(self, config):
        """Test 3 flows get assigned different phases"""
        # Create 3 flows
        explorers = [
            CooperativeExplorer(flow_id=i, config=config.cooperative_explorer)
            for i in range(3)
        ]

        # Test at different times within a cycle
        # Cycle = 500ms = 0.5s, divided into 3 phases
        # Phase 0: 0.0-0.167s, Phase 1: 0.167-0.333s, Phase 2: 0.333-0.5s

        times = [0.05, 0.2, 0.4]  # Times in each phase

        for i, time in enumerate(times):
            explore_decisions = [
                e.should_explore(time, ContentionLevel.MODERATE, 3)
                for e in explorers
            ]

            # Exactly one flow should explore in each phase
            assert sum(explore_decisions) == 1
            # The i-th flow should explore at time i
            assert explore_decisions[i] == True

    def test_phase_rotation(self, config):
        """Test phases rotate correctly over time"""
        explorer = CooperativeExplorer(flow_id=0, config=config.cooperative_explorer)

        # Flow 0 should explore in phase 0
        # With 3 flows, phase 0 is first ~167ms of each cycle

        # Test multiple cycles
        cycle_duration = 0.5  # 500ms
        for cycle in range(3):
            t_phase0 = cycle * cycle_duration + 0.05  # Early in cycle
            t_phase1 = cycle * cycle_duration + 0.2   # Middle of cycle
            t_phase2 = cycle * cycle_duration + 0.4   # Late in cycle

            # Should explore in phase 0, not in phase 1 or 2
            assert explorer.should_explore(t_phase0, ContentionLevel.MODERATE, 3) == True
            assert explorer.should_explore(t_phase1, ContentionLevel.MODERATE, 3) == False
            assert explorer.should_explore(t_phase2, ContentionLevel.MODERATE, 3) == False

    def test_flow_id_hash_assignment(self, config):
        """Test flow ID determines phase assignment"""
        # Create flows with IDs 0, 1, 2
        # With 3 flows: flow 0 → phase 0, flow 1 → phase 1, flow 2 → phase 2

        explorer0 = CooperativeExplorer(flow_id=0, config=config.cooperative_explorer)
        explorer1 = CooperativeExplorer(flow_id=1, config=config.cooperative_explorer)
        explorer2 = CooperativeExplorer(flow_id=2, config=config.cooperative_explorer)

        # Time = 0.2s (middle of cycle) should be phase 1 with 3 flows
        t = 0.2
        flow_count = 3

        assert explorer0.should_explore(t, ContentionLevel.MODERATE, flow_count) == False
        assert explorer1.should_explore(t, ContentionLevel.MODERATE, flow_count) == True
        assert explorer2.should_explore(t, ContentionLevel.MODERATE, flow_count) == False


class TestExplorationMagnitude:
    """Test exploration magnitude adjustment"""

    @pytest.fixture
    def explorer(self):
        config = Config()
        return CooperativeExplorer(flow_id=0, config=config.cooperative_explorer)

    def test_magnitude_solo(self, explorer):
        """Test magnitude when SOLO"""
        mag = explorer.get_exploration_magnitude(ContentionLevel.SOLO)
        assert mag == 0.1  # Base rate

    def test_magnitude_light(self, explorer):
        """Test magnitude when LIGHT contention"""
        mag = explorer.get_exploration_magnitude(ContentionLevel.LIGHT)
        assert mag == pytest.approx(0.08, abs=0.001)  # 80% of base

    def test_magnitude_moderate(self, explorer):
        """Test magnitude when MODERATE contention"""
        mag = explorer.get_exploration_magnitude(ContentionLevel.MODERATE)
        assert mag == 0.05  # Reduced rate

    def test_magnitude_heavy(self, explorer):
        """Test magnitude when HEAVY contention"""
        mag = explorer.get_exploration_magnitude(ContentionLevel.HEAVY)
        assert mag == pytest.approx(0.035, abs=0.001)  # 70% of reduced

    def test_magnitude_decreases_with_contention(self, explorer):
        """Test magnitude decreases as contention increases"""
        mag_solo = explorer.get_exploration_magnitude(ContentionLevel.SOLO)
        mag_light = explorer.get_exploration_magnitude(ContentionLevel.LIGHT)
        mag_moderate = explorer.get_exploration_magnitude(ContentionLevel.MODERATE)
        mag_heavy = explorer.get_exploration_magnitude(ContentionLevel.HEAVY)

        # Should decrease monotonically
        assert mag_solo > mag_light > mag_moderate > mag_heavy


class TestCollisionTracking:
    """Test collision tracking and statistics"""

    @pytest.fixture
    def explorer(self):
        config = Config()
        return CooperativeExplorer(flow_id=0, config=config.cooperative_explorer)

    def test_record_successful_exploration(self, explorer):
        """Test recording successful exploration"""
        explorer.record_exploration_outcome(
            timestamp=1.0,
            contention_level=ContentionLevel.MODERATE,
            exploration_magnitude=0.05,
            utility_before=10.0,
            utility_after=10.5  # Increased
        )

        assert explorer.exploration_count == 1
        assert explorer.successful_exploration_count == 1
        assert explorer.collision_count == 0
        assert len(explorer.exploration_history) == 1

    def test_record_collision(self, explorer):
        """Test recording collision"""
        explorer.record_exploration_outcome(
            timestamp=1.0,
            contention_level=ContentionLevel.MODERATE,
            exploration_magnitude=0.05,
            utility_before=10.0,
            utility_after=9.5  # Decreased
        )

        assert explorer.exploration_count == 1
        assert explorer.successful_exploration_count == 0
        assert explorer.collision_count == 1

    def test_collision_rate_calculation(self, explorer):
        """Test collision rate calculation"""
        # Record 3 successes and 2 collisions
        for i in range(3):
            explorer.record_exploration_outcome(
                timestamp=float(i),
                contention_level=ContentionLevel.MODERATE,
                exploration_magnitude=0.05,
                utility_before=10.0,
                utility_after=10.5  # Success
            )

        for i in range(2):
            explorer.record_exploration_outcome(
                timestamp=float(i + 3),
                contention_level=ContentionLevel.MODERATE,
                exploration_magnitude=0.05,
                utility_before=10.0,
                utility_after=9.5  # Collision
            )

        collision_rate = explorer.get_collision_rate()
        success_rate = explorer.get_success_rate()

        assert collision_rate == pytest.approx(0.4, abs=0.01)  # 2/5
        assert success_rate == pytest.approx(0.6, abs=0.01)   # 3/5

    def test_collision_rate_empty(self, explorer):
        """Test collision rate with no explorations"""
        assert explorer.get_collision_rate() == 0.0
        assert explorer.get_success_rate() == 1.0

    def test_history_bounded(self, explorer):
        """Test exploration history is bounded"""
        # Record 150 explorations (max = 100)
        for i in range(150):
            explorer.record_exploration_outcome(
                timestamp=float(i),
                contention_level=ContentionLevel.MODERATE,
                exploration_magnitude=0.05,
                utility_before=10.0,
                utility_after=10.5
            )

        # Should be bounded to 100
        assert len(explorer.exploration_history) == 100
        # Should keep most recent
        assert explorer.exploration_history[-1].timestamp == 149.0


class TestStatistics:
    """Test statistics reporting"""

    @pytest.fixture
    def explorer(self):
        config = Config()
        return CooperativeExplorer(flow_id=5, config=config.cooperative_explorer)

    def test_get_statistics_empty(self, explorer):
        """Test statistics with no data"""
        stats = explorer.get_statistics()

        assert stats['flow_id'] == 5
        assert stats['exploration']['total_count'] == 0
        assert stats['exploration']['collision_rate'] == 0.0
        assert stats['efficiency']['efficiency'] == 0.0

    def test_get_statistics_with_data(self, explorer):
        """Test statistics with actual data"""
        # Simulate some explorations
        for i in range(10):
            # Call should_explore
            explorer.should_explore(float(i) * 0.1, ContentionLevel.MODERATE, 3)

            # Record outcomes (7 successes, 3 collisions)
            utility_after = 10.5 if i < 7 else 9.5
            explorer.record_exploration_outcome(
                timestamp=float(i) * 0.1,
                contention_level=ContentionLevel.MODERATE,
                exploration_magnitude=0.05,
                utility_before=10.0,
                utility_after=utility_after
            )

        stats = explorer.get_statistics()

        assert stats['exploration']['total_count'] == 10
        assert stats['exploration']['collision_rate'] == 0.3
        assert stats['exploration']['success_rate'] == 0.7

    def test_get_summary(self, explorer):
        """Test human-readable summary"""
        # Add some data
        for i in range(5):
            explorer.record_exploration_outcome(
                timestamp=float(i),
                contention_level=ContentionLevel.MODERATE,
                exploration_magnitude=0.05,
                utility_before=10.0,
                utility_after=10.3
            )

        summary = explorer.get_summary()

        assert isinstance(summary, str)
        assert 'Cooperative Explorer Summary' in summary
        assert 'Flow 5' in summary
        assert 'Total Explorations' in summary
        assert 'Collision Rate' in summary


class TestReset:
    """Test reset functionality"""

    @pytest.fixture
    def explorer(self):
        config = Config()
        return CooperativeExplorer(flow_id=0, config=config.cooperative_explorer)

    def test_reset(self, explorer):
        """Test reset clears all state"""
        # Add data
        for i in range(10):
            explorer.should_explore(float(i) * 0.1, ContentionLevel.MODERATE, 3)
            explorer.record_exploration_outcome(
                timestamp=float(i) * 0.1,
                contention_level=ContentionLevel.MODERATE,
                exploration_magnitude=0.05,
                utility_before=10.0,
                utility_after=10.5
            )

        # Verify state exists
        assert explorer.exploration_count > 0
        assert len(explorer.exploration_history) > 0

        # Reset
        explorer.reset()

        # Verify state cleared
        assert explorer.exploration_count == 0
        assert explorer.collision_count == 0
        assert explorer.successful_exploration_count == 0
        assert len(explorer.exploration_history) == 0
        assert explorer.total_should_explore_calls == 0


class TestMultiFlowCoordination:
    """Test coordination between multiple flows"""

    @pytest.fixture
    def config(self):
        return Config()

    def test_no_collisions_moderate_contention(self, config):
        """Test flows take turns with MODERATE contention"""
        # Create 3 flows
        explorers = [
            CooperativeExplorer(flow_id=i, config=config.cooperative_explorer)
            for i in range(3)
        ]

        # Simulate 10 cycles
        cycle_duration = 0.5  # 500ms
        dt = 0.01  # 10ms timestep

        collision_count = 0
        simultaneous_explorations = 0

        for t in np.arange(0, 10 * cycle_duration, dt):
            # Check who wants to explore
            explore_decisions = [
                e.should_explore(t, ContentionLevel.MODERATE, 3)
                for e in explorers
            ]

            # Count simultaneous explorations
            if sum(explore_decisions) > 1:
                simultaneous_explorations += 1

        # With turn-taking, should have very few simultaneous explorations
        # (only at phase boundaries)
        assert simultaneous_explorations < 10

    def test_collision_reduction_with_coordination(self, config):
        """Test coordination reduces collisions vs random exploration"""
        # Simulate with coordination (CooperativeExplorer)
        explorers_coordinated = [
            CooperativeExplorer(flow_id=i, config=config.cooperative_explorer)
            for i in range(3)
        ]

        # Simulate without coordination (random)
        np.random.seed(42)
        cycle_duration = 0.5
        dt = 0.05

        coordinated_simultaneous = 0
        random_simultaneous = 0

        for t in np.arange(0, 5 * cycle_duration, dt):
            # Coordinated
            explore_coordinated = [
                e.should_explore(t, ContentionLevel.MODERATE, 3)
                for e in explorers_coordinated
            ]
            if sum(explore_coordinated) > 1:
                coordinated_simultaneous += 1

            # Random (30% probability each flow)
            explore_random = [np.random.random() < 0.3 for _ in range(3)]
            if sum(explore_random) > 1:
                random_simultaneous += 1

        # Coordinated should have fewer simultaneous explorations
        assert coordinated_simultaneous < random_simultaneous


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def explorer(self):
        config = Config()
        return CooperativeExplorer(flow_id=0, config=config.cooperative_explorer)

    def test_zero_flow_count(self, explorer):
        """Test handling of zero flow count"""
        # Should handle gracefully (default to 1)
        result = explorer.should_explore(1.0, ContentionLevel.MODERATE, 0)
        assert isinstance(result, bool)

    def test_negative_flow_count(self, explorer):
        """Test handling of negative flow count"""
        # Should handle gracefully (default to 1)
        result = explorer.should_explore(1.0, ContentionLevel.MODERATE, -1)
        assert isinstance(result, bool)

    def test_very_large_flow_count(self, explorer):
        """Test handling of very large flow count"""
        result = explorer.should_explore(1.0, ContentionLevel.HEAVY, 1000)
        assert isinstance(result, bool)

    def test_light_contention_relaxed_coordination(self, explorer):
        """Test LIGHT contention relaxes turn-taking"""
        # LIGHT contention allows adjacent phase exploration
        # Should explore more frequently than MODERATE

        moderate_explores = 0
        light_explores = 0

        for i in range(100):
            t = float(i) * 0.01
            if explorer.should_explore(t, ContentionLevel.MODERATE, 3):
                moderate_explores += 1
            if explorer.should_explore(t, ContentionLevel.LIGHT, 3):
                light_explores += 1

        # LIGHT should allow more exploration opportunities
        assert light_explores >= moderate_explores


class TestRealisticScenarios:
    """Test realistic multi-flow scenarios"""

    @pytest.fixture
    def config(self):
        return Config()

    def test_convergence_3_flows(self, config):
        """Test 3 flows converge to fair share with coordination"""
        np.random.seed(42)

        # Create 3 flows
        explorers = [
            CooperativeExplorer(flow_id=i, config=config.cooperative_explorer)
            for i in range(3)
        ]

        # Simulate exploration with utility feedback
        # Assume utility increases if flow gets its turn alone
        cycle_duration = 0.5
        dt = 0.05

        utilities = [10.0, 10.0, 10.0]

        for t in np.arange(0, 10 * cycle_duration, dt):
            explore_decisions = [
                e.should_explore(t, ContentionLevel.MODERATE, 3)
                for e in explorers
            ]

            # Simulate utility changes
            exploring_count = sum(explore_decisions)
            for i, exploring in enumerate(explore_decisions):
                if exploring:
                    if exploring_count == 1:
                        # Solo exploration → utility increases
                        new_utility = utilities[i] + 0.5
                    else:
                        # Collision → utility decreases
                        new_utility = utilities[i] - 0.3

                    # Record outcome
                    explorers[i].record_exploration_outcome(
                        timestamp=t,
                        contention_level=ContentionLevel.MODERATE,
                        exploration_magnitude=0.05,
                        utility_before=utilities[i],
                        utility_after=new_utility
                    )
                    utilities[i] = new_utility

        # All flows should have low collision rates due to coordination
        for explorer in explorers:
            collision_rate = explorer.get_collision_rate()
            assert collision_rate < 0.3  # Less than 30% collisions

    def test_efficiency_tracking(self, config):
        """Test exploration efficiency is tracked correctly"""
        explorer = CooperativeExplorer(flow_id=0, config=config.cooperative_explorer)

        # Call should_explore many times
        for i in range(100):
            t = float(i) * 0.01
            explorer.should_explore(t, ContentionLevel.MODERATE, 3)

        efficiency = explorer.get_exploration_efficiency()

        # With 3 flows, should explore ~1/3 of the time
        assert 0.2 < efficiency < 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
