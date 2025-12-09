"""
Unit tests for Extensions 3 & 4
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.multiflow_vivace import (
    CoordinatedVivace, FlowCoordinator, run_multiflow_experiment, compute_jain_fairness
)
from src.multipath_vivace import (
    MultipathNetwork, MultipathVivace, create_multipath_scenario
)


class TestMultiFlowCoordination:
    """Test suite for multi-flow coordination"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = Config()
        config.network.bandwidth_mbps = 10.0
        config.network.delay_ms = 50.0
        config.experiment.duration = 5.0
        return config
    
    def test_flow_coordinator_creation(self):
        """Test flow coordinator initialization"""
        coordinator = FlowCoordinator()
        assert coordinator.get_num_flows() == 0
    
    def test_flow_registration(self, config):
        """Test flow registration with coordinator"""
        from src.network_simulator import NetworkSimulator
        
        network = NetworkSimulator(config.network)
        coordinator = FlowCoordinator()
        
        flow = CoordinatedVivace(network, config, flow_id=0, coordinator=coordinator)
        coordinator.register_flow(0, flow)
        
        assert coordinator.get_num_flows() == 1
    
    def test_contention_detection(self, config):
        """Test contention detection mechanism"""
        from src.network_simulator import NetworkSimulator
        
        network = NetworkSimulator(config.network)
        flow = CoordinatedVivace(network, config, flow_id=0)
        
        # Add oscillating gradients
        for i in range(20):
            flow.gradient_history.append(1.0 if i % 2 == 0 else -1.0)
        
        contention = flow.detect_contention()
        assert contention == True
    
    def test_fair_share_estimation(self, config):
        """Test fair share estimation"""
        from src.network_simulator import NetworkSimulator
        
        network = NetworkSimulator(config.network)
        flow = CoordinatedVivace(network, config, flow_id=0)
        
        fair_share = flow.estimate_fair_share(total_capacity=10.0, num_flows=4)
        
        assert abs(fair_share - 2.5) < 0.01  # 10 / 4 = 2.5
    
    def test_alternating_exploration(self, config):
        """Test alternating exploration mechanism"""
        from src.network_simulator import NetworkSimulator
        
        network = NetworkSimulator(config.network)
        coordinator = FlowCoordinator()
        
        # Create 3 flows
        flows = []
        for i in range(3):
            flow = CoordinatedVivace(network, config, flow_id=i, coordinator=coordinator)
            coordinator.register_flow(i, flow)
            flows.append(flow)
        
        # Test alternating pattern
        for iteration in range(9):
            for flow_id, flow in enumerate(flows):
                flow.iteration = iteration
                should_explore = flow.should_explore()
                expected = (iteration % 3) == flow_id
                assert should_explore == expected
    
    def test_jain_fairness_calculation(self):
        """Test Jain's fairness index calculation"""
        # Perfect fairness
        fair_values = [10.0, 10.0, 10.0, 10.0]
        fairness = compute_jain_fairness(fair_values)
        assert abs(fairness - 1.0) < 0.01
        
        # Unfair
        unfair_values = [20.0, 5.0, 5.0, 5.0]
        fairness = compute_jain_fairness(unfair_values)
        assert fairness < 0.9
    
    def test_multiflow_experiment(self, config):
        """Test complete multi-flow experiment"""
        config.experiment.duration = 3.0  # Short for testing
        
        results = run_multiflow_experiment(config, num_flows=2, duration=3.0)
        
        assert 'num_flows' in results
        assert results['num_flows'] == 2
        assert 'flows' in results
        assert len(results['flows']) == 2
        assert 'fairness_index' in results
        assert 0 <= results['fairness_index'] <= 1.0
    
    def test_fairness_penalty(self, config):
        """Test fairness penalty computation"""
        from src.network_simulator import NetworkSimulator
        
        network = NetworkSimulator(config.network)
        flow = CoordinatedVivace(network, config, flow_id=0)
        
        flow.current_rate = 10.0
        flow.estimated_fair_share = 5.0
        
        penalty = flow.compute_fairness_penalty()
        assert penalty > 0  # Should penalize deviation
    
    def test_virtual_queue_estimation(self, config):
        """Test virtual queue estimation"""
        from src.network_simulator import NetworkSimulator
        
        network = NetworkSimulator(config.network)
        flow = CoordinatedVivace(network, config, flow_id=0)
        
        # Simulate RTT inflation
        baseline_rtt = 100.0  # ms
        inflated_rtt = 150.0  # ms
        
        flow.update_virtual_queue(inflated_rtt, baseline_rtt)
        
        assert flow.virtual_queue > 0  # Should detect queuing


class TestMultipathAllocation:
    """Test suite for multipath rate allocation"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = Config()
        config.experiment.duration = 3.0
        return config
    
    def test_multipath_network_creation(self):
        """Test multipath network initialization"""
        paths = [
            {'bandwidth_mbps': 10.0, 'delay_ms': 50.0, 'name': 'Path 1'},
            {'bandwidth_mbps': 5.0, 'delay_ms': 100.0, 'name': 'Path 2'}
        ]
        
        mp_network = MultipathNetwork(paths)
        
        assert mp_network.get_num_paths() == 2
        assert mp_network.path_names[0] == 'Path 1'
        assert mp_network.path_names[1] == 'Path 2'
    
    def test_predefined_scenarios(self):
        """Test predefined multipath scenarios"""
        scenarios = ['heterogeneous', 'symmetric', 'cellular_wifi']
        
        for scenario in scenarios:
            mp_network = create_multipath_scenario(scenario)
            assert mp_network.get_num_paths() >= 2
    
    def test_multipath_vivace_creation(self, config):
        """Test multipath Vivace initialization"""
        mp_network = create_multipath_scenario('symmetric')
        vivace = MultipathVivace(mp_network, config)
        
        assert vivace.num_paths == 2
        assert len(vivace.path_rates) == 2
        assert len(vivace.path_utilities) == 2
    
    def test_softmax_allocation(self, config):
        """Test softmax rate allocation"""
        mp_network = create_multipath_scenario('symmetric')
        vivace = MultipathVivace(mp_network, config)
        
        # Set different utilities for paths
        vivace.path_utilities = [10.0, 5.0]
        
        # Allocate 10 Mbps total
        allocated_rates = vivace.allocate_rates_softmax(10.0)
        
        # Check properties
        assert len(allocated_rates) == 2
        assert sum(allocated_rates) <= 10.1  # Allow small tolerance
        assert all(r >= vivace.min_path_rate for r in allocated_rates)
        
        # Higher utility path should get more rate
        assert allocated_rates[0] > allocated_rates[1]
    
    def test_path_correlation_learning(self, config):
        """Test path correlation learning"""
        mp_network = create_multipath_scenario('symmetric')
        vivace = MultipathVivace(mp_network, config)
        
        # Add correlated metrics
        for _ in range(10):
            for path_id in range(2):
                value = np.random.random() * 10
                metrics = {
                    'throughput': value,
                    'latency': 50.0,
                    'loss': 0.0
                }
                vivace.path_metrics_history[path_id].append(metrics)
        
        vivace.learn_path_correlations()
        
        # Check correlation matrix is valid
        assert vivace.path_correlations.shape == (2, 2)
        assert abs(vivace.path_correlations[0][0] - 1.0) < 0.1  # Diagonal should be ~1
    
    def test_correlation_adjustment(self, config):
        """Test utility adjustment for correlation"""
        mp_network = create_multipath_scenario('symmetric')
        vivace = MultipathVivace(mp_network, config)
        
        # Set high correlation
        vivace.path_correlations = np.array([[1.0, 0.9], [0.9, 1.0]])
        
        utilities = [10.0, 8.0]
        adjusted = vivace.account_for_correlation(utilities)
        
        # Lower utility path should be penalized
        assert adjusted[1] < utilities[1]
    
    def test_multipath_run(self, config):
        """Test complete multipath run"""
        mp_network = create_multipath_scenario('heterogeneous')
        vivace = MultipathVivace(mp_network, config)
        
        results = vivace.run(duration=3.0)
        
        assert 'avg_throughput' in results
        assert 'final_path_rates' in results
        assert 'per_path' in results
        assert len(results['per_path']) == 2
        
        # Check per-path stats
        for path_stats in results['per_path']:
            assert 'avg_throughput' in path_stats
            assert 'final_rate' in path_stats
    
    def test_aggregate_metrics(self, config):
        """Test aggregation of metrics across paths"""
        mp_network = create_multipath_scenario('symmetric')
        vivace = MultipathVivace(mp_network, config)
        
        # Add metrics for both paths
        vivace.path_rates = [5.0, 5.0]
        vivace.path_metrics_history[0].append({
            'throughput': 4.8,
            'latency': 60.0,
            'loss': 0.01
        })
        vivace.path_metrics_history[1].append({
            'throughput': 4.9,
            'latency': 70.0,
            'loss': 0.02
        })
        
        aggregate = vivace._compute_aggregate_metrics()
        
        # Throughput should be sum
        assert abs(aggregate['throughput'] - 9.7) < 0.1
        
        # Latency should be weighted average
        assert 60.0 <= aggregate['latency'] <= 70.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
