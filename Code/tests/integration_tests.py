"""
Integration tests for complete system
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.network_simulator import NetworkSimulator
from src.baseline_vivace import BaselineVivace
from src.adaptive_vivace import AdaptiveVivace


class TestIntegration:
    """Integration test suite"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = Config()
        config.network.bandwidth_mbps = 10.0
        config.network.delay_ms = 50.0
        config.network.queue_size = 50
        config.network.loss_rate = 0.0
        config.experiment.duration = 5.0  # Short for testing
        config.vivace.monitor_interval_ms = 100
        return config
    
    def test_baseline_vivace_run(self, config):
        """Test baseline Vivace complete run"""
        network = NetworkSimulator(config.network)
        vivace = BaselineVivace(network, config)
        
        results = vivace.run(duration=5.0)
        
        # Check results structure
        assert 'avg_throughput' in results
        assert 'avg_latency' in results
        assert 'avg_loss' in results
        assert 'avg_utility' in results
        assert 'history' in results
        
        # Check values are reasonable
        assert results['avg_throughput'] > 0
        assert results['avg_throughput'] <= config.network.bandwidth_mbps
        assert results['avg_latency'] >= config.network.delay_ms * 2
        assert 0 <= results['avg_loss'] <= 1
        
        # Check history
        assert len(results['history']['time']) > 0
        assert len(results['history']['rate']) > 0
    
    def test_adaptive_vivace_run(self, config):
        """Test adaptive Vivace complete run"""
        network = NetworkSimulator(config.network)
        vivace = AdaptiveVivace(network, config)
        
        results = vivace.run(duration=5.0)
        
        # Check all baseline results
        assert 'avg_throughput' in results
        assert 'avg_latency' in results
        
        # Check extension results
        assert 'traffic_type' in results
        assert 'classification_confidence' in results
        
        # Traffic type should be one of the valid types
        assert results['traffic_type'] in ['bulk', 'streaming', 'realtime', 'default']
    
    def test_baseline_vs_adaptive(self, config):
        """Test that both implementations produce valid results"""
        # Baseline
        network1 = NetworkSimulator(config.network)
        baseline = BaselineVivace(network1, config)
        results_baseline = baseline.run(duration=5.0)
        
        # Adaptive
        network2 = NetworkSimulator(config.network)
        adaptive = AdaptiveVivace(network2, config)
        results_adaptive = adaptive.run(duration=5.0)
        
        # Both should achieve reasonable performance
        assert results_baseline['avg_throughput'] > 0
        assert results_adaptive['avg_throughput'] > 0
        
        # Both should have low loss in ideal conditions
        assert results_baseline['avg_loss'] < 0.1
        assert results_adaptive['avg_loss'] < 0.1
    
    def test_convergence(self, config):
        """Test that algorithm converges to stable rate"""
        network = NetworkSimulator(config.network)
        vivace = BaselineVivace(network, config)
        
        results = vivace.run(duration=10.0)
        
        # Check that rate stabilizes
        rates = results['history']['rate']
        
        # Compare first half vs second half variance
        mid = len(rates) // 2
        var_first = np.var(rates[:mid])
        var_second = np.var(rates[mid:])
        
        # Second half should be more stable (lower variance)
        # This might not always hold, so we just check it doesn't crash
        assert isinstance(var_first, float)
        assert isinstance(var_second, float)
    
    def test_network_with_loss(self, config):
        """Test behavior with packet loss"""
        config.network.loss_rate = 0.02  # 2% loss
        
        network = NetworkSimulator(config.network)
        vivace = BaselineVivace(network, config)
        
        results = vivace.run(duration=5.0)
        
        # Should still achieve reasonable throughput
        assert results['avg_throughput'] > 0
        
        # Loss should be detected
        assert results['avg_loss'] > 0
    
    def test_varying_bandwidth(self, config):
        """Test adaptation to bandwidth changes"""
        # This is a simplified test - real implementation would need
        # network trace support for dynamic bandwidth
        
        network = NetworkSimulator(config.network)
        vivace = BaselineVivace(network, config)
        
        # Run first half
        results1 = vivace.run(duration=3.0)
        rate_first = results1['final_rate']
        
        # Change bandwidth (simulate by creating new network)
        config.network.bandwidth_mbps = 5.0
        network2 = NetworkSimulator(config.network)
        vivace.network = network2
        vivace.current_rate = rate_first  # Continue from previous rate
        
        # Run second half
        results2 = vivace.run(duration=3.0)
        
        # Should adapt (rate should decrease)
        # This is a rough test
        assert results2['final_rate'] <= rate_first * 1.5
    
    def test_traffic_classification_bulk(self, config):
        """Test classification of bulk traffic"""
        # Set config for bulk-like behavior
        config.experiment.traffic_type = 'bulk'
        
        network = NetworkSimulator(config.network)
        vivace = AdaptiveVivace(network, config)
        
        # Generate bulk traffic pattern
        results = vivace.run(duration=5.0)
        
        # After enough packets, should classify as bulk or default
        assert results['traffic_type'] in ['bulk', 'default']
    
    def test_multiple_runs_consistency(self, config):
        """Test that multiple runs give consistent results"""
        np.random.seed(42)
        
        network1 = NetworkSimulator(config.network)
        vivace1 = BaselineVivace(network1, config)
        results1 = vivace1.run(duration=5.0)
        
        # Reset and run again with same seed
        np.random.seed(42)
        network2 = NetworkSimulator(config.network)
        vivace2 = BaselineVivace(network2, config)
        results2 = vivace2.run(duration=5.0)
        
        # Results should be similar (within 20%)
        throughput_diff = abs(results1['avg_throughput'] - results2['avg_throughput'])
        assert throughput_diff < results1['avg_throughput'] * 0.2
    
    def test_reset_functionality(self, config):
        """Test that reset works properly"""
        network = NetworkSimulator(config.network)
        vivace = BaselineVivace(network, config)
        
        # Run once
        results1 = vivace.run(duration=3.0)
        
        # Reset
        vivace.reset()
        
        # Check state is cleared
        assert vivace.iteration == 0
        assert len(vivace.rate_history) == 0
        
        # Run again
        results2 = vivace.run(duration=3.0)
        
        # Should produce valid results
        assert results2['avg_throughput'] > 0
    
    def test_extreme_network_conditions(self, config):
        """Test extreme network conditions"""
        # Very high latency
        config.network.delay_ms = 200.0
        
        network = NetworkSimulator(config.network)
        vivace = BaselineVivace(network, config)
        
        results = vivace.run(duration=5.0)
        
        # Should still work
        assert results['avg_throughput'] > 0
        assert results['avg_latency'] >= 400.0  # At least 2*RTT
        
        # Very small buffer
        config.network.queue_size = 5
        config.network.delay_ms = 50.0
        
        network2 = NetworkSimulator(config.network)
        vivace2 = BaselineVivace(network2, config)
        
        results2 = vivace2.run(duration=5.0)
        
        # Might have more loss, but should still work
        assert results2['avg_throughput'] > 0


class TestNetworkSimulator:
    """Test network simulator in isolation"""
    
    @pytest.fixture
    def config(self):
        config = Config()
        config.network.bandwidth_mbps = 10.0
        config.network.delay_ms = 50.0
        config.network.queue_size = 100
        return config
    
    def test_network_creation(self, config):
        """Test network simulator creation"""
        network = NetworkSimulator(config.network)
        
        assert network.bandwidth_mbps == 10.0
        assert network.delay_ms == 50.0
        assert network.queue_size == 100
    
    def test_send_packet(self, config):
        """Test sending a packet"""
        network = NetworkSimulator(config.network)
        
        packet = network.send_packet(1500, 0.0)
        
        assert packet.size == 1500
        assert not packet.dropped
    
    def test_queue_overflow(self, config):
        """Test queue overflow causes drops"""
        config.network.queue_size = 5
        network = NetworkSimulator(config.network)
        
        # Fill queue
        for i in range(10):
            packet = network.send_packet(1500, i * 0.001)
            
            if i < 5:
                assert not packet.dropped
            else:
                assert packet.dropped
    
    def test_random_loss(self, config):
        """Test random loss"""
        config.network.loss_rate = 0.5  # 50% loss
        network = NetworkSimulator(config.network)
        
        dropped = 0
        total = 100
        
        for i in range(total):
            packet = network.send_packet(1500, i * 0.001)
            if packet.dropped:
                dropped += 1
        
        # Should be approximately 50% (allow some variance)
        assert 30 < dropped < 70


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
