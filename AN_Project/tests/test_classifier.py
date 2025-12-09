"""
Unit tests for Traffic Classifier
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.traffic_classifier import TrafficClassifier
from src.config import ClassifierConfig


class TestTrafficClassifier:
    """Test suite for TrafficClassifier"""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance"""
        config = ClassifierConfig()
        return TrafficClassifier(config)
    
    def test_initialization(self, classifier):
        """Test classifier initialization"""
        assert classifier.window_size == 50
        assert classifier.enabled == True
        assert len(classifier.packet_sizes) == 0
    
    def test_add_packet(self, classifier):
        """Test adding packets"""
        classifier.add_packet(1500, 0.0)
        classifier.add_packet(1500, 0.001)
        
        assert len(classifier.packet_sizes) == 2
        assert len(classifier.inter_arrivals) == 1
        assert classifier.inter_arrivals[0] == 0.001
    
    def test_bulk_traffic_classification(self, classifier):
        """Test classification of bulk transfer traffic"""
        # Generate bulk traffic: large packets, consistent size
        for i in range(60):
            classifier.add_packet(1500, i * 0.001)
        
        traffic_type, confidence = classifier.classify()
        
        assert traffic_type == 'bulk'
        assert confidence > 0.7
    
    def test_streaming_traffic_classification(self, classifier):
        """Test classification of streaming traffic"""
        # Generate streaming traffic: medium packets, constant rate
        packet_size = 1200
        for i in range(60):
            classifier.add_packet(packet_size, i * 0.002)
        
        traffic_type, confidence = classifier.classify()
        
        assert traffic_type in ['streaming', 'bulk']  # Can be either
        assert confidence > 0.5
    
    def test_realtime_traffic_classification(self, classifier):
        """Test classification of real-time traffic"""
        # Generate real-time traffic: small packets, bursty
        t = 0.0
        for i in range(80):
            if i % 10 < 5:
                # Burst
                classifier.add_packet(160, t)
                t += 0.001
            else:
                # Gap
                classifier.add_packet(160, t)
                t += 0.03
        
        traffic_type, confidence = classifier.classify()
        
        assert traffic_type in ['realtime', 'streaming']
        assert confidence > 0.4
    
    def test_entropy_computation(self, classifier):
        """Test entropy calculation"""
        # Uniform sizes - low entropy
        sizes_uniform = [1500] * 20
        entropy_uniform = classifier._compute_entropy(sizes_uniform)
        
        # Variable sizes - high entropy
        sizes_variable = list(range(100, 1500, 70))
        entropy_variable = classifier._compute_entropy(sizes_variable)
        
        assert entropy_variable > entropy_uniform
    
    def test_burst_detection(self, classifier):
        """Test burst ratio calculation"""
        # Smooth traffic
        smooth_iats = [0.01] * 20
        burst_ratio_smooth = classifier._detect_bursts(smooth_iats)
        
        # Bursty traffic
        bursty_iats = [0.001] * 10 + [0.1] * 10
        burst_ratio_bursty = classifier._detect_bursts(bursty_iats)
        
        assert burst_ratio_bursty > burst_ratio_smooth
    
    def test_insufficient_data(self, classifier):
        """Test behavior with insufficient data"""
        # Add only a few packets
        for i in range(5):
            classifier.add_packet(1500, i * 0.001)
        
        traffic_type, confidence = classifier.classify()
        
        assert traffic_type == 'default'
        assert confidence == 0.0
    
    def test_feature_extraction(self, classifier):
        """Test feature extraction"""
        # Add packets
        for i in range(60):
            classifier.add_packet(1500, i * 0.001)
        
        features = classifier.extract_features()
        
        assert 'avg_packet_size' in features
        assert 'std_packet_size' in features
        assert 'packet_size_entropy' in features
        assert 'burst_ratio' in features
        
        assert features['avg_packet_size'] == 1500
        assert features['std_packet_size'] == 0.0
    
    def test_reset(self, classifier):
        """Test reset functionality"""
        # Add data
        for i in range(20):
            classifier.add_packet(1500, i * 0.001)
        
        assert len(classifier.packet_sizes) > 0
        
        # Reset
        classifier.reset()
        
        assert len(classifier.packet_sizes) == 0
        assert len(classifier.inter_arrivals) == 0
    
    def test_window_management(self, classifier):
        """Test that old packets are removed"""
        # Add more than 2*window_size packets
        for i in range(150):
            classifier.add_packet(1500, i * 0.001)
        
        # Should keep only window_size
        assert len(classifier.packet_sizes) <= classifier.window_size
    
    def test_disabled_classifier(self):
        """Test classifier when disabled"""
        config = ClassifierConfig(enabled=False)
        classifier = TrafficClassifier(config)
        
        for i in range(60):
            classifier.add_packet(1500, i * 0.001)
        
        traffic_type, confidence = classifier.classify()
        
        assert traffic_type == 'default'
        assert confidence == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
