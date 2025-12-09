"""
Configuration management for PCC Vivace Extensions
"""
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class NetworkConfig:
    """Network simulation configuration"""
    bandwidth_mbps: float = 10.0
    delay_ms: float = 50.0
    queue_size: int = 100
    loss_rate: float = 0.0
    queue_type: str = 'fifo'  # fifo or red

    def __post_init__(self):
        """Validate network configuration parameters"""
        if self.bandwidth_mbps <= 0:
            raise ValueError(f"bandwidth_mbps must be > 0, got {self.bandwidth_mbps}")
        if self.delay_ms < 0:
            raise ValueError(f"delay_ms must be >= 0, got {self.delay_ms}")
        if self.queue_size <= 0:
            raise ValueError(f"queue_size must be > 0, got {self.queue_size}")
        if not 0 <= self.loss_rate <= 1:
            raise ValueError(f"loss_rate must be in [0, 1], got {self.loss_rate}")
        if self.queue_type not in ['fifo', 'red']:
            raise ValueError(f"queue_type must be 'fifo' or 'red', got {self.queue_type}")

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class VivaceConfig:
    """PCC Vivace algorithm configuration"""
    monitor_interval_ms: int = 100
    learning_rate: float = 0.1
    rate_min: float = 1.0  # Increased from 0.1 to prevent collapse
    rate_max: float = 100.0
    exploration_factor: float = 0.05
    momentum: float = 0.9
    rate_steps: int = 3  # number of rates to test per MI

    def __post_init__(self):
        """Validate Vivace configuration parameters"""
        if self.monitor_interval_ms <= 0:
            raise ValueError(f"monitor_interval_ms must be > 0, got {self.monitor_interval_ms}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.rate_min <= 0:
            raise ValueError(f"rate_min must be > 0, got {self.rate_min}")
        if self.rate_max <= self.rate_min:
            raise ValueError(f"rate_max ({self.rate_max}) must be > rate_min ({self.rate_min})")
        if not 0 <= self.exploration_factor <= 1:
            raise ValueError(f"exploration_factor must be in [0, 1], got {self.exploration_factor}")
        if not 0 <= self.momentum <= 1:
            raise ValueError(f"momentum must be in [0, 1], got {self.momentum}")
        if self.rate_steps <= 0:
            raise ValueError(f"rate_steps must be > 0, got {self.rate_steps}")

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class ClassifierConfig:
    """Traffic classifier configuration"""
    window_size: int = 50
    confidence_threshold: float = 0.55
    feature_set: list = field(default_factory=lambda: [
        'packet_size', 'inter_arrival', 'entropy', 'burst_ratio'
    ])
    enabled: bool = True
    expected_traffic_type: Optional[str] = None  # Override automatic classification

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class UtilityConfig:
    """Utility function parameters"""
    # Bulk transfer
    bulk_throughput_weight: float = 1.0
    bulk_latency_weight: float = 0.01
    bulk_loss_weight: float = 10.0
    
    # Streaming
    streaming_throughput_min: float = 5.0
    streaming_variance_weight: float = 5.0
    streaming_throughput_weight: float = 1.0
    
    # Real-time
    realtime_throughput_weight: float = 0.5
    realtime_latency_target: float = 50.0
    realtime_latency_slope: float = 0.1
    
    # Default
    default_throughput_weight: float = 1.0
    default_latency_sigmoid_center: float = 100.0
    default_latency_sigmoid_slope: float = 0.05
    default_loss_weight: float = 10.0
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class LossClassifierConfig:
    """Loss classification configuration"""
    # Event tracking
    window_size: int = 100  # Number of events to track
    min_events: int = 20  # Minimum events needed for classification

    # Loss detection
    loss_threshold: float = 0.001  # Minimum loss rate to consider (0.1%)

    # RTT inflation detection
    baseline_rtt_window: int = 100  # Samples for baseline RTT calculation
    rtt_inflation_margin: float = 10.0  # Absolute margin in ms
    rtt_inflation_percent: float = 0.20  # 20% relative increase

    # Correlation thresholds
    correlation_threshold_high: float = 0.5  # High correlation → congestion
    correlation_threshold_low: float = 0.2   # Low correlation → wireless

    # Utility adjustment
    wireless_penalty_reduction: float = 0.5  # Reduce loss penalty for wireless

    # Control
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class ContentionDetectorConfig:
    """Contention detection configuration (Extension 3)"""
    # Window settings
    window_size: int = 15  # Number of gradient samples to track (was 30, now 15 for faster SOLO detection)

    # Detection thresholds
    sign_change_threshold: float = 0.5  # 50% sign changes → contention (was 0.6, now 0.5 for better SOLO sensitivity)
    magnitude_threshold: float = 0.3    # Minimum gradient magnitude to count

    # Control
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class VirtualQueueEstimatorConfig:
    """Virtual queue estimation configuration (Extension 3)"""
    # Window settings
    baseline_window: int = 100  # Number of RTT samples for baseline
    throughput_window: int = 50  # Number of throughput samples for bandwidth

    # Minimum samples for estimates
    min_samples_baseline: int = 10  # Min RTT samples before baseline estimate
    min_samples_bandwidth: int = 5  # Min throughput samples before BW estimate

    # Smoothing
    smoothing_factor: float = 0.2  # EMA alpha for queue estimates

    # Control
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class CooperativeExplorerConfig:
    """Cooperative exploration configuration (Extension 3)"""
    # Exploration timing
    exploration_cycle_ms: float = 500.0  # Duration of exploration cycle (ms)

    # Exploration magnitude
    base_exploration_rate: float = 0.1  # 10% rate increase when exploring (SOLO/LIGHT)
    reduced_exploration_rate: float = 0.05  # 5% rate increase (MODERATE/HEAVY)

    # Solo behavior
    solo_exploration_probability: float = 0.9  # Probability of exploring when SOLO

    # Control
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class FairnessControllerConfig:
    """Fairness controller configuration (Extension 3)"""
    # Penalty weight
    fairness_penalty_weight: float = 0.5  # Base μ value for fairness penalty

    # Adaptation
    adaptive_mu: bool = True  # Adapt μ based on contention level

    # Smoothing
    smoothing_factor: float = 0.3  # EMA alpha for fair share estimates

    # Control
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    duration: float = 60.0
    num_flows: int = 1
    traffic_type: str = 'default'  # bulk, streaming, realtime, default
    save_results: bool = True
    output_dir: str = 'results'
    plot_realtime: bool = False
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class Config:
    """Master configuration class"""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    vivace: VivaceConfig = field(default_factory=VivaceConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    utilities: UtilityConfig = field(default_factory=UtilityConfig)
    loss_classifier: LossClassifierConfig = field(default_factory=LossClassifierConfig)
    contention_detector: ContentionDetectorConfig = field(default_factory=ContentionDetectorConfig)
    virtual_queue_estimator: VirtualQueueEstimatorConfig = field(default_factory=VirtualQueueEstimatorConfig)
    cooperative_explorer: CooperativeExplorerConfig = field(default_factory=CooperativeExplorerConfig)
    fairness_controller: FairnessControllerConfig = field(default_factory=FairnessControllerConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_file(cls, filepath: str) -> 'Config':
        """Load configuration from YAML file"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create config from dictionary"""
        config = cls()
        
        if 'network' in data:
            config.network = NetworkConfig(**data['network'])
        if 'vivace' in data:
            config.vivace = VivaceConfig(**data['vivace'])
        if 'classifier' in data:
            config.classifier = ClassifierConfig(**data['classifier'])
        if 'utilities' in data:
            config.utilities = UtilityConfig(**data['utilities'])
        if 'loss_classifier' in data:
            config.loss_classifier = LossClassifierConfig(**data['loss_classifier'])
        if 'contention_detector' in data:
            config.contention_detector = ContentionDetectorConfig(**data['contention_detector'])
        if 'virtual_queue_estimator' in data:
            config.virtual_queue_estimator = VirtualQueueEstimatorConfig(**data['virtual_queue_estimator'])
        if 'cooperative_explorer' in data:
            config.cooperative_explorer = CooperativeExplorerConfig(**data['cooperative_explorer'])
        if 'fairness_controller' in data:
            config.fairness_controller = FairnessControllerConfig(**data['fairness_controller'])
        if 'experiment' in data:
            config.experiment = ExperimentConfig(**data['experiment'])

        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'network': self.network.to_dict(),
            'vivace': self.vivace.to_dict(),
            'classifier': self.classifier.to_dict(),
            'utilities': self.utilities.to_dict(),
            'loss_classifier': self.loss_classifier.to_dict(),
            'contention_detector': self.contention_detector.to_dict(),
            'virtual_queue_estimator': self.virtual_queue_estimator.to_dict(),
            'cooperative_explorer': self.cooperative_explorer.to_dict(),
            'fairness_controller': self.fairness_controller.to_dict(),
            'experiment': self.experiment.to_dict()
        }
    
    def save(self, filepath: str):
        """Save configuration to YAML file"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def __repr__(self) -> str:
        return f"Config(network={self.network}, vivace={self.vivace})"


# Predefined configurations
def get_bulk_transfer_config() -> Config:
    """Configuration optimized for bulk transfers"""
    config = Config()
    config.experiment.traffic_type = 'bulk'
    config.utilities.bulk_throughput_weight = 1.5
    config.utilities.bulk_latency_weight = 0.005
    return config


def get_streaming_config() -> Config:
    """Configuration optimized for streaming"""
    config = Config()
    config.experiment.traffic_type = 'streaming'
    config.utilities.streaming_throughput_min = 5.0
    config.utilities.streaming_variance_weight = 7.0
    return config


def get_realtime_config() -> Config:
    """Configuration optimized for real-time applications"""
    config = Config()
    config.experiment.traffic_type = 'realtime'
    config.utilities.realtime_latency_target = 40.0
    config.utilities.realtime_throughput_weight = 0.3
    return config


def get_wireless_config() -> Config:
    """Configuration for wireless networks"""
    config = Config()
    config.network.loss_rate = 0.01
    config.loss_classifier.enabled = True
    config.loss_classifier.wireless_penalty_reduction = 0.6
    return config
