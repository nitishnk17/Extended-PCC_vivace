"""
Utility Function Bank
Different utility functions for different traffic types
"""
import numpy as np
from typing import Callable, Dict, List
import logging

logger = logging.getLogger(__name__)


class UtilityFunctionBank:
    """
    Bank of utility functions optimized for different traffic types
    """
    
    def __init__(self, config):
        """
        Initialize utility function bank
        
        Args:
            config: UtilityConfig object with weights for each traffic type
        """
        self.config = config
        
        # Throughput history for variance calculation
        self.throughput_history = []
        self.history_window = 10
        
        # Register utility functions
        self.functions = {
            'bulk': self.utility_bulk,
            'streaming': self.utility_streaming,
            'realtime': self.utility_realtime,
            'default': self.utility_default
        }
        
        logger.info("Utility function bank initialized")
    
    def get_utility_function(self, traffic_type: str) -> Callable:
        """
        Get utility function for given traffic type
        
        Args:
            traffic_type: 'bulk', 'streaming', 'realtime', or 'default'
            
        Returns:
            Utility function
        """
        func = self.functions.get(traffic_type, self.utility_default)
# TODO: add more utility functions (web browsing, file transfer, etc)
        logger.debug(f"Selected utility function: {traffic_type}")
        return func
    
    def utility_bulk(self, throughput: float, latency: float, loss: float) -> float:
        """
        Utility for bulk transfer: maximize throughput with latency awareness
        U = α₁·T^0.9·S(L) - α₂·T·l

        Similar to baseline but optimized for bulk transfer:
        - Uses sigmoid latency penalty to aggressively penalize queue buildup
        - Sublinear throughput (T^0.9) creates diminishing returns
        - This ensures proper convergence in PCC Vivace's gradient ascent

        Args:
            throughput: Throughput in Mbps
            latency: Latency in ms
            loss: Loss rate (0-1)

        Returns:
            Utility value
        """
        alpha1 = 1.0           # Throughput weight
        alpha2 = 11.0          # Loss penalty weight
        exponent = 0.9         # Sublinear throughput (diminishing returns)

        # Latency sigmoid parameters (with slight adjustment for high-latency tolerance)
        L_threshold = 105.0    # Latency threshold (slightly increased from 100.0)
        L_scale = 12.0         # Sigmoid slope (slightly increased from 10.0 for smoother transition)

        # Sublinear throughput term
        if throughput > 0:
            throughput_term = alpha1 * np.power(throughput, exponent)
        else:
            throughput_term = 0.0

        # Sigmoid latency penalty (penalizes queue buildup)
        if latency > 0:
            latency_factor = 1.0 / (1.0 + np.exp((latency - L_threshold) / L_scale))
        else:
            latency_factor = 1.0

        # Loss penalty
        loss_penalty = alpha2 * throughput * loss

        utility = throughput_term * latency_factor - loss_penalty

        # Allow negative values for extreme cases (high loss)
        return utility
    
    def utility_streaming(self, throughput: float, latency: float, loss: float) -> float:
        """
        Utility for streaming: stable throughput, minimize variance
        U = γ₁·T^0.9·I(T > T_min) - γ₂·Var(T) - γ₃·T·l

        Uses sublinear throughput for proper convergence, with variance penalty
        to encourage stable bitrate (important for video streaming).

        Args:
            throughput: Throughput in Mbps
            latency: Latency in ms
            loss: Loss rate (0-1)

        Returns:
            Utility value
        """
        # Update history
        self.throughput_history.append(throughput)
        if len(self.throughput_history) > self.history_window:
            self.throughput_history.pop(0)

        T_min = 2.0          # Minimum acceptable throughput (Mbps) - lowered for realistic scenarios
        gamma1 = 1.2         # Throughput weight (increased for better positive utilities)
        gamma2 = 0.5         # Variance penalty weight (reduced to avoid over-penalizing)
        gamma3 = 10.0        # Loss penalty weight (slightly increased from 8.0 for better stability)
        exponent = 0.9       # Sublinear throughput

        # Sublinear throughput with soft threshold
        if throughput > 0:
            throughput_term = gamma1 * np.power(throughput, exponent)
            # Apply gentler soft penalty if below minimum (using sqrt instead of linear)
            if throughput < T_min:
                throughput_term *= np.sqrt(throughput / T_min)
        else:
            throughput_term = 0.0

        # Variance penalty (penalize jitter) - keep it light
        if len(self.throughput_history) >= 5:
            variance = np.var(self.throughput_history)
            variance_penalty = gamma2 * variance
        else:
            variance_penalty = 0.0

        # Loss penalty
        loss_penalty = gamma3 * throughput * loss

        utility = throughput_term - variance_penalty - loss_penalty
        # Allow negative values for edge cases
        return utility
    
    def utility_realtime(self, throughput: float, latency: float, loss: float) -> float:
        """
        Utility for realtime: minimize latency aggressively
        U = β₁·T^0.9·f(L) - β₂·T·l

        Uses sublinear throughput with aggressive latency penalty.
        Realtime apps (VoIP, gaming) prioritize low latency over high throughput.

        Args:
            throughput: Throughput in Mbps
            latency: Latency in ms
            loss: Loss rate (0-1)

        Returns:
            Utility value
        """
        beta1 = 1.0           # Throughput weight (increased to match baseline scaling)
        beta2 = 12.0          # Loss penalty weight (balanced for stability)
        exponent = 0.9        # Sublinear throughput
        L_target = 50.0       # Target latency (ms)
        L_max = 200.0         # Maximum acceptable latency

        # Sublinear throughput
        if throughput > 0:
            throughput_term = beta1 * np.power(throughput, exponent)
        else:
            throughput_term = 0.0

        # Latency penalty function (balanced for stability)
        if latency <= L_target:
            # Small bonus for very low latency
            latency_factor = 1.0 + (L_target - latency) / (L_target * 8)
        elif latency <= L_max:
            # Linear degradation from 1.0 to 0.0
            latency_factor = max(0.0, 1.0 - (latency - L_target) / (L_max - L_target))
        else:
            # Small factor instead of zero to allow recovery
            latency_factor = 0.05

        # Loss has significant penalty for realtime
        loss_penalty = beta2 * throughput * loss

        utility = throughput_term * latency_factor - loss_penalty
        # Allow negative values for edge cases
        return utility
    
    def utility_default(self, throughput: float, latency: float, loss: float) -> float:
        """
        Default utility (original Vivace)
        
        U = T·S(L) - λ·T·l
        
        Args:
            throughput: Throughput in Mbps
            latency: Latency in ms
            loss: Loss rate (0-1)
            
        Returns:
            Utility value
        """
        weight = self.config.default_throughput_weight
        L_center = self.config.default_latency_sigmoid_center
        L_slope = self.config.default_latency_sigmoid_slope
        lambda_loss = self.config.default_loss_weight
        
        # Sigmoid for latency penalty
        sigmoid_value = 1.0 / (1.0 + np.exp(-L_slope * (latency - L_center)))
        
        utility = weight * throughput * sigmoid_value - lambda_loss * throughput * loss
        
        logger.debug(f"Default utility: {utility:.3f} (T={throughput:.2f}, L={latency:.2f}, l={loss:.4f})")
        return utility
    
    def register_custom_utility(self, name: str, func: Callable):
        """
        Register a custom utility function
        
        Args:
            name: Name for the utility function
            func: Function with signature (throughput, latency, loss) -> utility
        """
        self.functions[name] = func
        logger.info(f"Registered custom utility function: {name}")
    
    def reset_history(self):
        """Reset throughput history (for streaming variance)"""
        self.throughput_history.clear()
    
    def get_utility_gradient(self, 
                            throughput_samples: List[float],
                            latency_samples: List[float],
                            loss_samples: List[float],
                            rate_samples: List[float],
                            traffic_type: str) -> float:
        """
        Compute utility gradient using finite differences
        
        Args:
            throughput_samples: List of throughput measurements at different rates
            latency_samples: List of latency measurements
            loss_samples: List of loss measurements
            rate_samples: List of rates tested
            traffic_type: Traffic type for utility function selection
            
        Returns:
            Estimated gradient
        """
        utility_func = self.get_utility_function(traffic_type)
        
        # Compute utilities for all samples
        utilities = []
        for t, l, loss in zip(throughput_samples, latency_samples, loss_samples):
            u = utility_func(t, l, loss)
            utilities.append(u)
        
        # Finite difference gradient
        if len(utilities) >= 2 and len(rate_samples) >= 2:
            # Use highest and lowest rate samples
            max_idx = np.argmax(rate_samples)
            min_idx = np.argmin(rate_samples)
            
            rate_diff = rate_samples[max_idx] - rate_samples[min_idx]
            utility_diff = utilities[max_idx] - utilities[min_idx]
            
            if abs(rate_diff) > 1e-6:
                gradient = utility_diff / rate_diff
            else:
                gradient = 0.0
        else:
            gradient = 0.0
        
        return gradient
    
    def compare_utilities(self, 
                         throughput: float, 
                         latency: float, 
                         loss: float) -> Dict[str, float]:
        """
        Compare utility values across all traffic types for given metrics
        
        Args:
            throughput: Throughput in Mbps
            latency: Latency in ms
            loss: Loss rate
            
        Returns:
            Dictionary mapping traffic_type to utility value
        """
        results = {}
        for name, func in self.functions.items():
            results[name] = func(throughput, latency, loss)
        
        return results
