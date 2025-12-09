"""
Network Simulator for PCC Vivace testing
Simulates bottleneck link with queuing, loss, and delay
"""
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import time
import threading


@dataclass
class Packet:
    """Represents a network packet"""
    id: int
    flow_id: int  # Which flow this packet belongs to
    size: int  # bytes
    send_time: float  # seconds
    enqueue_time: Optional[float] = None
    dequeue_time: Optional[float] = None
    arrival_time: Optional[float] = None
    dropped: bool = False

    def __repr__(self):
        return f"Packet(id={self.id}, flow={self.flow_id}, size={self.size}, dropped={self.dropped})"

    def get_rtt(self) -> float:
        """
        Get RTT for this packet in milliseconds

        Returns:
            RTT in ms, or 0 if packet hasn't been acknowledged
        """
        if self.arrival_time is not None and self.send_time is not None:
            return (self.arrival_time - self.send_time) * 1000  # Convert to ms
        return 0.0


class NetworkSimulator:
    """
    Simulates a bottleneck network link with:
    - Configurable bandwidth
    - Propagation delay
    - Queue management (FIFO or RED)
    - Random packet loss
    - Dynamic bandwidth changes
    """
    
    def __init__(self, config):
        """
        Initialize network simulator
        
        Args:
            config: NetworkConfig object with bandwidth, delay, queue_size, loss_rate
        """
        self.bandwidth_mbps = config.bandwidth_mbps
        self.delay_ms = config.delay_ms
        self.queue_size = config.queue_size
        self.loss_rate = config.loss_rate
        self.queue_type = config.queue_type
        
        # State
        self.queue = deque()
        self.current_time = 0.0
        self.packet_counter = 0
        self.bytes_sent = 0
        self.packets_sent = 0
        self.packets_dropped = 0
        self.packets_received = 0

        # Window-based tracking for accurate metrics
        self.window_bytes_sent = 0
        self.window_packets_sent = 0
        self.window_packets_dropped = 0

        # Per-flow tracking
        self.flow_stats = {}  # flow_id -> stats

        # Metrics tracking (bounded to prevent memory leaks)
        self.max_history_length = 10000  # Keep last 10000 samples
        self.queue_occupancy_history = []
        self.bandwidth_utilization_history = []
        
        # Convert to convenient units
        self.bandwidth_bps = self.bandwidth_mbps * 1e6
        self.delay_s = self.delay_ms / 1000.0
        
        # For realistic simulation
        self.last_send_time = 0.0
        self.bytes_in_transmission = 0
        self.transmission_end_time = 0.0
        
        # Dynamic bandwidth support
        self.bandwidth_schedule = []  # List of (time, bandwidth) tuples
    # TODO: support more realistic queue management (RED, CoDel, etc)
        self.original_bandwidth = self.bandwidth_mbps

        # Thread safety for multi-flow scenarios
        self._lock = threading.Lock()
        
    def set_bandwidth_schedule(self, schedule: List[Tuple[float, float]]):
        """
        Set dynamic bandwidth changes
        
        Args:
            schedule: List of (time, bandwidth_mbps) tuples
        """
        self.bandwidth_schedule = sorted(schedule, key=lambda x: x[0])
    
    def get_current_bandwidth(self, time: float) -> float:
        """Get bandwidth at given time considering schedule"""
        if not self.bandwidth_schedule:
            return self.bandwidth_mbps
        
        # Find applicable bandwidth
        current_bw = self.original_bandwidth
        for sched_time, sched_bw in self.bandwidth_schedule:
            if time >= sched_time:
                current_bw = sched_bw
            else:
                break
        
        # Update if changed
        if current_bw != self.bandwidth_mbps:
            self.bandwidth_mbps = current_bw
            self.bandwidth_bps = current_bw * 1e6
        
        return current_bw
        
    def reset(self):
        """Reset simulator state"""
        self.queue.clear()
        self.current_time = 0.0
        self.packet_counter = 0
        self.bytes_sent = 0
        self.packets_sent = 0
        self.packets_dropped = 0
        self.packets_received = 0
        self.window_bytes_sent = 0
        self.window_packets_sent = 0
        self.window_packets_dropped = 0
        self.flow_stats.clear()
        self.queue_occupancy_history.clear()
        self.bandwidth_utilization_history.clear()
        self.last_send_time = 0.0
        self.bytes_in_transmission = 0
        self.transmission_end_time = 0.0
        self.bandwidth_mbps = self.original_bandwidth
        self.bandwidth_bps = self.original_bandwidth * 1e6

    def reset_window_stats(self):
        """Reset statistics for new measurement window (compatibility method)"""
        # This is a compatibility method for PCC implementations
        # NetworkSimulator doesn't maintain separate window stats, so this is a no-op
        pass

    def get_rtt_estimate(self) -> float:
        """Get current RTT estimate (propagation delay + queuing delay)"""
        # Base RTT is 2x propagation delay (forward + backward)
        base_rtt_ms = 2 * self.delay_ms

        # Add queuing delay estimate based on current queue occupancy
        queue_delay_ms = 0.0
        if len(self.queue) > 0 and self.bandwidth_bps > 0:
            # Estimate time to drain current queue
            queue_bytes = sum(p.size for p in self.queue)
            queue_delay_s = queue_bytes * 8 / self.bandwidth_bps
            queue_delay_ms = queue_delay_s * 1000

        return base_rtt_ms + queue_delay_ms

    def send_packet(self, size: int, current_time: float, flow_id: int = 0) -> Packet:
        """
        Send a packet through the network

        Args:
            size: Packet size in bytes
            current_time: Current simulation time
            flow_id: Flow identifier

        Returns:
            Packet object with timing information
        """
        with self._lock:
            self.current_time = current_time

            # Update bandwidth if schedule exists
            self.get_current_bandwidth(current_time)

            # Create packet
            packet = Packet(
                id=self.packet_counter,
                flow_id=flow_id,
                size=size,
                send_time=current_time
            )
            self.packet_counter += 1

            # Initialize flow stats if needed
            if flow_id not in self.flow_stats:
                self.flow_stats[flow_id] = {
                    'packets_sent': 0,
                    'packets_dropped': 0,
                    'bytes_sent': 0
                }

            # Random loss (wireless, corruption, etc.)
            if np.random.random() < self.loss_rate:
                packet.dropped = True
                self.packets_dropped += 1
                self.window_packets_dropped += 1
                self.flow_stats[flow_id]['packets_dropped'] += 1
                return packet

            # Check queue capacity (congestion loss)
            if len(self.queue) >= self.queue_size:
                packet.dropped = True
                self.packets_dropped += 1
                self.window_packets_dropped += 1
                self.flow_stats[flow_id]['packets_dropped'] += 1
                return packet

            # Add to queue
            packet.enqueue_time = current_time
            self.queue.append(packet)
            self._append_bounded_history(self.queue_occupancy_history, len(self.queue))

            return packet
    
    def process_queue(self, duration: float) -> List[Packet]:
        """
        Process queue for a given duration
        
        Args:
            duration: Time duration to process (seconds)
            
        Returns:
            List of packets that completed transmission
        """
        completed_packets = []
        time_remaining = duration
        
        while time_remaining > 0 and len(self.queue) > 0:
            # Check if currently transmitting a packet
            # FIXME: this doesnt account for partial packet transmission correctly
            if self.transmission_end_time > self.current_time:
                wait_time = min(self.transmission_end_time - self.current_time, time_remaining)
                self.current_time += wait_time
                time_remaining -= wait_time
                
                if time_remaining <= 0:
                    break
            
            # Start transmitting next packet
            packet = self.queue.popleft()
            
            # Calculate transmission time for this packet
            transmission_time = (packet.size * 8) / self.bandwidth_bps
            
            if transmission_time <= time_remaining:
                # Packet completes in this interval
                packet.dequeue_time = self.current_time
                
                # Add propagation delay
                packet.arrival_time = self.current_time + transmission_time + self.delay_s

                self.bytes_sent += packet.size
                self.packets_sent += 1
                self.packets_received += 1
                self.window_bytes_sent += packet.size
                self.window_packets_sent += 1

                # Update flow stats
                self.flow_stats[packet.flow_id]['packets_sent'] += 1
                self.flow_stats[packet.flow_id]['bytes_sent'] += packet.size
                
                completed_packets.append(packet)
                time_remaining -= transmission_time
                self.current_time += transmission_time
                self.transmission_end_time = self.current_time
            else:
                # Partial transmission - put back in queue
                self.queue.appendleft(packet)
                self.transmission_end_time = self.current_time + transmission_time
                time_remaining = 0
                
        return completed_packets
    
    def get_metrics(self, window_duration: float, flow_id: Optional[int] = None) -> dict:
        """
        Get current network metrics for the measurement window

        Args:
            window_duration: Time window for metrics calculation
            flow_id: If specified, return metrics for specific flow

        Returns:
            Dictionary with throughput, latency, loss rate
        """
        if flow_id is not None and flow_id in self.flow_stats:
            # Per-flow metrics
            stats = self.flow_stats[flow_id]

            if window_duration > 0:
                throughput = (stats['bytes_sent'] * 8) / (window_duration * 1e6)
            else:
                throughput = 0.0

            total = stats['packets_sent'] + stats['packets_dropped']
            loss_rate = stats['packets_dropped'] / total if total > 0 else 0.0
        else:
            # Aggregate metrics using window counters for accurate measurement
            if window_duration > 0:
                # Use window-based bytes sent for accurate throughput
                throughput = (self.window_bytes_sent * 8) / (window_duration * 1e6)
            else:
                throughput = 0.0

            # Use window-based packet counters for accurate loss rate
            total_packets = self.window_packets_sent + self.window_packets_dropped
            loss_rate = self.window_packets_dropped / total_packets if total_packets > 0 else 0.0
        
        # Average queue length (proxy for latency)
        if len(self.queue_occupancy_history) > 0:
            avg_queue = np.mean(list(self.queue_occupancy_history)[-100:])
            # Queuing delay + propagation delay
            queuing_delay = (avg_queue * 1500 * 8) / self.bandwidth_bps * 1000  # ms
            total_latency = self.delay_ms + queuing_delay
        else:
            total_latency = self.delay_ms

        # Prepare metrics to return
        metrics = {
            'throughput': throughput,
            'latency': total_latency,
            'loss': loss_rate,
            'queue_length': len(self.queue),
            'packets_sent': self.packets_sent if flow_id is None else self.flow_stats.get(flow_id, {}).get('packets_sent', 0),
            'packets_dropped': self.packets_dropped if flow_id is None else self.flow_stats.get(flow_id, {}).get('packets_dropped', 0)
        }

        # Reset window counters for next measurement period
        self.window_bytes_sent = 0
        self.window_packets_sent = 0
        self.window_packets_dropped = 0

        return metrics
    
    def get_current_queue_length(self) -> int:
        """Get current queue occupancy"""
        return len(self.queue)
    
    def get_rtt(self) -> float:
        """Estimate current RTT based on queue"""
        queue_delay = (len(self.queue) * 1500 * 8) / self.bandwidth_bps * 1000
        return self.delay_ms * 2 + queue_delay  # Round trip
    
    def get_flow_stats(self, flow_id: int) -> Dict:
        """Get statistics for specific flow"""
        return self.flow_stats.get(flow_id, {
            'packets_sent': 0,
            'packets_dropped': 0,
            'bytes_sent': 0
        })

    def _append_bounded_history(self, history_list: List, value):
        """Append to history list with maximum length bound to prevent memory leaks"""
        history_list.append(value)
        if len(history_list) > self.max_history_length:
            history_list.pop(0)  # Remove oldest element


class TrafficGenerator:
    """Generate different types of traffic patterns"""
    
    @staticmethod
    def generate_bulk_traffic(packet_size: int = 1500):
        """Generate bulk transfer traffic (large packets, continuous)"""
        while True:
            yield packet_size
    
    @staticmethod
    def generate_streaming_traffic(target_rate_mbps: float = 5.0):
        """Generate streaming traffic (constant bit rate)"""
        packet_size = 1200  # typical for video
        packets_per_sec = (target_rate_mbps * 1e6) / (8 * packet_size)
        inter_packet_time = 1.0 / packets_per_sec
        
        while True:
            yield packet_size, inter_packet_time
    
    @staticmethod
    def generate_realtime_traffic():
        """Generate real-time traffic (small packets, bursty)"""
        # VoIP-like: 50 packets/sec, 160 bytes
        # Gaming: occasional larger bursts
        while True:
            if np.random.random() < 0.1:  # 10% burst
                for _ in range(5):
                    yield 800  # Larger game state update
            else:
                yield 160  # Regular voice/control packet


def create_static_network(bandwidth_mbps=10, delay_ms=50, queue_size=100, loss_rate=0.0):
    """Helper to create static network configuration"""
    from .config import NetworkConfig
    config = NetworkConfig(
        bandwidth_mbps=bandwidth_mbps,
        delay_ms=delay_ms,
        queue_size=queue_size,
        loss_rate=loss_rate
    )
    return NetworkSimulator(config)


def create_dynamic_network(initial_bw=10, changes=None):
    """
    Helper to create dynamic bandwidth network
    
    Args:
        initial_bw: Initial bandwidth in Mbps
        changes: List of (time, bandwidth) tuples
    """
    from .config import NetworkConfig
    
    config = NetworkConfig(bandwidth_mbps=initial_bw, delay_ms=50, queue_size=100)
    network = NetworkSimulator(config)
    
    if changes:
        network.set_bandwidth_schedule(changes)
    
    return network
