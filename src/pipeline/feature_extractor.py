"""
Feature Extraction Pipeline for Network Traffic and System Logs
"""
"""
SENTINEL Feature Extraction Pipeline
====================================

Extracts security features from network flows and system logs.
Handles data normalization, feature engineering, and sequence building
for downstream AI models.

Classes:
    NetworkFlow: Dataclass for raw network traffic
    FeatureExtractor: Base abstract class
    NetworkFeatureExtractor: Extracts 13+ features from flows
    SequenceBuilder: Creates temporal sequences for LSTM/Transformers

Author: xorinf
Version: 1.0.0
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from loguru import logger


@dataclass
class NetworkFlow:
    """Represents a network flow with extracted features"""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    timestamp: float
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int
    duration: float
    flags: Optional[str] = None
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML models"""
        return np.array([
            self.src_port,
            self.dst_port,
            self.bytes_sent,
            self.bytes_recv,
            self.packets_sent,
            self.packets_recv,
            self.duration,
            self._protocol_to_int(),
        ], dtype=np.float32)
    
    def _protocol_to_int(self) -> int:
        protocols = {"TCP": 0, "UDP": 1, "ICMP": 2, "OTHER": 3}
        return protocols.get(self.protocol.upper(), 3)


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors"""
    
    @abstractmethod
    def extract(self, raw_data: Any) -> np.ndarray:
        """Extract features from raw data"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return list of feature names for explainability"""
        pass


class NetworkFeatureExtractor(FeatureExtractor):
    """Extract features from network traffic"""
    
    FEATURE_NAMES = [
        "src_port", "dst_port", "bytes_sent", "bytes_recv",
        "packets_sent", "packets_recv", "duration", "protocol",
        "bytes_per_packet", "packet_rate", "byte_ratio",
        "port_entropy", "connection_count"
    ]
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.flow_buffer: List[NetworkFlow] = []
        
    def extract(self, flow: NetworkFlow) -> np.ndarray:
        """Extract features from a single flow with context"""
        self.flow_buffer.append(flow)
        if len(self.flow_buffer) > self.window_size:
            self.flow_buffer.pop(0)
        
        # Base features
        base = flow.to_vector()
        
        # Derived features
        total_packets = flow.packets_sent + flow.packets_recv + 1e-6
        total_bytes = flow.bytes_sent + flow.bytes_recv + 1e-6
        
        bytes_per_packet = total_bytes / total_packets
        packet_rate = total_packets / (flow.duration + 1e-6)
        byte_ratio = flow.bytes_sent / (flow.bytes_recv + 1e-6)
        
        # Contextual features (from window)
        recent_ports = [f.dst_port for f in self.flow_buffer[-20:]]
        port_entropy = self._calculate_entropy(recent_ports)
        connection_count = len(self.flow_buffer)
        
        derived = np.array([
            bytes_per_packet,
            packet_rate,
            byte_ratio,
            port_entropy,
            connection_count
        ], dtype=np.float32)
        
        return np.concatenate([base, derived])
    
    def get_feature_names(self) -> List[str]:
        return self.FEATURE_NAMES
    
    @staticmethod
    def _calculate_entropy(values: List[int]) -> float:
        """Calculate Shannon entropy"""
        if not values:
            return 0.0
        _, counts = np.unique(values, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-10))


class SystemLogExtractor(FeatureExtractor):
    """Extract features from system logs (Sysmon, Windows Events)"""
    
    FEATURE_NAMES = [
        "event_type", "process_depth", "parent_child_mismatch",
        "unusual_time", "admin_action", "network_connection",
        "file_creation", "registry_modification"
    ]
    
    def extract(self, log_entry: Dict[str, Any]) -> np.ndarray:
        """Extract features from a log entry"""
        features = np.zeros(len(self.FEATURE_NAMES), dtype=np.float32)
        
        # Event type encoding
        event_types = {
            "ProcessCreate": 1, "NetworkConnect": 2,
            "FileCreate": 3, "RegistryEvent": 4
        }
        features[0] = event_types.get(log_entry.get("EventType", ""), 0)
        
        # Process tree depth (deeper = more suspicious)
        features[1] = log_entry.get("ProcessDepth", 0)
        
        # Parent-child mismatch (e.g., Word spawning PowerShell)
        features[2] = float(log_entry.get("ParentChildMismatch", False))
        
        # Time-based features
        hour = log_entry.get("Hour", 12)
        features[3] = float(hour < 6 or hour > 22)  # Unusual hours
        
        # Action flags
        features[4] = float(log_entry.get("IsAdminAction", False))
        features[5] = float(log_entry.get("HasNetworkConn", False))
        features[6] = float(log_entry.get("FileCreated", False))
        features[7] = float(log_entry.get("RegistryMod", False))
        
        return features
    
    def get_feature_names(self) -> List[str]:
        return self.FEATURE_NAMES


class SequenceBuilder:
    """Build sequences of events for LSTM/Transformer models"""
    
    def __init__(self, sequence_length: int = 50):
        self.sequence_length = sequence_length
        self.session_buffers: Dict[str, List[np.ndarray]] = {}
    
    def add_event(self, session_id: str, features: np.ndarray) -> Optional[np.ndarray]:
        """Add event to session, return sequence if ready"""
        if session_id not in self.session_buffers:
            self.session_buffers[session_id] = []
        
        self.session_buffers[session_id].append(features)
        
        # Return sequence when buffer is full
        if len(self.session_buffers[session_id]) >= self.sequence_length:
            sequence = np.stack(self.session_buffers[session_id][-self.sequence_length:])
            return sequence
        
        return None
    
    def get_partial_sequence(self, session_id: str) -> Optional[np.ndarray]:
        """Get current partial sequence (padded if needed)"""
        if session_id not in self.session_buffers:
            return None
        
        events = self.session_buffers[session_id]
        if not events:
            return None
        
        # Pad if needed
        feature_dim = events[0].shape[0]
        padded = np.zeros((self.sequence_length, feature_dim), dtype=np.float32)
        
        n_events = min(len(events), self.sequence_length)
        padded[-n_events:] = np.stack(events[-n_events:])
        
        return padded


logger.info("Feature extraction pipeline initialized")
