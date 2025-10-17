from dataclasses import dataclass
from typing import Dict, List, Optional
import scapy.all as scapy


@dataclass
class PacketData:
    packet: scapy.Packet
    timestamp: float
    size: int
    protocol: str
    source_ip: str = ""
    dest_ip: str = ""
    source_port: int = 0
    dest_port: int = 0
    
    @classmethod
    def from_packet(cls, packet: scapy.Packet) -> 'PacketData':
        timestamp = getattr(packet, 'time', 0.0)
        size = len(packet)
        
        protocol = "Unknown"
        source_ip = ""
        dest_ip = ""
        source_port = 0
        dest_port = 0
        
        try:
            if packet.haslayer('IP'):
                source_ip = getattr(packet['IP'], 'src', '')
                dest_ip = getattr(packet['IP'], 'dst', '')
                protocol = 'IP'
            
            if packet.haslayer('TCP'):
                protocol = 'TCP'
                source_port = getattr(packet['TCP'], 'sport', 0)
                dest_port = getattr(packet['TCP'], 'dport', 0)
            elif packet.haslayer('UDP'):
                protocol = 'UDP'
                source_port = getattr(packet['UDP'], 'sport', 0)
                dest_port = getattr(packet['UDP'], 'dport', 0)
            elif packet.haslayer('ICMP'):
                protocol = 'ICMP'
        except Exception:
            pass
            
        return cls(
            packet=packet,
            timestamp=timestamp,
            size=size,
            protocol=protocol,
            source_ip=source_ip,
            dest_ip=dest_ip,
            source_port=source_port,
            dest_port=dest_port
        )


@dataclass
class NetworkStats:
    total_packets: int = 0
    total_bytes: int = 0
    protocol_counts: Dict[str, int] = None
    
    def __post_init__(self):
        if self.protocol_counts is None:
            self.protocol_counts = {}
    
    def add_packet(self, packet_data: PacketData) -> None:
        self.total_packets += 1
        self.total_bytes += packet_data.size
        self.protocol_counts[packet_data.protocol] = self.protocol_counts.get(packet_data.protocol, 0) + 1
    
    def clear(self) -> None:
        self.total_packets = 0
        self.total_bytes = 0
        self.protocol_counts.clear()
    
    def get_summary(self) -> str:
        return f"Пакетов: {self.total_packets} | Байт: {self.total_bytes}"
