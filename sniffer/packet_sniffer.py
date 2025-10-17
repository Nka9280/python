import logging
import time
from threading import Thread
from typing import Callable, Dict, List, Optional

import scapy.all as scapy
import winsound


class PacketSniffer:
    def __init__(self) -> None:
        self.is_sniffing: bool = False
        self.packet_sizes_per_second: List[int] = []
        self.start_time: Optional[float] = None
        self.packet_log: List[scapy.Packet] = []
        self.log_file: str = "packet_log.txt"
        self._on_packet: Optional[Callable[[scapy.Packet, str], None]] = None
        self._sniff_thread: Optional[Thread] = None
        self.beep_enabled: bool = True
        self.max_packets: int = 10000
        self.total_bytes: int = 0
        self.protocol_counts: Dict[str, int] = {}
        self.setup_logging()

    def setup_logging(self) -> None:
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
        )

    def set_on_packet(self, callback: Optional[Callable[[scapy.Packet, str], None]]) -> None:
        self._on_packet = callback

    def set_config(self, beep_enabled: Optional[bool] = None, max_packets: Optional[int] = None) -> None:
        if beep_enabled is not None:
            self.beep_enabled = bool(beep_enabled)
        if isinstance(max_packets, int) and max_packets > 0:
            self.max_packets = max_packets

    def packet_callback(self, packet: scapy.Packet) -> None:
        if not self.is_sniffing:
            return

        elapsed_time = int(time.time() - self.start_time)

        # Extracting packet size and adding it to the current time
        packet_size = len(packet)
        if len(self.packet_sizes_per_second) <= elapsed_time:
            self.packet_sizes_per_second.extend([0] * (elapsed_time - len(self.packet_sizes_per_second) + 1))
        self.packet_sizes_per_second[elapsed_time] += packet_size  # Adding packet size

        try:
            proto = packet.sprintf('%IP.proto%')
        except Exception:
            proto = 'UNKNOWN'
        packet_info = f"Captured packet: {proto} - {packet.summary()}"
        self.packet_log.append(packet)
        self.total_bytes += len(packet)
        self.protocol_counts[proto] = self.protocol_counts.get(proto, 0) + 1
        if len(self.packet_log) > self.max_packets:
            old = self.packet_log.pop(0)
            self.total_bytes = max(0, self.total_bytes - len(old))

        logging.info(packet_info)

        if self.beep_enabled:
            try:
                winsound.Beep(1000, 100)
            except RuntimeError:
                pass

        if self._on_packet is not None:
            try:
                self._on_packet(packet, packet_info)
            except Exception:
                pass

    def capture_packets(self, interface: str, filter_string: str) -> None:
        self.is_sniffing = True
        self.start_time = time.time()
        print(f"Starting packet capture on interface: {interface} with filter: {filter_string}")
        try:
            scapy.sniff(iface=interface, prn=self.packet_callback, stop_filter=lambda x: not self.is_sniffing,
                         store=0, filter=filter_string)
        except Exception as e:
            print(f"Error capturing packets: {e}")

    def start_sniffing(self, interface: str, filter_string: str) -> None:
        self.packet_sizes_per_second.clear()
        self.packet_log.clear()
        self.is_sniffing = True

        self._sniff_thread = Thread(target=self.capture_packets, args=(interface, filter_string))
        self._sniff_thread.daemon = True
        self._sniff_thread.start()

    def stop_sniffing(self) -> None:
        self.is_sniffing = False
        if self._sniff_thread is not None and self._sniff_thread.is_alive():
            try:
                self._sniff_thread.join(timeout=1.0)
            except Exception:
                pass

    def get_stats(self) -> Dict[str, object]:
        return {
            "is_sniffing": self.is_sniffing,
            "total_packets": len(self.packet_log),
            "total_bytes": self.total_bytes,
            "protocol_counts": dict(self.protocol_counts),
        }

    def clear(self) -> None:
        self.packet_sizes_per_second.clear()
        self.packet_log.clear()
        self.total_bytes = 0
        self.protocol_counts.clear()
