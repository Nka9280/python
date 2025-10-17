import json
import csv
import os
from typing import List, Optional
from tkinter import messagebox
from models import PacketData
from anomaly_analysis import AnomalyAnalyzer


class DataExportService:
    @staticmethod
    def export_csv(packets: List[PacketData], filename: str = 'packet_log.csv') -> bool:
        try:
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Время', 'Протокол', 'Источник', 'Назначение', 'Размер', 'Сводка'])
                for packet_data in packets:
                    try:
                        summary = packet_data.packet.summary()
                    except Exception:
                        summary = "Ошибка получения сводки"
                    writer.writerow([
                        packet_data.timestamp,
                        packet_data.protocol,
                        packet_data.source_ip,
                        packet_data.dest_ip,
                        packet_data.size,
                        summary
                    ])
            return True
        except Exception:
            return False
    
    @staticmethod
    def export_json(packets: List[PacketData], filename: str = 'packet_log.json') -> bool:
        try:
            data = []
            for packet_data in packets:
                item = {
                    "timestamp": packet_data.timestamp,
                    "protocol": packet_data.protocol,
                    "source_ip": packet_data.source_ip,
                    "dest_ip": packet_data.dest_ip,
                    "source_port": packet_data.source_port,
                    "dest_port": packet_data.dest_port,
                    "size": packet_data.size,
                    "summary": ""
                }
                try:
                    item["summary"] = packet_data.packet.summary()
                except Exception:
                    pass
                data.append(item)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False


class ModelService:
    def __init__(self, model_path: str = "anomaly_model.joblib"):
        self.model_path = model_path
        self.analyzer = AnomalyAnalyzer()
        self.load_model()
    
    def load_model(self) -> bool:
        try:
            if os.path.exists(self.model_path):
                self.analyzer.load(self.model_path)
                return True
        except Exception:
            pass
        return False
    
    def save_model(self) -> bool:
        try:
            self.analyzer.save(self.model_path)
            return True
        except Exception:
            return False
    
    def train_model(self, packets: List[PacketData]) -> bool:
        try:
            packet_dicts = []
            for packet_data in packets:
                packet_dicts.append({
                    'time': packet_data.timestamp,
                    'src': packet_data.source_ip,
                    'dst': packet_data.dest_ip,
                    'sport': packet_data.source_port,
                    'dport': packet_data.dest_port,
                    'name': packet_data.protocol,
                    'ttl': 64,
                    'flags': '',
                    'len': packet_data.size
                })
            self.analyzer.fit(packet_dicts)
            return True
        except Exception:
            return False
    
    def predict_anomalies(self, packets: List[PacketData]) -> Optional[List[int]]:
        try:
            packet_dicts = []
            for packet_data in packets:
                packet_dicts.append({
                    'time': packet_data.timestamp,
                    'src': packet_data.source_ip,
                    'dst': packet_data.dest_ip,
                    'sport': packet_data.source_port,
                    'dport': packet_data.dest_port,
                    'name': packet_data.protocol,
                    'ttl': 64,
                    'flags': '',
                    'len': packet_data.size
                })
            return self.analyzer.predict(packet_dicts)
        except Exception:
            return None
