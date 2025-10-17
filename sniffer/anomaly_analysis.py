from typing import Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib


class AnomalyAnalyzer:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key
        self._suspicious_ips: Optional[set] = None
        self._model: Optional[RandomForestClassifier] = None

    def fetch_suspicious_ips(self) -> set:
        if not self.api_key:
            return set()
        if self._suspicious_ips is not None:
            return self._suspicious_ips
        url = "https://api.abuseipdb.com/api/v2/blacklist"
        headers = {
            'Key': self.api_key,
            'Accept': 'application/json',
        }
        params = {
            'limit': 1000,
        }
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            self._suspicious_ips = {entry['ipAddress'] for entry in data.get('data', [])}
            return self._suspicious_ips
        except requests.exceptions.RequestException:
            return set()

    @staticmethod
    def ip_to_int(ip: str) -> int:
        try:
            octets = [int(x) for x in ip.split('.')]
            if len(octets) != 4:
                return 0
            return (octets[0] << 24) + (octets[1] << 16) + (octets[2] << 8) + octets[3]
        except Exception:
            return 0

    def extract_features(self, packet_log: Iterable[Mapping]) -> pd.DataFrame:
        suspicious = self.fetch_suspicious_ips()
        features: List[Mapping] = []
        for packet in packet_log:
            try:
                src_ip = packet.get('src', '0.0.0.0')
                dst_ip = packet.get('dst', '0.0.0.0')
                feature = {
                    'time': float(packet.get('time', 0.0)),
                    'source_ip': self.ip_to_int(src_ip),
                    'destination_ip': self.ip_to_int(dst_ip),
                    'source_port': int(packet.get('sport', 0) or 0),
                    'destination_port': int(packet.get('dport', 0) or 0),
                    'protocol': str(packet.get('name', 'UNK')),
                    'ttl': int(packet.get('ttl', 0) or 0),
                    'flags': str(packet.get('flags', '')),
                    'packet_length': int(packet.get('len', 0) or 0),
                    'is_suspicious_ip': int(src_ip in suspicious),
                    'time_bucket': int(float(packet.get('time', 0.0)) // 60),
                }
                features.append(feature)
            except Exception:
                # Skip malformed packets
                continue
        return pd.DataFrame(features)

    @staticmethod
    def calculate_packet_delays(features: pd.DataFrame) -> pd.DataFrame:
        if features.empty:
            return features
        features = features.sort_values(by=['source_ip', 'destination_ip', 'time']).copy()
        features['packet_delay'] = (
            features.groupby(['source_ip', 'destination_ip'])['time'].diff().fillna(0)
        )
        return features

    def prepare_data(self, features: pd.DataFrame) -> pd.DataFrame:
        if features.empty:
            return features
        features = self.calculate_packet_delays(features)
        features = pd.get_dummies(features, columns=['protocol'], drop_first=True)
        features = features.fillna(0)
        return features

    def fit(self, packet_log: Iterable[Mapping]) -> None:
        df = self.extract_features(packet_log)
        df = self.prepare_data(df)
        if df.empty:
            self._model = RandomForestClassifier(n_estimators=50, random_state=42)
            return
        df['target'] = np.random.choice([0, 1], len(df))
        X = df.drop(columns=['target'])
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self._model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        self._model.fit(X_train, y_train)
        y_pred = self._model.predict(X_test)
        # For now we just print; a junior might log/print simple metrics
        print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
        print('Classification Report:')
        print(classification_report(y_test, y_pred))

    def predict(self, new_packet_log: Iterable[Mapping]) -> Optional[np.ndarray]:
        if self._model is None:
            return None
        new_features = self.extract_features(new_packet_log)
        new_features = self.prepare_data(new_features)
        if new_features.empty:
            return None
        return self._model.predict(new_features)

    def save(self, path: str) -> None:
        if self._model is None:
            return
        joblib.dump(self._model, path)

    def load(self, path: str) -> None:
        try:
            self._model = joblib.load(path)
        except Exception:
            self._model = None


__all__ = ["AnomalyAnalyzer"]
