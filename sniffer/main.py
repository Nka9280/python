import sqlite3
import uuid
from datetime import datetime

class NetworkTrafficDB:
    def __init__(self, db_name="network_traffic.db"):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS network_traffic (
            session_id TEXT PRIMARY KEY,
            src_ip TEXT NOT NULL,
            dst_ip TEXT NOT NULL,
            src_port INTEGER NOT NULL,
            dst_port INTEGER NOT NULL,
            protocol TEXT NOT NULL,
            packet_size INTEGER NOT NULL,
            timestamp DATETIME NOT NULL,
            flag TEXT NOT NULL,
            duration INTEGER NOT NULL
        )''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS anomalies (
            anomaly_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            anomaly_type TEXT NOT NULL,
            severity INTEGER NOT NULL,
            timestamp DATETIME NOT NULL,
            description TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES network_traffic(session_id) ON DELETE CASCADE
        )''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_behavior (
            user_id TEXT PRIMARY KEY,
            src_ip TEXT NOT NULL,
            request_count INTEGER NOT NULL,
            avg_packet_size INTEGER NOT NULL,
            timestamp DATETIME NOT NULL
        )''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS network_devices (
            device_id TEXT PRIMARY KEY,
            device_type TEXT NOT NULL,
            device_ip TEXT NOT NULL,
            last_seen DATETIME NOT NULL
        )''')

        self.conn.commit()

    def insert_network_traffic(self, src_ip, dst_ip, src_port, dst_port, protocol, packet_size, flag, duration):
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.cursor.execute('''
        INSERT INTO network_traffic (session_id, src_ip, dst_ip, src_port, dst_port, protocol, packet_size, timestamp, flag, duration)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, src_ip, dst_ip, src_port, dst_port, protocol, packet_size, timestamp, flag, duration))

        self.conn.commit()
        return session_id

    def insert_anomaly(self, session_id, anomaly_type, severity, description):
        anomaly_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.cursor.execute('''
        INSERT INTO anomalies (anomaly_id, session_id, anomaly_type, severity, timestamp, description)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (anomaly_id, session_id, anomaly_type, severity, timestamp, description))

        self.conn.commit()

    def insert_user_behavior(self, src_ip, request_count, avg_packet_size):
        user_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.cursor.execute('''
        INSERT INTO user_behavior (user_id, src_ip, request_count, avg_packet_size, timestamp)
        VALUES (?, ?, ?, ?, ?)
        ''', (user_id, src_ip, request_count, avg_packet_size, timestamp))

        self.conn.commit()

    def insert_network_device(self, device_type, device_ip):
        device_id = str(uuid.uuid4())
        last_seen = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.cursor.execute('''
        INSERT INTO network_devices (device_id, device_type, device_ip, last_seen)
        VALUES (?, ?, ?, ?)
        ''', (device_id, device_type, device_ip, last_seen))

        self.conn.commit()

    def get_all_traffic(self):
        self.cursor.execute('SELECT * FROM network_traffic')
        return self.cursor.fetchall()

    def get_all_anomalies(self):
        self.cursor.execute('SELECT * FROM anomalies')
        return self.cursor.fetchall()

    def get_all_user_behavior(self):
        self.cursor.execute('SELECT * FROM user_behavior')
        return self.cursor.fetchall()

    def get_all_devices(self):
        self.cursor.execute('SELECT * FROM network_devices')
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()

# Пример использования
if __name__ == "__main__":
    db = NetworkTrafficDB()

    # Вставка данных о трафике
    session_id = db.insert_network_traffic('192.168.0.1', '192.168.0.2', 12345, 80, 'TCP', 1500, 'SYN', 60)

    # Вставка аномалии
    db.insert_anomaly(session_id, 'DDoS', 5, 'Превышение порога запросов')

    # Вставка поведения пользователя
    db.insert_user_behavior('192.168.0.3', 100, 1200)

    # Вставка сетевого устройства
    db.insert_network_device('Firewall', '192.168.0.254')

    # Получение всех данных
    print("Network Traffic:", db.get_all_traffic())
    print("Anomalies:", db.get_all_anomalies())
    print("User Behavior:", db.get_all_user_behavior())
    print("Network Devices:", db.get_all_devices())

    db.close()
