import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from typing import List, Callable, Optional
from models import PacketData, NetworkStats


class PacketListWidget:
    def __init__(self, parent: ttk.Frame, on_packet_select: Optional[Callable] = None):
        self.parent = parent
        self.on_packet_select = on_packet_select
        self.packets: List[PacketData] = []
        self.filtered_packets: List[PacketData] = []
        self.search_term = ""
        
        self.create_widgets()
    
    def create_widgets(self):
        self.frame = ttk.LabelFrame(self.parent, text="Список пакетов", padding="10")
        self.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.listbox = tk.Listbox(
            self.frame, 
            height=20, 
            width=60, 
            bg="#2C2C2C", 
            fg="#FFFFFF", 
            font=('Arial', 12)
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind('<<ListboxSelect>>', self._on_select)
        
        self.scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.listbox.yview)
        self.listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def add_packet(self, packet_data: PacketData):
        self.packets.append(packet_data)
        if self._matches_search(packet_data):
            self.filtered_packets.append(packet_data)
            self._update_display()
    
    def clear(self):
        self.packets.clear()
        self.filtered_packets.clear()
        self.listbox.delete(0, tk.END)
    
    def search(self, term: str):
        self.search_term = term.lower()
        self.filtered_packets = [p for p in self.packets if self._matches_search(p)]
        self._update_display()
    
    def _matches_search(self, packet_data: PacketData) -> bool:
        if not self.search_term:
            return True
        summary = packet_data.packet.summary().lower()
        return self.search_term in summary
    
    def _update_display(self):
        self.listbox.delete(0, tk.END)
        for packet_data in self.filtered_packets:
            try:
                summary = packet_data.packet.summary()
            except Exception:
                summary = "Пакет (не удалось получить сводку)"
            self.listbox.insert(tk.END, summary)
    
    def _on_select(self, event):
        selected = self.listbox.curselection()
        if selected and self.on_packet_select:
            try:
                packet_data = self.filtered_packets[selected[0]]
                self.on_packet_select(packet_data)
            except Exception:
                pass


class StatsWidget:
    def __init__(self, parent: ttk.Frame):
        self.parent = parent
        self.stats = NetworkStats()
        self.create_widgets()
    
    def create_widgets(self):
        self.frame = ttk.Frame(self.parent, padding="10")
        self.frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.stats_label = ttk.Label(self.frame, text="Статистика: 0 байт захвачено", foreground="#FFFFFF")
        self.stats_label.pack(pady=5)
        
        self.status_label = ttk.Label(
            self.frame, 
            text="Ожидание запуска захвата", 
            relief=tk.SUNKEN, 
            anchor="w", 
            background="#2C2C2C", 
            font=('Arial', 12)
        )
        self.status_label.pack(fill=tk.X)
        
        self.protocol_frame = ttk.LabelFrame(self.frame, text="Статистика по протоколам", padding="10")
        self.protocol_frame.pack(fill=tk.X)
        
        self.protocol_label = ttk.Label(self.protocol_frame, text="", font=('Arial', 12), foreground="#FFFFFF")
        self.protocol_label.pack()
    
    def update_stats(self, stats: NetworkStats):
        self.stats = stats
        self.stats_label.config(text=stats.get_summary())
        self._update_protocol_stats()
    
    def set_status(self, text: str):
        self.status_label.config(text=text)
    
    def _update_protocol_stats(self):
        if not self.stats.protocol_counts:
            self.protocol_label.config(text="")
            return
        
        total = self.stats.total_packets
        lines = []
        for proto, count in self.stats.protocol_counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            lines.append(f"{proto}: {count} пакетов ({percentage:.1f}%)")
        
        self.protocol_label.config(text="\n".join(lines))


class GraphWidget:
    def __init__(self, parent):
        self.parent = parent
        self.traffic_data = []
        self.time_intervals = []
        self.create_widgets()
    
    def create_widgets(self):
        self.frame = ttk.LabelFrame(self.parent, text="График нагрузки сети", padding="10")
        self.frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.ax.set_facecolor('#1E1E1E')
        self.fig.patch.set_facecolor('#1E1E1E')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['top'].set_color('white')
    
    def update_graph(self, packet_count: int):
        self.traffic_data.append(packet_count)
        self.time_intervals.append(len(self.traffic_data))
        
        self.ax.clear()
        self.ax.set_facecolor('#1E1E1E')
        self.fig.patch.set_facecolor('#1E1E1E')
        
        if self.traffic_data:
            self.ax.plot(self.time_intervals, self.traffic_data, color='cyan', linewidth=2, label='Трафик')
            self.ax.fill_between(self.time_intervals, self.traffic_data, color='cyan', alpha=0.3)
            
            avg_traffic = np.mean(self.traffic_data)
            max_traffic = max(self.traffic_data)
            self.ax.axhline(avg_traffic, color='yellow', linestyle='--', linewidth=1, 
                           label=f'Средний: {avg_traffic:.1f}')
            self.ax.axhline(max_traffic, color='red', linestyle='--', linewidth=1, 
                           label=f'Максимум: {max_traffic:.1f}')
        
        self.ax.set_title("Нагрузка сети", fontsize=14, color='white')
        self.ax.set_xlabel("Время (с)", fontsize=12, color='white')
        self.ax.set_ylabel("Количество пакетов", fontsize=12, color='white')
        self.ax.grid(True, color='grey', linestyle='--', alpha=0.5)
        
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['top'].set_color('white')
        
        self.ax.legend(loc='upper right', facecolor='#2C2C2C', framealpha=0.8, 
                      edgecolor='white', fontsize=10)
        
        for label in (self.ax.get_xticklabels() + self.ax.get_yticklabels()):
            label.set_color('white')
        
        self.canvas.draw()
    
    def clear(self):
        self.traffic_data.clear()
        self.time_intervals.clear()
        self.ax.clear()
        self.canvas.draw()
