import tkinter as tk
from tkinter import ttk, messagebox
import psutil
import threading
from packet_sniffer import PacketSniffer
from models import PacketData, NetworkStats
from ui_components import PacketListWidget, StatsWidget, GraphWidget
from services import DataExportService, ModelService


class SnifferApp:
    def __init__(self, root):
        self.root = root
        self.sniffer = PacketSniffer()
        self.stats = NetworkStats()
        self.packets: list[PacketData] = []
        
        self.export_service = DataExportService()
        self.model_service = ModelService()
        
        self.setup_ui()
        self.setup_sniffer()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def setup_ui(self):
        self.root.title("Пакетный захватчик Пятерочка")
        self.root.geometry("900x700")
        self.root.configure(bg="#1E1E1E")

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TFrame", background="#1E1E1E")
        self.style.configure("TLabel", background="#1E1E1E", foreground="#FFFFFF", font=('Arial', 12))
        self.style.configure("TButton", background="#2C2C2C", foreground="#FFFFFF", borderwidth=0, font=('Arial', 12))
        self.style.map("TButton", background=[('active', '#3C3C3C')])
        self.style.configure("TEntry", background="#2C2C2C", foreground="#FFFFFF", font=('Arial', 12))
        self.style.configure("TCombobox", fieldbackground="#2C2C2C", background="#2C2C2C", foreground="#FFFFFF")

        self.create_widgets()
        self.create_graph()
    
    def setup_sniffer(self):
        def on_packet_callback(packet, packet_info):
            packet_data = PacketData.from_packet(packet)
            self.packets.append(packet_data)
            self.stats.add_packet(packet_data)
            self.packet_list.add_packet(packet_data)
            self.stats_widget.update_stats(self.stats)
            self.graph_widget.update_graph(len(self.packets))
        
        self.sniffer.set_on_packet(on_packet_callback)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.create_interface_controls(main_frame)
        self.create_search_controls(main_frame)
        self.create_action_buttons(main_frame)
        self.create_main_content(main_frame)

    def create_interface_controls(self, parent):
        frame = ttk.Frame(parent, padding="10", relief=tk.RAISED, borderwidth=1)
        frame.pack(fill=tk.X)
        
        ttk.Label(frame, text="Сетевой интерфейс:").grid(row=0, column=0, padx=5, pady=5)
        self.interface_combo = ttk.Combobox(frame, values=self.get_network_interfaces(), state="readonly")
        self.interface_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(frame, text="Фильтр (например, tcp):").grid(row=0, column=2, padx=5, pady=5)
        self.filter_entry = ttk.Entry(frame, width=10)
        self.filter_entry.grid(row=0, column=3, padx=5, pady=5)
    
    def create_search_controls(self, parent):
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.X)
        
        ttk.Label(frame, text="Поиск по пакетам:").grid(row=0, column=0, padx=5, pady=5)
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(frame, textvariable=self.search_var, width=30)
        self.search_entry.grid(row=0, column=1, padx=5, pady=5)
        self.search_entry.bind("<KeyRelease>", self.on_search)
        
        clear_search_btn = ttk.Button(frame, text="Очистить поиск", command=self.clear_search)
        clear_search_btn.grid(row=0, column=2, padx=5, pady=5)
    
    def create_action_buttons(self, parent):
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.X)
        
        self.start_button = ttk.Button(frame, text="Запустить захват", command=self.start_sniffing)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.stop_button = ttk.Button(frame, text="Остановить захват", command=self.stop_sniffing, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)

        save_csv_btn = ttk.Button(frame, text="Сохранить CSV", command=self.save_csv)
        save_csv_btn.grid(row=0, column=2, padx=5, pady=5)

        save_json_btn = ttk.Button(frame, text="Сохранить JSON", command=self.save_json)
        save_json_btn.grid(row=0, column=3, padx=5, pady=5)

        clear_btn = ttk.Button(frame, text="Очистить", command=self.clear_all)
        clear_btn.grid(row=0, column=4, padx=5, pady=5)
    
    def create_main_content(self, parent):
        output_frame = ttk.Frame(parent, padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        self.packet_list = PacketListWidget(output_frame, self.on_packet_select)
        self.stats_widget = StatsWidget(output_frame)
        
        controls = ttk.Frame(self.stats_widget.frame, padding="5")
        controls.pack(fill=tk.X)
        
        self.beep_var = tk.BooleanVar(value=True)
        beep_toggle = ttk.Checkbutton(controls, text="Звук", variable=self.beep_var, command=self.on_toggle_beep)
        beep_toggle.pack(side=tk.LEFT)
        
        train_btn = ttk.Button(controls, text="Обучить модель", command=self.train_model)
        train_btn.pack(side=tk.LEFT, padx=5)

    def create_graph(self):
        self.graph_widget = GraphWidget(self.root)

    def start_sniffing(self):
        interface = self.interface_combo.get()
        filter_string = self.filter_entry.get()

        if not interface:
            messagebox.showwarning("Предупреждение", "Выберите сетевой интерфейс.")
            return

        if self.sniffer.is_sniffing:
            messagebox.showwarning("Предупреждение", "Захват уже запущен.")
            return

        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.stats_widget.set_status("Захват пакетов запущен...")

        threading.Thread(target=self.sniffer.start_sniffing, args=(interface, filter_string), daemon=True).start()

    def stop_sniffing(self):
        self.sniffer.stop_sniffing()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.stats_widget.set_status("Ожидание запуска захвата")

    def save_csv(self):
        if self.export_service.export_csv(self.packets):
            messagebox.showinfo("Информация", "Журнал пакетов сохранён в packet_log.csv")
        else:
            messagebox.showerror("Ошибка", "Не удалось сохранить CSV файл")

    def save_json(self):
        if self.export_service.export_json(self.packets):
            messagebox.showinfo("Информация", "Журнал пакетов сохранён в packet_log.json")
        else:
            messagebox.showerror("Ошибка", "Не удалось сохранить JSON файл")

    def clear_all(self):
        self.sniffer.clear()
        self.packets.clear()
        self.stats.clear()
        self.packet_list.clear()
        self.graph_widget.clear()
        self.stats_widget.update_stats(self.stats)

    def on_search(self, event):
        search_term = self.search_var.get()
        self.packet_list.search(search_term)

    def clear_search(self):
        self.search_var.set("")
        self.packet_list.search("")

    def on_toggle_beep(self):
        self.sniffer.set_config(beep_enabled=bool(self.beep_var.get()))

    def train_model(self):
        if not self.packets:
            messagebox.showwarning("Предупреждение", "Нет данных для обучения модели.")
            return
        
        if self.model_service.train_model(self.packets):
            messagebox.showinfo("Информация", "Модель успешно обучена и сохранена.")
        else:
            messagebox.showerror("Ошибка", "Не удалось обучить модель.")

    def on_packet_select(self, packet_data: PacketData):
        info = self.extract_packet_info(packet_data)
        messagebox.showinfo("Информация о пакете", info)

    def extract_packet_info(self, packet_data: PacketData) -> str:
        try:
            return (
                f"Время: {packet_data.timestamp}\n"
                f"Источник: {packet_data.source_ip}\n"
                f"Назначение: {packet_data.dest_ip}\n"
                f"Протокол: {packet_data.protocol}\n"
                f"Размер: {packet_data.size} байт\n"
                f"Порт источника: {packet_data.source_port}\n"
                f"Порт назначения: {packet_data.dest_port}\n"
                f"Сводка: {packet_data.packet.summary()}\n"
            )
        except Exception as e:
            return f"Ошибка извлечения информации: {str(e)}"

    def get_network_interfaces(self):
        return list(psutil.net_if_addrs().keys())

    def on_close(self):
        try:
            self.sniffer.stop_sniffing()
        except Exception:
            pass
        try:
            self.model_service.save_model()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SnifferApp(root)
    root.mainloop()