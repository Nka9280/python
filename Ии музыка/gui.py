"""
Модуль графического интерфейса для классификатора музыкальных жанров
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import logging
from typing import Optional
from model_manager import ModelManager
from audio_processor import AudioProcessor
from config import SUPPORTED_AUDIO_FORMATS, WINDOW_TITLE, WINDOW_SIZE

# Настройка логирования
logger = logging.getLogger(__name__)

class MusicGenreClassifierGUI:
    """Графический интерфейс для классификатора музыкальных жанров"""
    
    def __init__(self):
        """Инициализация GUI"""
        self.model_manager = ModelManager()
        self.audio_processor = AudioProcessor()
        self.root = None
        self.result_label = None
        self.progress_bar = None
        self.status_label = None
        
    def create_gui(self) -> None:
        """Создает графический интерфейс"""
        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.resizable(True, True)
        
        # Центрирование окна
        self._center_window()
        
        # Создание стиля
        self._create_styles()
        
        # Создание элементов интерфейса
        self._create_widgets()
        
        # Загрузка модели
        self._load_model()
        
    def _center_window(self) -> None:
        """Центрирует окно на экране"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def _create_styles(self) -> None:
        """Создает стили для элементов интерфейса"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Настройка стилей
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Result.TLabel', font=('Arial', 12))
        style.configure('Status.TLabel', font=('Arial', 10))
        style.configure('Custom.TButton', font=('Arial', 11))
    
    def _create_widgets(self) -> None:
        """Создает элементы интерфейса"""
        # Главный фрейм
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Настройка весов для растягивания
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Заголовок
        title_label = ttk.Label(main_frame, text="Классификатор музыкальных жанров", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, pady=(0, 20), sticky=tk.W+tk.E)
        
        # Кнопка выбора файла
        self.select_button = ttk.Button(main_frame, text="Выбрать музыкальный файл", 
                                       command=self._select_file, style='Custom.TButton')
        self.select_button.grid(row=1, column=0, pady=10, sticky=tk.W+tk.E)
        
        # Прогресс-бар
        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress_bar.grid(row=2, column=0, pady=10, sticky=tk.W+tk.E)
        
        # Результат
        self.result_label = ttk.Label(main_frame, text="Выберите музыкальный файл для анализа", 
                                     style='Result.TLabel', wraplength=400)
        self.result_label.grid(row=3, column=0, pady=20, sticky=tk.W+tk.E)
        
        # Статус
        self.status_label = ttk.Label(main_frame, text="Готов к работе", 
                                    style='Status.TLabel')
        self.status_label.grid(row=4, column=0, pady=5, sticky=tk.W+tk.E)
        
        # Информационная панель
        self._create_info_panel(main_frame)
    
    def _create_info_panel(self, parent: ttk.Frame) -> None:
        """Создает информационную панель"""
        info_frame = ttk.LabelFrame(parent, text="Информация", padding="10")
        info_frame.grid(row=5, column=0, pady=20, sticky=tk.W+tk.E)
        
        info_text = """
Поддерживаемые форматы: MP3, WAV, FLAC, M4A
Поддерживаемые жанры: Pop, Rock, Hip-Hop, Classical, Disco, Reggae, Metal, Country
Рекомендуемая длительность: до 10 секунд
        """
        
        info_label = ttk.Label(info_frame, text=info_text.strip(), 
                              font=('Arial', 9), justify=tk.LEFT)
        info_label.grid(row=0, column=0, sticky=tk.W+tk.E)
    
    def _load_model(self) -> None:
        """Загружает обученную модель"""
        def load_in_background():
            self._update_status("Загрузка модели...")
            self.progress_bar.start()
            
            try:
                if self.model_manager.load_trained_model():
                    self._update_status("Модель загружена успешно")
                    self.select_button.config(state='normal')
                else:
                    self._update_status("Ошибка загрузки модели")
                    messagebox.showerror("Ошибка", "Не удалось загрузить модель")
                    self.select_button.config(state='disabled')
            except Exception as e:
                logger.error(f"Ошибка загрузки модели: {e}")
                self._update_status("Ошибка загрузки модели")
                messagebox.showerror("Ошибка", f"Ошибка загрузки модели: {e}")
            finally:
                self.progress_bar.stop()
        
        # Запуск загрузки в отдельном потоке
        threading.Thread(target=load_in_background, daemon=True).start()
    
    def _select_file(self) -> None:
        """Обрабатывает выбор файла"""
        file_path = filedialog.askopenfilename(
            title="Выберите музыкальный файл",
            filetypes=SUPPORTED_AUDIO_FORMATS
        )
        
        if file_path:
            self._process_file(file_path)
    
    def _process_file(self, file_path: str) -> None:
        """Обрабатывает выбранный файл"""
        def process_in_background():
            self._update_status("Обработка файла...")
            self.progress_bar.start()
            self.select_button.config(state='disabled')
            
            try:
                # Валидация файла
                if not self.audio_processor.validate_audio_file(file_path):
                    self._update_result("Ошибка: Неверный аудиофайл", "error")
                    return
                
                # Конвертация при необходимости
                if file_path.lower().endswith('.mp3'):
                    converted_path = self.audio_processor.convert_audio_format(file_path)
                    if converted_path:
                        file_path = converted_path
                    else:
                        self._update_result("Ошибка конвертации MP3 файла", "error")
                        return
                
                # Предсказание жанра
                predicted_genre = self.model_manager.predict_genre(file_path)
                
                if predicted_genre:
                    self._update_result(f"Предсказанный жанр: {predicted_genre}", "success")
                    self._update_status("Анализ завершен")
                else:
                    self._update_result("Ошибка анализа файла", "error")
                    self._update_status("Ошибка анализа")
                    
            except Exception as e:
                logger.error(f"Ошибка обработки файла: {e}")
                self._update_result(f"Ошибка: {e}", "error")
                self._update_status("Ошибка обработки")
            finally:
                self.progress_bar.stop()
                self.select_button.config(state='normal')
        
        # Запуск обработки в отдельном потоке
        threading.Thread(target=process_in_background, daemon=True).start()
    
    def _update_result(self, text: str, status_type: str = "normal") -> None:
        """Обновляет текст результата"""
        self.result_label.config(text=text)
        
        # Изменение цвета в зависимости от статуса
        if status_type == "success":
            self.result_label.config(foreground="green")
        elif status_type == "error":
            self.result_label.config(foreground="red")
        else:
            self.result_label.config(foreground="black")
    
    def _update_status(self, text: str) -> None:
        """Обновляет статус"""
        self.status_label.config(text=text)
        self.root.update_idletasks()
    
    def run(self) -> None:
        """Запускает GUI"""
        if self.root:
            self.root.mainloop()
        else:
            logger.error("GUI не инициализирован")
