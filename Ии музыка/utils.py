"""
Утилиты для проекта классификации музыкальных жанров
"""
import logging
import os
import sys
from typing import Optional
from config import LOG_LEVEL, LOG_FORMAT

def setup_logging(log_file: Optional[str] = None) -> None:
    """
    Настраивает систему логирования
    
    Args:
        log_file: Путь к файлу лога (опционально)
    """
    # Создаем форматтер
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Настраиваем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    
    # Очищаем существующие обработчики
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Консольный обработчик
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Файловый обработчик (если указан файл)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Настройка логирования для внешних библиотек
    logging.getLogger('librosa').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)

def check_dependencies() -> bool:
    """
    Проверяет наличие необходимых зависимостей
    
    Returns:
        True если все зависимости доступны, False иначе
    """
    required_modules = [
        'numpy', 'librosa', 'sklearn', 'matplotlib', 
        'seaborn', 'pydub', 'tkinter'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"Отсутствуют необходимые модули: {', '.join(missing_modules)}")
        print("Установите их с помощью: pip install -r requirements.txt")
        return False
    
    return True

def check_ffmpeg() -> bool:
    """
    Проверяет наличие FFmpeg
    
    Returns:
        True если FFmpeg доступен, False иначе
    """
    try:
        from pydub import AudioSegment
        # Пробуем создать простой аудиосегмент
        AudioSegment.silent(duration=1000)
        return True
    except Exception as e:
        print(f"FFmpeg не найден или не настроен: {e}")
        print("Установите FFmpeg и укажите правильный путь в config.py")
        return False

def validate_paths() -> bool:
    """
    Проверяет существование необходимых путей
    
    Returns:
        True если все пути валидны, False иначе
    """
    from config import DATA_PATH, MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH
    
    paths_to_check = [
        (DATA_PATH, "Путь к датасету"),
        (MODEL_PATH, "Файл модели"),
        (SCALER_PATH, "Файл скейлера"),
        (LABEL_ENCODER_PATH, "Файл кодировщика меток")
    ]
    
    missing_paths = []
    
    for path, description in paths_to_check:
        if not os.path.exists(path):
            missing_paths.append(f"{description}: {path}")
    
    if missing_paths:
        print("Отсутствуют необходимые файлы/папки:")
        for missing in missing_paths:
            print(f"  - {missing}")
        return False
    
    return True

def get_system_info() -> dict:
    """
    Возвращает информацию о системе
    
    Returns:
        Словарь с информацией о системе
    """
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': sys.version,
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available
    }

def create_requirements_file() -> None:
    """Создает файл requirements.txt с необходимыми зависимостями"""
    requirements = """numpy>=1.21.0
librosa>=0.9.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
pydub>=0.25.0
psutil>=5.8.0
"""
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements)
    
    print("Файл requirements.txt создан")

def print_system_info() -> None:
    """Выводит информацию о системе"""
    info = get_system_info()
    
    print("=== Информация о системе ===")
    print(f"Платформа: {info['platform']}")
    print(f"Python: {info['python_version']}")
    print(f"CPU ядер: {info['cpu_count']}")
    print(f"Память: {info['memory_total'] // (1024**3)} GB")
    print(f"Доступно памяти: {info['memory_available'] // (1024**3)} GB")
    print("=" * 30)
