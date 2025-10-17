"""
Главный модуль приложения для классификации музыкальных жанров
"""
import sys
import logging
from utils import setup_logging, check_dependencies, check_ffmpeg, validate_paths, print_system_info
from gui import MusicGenreClassifierGUI
from model_manager import ModelManager

def main():
    """Главная функция приложения"""
    # Настройка логирования
    setup_logging('music_classifier.log')
    logger = logging.getLogger(__name__)
    
    logger.info("Запуск приложения классификации музыкальных жанров")
    
    # Вывод информации о системе
    print_system_info()
    
    # Проверка зависимостей
    if not check_dependencies():
        logger.error("Не все зависимости установлены")
        return 1
    
    # Проверка FFmpeg
    if not check_ffmpeg():
        logger.warning("FFmpeg не найден, конвертация MP3 может не работать")
    
    # Проверка путей (только для предсказания, не для обучения)
    try:
        if not validate_paths():
            logger.warning("Некоторые файлы модели не найдены")
            print("Для обучения модели запустите: python train_model.py")
    except Exception as e:
        logger.warning(f"Ошибка проверки путей: {e}")
    
    try:
        # Создание и запуск GUI
        app = MusicGenreClassifierGUI()
        app.create_gui()
        app.run()
        
        logger.info("Приложение завершено")
        return 0
        
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
