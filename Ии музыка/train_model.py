"""
Скрипт для обучения модели классификации музыкальных жанров
"""
import sys
import logging
from utils import setup_logging, check_dependencies, check_ffmpeg, print_system_info
from model_manager import ModelManager

def main():
    """Главная функция для обучения модели"""
    # Настройка логирования
    setup_logging('training.log')
    logger = logging.getLogger(__name__)
    
    logger.info("Запуск обучения модели классификации музыкальных жанров")
    
    # Вывод информации о системе
    print_system_info()
    
    # Проверка зависимостей
    if not check_dependencies():
        logger.error("Не все зависимости установлены")
        return 1
    
    # Проверка FFmpeg
    if not check_ffmpeg():
        logger.warning("FFmpeg не найден, но это не критично для обучения")
    
    try:
        # Создание менеджера модели
        model_manager = ModelManager()
        
        # Обучение модели
        logger.info("Начало обучения модели...")
        success = model_manager.train_model()
        
        if success:
            logger.info("Обучение модели завершено успешно")
            print("Модель обучена и сохранена!")
            return 0
        else:
            logger.error("Ошибка обучения модели")
            print("Ошибка обучения модели. Проверьте логи.")
            return 1
            
    except Exception as e:
        logger.error(f"Критическая ошибка при обучении: {e}")
        print(f"Критическая ошибка: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
