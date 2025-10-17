"""
Примеры использования модулей проекта классификации музыкальных жанров
"""
import os
import logging
from audio_processor import AudioProcessor
from model_manager import ModelManager
from utils import setup_logging

def example_audio_processing():
    """Пример использования AudioProcessor"""
    print("=== Пример обработки аудио ===")
    
    # Настройка логирования
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Создание процессора
    processor = AudioProcessor()
    
    # Пример валидации файла
    audio_file = "path_to_music_dataset/classical/classical.00000.wav"
    if os.path.exists(audio_file):
        is_valid = processor.validate_audio_file(audio_file)
        print(f"Файл {audio_file} валиден: {is_valid}")
        
        # Извлечение признаков
        features = processor.extract_features(audio_file)
        if features is not None:
            print(f"Извлечено {len(features)} признаков")
            print(f"Первые 5 признаков: {features[:5]}")
        else:
            print("Не удалось извлечь признаки")
    else:
        print(f"Файл {audio_file} не найден")

def example_model_prediction():
    """Пример использования ModelManager для предсказания"""
    print("\n=== Пример предсказания жанра ===")
    
    # Создание менеджера модели
    model_manager = ModelManager()
    
    # Загрузка модели
    if model_manager.load_trained_model():
        print("Модель загружена успешно")
        
        # Предсказание жанра
        audio_file = "path_to_music_dataset/rock/rock.00000.wav"
        if os.path.exists(audio_file):
            predicted_genre = model_manager.predict_genre(audio_file)
            if predicted_genre:
                print(f"Предсказанный жанр: {predicted_genre}")
            else:
                print("Ошибка предсказания")
        else:
            print(f"Файл {audio_file} не найден")
    else:
        print("Ошибка загрузки модели")

def example_batch_processing():
    """Пример пакетной обработки файлов"""
    print("\n=== Пример пакетной обработки ===")
    
    # Настройка
    processor = AudioProcessor()
    model_manager = ModelManager()
    
    if not model_manager.load_trained_model():
        print("Модель не загружена")
        return
    
    # Обработка нескольких файлов
    test_files = [
        "path_to_music_dataset/classical/classical.00000.wav",
        "path_to_music_dataset/rock/rock.00000.wav",
        "path_to_music_dataset/pop/pop.00000.wav"
    ]
    
    results = []
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"Обработка: {file_path}")
            predicted_genre = model_manager.predict_genre(file_path)
            if predicted_genre:
                results.append((file_path, predicted_genre))
                print(f"  -> {predicted_genre}")
            else:
                print(f"  -> Ошибка обработки")
        else:
            print(f"Файл не найден: {file_path}")
    
    # Вывод результатов
    print(f"\nОбработано {len(results)} файлов:")
    for file_path, genre in results:
        print(f"  {os.path.basename(file_path)}: {genre}")

def example_feature_analysis():
    """Пример анализа признаков"""
    print("\n=== Пример анализа признаков ===")
    
    processor = AudioProcessor()
    audio_file = "path_to_music_dataset/classical/classical.00000.wav"
    
    if os.path.exists(audio_file):
        # Извлечение признаков
        features = processor.extract_features(audio_file)
        
        if features is not None:
            print(f"Общее количество признаков: {len(features)}")
            
            # Анализ различных типов признаков
            mfcc_features = features[:13]  # MFCC
            chroma_features = features[13:25]  # Chroma
            spectral_contrast = features[25:32]  # Spectral Contrast
            tonnetz_features = features[32:38]  # Tonnetz
            other_features = features[38:]  # Остальные
            
            print(f"MFCC (13): среднее = {mfcc_features.mean():.4f}, std = {mfcc_features.std():.4f}")
            print(f"Chroma (12): среднее = {chroma_features.mean():.4f}, std = {chroma_features.std():.4f}")
            print(f"Spectral Contrast (7): среднее = {spectral_contrast.mean():.4f}, std = {spectral_contrast.std():.4f}")
            print(f"Tonnetz (6): среднее = {tonnetz_features.mean():.4f}, std = {tonnetz_features.std():.4f}")
            print(f"Остальные ({len(other_features)}): среднее = {other_features.mean():.4f}, std = {other_features.std():.4f}")

def main():
    """Главная функция с примерами"""
    print("Примеры использования классификатора музыкальных жанров")
    print("=" * 60)
    
    try:
        # Запуск примеров
        example_audio_processing()
        example_model_prediction()
        example_batch_processing()
        example_feature_analysis()
        
        print("\n" + "=" * 60)
        print("Все примеры выполнены успешно!")
        
    except Exception as e:
        print(f"Ошибка выполнения примеров: {e}")
        logging.error(f"Ошибка в примерах: {e}")

if __name__ == "__main__":
    main()
