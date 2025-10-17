"""
Модуль для обработки аудиофайлов и извлечения признаков
"""
import os
import numpy as np
import librosa
import logging
from typing import Optional, Tuple
from pydub import AudioSegment
from config import (
    MAX_DURATION, SAMPLE_RATE, N_MFCC, N_CHROMA, 
    N_SPECTRAL_CONTRAST, N_TONNETZ, FFMPEG_PATH
)

# Настройка логирования
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Класс для обработки аудиофайлов и извлечения признаков"""
    
    def __init__(self):
        """Инициализация процессора аудио"""
        # Устанавливаем путь к FFmpeg
        AudioSegment.converter = FFMPEG_PATH
        
    def convert_audio_format(self, input_path: str, output_format: str = "wav") -> Optional[str]:
        """
        Конвертирует аудиофайл в указанный формат
        
        Args:
            input_path: Путь к входному файлу
            output_format: Целевой формат (по умолчанию wav)
            
        Returns:
            Путь к сконвертированному файлу или None в случае ошибки
        """
        try:
            output_path = input_path.rsplit('.', 1)[0] + f".{output_format}"
            
            # Определяем формат входного файла
            input_format = input_path.rsplit('.', 1)[1].lower()
            
            audio = AudioSegment.from_file(input_path, format=input_format)
            audio.export(output_path, format=output_format)
            
            logger.info(f"Файл успешно сконвертирован: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Ошибка конвертации {input_path}: {e}")
            return None
    
    def extract_features(self, file_path: str, expected_num_features: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Извлекает признаки из аудиофайла
        
        Args:
            file_path: Путь к аудиофайлу
            expected_num_features: Ожидаемое количество признаков для валидации
            
        Returns:
            Массив признаков или None в случае ошибки
        """
        try:
            # Загрузка аудиофайла
            audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
            
            if len(audio) == 0:
                logger.warning(f"Пустой аудиофайл: {file_path}")
                return None
            
            # Извлечение MFCC
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
            mfccs_mean = np.mean(mfccs, axis=1)
            
            # Извлечение хроматических признаков
            chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Извлечение спектрального контраста
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
            
            # Извлечение тональных признаков
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sample_rate)
            tonnetz_mean = np.mean(tonnetz, axis=1)
            
            # Дополнительные признаки
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
            spectral_bandwidth_mean = np.mean(spectral_bandwidth)
            
            rms = librosa.feature.rms(y=audio)
            rms_mean = np.mean(rms)
            
            zcr = librosa.feature.zero_crossing_rate(y=audio)
            zcr_mean = np.mean(zcr)
            
            # Извлечение темпа
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
            
            # Объединение всех признаков
            features = np.hstack([
                mfccs_mean, 
                chroma_mean, 
                spectral_contrast_mean, 
                tonnetz_mean,
                spectral_bandwidth_mean, 
                rms_mean, 
                zcr_mean, 
                tempo
            ])
            
            # Валидация количества признаков
            if expected_num_features is not None and len(features) != expected_num_features:
                logger.error(f"Несоответствие количества признаков. Ожидается {expected_num_features}, получено {len(features)}")
                return None
            
            logger.info(f"Успешно извлечено {len(features)} признаков из {file_path}")
            return features
            
        except Exception as e:
            logger.error(f"Ошибка извлечения признаков из {file_path}: {e}")
            return None
    
    def validate_audio_file(self, file_path: str) -> bool:
        """
        Проверяет валидность аудиофайла
        
        Args:
            file_path: Путь к аудиофайлу
            
        Returns:
            True если файл валиден, False иначе
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Файл не найден: {file_path}")
                return False
            
            # Проверяем, что файл не пустой
            if os.path.getsize(file_path) == 0:
                logger.error(f"Пустой файл: {file_path}")
                return False
            
            # Пробуем загрузить файл
            audio, _ = librosa.load(file_path, duration=1)  # Загружаем только 1 секунду для проверки
            
            if len(audio) == 0:
                logger.error(f"Не удалось загрузить аудио из {file_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка валидации файла {file_path}: {e}")
            return False
