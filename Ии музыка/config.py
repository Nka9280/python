"""
Конфигурационный файл для проекта классификации музыкальных жанров
"""
import os

# Пути к файлам
DATA_PATH = "path_to_music_dataset"
MODEL_PATH = "best_model.pkl"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# Настройки аудио
MAX_DURATION = 10  # секунд
SAMPLE_RATE = None  # None для использования оригинальной частоты дискретизации

# Настройки признаков
N_MFCC = 13
N_CHROMA = 12
N_SPECTRAL_CONTRAST = 7
N_TONNETZ = 6

# Настройки модели
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Поддерживаемые жанры
GENRES = ['pop', 'rock', 'hiphop', 'classical', 'disco', 'reggae', 'metal', 'country']

# Настройки GUI
WINDOW_TITLE = "Классификатор музыкальных жанров"
WINDOW_SIZE = "500x300"

# Настройки FFmpeg
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"

# Поддерживаемые форматы файлов
SUPPORTED_AUDIO_FORMATS = [("Audio Files", "*.mp3;*.wav;*.flac;*.m4a")]

# Настройки логирования
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
