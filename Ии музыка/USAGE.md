# Руководство по использованию

## Быстрый старт

### 1. Запуск приложения

```bash
python app.py
```

### 2. Обучение модели

```bash
python train_model.py
```

### 3. Запуск тестов

```bash
python tests.py
```

## Подробное руководство

### Обучение модели

#### Подготовка данных

1. Убедитесь, что у вас есть папка `path_to_music_dataset` с подпапками для каждого жанра:
   ```
   path_to_music_dataset/
   ├── classical/
   ├── country/
   ├── disco/
   ├── hiphop/
   ├── metal/
   ├── pop/
   ├── reggae/
   └── rock/
   ```

2. Поместите аудиофайлы (.wav) в соответствующие папки

3. Запустите обучение:
   ```bash
   python train_model.py
   ```

#### Процесс обучения

Обучение включает:
- Извлечение признаков из всех аудиофайлов
- Нормализация признаков
- Разделение на обучающую и тестовую выборки
- Оптимизация параметров с помощью GridSearch
- Обучение модели Random Forest
- Сохранение модели, скейлера и кодировщика меток

### Использование приложения

#### Графический интерфейс

1. Запустите приложение:
   ```bash
   python app.py
   ```

2. Нажмите кнопку "Выбрать музыкальный файл"

3. Выберите аудиофайл (поддерживаются MP3, WAV, FLAC, M4A)

4. Дождитесь результата анализа

#### Программное использование

```python
from model_manager import ModelManager
from audio_processor import AudioProcessor

# Создание менеджера модели
model_manager = ModelManager()

# Загрузка модели
if model_manager.load_trained_model():
    # Предсказание жанра
    predicted_genre = model_manager.predict_genre("path/to/audio.wav")
    print(f"Предсказанный жанр: {predicted_genre}")
```

### Примеры использования

#### Базовый пример

```python
from model_manager import ModelManager

# Загрузка и предсказание
model_manager = ModelManager()
model_manager.load_trained_model()

# Предсказание для одного файла
genre = model_manager.predict_genre("music.wav")
print(f"Жанр: {genre}")
```

#### Пакетная обработка

```python
import os
from model_manager import ModelManager

model_manager = ModelManager()
model_manager.load_trained_model()

# Обработка нескольких файлов
files = ["song1.wav", "song2.wav", "song3.wav"]
results = []

for file_path in files:
    if os.path.exists(file_path):
        genre = model_manager.predict_genre(file_path)
        results.append((file_path, genre))
        print(f"{file_path}: {genre}")
```

#### Извлечение признаков

```python
from audio_processor import AudioProcessor

processor = AudioProcessor()

# Валидация файла
if processor.validate_audio_file("music.wav"):
    # Извлечение признаков
    features = processor.extract_features("music.wav")
    print(f"Извлечено {len(features)} признаков")
    print(f"Первые 5 признаков: {features[:5]}")
```

### Конфигурация

#### Основные настройки

Отредактируйте `config.py`:

```python
# Пути к файлам
DATA_PATH = "path_to_music_dataset"
MODEL_PATH = "best_model.pkl"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# Настройки аудио
MAX_DURATION = 10  # секунд
SAMPLE_RATE = None  # None для оригинальной частоты

# Настройки модели
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Поддерживаемые жанры
GENRES = ['pop', 'rock', 'hiphop', 'classical', 'disco', 'reggae', 'metal', 'country']
```

#### Настройка FFmpeg

Убедитесь, что FFmpeg установлен и путь правильный:

```python
# В config.py
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"  # Windows
# FFMPEG_PATH = "/usr/bin/ffmpeg"  # Linux
# FFMPEG_PATH = "/usr/local/bin/ffmpeg"  # macOS
```

### Логирование

#### Настройка логов

```python
from utils import setup_logging

# Настройка логирования
setup_logging('my_app.log')

# Использование в коде
import logging
logger = logging.getLogger(__name__)
logger.info("Приложение запущено")
```

#### Уровни логирования

- `DEBUG` - подробная информация для отладки
- `INFO` - общая информация о работе
- `WARNING` - предупреждения
- `ERROR` - ошибки
- `CRITICAL` - критические ошибки

### Тестирование

#### Запуск всех тестов

```bash
python tests.py
```

#### Запуск отдельных тестов

```python
import unittest
from tests import TestAudioProcessor, TestModelManager

# Создание test suite
suite = unittest.TestSuite()
suite.addTest(unittest.makeSuite(TestAudioProcessor))
suite.addTest(unittest.makeSuite(TestModelManager))

# Запуск тестов
runner = unittest.TextTestRunner()
runner.run(suite)
```

### Отладка

#### Проверка зависимостей

```python
from utils import check_dependencies, check_ffmpeg

# Проверка Python зависимостей
if check_dependencies():
    print("Все зависимости установлены")
else:
    print("Некоторые зависимости отсутствуют")

# Проверка FFmpeg
if check_ffmpeg():
    print("FFmpeg доступен")
else:
    print("FFmpeg не найден")
```

#### Проверка системы

```python
from utils import get_system_info

info = get_system_info()
print(f"Платформа: {info['platform']}")
print(f"Python: {info['python_version']}")
print(f"CPU ядер: {info['cpu_count']}")
print(f"Память: {info['memory_total'] // (1024**3)} GB")
```

### Производительность

#### Оптимизация обучения

1. Увеличьте количество ядер CPU:
   ```python
   # В model_manager.py
   grid_search = GridSearchCV(..., n_jobs=-1)  # Использует все ядра
   ```

2. Уменьшите размер датасета для тестирования:
   ```python
   # В config.py
   MAX_DURATION = 5  # Уменьшите с 10 до 5 секунд
   ```

#### Оптимизация предсказания

1. Кэширование модели:
   ```python
   # Модель загружается один раз при инициализации
   model_manager = ModelManager()
   model_manager.load_trained_model()
   ```

2. Валидация файлов:
   ```python
   # Проверяйте файлы перед обработкой
   if processor.validate_audio_file(file_path):
       genre = model_manager.predict_genre(file_path)
   ```

### Устранение неполадок

#### Частые проблемы

1. **"Модель не загружена"**
   - Убедитесь, что файлы модели существуют
   - Запустите `python train_model.py` для обучения

2. **"FFmpeg не найден"**
   - Установите FFmpeg
   - Проверьте путь в `config.py`

3. **"Ошибка извлечения признаков"**
   - Проверьте формат аудиофайла
   - Убедитесь, что файл не поврежден

4. **Низкая точность**
   - Увеличьте размер датасета
   - Добавьте больше признаков
   - Попробуйте другие алгоритмы

#### Отладочные команды

```bash
# Проверка зависимостей
python -c "from utils import check_dependencies; check_dependencies()"

# Проверка FFmpeg
python -c "from utils import check_ffmpeg; check_ffmpeg()"

# Проверка путей
python -c "from utils import validate_paths; validate_paths()"

# Информация о системе
python -c "from utils import get_system_info; print(get_system_info())"
```

### Расширение функциональности

#### Добавление новых жанров

1. Добавьте жанр в `GENRES` в `config.py`
2. Создайте папку с файлами в `DATA_PATH`
3. Переобучите модель

#### Добавление новых признаков

1. Модифицируйте `extract_features()` в `AudioProcessor`
2. Обновите `expected_num_features` в `ModelManager`
3. Переобучите модель

#### Замена алгоритма ML

1. Измените алгоритм в `ModelManager.train_model()`
2. Обновите параметры GridSearch
3. Переобучите модель

### Мониторинг

#### Логи приложения

```bash
# Просмотр логов в реальном времени
tail -f music_classifier.log

# Поиск ошибок
grep "ERROR" music_classifier.log

# Статистика
grep "INFO" music_classifier.log | wc -l
```

#### Производительность

```bash
# Использование CPU
top -p $(pgrep -f "python app.py")

# Использование памяти
ps aux | grep python

# Использование диска
du -sh path_to_music_dataset/
```
