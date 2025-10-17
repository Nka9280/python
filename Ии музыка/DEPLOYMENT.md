# Руководство по развертыванию

## Быстрый старт

### 1. Установка зависимостей

```bash
# Клонирование репозитория (если используется Git)
git clone <repository-url>
cd music-genre-classifier

# Установка Python зависимостей
pip install -r requirements.txt
```

### 2. Настройка FFmpeg

**Windows:**
1. Скачайте FFmpeg с https://ffmpeg.org/download.html
2. Распакуйте в `C:\ffmpeg\`
3. Убедитесь, что путь в `config.py` правильный

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### 3. Подготовка данных

Создайте структуру папок для датасета:
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

Поместите аудиофайлы (.wav) в соответствующие папки.

### 4. Обучение модели

```bash
python train_model.py
```

### 5. Запуск приложения

```bash
python app.py
```

## Детальная настройка

### Конфигурация

Отредактируйте `config.py` для ваших нужд:

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

### Переменные окружения

Создайте файл `.env` (опционально):

```env
# Пути
DATA_PATH=path_to_music_dataset
MODEL_PATH=best_model.pkl
SCALER_PATH=scaler.pkl
LABEL_ENCODER_PATH=label_encoder.pkl

# Настройки логирования
LOG_LEVEL=INFO
LOG_FILE=music_classifier.log

# Настройки FFmpeg
FFMPEG_PATH=C:\ffmpeg\bin\ffmpeg.exe
```

### Логирование

Настройте логирование в `utils.py`:

```python
def setup_logging(log_file=None):
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file) if log_file else logging.StreamHandler()
        ]
    )
```

## Развертывание в продакшене

### Docker (рекомендуется)

Создайте `Dockerfile`:

```dockerfile
FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Установка Python зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . /app
WORKDIR /app

# Запуск приложения
CMD ["python", "app.py"]
```

Создайте `docker-compose.yml`:

```yaml
version: '3.8'
services:
  music-classifier:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - DATA_PATH=/app/data
      - MODEL_PATH=/app/models/best_model.pkl
```

### Виртуальное окружение

```bash
# Создание виртуального окружения
python -m venv venv

# Активация (Windows)
venv\Scripts\activate

# Активация (Linux/macOS)
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### Системный сервис (Linux)

Создайте файл `/etc/systemd/system/music-classifier.service`:

```ini
[Unit]
Description=Music Genre Classifier
After=network.target

[Service]
Type=simple
User=music
WorkingDirectory=/opt/music-classifier
ExecStart=/opt/music-classifier/venv/bin/python app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Активация сервиса:

```bash
sudo systemctl daemon-reload
sudo systemctl enable music-classifier
sudo systemctl start music-classifier
```

## Мониторинг и обслуживание

### Логи

Проверьте логи:

```bash
# Основное приложение
tail -f music_classifier.log

# Обучение модели
tail -f training.log

# Системные логи (Linux)
journalctl -u music-classifier -f
```

### Производительность

Мониторинг ресурсов:

```bash
# Использование CPU и памяти
htop

# Использование диска
df -h

# Сетевые соединения
netstat -tulpn
```

### Резервное копирование

Создайте скрипт резервного копирования:

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/music-classifier-$DATE"

mkdir -p $BACKUP_DIR

# Копирование моделей
cp *.pkl $BACKUP_DIR/

# Копирование логов
cp *.log $BACKUP_DIR/

# Копирование конфигурации
cp config.py $BACKUP_DIR/

echo "Backup created: $BACKUP_DIR"
```

## Устранение неполадок

### Частые проблемы

1. **FFmpeg не найден**
   ```bash
   # Проверка установки
   ffmpeg -version
   
   # Установка (Ubuntu/Debian)
   sudo apt install ffmpeg
   ```

2. **Ошибки памяти**
   ```bash
   # Увеличение лимита памяти
   export PYTHONHASHSEED=0
   export OMP_NUM_THREADS=1
   ```

3. **Проблемы с аудио**
   ```bash
   # Проверка поддержки форматов
   python -c "import librosa; print(librosa.__version__)"
   ```

### Отладка

Включите отладочное логирование:

```python
# В config.py
LOG_LEVEL = "DEBUG"
```

Проверьте зависимости:

```bash
python -c "from utils import check_dependencies; check_dependencies()"
```

## Масштабирование

### Горизонтальное масштабирование

Используйте балансировщик нагрузки:

```nginx
# nginx.conf
upstream music_classifier {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    location / {
        proxy_pass http://music_classifier;
    }
}
```

### Вертикальное масштабирование

Увеличьте ресурсы сервера:
- CPU: 4+ ядер
- RAM: 8+ GB
- Диск: SSD для быстрого доступа к данным

## Безопасность

### Рекомендации

1. **Ограничение доступа**
   ```bash
   # Firewall
   sudo ufw allow 22
   sudo ufw allow 80
   sudo ufw enable
   ```

2. **Обновления**
   ```bash
   # Обновление зависимостей
   pip install --upgrade -r requirements.txt
   ```

3. **Мониторинг**
   ```bash
   # Проверка процессов
   ps aux | grep python
   
   # Проверка портов
   netstat -tulpn | grep python
   ```
