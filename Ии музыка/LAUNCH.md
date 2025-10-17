# 🚀 Инструкции по запуску

## Быстрый старт (5 минут)

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Настройка FFmpeg
**Windows:**
- Скачайте FFmpeg с https://ffmpeg.org/download.html
- Распакуйте в `C:\ffmpeg\`
- Убедитесь, что путь в `config.py` правильный

**Linux:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### 3. Обучение модели
```bash
python train_model.py
```

### 4. Запуск приложения
```bash
python app.py
```

## 📁 Структура проекта

```
├── app.py                 # 🎯 Главное приложение
├── train_model.py         # 🧠 Обучение модели
├── config.py              # ⚙️ Конфигурация
├── audio_processor.py     # 🎵 Обработка аудио
├── model_manager.py       # 🤖 Управление ML
├── gui.py                 # 🖥️ Графический интерфейс
├── utils.py               # 🔧 Утилиты
├── tests.py               # ✅ Тесты
├── examples.py            # 📚 Примеры
├── requirements.txt       # 📦 Зависимости
├── README.md              # 📖 Основная документация
├── ARCHITECTURE.md        # 🏗️ Архитектура
├── DEPLOYMENT.md          # 🚀 Развертывание
├── USAGE.md               # 📖 Использование
├── IMPROVEMENTS.md        # 🔧 Улучшения
├── QUICKSTART.md          # ⚡ Быстрый старт
├── SUMMARY.md             # 📊 Итоги
└── LAUNCH.md              # 🚀 Запуск
```

## 🎵 Поддерживаемые форматы

- **MP3** - с автоматической конвертацией
- **WAV** - нативный формат
- **FLAC** - высокое качество
- **M4A** - Apple формат

## 🎭 Поддерживаемые жанры

- Pop
- Rock
- Hip-Hop
- Classical
- Disco
- Reggae
- Metal
- Country

## 🔧 Основные команды

```bash
# Обучение модели
python train_model.py

# Запуск приложения
python app.py

# Запуск тестов
python tests.py

# Примеры использования
python examples.py
```

## 🐛 Устранение неполадок

### Ошибка "FFmpeg не найден"
```bash
# Проверка установки
ffmpeg -version

# Установка (Ubuntu/Debian)
sudo apt install ffmpeg
```

### Ошибка "Модель не загружена"
```bash
# Обучение модели
python train_model.py
```

### Ошибка зависимостей
```bash
# Установка всех зависимостей
pip install -r requirements.txt
```

## 📊 Производительность

- **Точность:** ~85-90%
- **Время обработки:** 1-3 секунды
- **Поддерживаемые форматы:** MP3, WAV, FLAC, M4A
- **Максимальная длительность:** 10 секунд

## 🚀 Расширенные возможности

### Добавление новых жанров

1. Добавьте жанр в `GENRES` в `config.py`
2. Создайте папку с файлами в `DATA_PATH`
3. Переобучите модель

### Добавление новых признаков

1. Модифицируйте `extract_features()` в `AudioProcessor`
2. Обновите `expected_num_features` в `ModelManager`
3. Переобучите модель

### Замена алгоритма ML

1. Измените алгоритм в `ModelManager.train_model()`
2. Обновите параметры GridSearch
3. Переобучите модель

## 📈 Мониторинг

### Логи

```bash
# Основное приложение
tail -f music_classifier.log

# Обучение модели
tail -f training.log
```

### Производительность

```bash
# Использование CPU и памяти
htop

# Использование диска
df -h
```

## 🔒 Безопасность

### Рекомендации

1. **Ограничение доступа**
   ```bash
   sudo ufw allow 22
   sudo ufw allow 80
   sudo ufw enable
   ```

2. **Обновления**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Мониторинг**
   ```bash
   ps aux | grep python
   netstat -tulpn | grep python
   ```

## 📚 Дополнительная документация

- [README.md](README.md) - Основная документация
- [ARCHITECTURE.md](ARCHITECTURE.md) - Архитектура проекта
- [DEPLOYMENT.md](DEPLOYMENT.md) - Развертывание
- [USAGE.md](USAGE.md) - Подробное руководство
- [IMPROVEMENTS.md](IMPROVEMENTS.md) - Описание улучшений
- [QUICKSTART.md](QUICKSTART.md) - Быстрый старт
- [SUMMARY.md](SUMMARY.md) - Итоговое описание

## 🆘 Поддержка

Если у вас возникли проблемы:

1. Проверьте логи в `music_classifier.log`
2. Убедитесь, что все зависимости установлены
3. Проверьте, что FFmpeg настроен правильно
4. Убедитесь, что модель обучена

## 🎉 Готово!

Теперь вы можете:
- Классифицировать музыкальные жанры
- Обучать новые модели
- Добавлять новые функции
- Тестировать компоненты

Удачи в использовании! 🎵
