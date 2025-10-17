"""
Модуль для управления моделью машинного обучения
"""
import os
import pickle
import numpy as np
import logging
from typing import Tuple, Optional, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DATA_PATH, MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH,
    TEST_SIZE, RANDOM_STATE, CV_FOLDS, GENRES
)
from audio_processor import AudioProcessor

# Настройка логирования
logger = logging.getLogger(__name__)

class ModelManager:
    """Класс для управления моделью машинного обучения"""
    
    def __init__(self):
        """Инициализация менеджера модели"""
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.audio_processor = AudioProcessor()
        
    def load_trained_model(self) -> bool:
        """
        Загружает обученную модель, скейлер и кодировщик меток
        
        Returns:
            True если загрузка успешна, False иначе
        """
        try:
            # Загрузка модели
            if not os.path.exists(MODEL_PATH):
                logger.error(f"Файл модели не найден: {MODEL_PATH}")
                return False
                
            with open(MODEL_PATH, 'rb') as model_file:
                self.model = pickle.load(model_file)
            
            # Загрузка скейлера
            if not os.path.exists(SCALER_PATH):
                logger.error(f"Файл скейлера не найден: {SCALER_PATH}")
                return False
                
            with open(SCALER_PATH, 'rb') as scaler_file:
                self.scaler = pickle.load(scaler_file)
            
            # Загрузка кодировщика меток
            if not os.path.exists(LABEL_ENCODER_PATH):
                logger.error(f"Файл кодировщика меток не найден: {LABEL_ENCODER_PATH}")
                return False
                
            with open(LABEL_ENCODER_PATH, 'rb') as le_file:
                self.label_encoder = pickle.load(le_file)
            
            logger.info("Модель, скейлер и кодировщик меток успешно загружены")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return False
    
    def train_model(self) -> bool:
        """
        Обучает модель на данных из датасета
        
        Returns:
            True если обучение успешно, False иначе
        """
        try:
            if not os.path.exists(DATA_PATH):
                logger.error(f"Путь к датасету не найден: {DATA_PATH}")
                return False
            
            logger.info("Начало извлечения признаков из датасета...")
            features, labels = self._extract_dataset_features()
            
            if len(features) == 0:
                logger.error("Не удалось извлечь признаки из датасета")
                return False
            
            logger.info(f"Извлечено {len(features)} образцов с {features.shape[1]} признаками")
            
            # Кодирование меток
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(labels)
            
            # Нормализация признаков
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(features)
            
            # Разделение данных
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            
            # Параметры для GridSearch
            param_grid = {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            
            # Обучение модели
            logger.info("Начало обучения модели...")
            model = RandomForestClassifier(random_state=RANDOM_STATE)
            grid_search = GridSearchCV(
                estimator=model, 
                param_grid=param_grid,
                scoring='accuracy', 
                cv=CV_FOLDS, 
                verbose=1, 
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            
            logger.info(f"Лучшие параметры: {grid_search.best_params_}")
            logger.info(f"Лучшая точность: {grid_search.best_score_}")
            
            # Оценка модели
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Точность на тестовой выборке: {accuracy:.4f}")
            
            # Сохранение модели
            self._save_model()
            
            # Визуализация результатов
            self._plot_results(y_test, y_pred)
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обучения модели: {e}")
            return False
    
    def predict_genre(self, file_path: str) -> Optional[str]:
        """
        Предсказывает жанр музыкального файла
        
        Args:
            file_path: Путь к аудиофайлу
            
        Returns:
            Предсказанный жанр или None в случае ошибки
        """
        try:
            if self.model is None or self.scaler is None or self.label_encoder is None:
                logger.error("Модель не загружена")
                return None
            
            # Валидация файла
            if not self.audio_processor.validate_audio_file(file_path):
                return None
            
            # Извлечение признаков
            expected_num_features = self.scaler.n_features_in_
            features = self.audio_processor.extract_features(file_path, expected_num_features)
            
            if features is None:
                logger.error("Не удалось извлечь признаки из файла")
                return None
            
            # Масштабирование признаков
            features_scaled = self.scaler.transform([features])
            
            # Предсказание
            predicted_label = self.model.predict(features_scaled)
            predicted_genre = self.label_encoder.inverse_transform(predicted_label)
            
            logger.info(f"Предсказанный жанр для {file_path}: {predicted_genre[0]}")
            return predicted_genre[0]
            
        except Exception as e:
            logger.error(f"Ошибка предсказания жанра: {e}")
            return None
    
    def _extract_dataset_features(self) -> Tuple[np.ndarray, List[str]]:
        """
        Извлекает признаки из всего датасета
        
        Returns:
            Кортеж (признаки, метки)
        """
        features = []
        labels = []
        
        for genre in GENRES:
            genre_path = os.path.join(DATA_PATH, genre)
            if not os.path.exists(genre_path):
                logger.warning(f"Папка жанра не найдена: {genre_path}")
                continue
            
            genre_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
            logger.info(f"Обработка {len(genre_files)} файлов жанра {genre}")
            
            for file_name in genre_files:
                file_path = os.path.join(genre_path, file_name)
                feature = self.audio_processor.extract_features(file_path)
                
                if feature is not None:
                    features.append(feature)
                    labels.append(genre)
        
        return np.array(features), labels
    
    def _save_model(self) -> None:
        """Сохраняет модель, скейлер и кодировщик меток"""
        try:
            # Сохранение модели
            with open(MODEL_PATH, 'wb') as model_file:
                pickle.dump(self.model, model_file)
            logger.info(f"Модель сохранена в {MODEL_PATH}")
            
            # Сохранение скейлера
            with open(SCALER_PATH, 'wb') as scaler_file:
                pickle.dump(self.scaler, scaler_file)
            logger.info(f"Скейлер сохранен в {SCALER_PATH}")
            
            # Сохранение кодировщика меток
            with open(LABEL_ENCODER_PATH, 'wb') as le_file:
                pickle.dump(self.label_encoder, le_file)
            logger.info(f"Кодировщик меток сохранен в {LABEL_ENCODER_PATH}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")
    
    def _plot_results(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Создает визуализации результатов
        
        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
        """
        try:
            # Матрица ошибок
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=GENRES, yticklabels=GENRES)
            plt.xlabel('Предсказанный жанр')
            plt.ylabel('Истинный жанр')
            plt.title('Матрица ошибок')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Важность признаков
            if hasattr(self.model, 'feature_importances_'):
                feature_importances = self.model.feature_importances_
                sorted_idx = np.argsort(feature_importances)[::-1]
                
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(feature_importances)), feature_importances[sorted_idx])
                plt.xlabel('Важность признака')
                plt.title('Важность признаков (Random Forest)')
                plt.tight_layout()
                plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
                plt.show()
            
        except Exception as e:
            logger.error(f"Ошибка создания визуализаций: {e}")
