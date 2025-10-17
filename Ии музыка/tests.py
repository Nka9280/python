"""
Unit тесты для проекта классификации музыкальных жанров
"""
import unittest
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
import sys

# Добавляем текущую директорию в путь для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_processor import AudioProcessor
from model_manager import ModelManager
from utils import check_dependencies, check_ffmpeg, get_system_info

class TestAudioProcessor(unittest.TestCase):
    """Тесты для класса AudioProcessor"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.processor = AudioProcessor()
    
    def test_audio_processor_initialization(self):
        """Тест инициализации AudioProcessor"""
        self.assertIsNotNone(self.processor)
    
    def test_validate_audio_file_nonexistent(self):
        """Тест валидации несуществующего файла"""
        result = self.processor.validate_audio_file("nonexistent.wav")
        self.assertFalse(result)
    
    def test_validate_audio_file_empty_path(self):
        """Тест валидации пустого пути"""
        result = self.processor.validate_audio_file("")
        self.assertFalse(result)
    
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_validate_audio_file_empty_file(self, mock_getsize, mock_exists):
        """Тест валидации пустого файла"""
        mock_exists.return_value = True
        mock_getsize.return_value = 0
        
        result = self.processor.validate_audio_file("empty.wav")
        self.assertFalse(result)
    
    @patch('librosa.load')
    def test_extract_features_success(self, mock_load):
        """Тест успешного извлечения признаков"""
        # Мокаем librosa.load
        mock_audio = np.random.randn(1000)
        mock_sr = 22050
        mock_load.return_value = (mock_audio, mock_sr)
        
        # Мокаем другие функции librosa
        with patch('librosa.feature.mfcc') as mock_mfcc, \
             patch('librosa.feature.chroma_stft') as mock_chroma, \
             patch('librosa.feature.spectral_contrast') as mock_contrast, \
             patch('librosa.feature.tonnetz') as mock_tonnetz, \
             patch('librosa.feature.spectral_bandwidth') as mock_bandwidth, \
             patch('librosa.feature.rms') as mock_rms, \
             patch('librosa.feature.zero_crossing_rate') as mock_zcr, \
             patch('librosa.beat.beat_track') as mock_beat:
            
            # Настраиваем моки
            mock_mfcc.return_value = np.random.randn(13, 10)
            mock_chroma.return_value = np.random.randn(12, 10)
            mock_contrast.return_value = np.random.randn(7, 10)
            mock_tonnetz.return_value = np.random.randn(6, 10)
            mock_bandwidth.return_value = np.random.randn(1, 10)
            mock_rms.return_value = np.random.randn(1, 10)
            mock_zcr.return_value = np.random.randn(1, 10)
            mock_beat.return_value = (120.0, np.array([0, 10, 20]))
            
            result = self.processor.extract_features("test.wav")
            
            self.assertIsNotNone(result)
            self.assertIsInstance(result, np.ndarray)
            self.assertGreater(len(result), 0)
    
    def test_extract_features_nonexistent_file(self):
        """Тест извлечения признаков из несуществующего файла"""
        result = self.processor.extract_features("nonexistent.wav")
        self.assertIsNone(result)
    
    @patch('librosa.load')
    def test_extract_features_empty_audio(self, mock_load):
        """Тест извлечения признаков из пустого аудио"""
        mock_load.return_value = (np.array([]), 22050)
        
        result = self.processor.extract_features("empty.wav")
        self.assertIsNone(result)

class TestModelManager(unittest.TestCase):
    """Тесты для класса ModelManager"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.model_manager = ModelManager()
    
    def test_model_manager_initialization(self):
        """Тест инициализации ModelManager"""
        self.assertIsNotNone(self.model_manager)
        self.assertIsNone(self.model_manager.model)
        self.assertIsNone(self.model_manager.scaler)
        self.assertIsNone(self.model_manager.label_encoder)
    
    @patch('os.path.exists')
    def test_load_trained_model_missing_files(self, mock_exists):
        """Тест загрузки модели с отсутствующими файлами"""
        mock_exists.return_value = False
        
        result = self.model_manager.load_trained_model()
        self.assertFalse(result)
    
    @patch('pickle.load')
    @patch('os.path.exists')
    def test_load_trained_model_success(self, mock_exists, mock_pickle_load):
        """Тест успешной загрузки модели"""
        mock_exists.return_value = True
        mock_pickle_load.side_effect = [MagicMock(), MagicMock(), MagicMock()]
        
        result = self.model_manager.load_trained_model()
        self.assertTrue(result)
    
    def test_predict_genre_no_model(self):
        """Тест предсказания без загруженной модели"""
        result = self.model_manager.predict_genre("test.wav")
        self.assertIsNone(result)

class TestUtils(unittest.TestCase):
    """Тесты для утилит"""
    
    def test_check_dependencies(self):
        """Тест проверки зависимостей"""
        result = check_dependencies()
        # Этот тест может провалиться если не все зависимости установлены
        # В реальном проекте нужно было бы мокать импорты
        self.assertIsInstance(result, bool)
    
    def test_get_system_info(self):
        """Тест получения информации о системе"""
        info = get_system_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn('platform', info)
        self.assertIn('python_version', info)
        self.assertIn('cpu_count', info)
        self.assertIn('memory_total', info)
        self.assertIn('memory_available', info)

class TestIntegration(unittest.TestCase):
    """Интеграционные тесты"""
    
    def test_audio_processor_with_mock_data(self):
        """Тест интеграции AudioProcessor с мок-данными"""
        processor = AudioProcessor()
        
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Мокаем librosa для создания "валидного" аудио
            with patch('librosa.load') as mock_load, \
                 patch('librosa.feature.mfcc') as mock_mfcc, \
                 patch('librosa.feature.chroma_stft') as mock_chroma, \
                 patch('librosa.feature.spectral_contrast') as mock_contrast, \
                 patch('librosa.feature.tonnetz') as mock_tonnetz, \
                 patch('librosa.feature.spectral_bandwidth') as mock_bandwidth, \
                 patch('librosa.feature.rms') as mock_rms, \
                 patch('librosa.feature.zero_crossing_rate') as mock_zcr, \
                 patch('librosa.beat.beat_track') as mock_beat:
                
                # Настраиваем моки
                mock_load.return_value = (np.random.randn(1000), 22050)
                mock_mfcc.return_value = np.random.randn(13, 10)
                mock_chroma.return_value = np.random.randn(12, 10)
                mock_contrast.return_value = np.random.randn(7, 10)
                mock_tonnetz.return_value = np.random.randn(6, 10)
                mock_bandwidth.return_value = np.random.randn(1, 10)
                mock_rms.return_value = np.random.randn(1, 10)
                mock_zcr.return_value = np.random.randn(1, 10)
                mock_beat.return_value = (120.0, np.array([0, 10, 20]))
                
                # Тестируем валидацию
                is_valid = processor.validate_audio_file(tmp_path)
                self.assertTrue(is_valid)
                
                # Тестируем извлечение признаков
                features = processor.extract_features(tmp_path)
                self.assertIsNotNone(features)
                self.assertIsInstance(features, np.ndarray)
                
        finally:
            # Удаляем временный файл
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

def run_tests():
    """Запуск всех тестов"""
    # Создаем тестовый suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Добавляем тесты
    suite.addTests(loader.loadTestsFromTestCase(TestAudioProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestModelManager))
    suite.addTests(loader.loadTestsFromTestCase(TestUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
