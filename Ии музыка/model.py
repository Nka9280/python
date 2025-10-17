import os
import numpy as np
import librosa
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score, confusion_matrix, precision_recall_curve
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns

# Путь к папке с набором данных
data_path = "path_to_music_dataset"  
genres = ['pop', 'rock', 'hiphop', 'classical', 'disco', 'reggae', 'metal', 'country']

# Функция для извлечения признаков из аудиофайла
def extract_features(file_path, expected_num_features=None):
    try:
        # Загрузка аудиофайла
        audio, sample_rate = librosa.load(file_path, sr=None, duration=10)

        # Извлечение MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)

        # Извлечение хрома
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_mean = np.mean(chroma, axis=1)

        # Извлечение спектрального контраста
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

        # Извлечение тоннета
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
        features = np.hstack([mfccs_mean, chroma_mean, spectral_contrast_mean, tonnetz_mean,
                              spectral_bandwidth_mean, rms_mean, zcr_mean, tempo])

        # Проверка на соответствие количества признаков
        if expected_num_features is not None and len(features) != expected_num_features:
            print(f"Ожидается {expected_num_features} признаков, но получено {len(features)}")
            return None

        return features
    except Exception as e:
        print(f"Ошибка загрузки {file_path}: {e}")
        return None

# Создание списков для признаков и меток
features = []
labels = []

# Извлечение признаков из всех аудиофайлов
for genre in genres:
    genre_path = os.path.join(data_path, genre)
    for file_name in os.listdir(genre_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(genre_path, file_name)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(genre)

# Преобразование списков в массивы numpy
features = np.array(features)
labels = np.array(labels)

# Проверка количества извлеченных признаков
print(f"Количество извлеченных признаков: {features.shape[1]}")

# Кодирование меток (жанров) в числовой формат
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# Нормализация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Определение параметров для поиска
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Создание экземпляра Random Forest
model = RandomForestClassifier(random_state=42)

# Настройка GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

# Обучение модели с GridSearchCV
print("Начало обучения модели...")
grid_search.fit(X_train, y_train)
print("Обучение завершено.")

# Вывод лучших параметров и их оценки
print("Лучшие параметры:", grid_search.best_params_)
print("Лучшая точность:", grid_search.best_score_)

# Получаем оптимизированную модель
best_model = grid_search.best_estimator_

# Оценка оптимизированной модели на тестовой выборке
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=genres)
print(report)

# Дополнительные метрики
def print_additional_metrics(y_true, y_pred, genres):
    print("Точность (Accuracy):", accuracy_score(y_true, y_pred))
    print("Средняя точность (Precision):", precision_score(y_true, y_pred, average='weighted'))
    print("Средняя полнота (Recall):", recall_score(y_true, y_pred, average='weighted'))
    print("Средняя F1-меры (F1-score):", f1_score(y_true, y_pred, average='weighted'))

print_additional_metrics(y_test, y_pred, genres)

# Построение матрицы ошибок
def plot_confusion_matrix(y_true, y_pred, genres):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genres, yticklabels=genres)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(y_test, y_pred, genres)

# Визуализация важности признаков
def plot_feature_importance(model, feature_names):
    feature_importances = model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    
    plt.figure(figsize=(10, 7))
    plt.barh(range(len(feature_importances)), feature_importances[sorted_idx], align='center')
    plt.yticks(range(len(feature_importances)), np.array(feature_names)[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importances from RandomForest')
    plt.show()

# Формируем имена признаков
feature_names = []
feature_names.extend([f'MFCC_{i+1}' for i in range(13)])  # MFCC
feature_names.extend([f'Chroma_{i+1}' for i in range(12)])  # Chroma
feature_names.extend([f'Spectral_Contrast_{i+1}' for i in range(7)])  # Spectral Contrast
feature_names.extend([f'Tonnetz_{i+1}' for i in range(6)])  # Tonnetz
feature_names.append('Spectral_Bandwidth')  # Spectral Bandwidth
feature_names.append('RMS')  # RMS
feature_names.append('Zero_Crossing_Rate')  # Zero Crossing Rate
feature_names.append('Tempo')  # Tempo

plot_feature_importance(best_model, feature_names)

# Precision-Recall Curve
def plot_precision_recall_curve(model, X_test, y_test, label_encoder):
    y_prob = model.predict_proba(X_test)
    plt.figure(figsize=(10, 7))
    
    for i, genre in enumerate(label_encoder.classes_):
        precision, recall, _ = precision_recall_curve(y_test == i, y_prob[:, i])
        plt.plot(recall, precision, label=genre)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

plot_precision_recall_curve(best_model, X_test, y_test, label_encoder)

# Сохранение модели
with open('best_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)
print("Модель успешно сохранена в best_model.pkl")

# Сохранение скейлера
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
print("Скейлер успешно сохранён в scaler.pkl")

# Сохранение кодировщика меток
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)
print("Кодировщик меток успешно сохранён в label_encoder.pkl")

# Функция для предсказания жанра
def predict_genre(file_path, model, scaler, label_encoder):
    expected_num_features = scaler.n_features_in_

    # Извлекаем признаки из нового аудиофайла
    features = extract_features(file_path, expected_num_features)

    # Проверка, удалось ли извлечь признаки
    if features is None:
        return "Ошибка извлечения признаков или их количество не совпадает с ожидаемым."

    # Масштабируем признаки
    features_scaled = scaler.transform([features])

    # Предсказываем жанр
    predicted_label = model.predict(features_scaled)
    predicted_genre = label_encoder.inverse_transform(predicted_label)

    return predicted_genre[0]

# Интерфейс Tkinter для выбора файла и отображения предсказанного жанра
def open_file_dialog(model, scaler, label_encoder, result_label):
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        predicted_genre = predict_genre(file_path, model, scaler, label_encoder)
        result_label.config(text=f"Предсказанный жанр: {predicted_genre}")

# Запуск Tkinter приложения
def run_app():
    # Загрузка обученной модели, скейлера и кодировщика меток
    with open('best_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('label_encoder.pkl', 'rb') as le_file:
        label_encoder = pickle.load(le_file)

    # Настройка окна Tkinter
    root = tk.Tk()
    root.title("Music Genre Classifier")

    # Элементы интерфейса
    open_button = tk.Button(root, text="Выбрать файл", 
                            command=lambda: open_file_dialog(model, scaler, label_encoder, result_label))
    open_button.pack(pady=20)

    result_label = tk.Label(root, text="Предсказанный жанр: ")
    result_label.pack(pady=20)

    # Запуск интерфейса
    root.mainloop()

# Запуск приложения
if __name__ == "__main__":
    run_app()
