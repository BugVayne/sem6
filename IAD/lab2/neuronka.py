# Импортируем необходимые библиотеки
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras import layers
from sklearn.datasets import fetch_openml

# Загрузка данных MNIST
print("Loading MNIST data...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target

# Преобразование данных в numpy массивы и нормализация
X = X.to_numpy()
X = X / 255.0  # Нормализация в диапазоне [0, 1]
y = y.astype(int).to_numpy()

# Уменьшение объема данных (10,000 образцов)
subset_size = 10000
X = X[:subset_size]
y = y[:subset_size]

# Разделение данных на обучающую и тестовую выборки
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Масштабирование данных
print("Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Изменение формы входных данных для модели
X_train_scaled = X_train_scaled.reshape(-1, 28, 28)
X_test_scaled = X_test_scaled.reshape(-1, 28, 28)

# Преобразование меток
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
y_test_classes = np.argmax(y_test, axis=1)  # Для оценки точности

# Создание модели
model = keras.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation='relu', name='dense_1'),
    layers.Dense(10, activation='softmax', name='output')
])

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train_scaled, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Предсказание на тестовом наборе
y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)  # Предсказанные классы

# Оценка производительности
accuracy = accuracy_score(y_test_classes, y_pred)
report = classification_report(y_test_classes, y_pred, output_dict=True)

# Сохранение результатов в словарь
results = {}
model_name = "Neural Network"
results[model_name] = {
    "accuracy": accuracy,
    "precision": report["weighted avg"]["precision"],
    "recall": report["weighted avg"]["recall"],
    "f1_score": report["weighted avg"]["f1-score"]
}

# Вывод результатов
print(f"\n{model_name} Accuracy: {accuracy:.4f}")
print(classification_report(y_test_classes, y_pred))

# Функция для визуализации матрицы путаницы
def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    # Добавление