#%%
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#%%

# Load MNIST data
print("Loading MNIST data...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target

# Convert data to numpy arrays and normalize
X = X.to_numpy()
X = X / 255.0  # Normalize to [0, 1]
y = y.astype(int).to_numpy()

# Take a smaller subset of the data (10,000 samples)
subset_size = 10000
X = X[:subset_size]
y = y[:subset_size]

# Split data into train and test sets
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the data
print("Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


results = {}
#%%

# Function to visualize sample images
def plot_sample_images(X, y, n=10):
    plt.figure(figsize=(10, 2))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.show()


# Visualize first 10 images
print("Sample images:")
plot_sample_images(X_train, y_train)
#%%
def plot_confusion_and_roc(y_test, y_pred, y_proba, model_name):
    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Plot ROC curve (if probabilities are available)
    if y_proba is not None:
        lb = LabelBinarizer()
        y_test_binarized = lb.fit_transform(y_test)
        fpr, tpr, _ = roc_curve(y_test_binarized.ravel(), y_proba.ravel())
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f"ROC Curve - {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()
#%%

# Train and evaluate Logistic Regression
model_name = "Logistic Regression"
model = LogisticRegression(max_iter=1000, solver='lbfgs', C = 0.1)

print(f"\nTraining {model_name}...")
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

results[model_name] = {
    "accuracy": accuracy,
    "precision": report["weighted avg"]["precision"],
    "recall": report["weighted avg"]["recall"],
    "f1_score": report["weighted avg"]["f1-score"]
}

print(f"\n{model_name} Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

plot_confusion_and_roc(y_test, y_pred, y_proba, model_name)

#%%

# Train and evaluate SVM
model_name = "SVM"
model = SVC(kernel='poly', shrinking=True)

print(f"\nTraining {model_name}...")
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

results[model_name] = {
    "accuracy": accuracy,
    "precision": report["weighted avg"]["precision"],
    "recall": report["weighted avg"]["recall"],
    "f1_score": report["weighted avg"]["f1-score"]
}

print(f"\n{model_name} Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

plot_confusion_and_roc(y_test, y_pred, y_proba, model_name)
#%%

param_grid = {
    'n_estimators': [100, 200, 300], 
    'max_depth': [10, 20, 30, 40], 
    'min_samples_split': [2, 5, 10],   
    'min_samples_leaf': [1, 2, 4]    
}

model_name = "Random Forest"
model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

print(f"\nTraining {model_name} with GridSearchCV...")
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled) if hasattr(best_model, "predict_proba") else None

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

results[model_name] = {
    "accuracy": accuracy,
    "precision": report["weighted avg"]["precision"],
    "recall": report["weighted avg"]["recall"],
    "f1_score": report["weighted avg"]["f1-score"],
}

print(f"\n{model_name} Accuracy: {accuracy:.4f}")
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred))

plot_confusion_and_roc(y_test, y_pred, y_proba, model_name)
#%%

# Определите параметры для поиска
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.2],
    'subsample': [0.8, 1.0],
}

# Создайте модель XGBoost
model = XGBClassifier(eval_metric='mlogloss', tree_method='hist', use_label_encoder=False, device='cuda')

# Создайте объект GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring='accuracy', cv=3, verbose=1, n_jobs=-1)

print("\nTraining XGBoost with GridSearchCV...")
grid_search.fit(X_train_scaled, y_train)

# Получите лучшие параметры и модель
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Прогнозирование и оценка модели
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled) if hasattr(best_model, "predict_proba") else None

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

results["XGBoost"] = {
    "accuracy": accuracy,
    "precision": report["weighted avg"]["precision"],
    "recall": report["weighted avg"]["recall"],
    "f1_score": report["weighted avg"]["f1-score"]  
}

print(f"\nBest Parameters: {best_params}")
print(f"\nXGBoost Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

plot_confusion_and_roc(y_test, y_pred, y_proba, "XGBoost")
#%%

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

    # Добавление текста с количеством
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# Визуализация матрицы путаницы
plot_confusion_matrix(y_test_classes, y_pred)
#%%

# Display comparative results
results_df = pd.DataFrame(results).T
print("\nModel Comparison:")        
print(results_df)
