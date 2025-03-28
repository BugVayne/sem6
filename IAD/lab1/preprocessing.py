import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Шаг 1: Загрузка данных
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# Объединение данных
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Преобразование в DataFrame
columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
df = pd.DataFrame(data, columns=columns)

# Шаг 2: Предобработка данных
# 2.1 Масштабирование
scaler_standard = StandardScaler()
X_standard = scaler_standard.fit_transform(df)

# 2.2 Нормализация
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(df)

# Шаг 3: Применение PCA
pca_standard = PCA()
X_pca_standard = pca_standard.fit_transform(X_standard)

pca_minmax = PCA()
X_pca_minmax = pca_minmax.fit_transform(X_minmax)

# Шаг 4: Сравнение результатов
# Объясненная дисперсия
explained_variance_standard = pca_standard.explained_variance_ratio_
explained_variance_minmax = pca_minmax.explained_variance_ratio_

# Визуализация
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance_standard) + 1), explained_variance_standard)
plt.title('PCA with StandardScaler')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')

plt.subplot(1, 2, 2)
plt.bar(range(1, len(explained_variance_minmax) + 1), explained_variance_minmax)
plt.title('PCA with MinMaxScaler')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')

plt.tight_layout()
plt.show()