import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from matplotlib.colors import ListedColormap

# Загрузка данных Iris
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Применение LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Применение QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X[:, :2], y)  # Используем только первые два признака для визуализации QDA

# Создание сетки для границ классификации
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Предсказание классов для сетки
Z = qda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Построение графиков
plt.figure(figsize=(18, 5))

# График LDA
plt.subplot(1, 3, 1)
for i, target_name in enumerate(target_names):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], label=target_name)
plt.title('LDA на наборе данных Iris')
plt.xlabel('LDA 1')
plt.ylabel('LDA 2')
plt.legend()

# График исходных данных
plt.subplot(1, 3, 2)
for i, target_name in enumerate(target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=target_name)
plt.title('Исходные данные (первая пара признаков)')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()

# График QDA с границами классификации
plt.subplot(1, 3, 3)
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ['#FF0000', '#00FF00', '#0000FF']
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
for i, target_name in enumerate(target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], color=cmap_bold[i], label=target_name, edgecolor='k')
plt.title('Границы классификации QDA')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()

plt.tight_layout()
plt.show()