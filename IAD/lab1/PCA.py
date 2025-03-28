import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Загружаем набор данных Iris
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Применяем PCA для снижения размерности до 2-х
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Визуализация результатов
plt.figure(figsize=(8, 6))
colors = ['navy', 'turquoise', 'darkorange']

for i, target_name in enumerate(target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=colors[i], label=target_name)

plt.title('PCA на наборе данных Iris')
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.legend()
plt.grid()
plt.show()