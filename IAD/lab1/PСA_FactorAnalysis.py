import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA, FactorAnalysis

# Загрузка данных
wine = load_wine()
X = wine.data
y = wine.target
target_names = wine.target_names

# Применение PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Применение Factor Analysis
fa = FactorAnalysis(n_components=2)
X_fa = fa.fit_transform(X)

# Создание DataFrame для удобства визуализации
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['target'] = y

fa_df = pd.DataFrame(X_fa, columns=['FA1', 'FA2'])
fa_df['target'] = y

# Визуализация PCA
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='target', palette='viridis', s=100)
plt.title('PCA Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Визуализация Factor Analysis
plt.subplot(1, 2, 2)
sns.scatterplot(data=fa_df, x='FA1', y='FA2', hue='target', palette='viridis', s=100)
plt.title('Factor Analysis Results')
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')

plt.tight_layout()
plt.show()