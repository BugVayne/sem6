import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression  # Change to Logistic Regression
from sklearn.metrics import classification_report

try:
    # Загрузка данных MNIST
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data[:1000]  # Используем только 1,000 образцов
    y = mnist.target.astype(int)[:1000]  # Исправлено

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Применение PCA
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Обучение классификатора на PCA
    classifier_pca = LogisticRegression(max_iter=1000)  # Use Logistic Regression
    classifier_pca.fit(X_train_pca, y_train)
    y_pred_pca = classifier_pca.predict(X_test_pca)

    # Отчет о классификации для PCA
    print("PCA Classification Report:")
    print(classification_report(y_test, y_pred_pca))

    # Применение LDA
    lda = LDA(n_components=9)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    # Обучение классификатора на LDA
    classifier_lda = LogisticRegression(max_iter=1000)  # Use Logistic Regression
    classifier_lda.fit(X_train_lda, y_train)
    y_pred_lda = classifier_lda.predict(X_test_lda)

    # Отчет о классификации для LDA
    print("LDA Classification Report:")
    print(classification_report(y_test, y_pred_lda))

except Exception as e:
    print("An error occurred:", e)