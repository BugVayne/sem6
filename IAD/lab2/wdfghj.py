# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
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


# Define models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs'),
    "SVM": SVC(kernel='linear', probability=True, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=True, eval_metric='mlogloss')
}

# Dictionary to store results
results = {}

# Train and evaluate models
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    # Train model with progress bar
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None

    # Calculate metrics
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
        results[model_name]["auc_roc"] = roc_auc

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f"ROC Curve - {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()

# Display comparative results
results_df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(results_df)

# Hyperparameter tuning for Random Forest
print("\nPerforming hyperparameter tuning for Random Forest...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

print("\nBest parameters for Random Forest:", grid_search.best_params_)
print("Best score for Random Forest:", grid_search.best_score_)

best_rf_model = grid_search.best_estimator_
feature_importances = best_rf_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances)
plt.title("Feature Importances - Random Forest")
plt.show()