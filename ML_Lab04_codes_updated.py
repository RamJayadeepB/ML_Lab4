import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_percentage_error, r2_score
)

# --------------------------
# A1: Confusion Matrix & Metrics
# --------------------------
def evaluate_model_performance(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)  # Predict on training data
    test_pred = model.predict(X_test)    # Predict on test data

    results = {
        "train_cm": confusion_matrix(y_train, train_pred),  # Confusion matrix (train)
        "test_cm": confusion_matrix(y_test, test_pred),     # Confusion matrix (test)
        "train_report": classification_report(y_train, train_pred, output_dict=True),  # Train metrics
        "test_report": classification_report(y_test, test_pred, output_dict=True),     # Test metrics
    }
    return results

# --------------------------
# A2: Error Metrics (Regression)
# --------------------------
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)        # Mean Squared Error
    rmse = np.sqrt(mse)                             # Root Mean Squared Error
    mape = mean_absolute_percentage_error(y_true, y_pred)  # Mean Absolute Percentage Error
    r2 = r2_score(y_true, y_pred)                   # R-squared score
    return mse, rmse, mape, r2

# --------------------------
# A3: Generate Random Points
# --------------------------
def generate_random_training_points(n=20, seed=42):
    np.random.seed(seed)                            # Set seed for reproducibility
    X = np.random.randint(1, 11, (n, 2))            # Generate 2D features (1-10)
    y = np.random.choice([0, 1], size=n)            # Random binary labels
    return X, y

def plot_training_points(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')  # Scatter plot with color by class
    plt.title("A3: Training Points (Random 2D Data)")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.show()

# --------------------------
# A4: Classify Grid Points with kNN
# --------------------------
def classify_grid_with_knn(X_train, y_train, k=3):
    model = KNeighborsClassifier(n_neighbors=k)     # Initialize kNN
    model.fit(X_train, y_train)                     # Train kNN

    x_test, y_test = np.meshgrid(np.arange(0, 10.1, 0.1), np.arange(0, 10.1, 0.1))  # Grid
    grid_points = np.c_[x_test.ravel(), y_test.ravel()]  # Flatten grid to (x,y) pairs

    preds = model.predict(grid_points)             # Predict on grid
    return x_test, y_test, preds.reshape(x_test.shape)  # Reshape predictions

def plot_classified_grid(x_test, y_test, grid_preds, title="A4: kNN Decision Boundary"):
    plt.contourf(x_test, y_test, grid_preds, cmap='bwr', alpha=0.3)  # Plot decision boundary
    plt.title(title)
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.show()

# --------------------------
# A5: Decision Boundaries for Multiple k
# --------------------------
def decision_boundaries_for_k(X_train, y_train, k_values=[1, 3, 5, 7]):
    for k in k_values:                               # Loop over k values
        x_test, y_test, grid_preds = classify_grid_with_knn(X_train, y_train, k=k)
        plot_classified_grid(x_test, y_test, grid_preds, title=f"A5: Decision Boundary (k={k})")

# --------------------------
# A6: Decision Boundary for Project Data
# --------------------------
def project_knn_boundary(X, y, feature_indices=[0, 1], classes=("rice", "maize"), k=3):
    mask = np.isin(y, classes)                      # Filter two classes
    X_sub, y_sub = X[mask][:, feature_indices], y[mask]

    y_sub = np.where(y_sub == classes[0], 0, 1)     # Convert to binary labels

    model = KNeighborsClassifier(n_neighbors=k)     # Train kNN on filtered data
    model.fit(X_sub, y_sub)

    x_min, x_max = X_sub[:, 0].min()-1, X_sub[:, 0].max()+1  # Range for feature 1
    y_min, y_max = X_sub[:, 1].min()-1, X_sub[:, 1].max()+1  # Range for feature 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))      # Grid for plotting

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points).reshape(xx.shape)          # Predict on grid

    plt.contourf(xx, yy, Z, cmap='bwr', alpha=0.3)            # Decision boundary plot
    plt.scatter(X_sub[:, 0], X_sub[:, 1], c=y_sub, cmap='bwr', edgecolor='k')  # Plot data points
    plt.title(f"A6: kNN Decision Boundary ({classes[0]} vs {classes[1]})")
    plt.xlabel(feature_indices[0])
    plt.ylabel(feature_indices[1])
    plt.show()

# --------------------------
# A7: Hyperparameter Tuning
# --------------------------
def tune_knn_hyperparameter(X_train, y_train, param_grid=None, search="grid"):
    if param_grid is None:
        param_grid = {"n_neighbors": list(range(1, 15))}      # Range of k values

    knn = KNeighborsClassifier()
    if search == "grid":
        search_cv = GridSearchCV(knn, param_grid, cv=5)       # Grid Search
    else:
        search_cv = RandomizedSearchCV(knn, param_grid, cv=5, n_iter=10, random_state=42)  # Random Search

    search_cv.fit(X_train, y_train)                           # Fit search
    return search_cv.best_params_, search_cv.best_score_      # Return best k and score

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\DELL\Downloads\Crop_recommendation.csv")  # Load dataset
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target_col = 'label'

    X = df[feature_cols].values           # Features
    y = df[target_col].values            # Target labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)  # Split data

    # A1: Classification metrics
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    results = evaluate_model_performance(knn_model, X_train, y_train, X_test, y_test)
    print("A1 Confusion Matrix (Train):\n", results["train_cm"])
    print("A1 Confusion Matrix (Test):\n", results["test_cm"])
    print("A1 Classification Report (Test):\n", pd.DataFrame(results["test_report"]).T)

    # A2: Regression metrics example
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X_train, X_train[:, 4])        # Train to predict humidity
    humidity_pred = reg.predict(X_test)
    mse, rmse, mape, r2 = regression_metrics(X_test[:, 4], humidity_pred)
    print("A2 Regression Metrics - MSE:", mse, "RMSE:", rmse, "MAPE:", mape, "R2:", r2)

    # A3: Random 2D dataset visualization
    X_rand, y_rand = generate_random_training_points()
    plot_training_points(X_rand, y_rand)

    # A4: Decision boundary for random data
    x_test, y_test, preds = classify_grid_with_knn(X_rand, y_rand, k=3)
    plot_classified_grid(x_test, y_test, preds, title="A4: Decision Boundary (k=3)")

    # A5: Decision boundaries for different k
    decision_boundaries_for_k(X_rand, y_rand, k_values=[1, 3, 5, 7])

    # A6: Project data (two features, two classes)
    project_knn_boundary(X, y, feature_indices=[0, 1], classes=("rice", "maize"), k=1)
    project_knn_boundary(X, y, feature_indices=[0, 1], classes=("rice", "maize"), k=3)
    project_knn_boundary(X, y, feature_indices=[0, 1], classes=("rice", "maize"), k=5)
    project_knn_boundary(X, y, feature_indices=[0, 1], classes=("rice", "maize"), k=7)

    # A7: Hyperparameter tuning
    best_params, best_score = tune_knn_hyperparameter(X_train, y_train, search="grid")
    print("A7 Best Params:", best_params, "Best Score:", best_score)
