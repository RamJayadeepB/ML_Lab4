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
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    results = {
        "train_cm": confusion_matrix(y_train, train_pred),
        "test_cm": confusion_matrix(y_test, test_pred),
        "train_report": classification_report(y_train, train_pred, output_dict=True),
        "test_report": classification_report(y_test, test_pred, output_dict=True),
    }
    return results

# --------------------------
# A2: Error Metrics (using regression approach on one feature)
# --------------------------
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

# --------------------------
# A3: Generate 20 random points with 2 features + classes
# --------------------------
def generate_random_training_points(n=20, seed=42):
    np.random.seed(seed)
    X = np.random.randint(1, 11, (n, 2))
    y = np.random.choice([0, 1], size=n)
    return X, y

def plot_training_points(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
    plt.title("A3: Training Points (Random 2D Data)")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.show()

# --------------------------
# A4: Classify 10,000 grid test points using kNN (k=3)
# --------------------------
def classify_grid_with_knn(X_train, y_train, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    x_test, y_test = np.meshgrid(np.arange(0, 10.1, 0.1), np.arange(0, 10.1, 0.1))
    grid_points = np.c_[x_test.ravel(), y_test.ravel()]

    preds = model.predict(grid_points)
    return x_test, y_test, preds.reshape(x_test.shape)

def plot_classified_grid(x_test, y_test, grid_preds, title="A4: kNN Decision Boundary"):
    plt.contourf(x_test, y_test, grid_preds, cmap='bwr', alpha=0.3)
    plt.title(title)
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.show()

# --------------------------
# A5: Repeat A4 for various k values
# --------------------------
def decision_boundaries_for_k(X_train, y_train, k_values=[1, 3, 5, 7]):
    for k in k_values:
        x_test, y_test, grid_preds = classify_grid_with_knn(X_train, y_train, k=k)
        plot_classified_grid(x_test, y_test, grid_preds, title=f"A5: Decision Boundary (k={k})")

# --------------------------
# A6: Repeat A3–A5 for project data (two features & two classes)
# --------------------------
def project_knn_boundary(X, y, feature_indices=[0, 1], classes=("rice", "maize"), k=3):
    # Filter only two classes
    mask = np.isin(y, classes)
    X_sub, y_sub = X[mask][:, feature_indices], y[mask]

    # Convert class labels to 0/1
    y_sub = np.where(y_sub == classes[0], 0, 1)

    # Train kNN
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_sub, y_sub)

    # Grid for plotting
    x_min, x_max = X_sub[:, 0].min()-1, X_sub[:, 0].max()+1
    y_min, y_max = X_sub[:, 1].min()-1, X_sub[:, 1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, cmap='bwr', alpha=0.3)
    plt.scatter(X_sub[:, 0], X_sub[:, 1], c=y_sub, cmap='bwr', edgecolor='k')
    plt.title(f"A6: kNN Decision Boundary ({classes[0]} vs {classes[1]})")
    plt.xlabel(feature_indices[0])
    plt.ylabel(feature_indices[1])
    plt.show()

# --------------------------
# A7: Hyperparameter Tuning with GridSearchCV
# --------------------------
def tune_knn_hyperparameter(X_train, y_train, param_grid=None, search="grid"):
    if param_grid is None:
        param_grid = {"n_neighbors": list(range(1, 15))}

    knn = KNeighborsClassifier()
    if search == "grid":
        search_cv = GridSearchCV(knn, param_grid, cv=5)
    else:
        search_cv = RandomizedSearchCV(knn, param_grid, cv=5, n_iter=10, random_state=42)

    search_cv.fit(X_train, y_train)
    return search_cv.best_params_, search_cv.best_score_

# --------------------------
# Main Section
# --------------------------
if __name__ == "__main__":
    # Load Crop dataset
    df = pd.read_csv("C:/Users/bramj/OneDrive/Desktop/Crop_recommendation.csv")
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target_col = 'label'

    X = df[feature_cols].values
    y = df[target_col].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # A1
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    results = evaluate_model_performance(knn_model, X_train, y_train, X_test, y_test)
    print("A1 Confusion Matrix (Train):\n", results["train_cm"])
    print("A1 Confusion Matrix (Test):\n", results["test_cm"])
    print("A1 Classification Report (Test):\n", pd.DataFrame(results["test_report"]).T)

    # A2 (for illustration, predicting humidity from features using regression)
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X_train, X_train[:, 4])   # train on features, predict humidity
    humidity_pred = reg.predict(X_test)
    mse, rmse, mape, r2 = regression_metrics(X_test[:, 4], humidity_pred)
    print("A2 Regression Metrics - MSE:", mse, "RMSE:", rmse, "MAPE:", mape, "R2:", r2)

    # A3
    X_rand, y_rand = generate_random_training_points()
    plot_training_points(X_rand, y_rand)

    # A4
    x_test, y_test, preds = classify_grid_with_knn(X_rand, y_rand, k=3)
    plot_classified_grid(x_test, y_test, preds, title="A4: Decision Boundary (k=3)")

    # A5
    decision_boundaries_for_k(X_rand, y_rand, k_values=[1, 3, 5, 7])

    # A6 (Crop dataset, only two classes + two features)
    project_knn_boundary(X, y, feature_indices=[0, 1], classes=("rice", "maize"), k=3)

    # A7
    best_params, best_score = tune_knn_hyperparameter(X_train, y_train, search="grid")
    print("A7 Best Params:", best_params, "Best Score:", best_score)
