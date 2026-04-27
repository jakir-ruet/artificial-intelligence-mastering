import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Enable MLflow
mlflow.set_experiment("iris-classification")

with mlflow.start_run():

    # Parameters
    n_estimators = 50
    max_depth = 3

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metric
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    # Save plot (artifact)
    plt.figure()
    plt.hist(y_pred)
    plt.title("Prediction Distribution")
    plt.savefig("pred_plot.png")

    mlflow.log_artifact("pred_plot.png")

    # Log model
    # mlflow.sklearn.log_model(model, "model")
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="IrisClassifier"
    )

    print(f"Accuracy: {acc}")
