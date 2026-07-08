import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess_data import preprocess

MODELS = {
    "Logistic Regression": "models/dropout_model.pkl",
    "Random Forest": "models/dropout_random_forest_model.pkl",
    "Tuned Random Forest": "models/dropout_random_forest_tuned_model.pkl",
}


def evaluate_model(name, model_path, X_test, y_test):
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }


def compare_models():
    X_train, X_test, y_train, y_test = preprocess()

    results = []

    for name, path in MODELS.items():
        result = evaluate_model(name, path, X_test, y_test)
        results.append(result)

    print("Model Comparison")
    print("----------------")

    for result in results:
        print()
        print(result["model"])
        print("Accuracy :", result["accuracy"])
        print("Precision:", result["precision"])
        print("Recall   :", result["recall"])
        print("F1 Score :", result["f1_score"])


if __name__ == "__main__":
    compare_models()
