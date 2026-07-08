import shutil
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess_data import preprocess

MODELS = {
    "Logistic Regression": "models/dropout_model.pkl",
    "Random Forest": "models/dropout_random_forest_model.pkl",
    "Tuned Random Forest": "models/dropout_random_forest_tuned_model.pkl",
}

BEST_MODEL_PATH = "models/best_dropout_model.pkl"


def evaluate_model(name, model_path, X_test, y_test):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    return {
        "name": name,
        "path": model_path,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }


def select_best_model():
    _, X_test, _, y_test = preprocess()

    results = []

    for name, path in MODELS.items():
        results.append(evaluate_model(name, path, X_test, y_test))

    best_model = max(results, key=lambda item: item["f1_score"])

    shutil.copy(best_model["path"], BEST_MODEL_PATH)

    print("Best Model Selected")
    print("-------------------")
    print("Model    :", best_model["name"])
    print("Accuracy :", best_model["accuracy"])
    print("Precision:", best_model["precision"])
    print("Recall   :", best_model["recall"])
    print("F1 Score :", best_model["f1_score"])
    print("Saved As :", BEST_MODEL_PATH)


if __name__ == "__main__":
    select_best_model()
