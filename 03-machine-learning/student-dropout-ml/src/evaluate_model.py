import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from preprocess_data import preprocess

MODEL_PATH = "models/dropout_model.pkl"


def evaluate_model():
    X_train, X_test, y_train, y_test = preprocess()

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)

    print("Model Evaluation Result")
    print("-----------------------")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model()
