import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess_data import preprocess

MODEL_PATH = "models/dropout_random_forest_model.pkl"


def train_random_forest():
    X_train, X_test, y_train, y_test = preprocess()

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Random Forest Training Completed")
    print("--------------------------------")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_random_forest()
