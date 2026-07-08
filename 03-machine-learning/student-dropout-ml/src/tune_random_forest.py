import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess_data import preprocess

MODEL_PATH = "models/dropout_random_forest_tuned_model.pkl"


def tune_random_forest():
    X_train, X_test, y_train, y_test = preprocess()

    model = RandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, scoring="f1", n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    print("Random Forest Hyperparameter Tuning Completed")
    print("---------------------------------------------")
    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

    joblib.dump(best_model, MODEL_PATH)
    print(f"Best model saved to {MODEL_PATH}")


if __name__ == "__main__":
    tune_random_forest()
