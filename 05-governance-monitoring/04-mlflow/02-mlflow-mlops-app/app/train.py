from .data import load_data
from .models import get_models
from .evaluate import evaluate_model
from .validation import cross_validate_model
from .mlflow_utils import setup_mlflow, log_run

def run_training():
    setup_mlflow()

    X_train, X_test, y_train, y_test = load_data()
    models = get_models()

    for name, model in models.items():

        # 🔹 Cross-validation BEFORE final training
        cv_metrics = cross_validate_model(model, X_train, y_train)

        # Train final model
        model.fit(X_train, y_train)

        # Test evaluation
        y_pred = model.predict(X_test)
        test_metrics = evaluate_model(y_test, y_pred)

        # Merge metrics
        all_metrics = {**cv_metrics, **test_metrics}

        log_run(name, model, all_metrics)

        print(f"{name}: {all_metrics}")
