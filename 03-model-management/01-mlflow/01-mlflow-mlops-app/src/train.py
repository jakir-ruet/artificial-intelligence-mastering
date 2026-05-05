import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error

with mlflow.start_run() as run:

    model = get_model()
    model.fit(x_train, y_train)

    # Correct RUN ID
    print("RUN ID:", run.info.run_id)

    preds = model.predict(x_test)
    mse = mean_squared_error(y_test, preds)

    # logs
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)

    # Correct way to SAVE model
    mlflow.sklearn.log_model(model, "model")

    print("Training complete, MSE:", mse)
