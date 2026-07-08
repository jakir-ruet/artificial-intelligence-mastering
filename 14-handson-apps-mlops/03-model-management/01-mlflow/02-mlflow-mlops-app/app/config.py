import os
class Config:
	MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
	EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "mlflow-mlops-app")
	TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
	RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
