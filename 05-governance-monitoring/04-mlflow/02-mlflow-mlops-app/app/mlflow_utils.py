import mlflow
import mlflow.sklearn
from .config import Config

def setup_mlflow():
	mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
	mlflow.set_experiment(Config.EXPERIMENT_NAME)

def log_run(model_name, model, metrics):
	with mlflow.start_run(run_name=model_name):
		for k, v in metrics.items():
			mlflow.log_metric(k, v)
		mlflow.sklearn.log_model(
			model,
			artifact_path=model_name,
			registered_model_name=model_name # Enables Model Registry
		)
