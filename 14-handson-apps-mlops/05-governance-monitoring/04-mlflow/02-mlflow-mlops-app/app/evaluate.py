from sklearn.metrics import(
	mean_squared_error,
	mean_absolute_error,
	r2_score,
	explained_variance_score
)
import numpy as np

def evaluate_model(y_true, y_pred):
	mse = mean_squared_error(y_true, y_pred)
	return{
		"mse": mse,
		"rmse": np.sqrt(mse),
		"mae": mean_absolute_error(y_true, y_pred),
		"r2": r2_score(y_true, y_pred),
		"explained_variance": explained_variance_score(y_true, y_pred)
	}
