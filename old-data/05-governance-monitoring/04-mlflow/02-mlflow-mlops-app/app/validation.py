from sklearn.model_selection import cross_val_score
import numpy as np

def cross_validate_model(model, X_train, y_train, cv=5):
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        scoring="neg_mean_squared_error",
        cv=cv
    )

    mse_scores = -scores

    return {
        "cv_mse_mean": np.mean(mse_scores),
        "cv_mse_std": np.std(mse_scores),
        "cv_rmse_mean": np.sqrt(np.mean(mse_scores))
    }
