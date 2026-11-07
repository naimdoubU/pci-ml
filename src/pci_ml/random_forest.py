"""Random Forest training and evaluation helpers."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_random_forest(X_train, y_train, params, random_state):
    """Train a Random Forest regressor from a parameter snapshot.

    ``max_depth`` uses 0 to mean no depth limit.
    """
    max_depth = int(params["max_depth"])
    model = RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=None if max_depth == 0 else max_depth,
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=float(params["max_features"]),
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_random_forest(model, X, y):
    """Return MAE, RMSE, R2, and predictions for the given data."""
    predictions = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "r2": float(r2_score(y, predictions)),
        "predictions": predictions,
    }
