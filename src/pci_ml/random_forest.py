"""Random Forest training, tuning, and evaluation helpers."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV


def _max_depth(value):
    return None if value in (0, None) else int(value)


def _max_samples(value):
    return None if value in (0, None) else float(value)


def train_random_forest(X_train, y_train, params, random_state):
    """Train a Random Forest regressor from a parameter snapshot.

    ``max_depth`` and ``max_samples`` use 0 or None to mean no limit.
    """
    model = RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=_max_depth(params["max_depth"]),
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=float(params["max_features"]),
        max_samples=_max_samples(params.get("max_samples", 1.0)),
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def tune_random_forest(X_train, y_train, groups, param_distributions,
                       n_iter, cv_splits, random_state):
    """Search Random Forest hyperparameters with GroupKFold cross-validation."""
    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=random_state),
        param_distributions=param_distributions,
        n_iter=int(n_iter),
        scoring="neg_mean_absolute_error",
        cv=GroupKFold(n_splits=int(cv_splits)),
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train, groups=groups)
    return search


def evaluate_random_forest(model, X, y):
    """Return MAE, RMSE, R2, and predictions for the given data."""
    predictions = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "r2": float(r2_score(y, predictions)),
        "predictions": predictions,
    }
