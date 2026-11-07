"""Random Forest training, tuning, and evaluation helpers."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GroupKFold, ParameterSampler,
                                     cross_val_predict, cross_validate)


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
                       n_iter, cv_splits, random_state, progress=None):
    """Search Random Forest hyperparameters with GroupKFold cross-validation.

    Returns the best model, its parameters, mean per-fold CV scores, and
    pooled out-of-fold CV scores. ``progress(done, total, best_mae)`` is
    called after each candidate.
    """
    cv = GroupKFold(n_splits=int(cv_splits))
    sampler = ParameterSampler(
        param_distributions,
        n_iter=int(n_iter),
        random_state=random_state,
    )
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    best_params = None
    best_scores = None
    for done, params in enumerate(sampler, start=1):
        scores = cross_validate(
            RandomForestRegressor(random_state=random_state, **params),
            X_train, y_train,
            groups=groups,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        cv_scores = {
            "mae": -float(np.mean(scores["test_mae"])),
            "rmse": -float(np.mean(scores["test_rmse"])),
            "r2": float(np.mean(scores["test_r2"])),
        }

        if best_scores is None or cv_scores["mae"] < best_scores["mae"]:
            best_params = params
            best_scores = cv_scores

        if progress is not None:
            progress(done, int(n_iter), best_scores["mae"])

    model = RandomForestRegressor(random_state=random_state, **best_params)
    model.fit(X_train, y_train)

    oof_predictions = cross_val_predict(
        RandomForestRegressor(random_state=random_state, **best_params),
        X_train, y_train,
        groups=groups,
        cv=cv,
        n_jobs=-1,
    )
    pooled_scores = {
        "mae": float(mean_absolute_error(y_train, oof_predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_train, oof_predictions))),
        "r2": float(r2_score(y_train, oof_predictions)),
    }
    return {
        "model": model,
        "params": best_params,
        "cv": best_scores,
        "cv_pooled": pooled_scores,
    }


def evaluate_random_forest(model, X, y):
    """Return MAE, RMSE, R2, and predictions for the given data."""
    predictions = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "r2": float(r2_score(y, predictions)),
        "predictions": predictions,
    }
