"""Random Forest estimator and tuning helpers."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterSampler, cross_validate

from pci_ml.experiments import fit_full, make_folds, oof_predict


def _max_depth(value):
    return None if value in (0, None) else int(value)


def _max_samples(value):
    return None if value in (0, None) else float(value)


def rf_estimator(params, random_state):
    """Build an unfitted Random Forest from a parameter snapshot."""
    return RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=_max_depth(params["max_depth"]),
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=float(params["max_features"]),
        max_samples=_max_samples(params.get("max_samples", 1.0)),
        random_state=random_state,
    )


def tune_random_forest(X_train, y_train, groups, param_distributions,
                       n_iter, cv_splits, random_state, progress=None):
    """Search Random Forest hyperparameters with GroupKFold cross-validation.

    Returns the best model, its parameters, the shared folds, and pooled
    out-of-fold predictions. ``progress(done, total, best_mae)`` is called
    after each candidate.
    """
    folds = make_folds(groups, n_splits=int(cv_splits))
    sampler = ParameterSampler(
        param_distributions,
        n_iter=int(n_iter),
        random_state=random_state,
    )
    scoring = {"mae": "neg_mean_absolute_error"}

    best_params = None
    best_mae = None
    for done, params in enumerate(sampler, start=1):
        scores = cross_validate(
            rf_estimator(params, random_state),
            X_train, y_train,
            cv=folds,
            scoring=scoring,
            n_jobs=-1,
        )
        mae = -float(np.mean(scores["test_mae"]))

        if best_mae is None or mae < best_mae:
            best_params = params
            best_mae = mae

        if progress is not None:
            progress(done, int(n_iter), best_mae)

    model = fit_full(rf_estimator(best_params, random_state), X_train, y_train)
    oof_predictions = oof_predict(
        rf_estimator(best_params, random_state), X_train, y_train, folds
    )
    return {
        "model": model,
        "params": best_params,
        "folds": folds,
        "oof": oof_predictions,
    }
