"""Model-agnostic helpers for the PCI experiments.

The same folds, metrics, weights, and calibration can be reused by the
Random Forest, XGBoost, and ANN sections. Pass any scikit-learn compatible
regressor to :func:`oof_predict`, :func:`fit_full`, or
:func:`evaluate_variant`.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

PCI_BINS = [0, 25, 40, 55, 70, 85, 100]


def make_folds(groups, n_splits=5, shuffle=False, random_state=None):
    """Return ``(train_idx, valid_idx)`` group-aware folds.

    ``GroupKFold`` gained ``shuffle``/``random_state`` in scikit-learn 1.6.
    The project pins 1.9+, so the keyword arguments are safe here.
    """
    groups = np.asarray(groups)
    cv = GroupKFold(n_splits=n_splits, shuffle=shuffle,
                    random_state=random_state)
    return list(cv.split(np.zeros(len(groups)), groups=groups))


def oof_predict(estimator, X, y, folds, sample_weight=None):
    """Out-of-fold predictions; ``estimator`` is cloned for every fold."""
    X = np.asarray(X)
    y = np.asarray(y)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)

    oof = np.empty(len(y), dtype=float)
    for train_idx, valid_idx in folds:
        model = clone(estimator)
        if sample_weight is None:
            model.fit(X[train_idx], y[train_idx])
        else:
            model.fit(X[train_idx], y[train_idx],
                      sample_weight=sample_weight[train_idx])
        oof[valid_idx] = model.predict(X[valid_idx])
    return oof


def fit_full(estimator, X, y, sample_weight=None):
    """Clone and fit ``estimator`` on the full training data."""
    model = clone(estimator)
    if sample_weight is None:
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=np.asarray(sample_weight, dtype=float))
    return model


def evaluate_oof(y, oof, groups, folds, bins=PCI_BINS):
    """Summarise out-of-fold predictions.

    Returns pooled MAE/RMSE/R2, per-fold MAE, a per-PCI-bin table, a
    per-road table, and the PCI<=40 / PCI>85 slices. Bins include the
    lower edge so PCI == 0 is counted.
    """
    y = np.asarray(y)
    oof = np.asarray(oof)
    groups = np.asarray(groups)

    frame = pd.DataFrame({
        "road": groups,
        "pci": y,
        "pred": oof,
        "err": np.abs(y - oof),
        "signed": oof - y,
    })
    frame["fold"] = 0
    for number, (_, valid_idx) in enumerate(folds, start=1):
        frame.loc[valid_idx, "fold"] = number
    frame["bin"] = pd.cut(frame["pci"], bins, include_lowest=True)

    per_bin = frame.groupby("bin", observed=True).agg(
        n=("err", "size"),
        mae=("err", "mean"),
        bias=("signed", "mean"),
    )
    per_road = frame.groupby("road").agg(
        n=("err", "size"),
        pci_mean=("pci", "mean"),
        mae=("err", "mean"),
        bias=("signed", "mean"),
        fold=("fold", "first"),
    ).sort_values("mae", ascending=False)

    per_fold = [float(np.mean(frame.loc[valid_idx, "err"]))
                for _, valid_idx in folds]
    low = frame["pci"] <= 40
    high = frame["pci"] > 85

    return {
        "pooled": {
            "mae": float(mean_absolute_error(y, oof)),
            "rmse": float(np.sqrt(mean_squared_error(y, oof))),
            "r2": float(r2_score(y, oof)),
        },
        "per_fold": per_fold,
        "per_bin": per_bin,
        "per_road": per_road,
        "low": {
            "n": int(low.sum()),
            "mae": float(frame.loc[low, "err"].mean()),
            "bias": float(frame.loc[low, "signed"].mean()),
        },
        "high": {
            "n": int(high.sum()),
            "mae": float(frame.loc[high, "err"].mean()),
        },
        "frame": frame,
    }


def condition_weights(y, alpha):
    """Weight poor segments more: ``w = 1 + alpha * (100 - PCI) / 100``."""
    y = np.asarray(y, dtype=float)
    return 1.0 + float(alpha) * (100.0 - y) / 100.0


def _fit_calibrator(predictions, y, method):
    predictions = np.asarray(predictions, dtype=float)
    y = np.asarray(y, dtype=float)
    if method == "linear":
        return LinearRegression().fit(predictions[:, None], y)
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(predictions, y)
    raise ValueError(f"unknown calibration method: {method!r}")


def _apply_calibrator(calibrator, predictions, method):
    predictions = np.asarray(predictions, dtype=float)
    if method == "linear":
        return calibrator.predict(predictions[:, None])
    if method == "isotonic":
        return calibrator.predict(predictions)
    raise ValueError(f"unknown calibration method: {method!r}")


def calibrate_oof(oof, y, folds, method):
    """Leave-one-fold-out calibration learned from OOF predictions only."""
    oof = np.asarray(oof, dtype=float)
    y = np.asarray(y, dtype=float)
    calibrated = np.empty(len(y), dtype=float)

    for _, valid_idx in folds:
        mask = np.ones(len(y), dtype=bool)
        mask[valid_idx] = False
        calibrator = _fit_calibrator(oof[mask], y[mask], method)
        calibrated[valid_idx] = _apply_calibrator(
            calibrator, oof[valid_idx], method
        )
    return calibrated


def fit_final_calibrator(oof, y, method):
    """Fit the calibrator on all OOF predictions (for the final model)."""
    return _fit_calibrator(oof, y, method)


def apply_calibrator(calibrator, predictions, method):
    """Apply a calibrator returned by :func:`fit_final_calibrator`."""
    return _apply_calibrator(calibrator, predictions, method)


def evaluate_variant(estimator, X_train, y_train, groups, folds,
                     X_test=None, y_test=None, sample_weight=None,
                     calibrate=None):
    """Evaluate one estimator configuration with CV and, optionally, test.

    ``estimator`` may be any scikit-learn compatible regressor (RF,
    XGBoost, ...). When ``X_test``/``y_test`` are provided, the model is
    refit on all training rows and scored on the test set. A calibrator is
    always learned from out-of-fold predictions only.
    """
    raw_oof = oof_predict(estimator, X_train, y_train, folds, sample_weight)
    oof = raw_oof
    if calibrate is not None:
        oof = calibrate_oof(raw_oof, y_train, folds, calibrate)

    result = evaluate_oof(y_train, oof, groups, folds)
    result["oof"] = oof

    if X_test is not None and y_test is not None:
        model = fit_full(estimator, X_train, y_train, sample_weight)
        predictions = model.predict(X_test)
        if calibrate is not None:
            calibrator = fit_final_calibrator(raw_oof, y_train, calibrate)
            predictions = apply_calibrator(calibrator, predictions, calibrate)
        result["test"] = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "r2": float(r2_score(y_test, predictions)),
        }
    else:
        result["test"] = {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    return result


def shuffled_cv_mae(estimator, X, y, groups, n_splits=5, seeds=range(5),
                    sample_weight=None, calibrate=None):
    """Pooled MAE over several shuffled group-fold assignments."""
    values = []
    for seed in seeds:
        folds = make_folds(groups, n_splits=n_splits, shuffle=True,
                           random_state=seed)
        oof = oof_predict(estimator, X, y, folds, sample_weight)
        if calibrate is not None:
            oof = calibrate_oof(oof, y, folds, calibrate)
        values.append(float(mean_absolute_error(y, oof)))
    return np.asarray(values)
