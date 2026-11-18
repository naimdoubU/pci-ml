import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

def make_folds(groups, n_splits):
    kfold = GroupKFold(n_splits=n_splits)
    return list(kfold.split(np.zeros(len(groups)), groups=groups))

def score_prediction(actual, predicted):
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "r2": r2_score(actual, predicted),
    }

def mean_fold_mae(actual, predicted, folds):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return float(np.mean([
        score_prediction(actual[valid_idx], predicted[valid_idx])["mae"]
        for _, valid_idx in folds
    ]))


# Galat baku rata-rata MAE per fold
def fold_mae_se(cv_folds):
    maes = np.array([row["mae"] for row in cv_folds], dtype=float)
    return float(np.std(maes, ddof=1) / np.sqrt(len(maes)))


# Peringkat model: CV MAE terendah, dengan 1-SE rule dan CV RMSE sebagai tie-break
def rank_models(scores):
    best = min(scores, key=lambda name: scores[name]["cv_score"]["mae"])
    se = fold_mae_se(scores[best]["cv_folds"])
    band = scores[best]["cv_score"]["mae"] + se

    tied = [name for name in scores if scores[name]["cv_score"]["mae"] <= band]
    winner = min(tied, key=lambda name: scores[name]["cv_score"]["rmse"])

    rest = sorted(
        (name for name in scores if name != winner),
        key=lambda name: (scores[name]["cv_score"]["mae"], scores[name]["cv_score"]["rmse"]),
    )
    return [(name, scores[name]) for name in (winner, *rest)], se, band, len(tied) > 1