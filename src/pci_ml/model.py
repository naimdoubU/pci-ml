import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

def make_folds(groups, n_splits):
    kfold = GroupKFold(n_splits=n_splits)
    return  list(kfold.split(np.zeros(len(groups)), groups=groups))

def score_prediction(actual, predicted):
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "r2": r2_score(actual,predicted),
    }

def mean_fold_mae(actual, predicted, folds):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return float(np.mean([
        score_prediction(actual[valid_idx], predicted[valid_idx])["mae"]
        for _, valid_idx in folds
    ]))

