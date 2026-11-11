from pathlib import Path
from sklearn.model_selection import cross_val_predict
from xgboost import XGBRegressor

from pci_ml.model import clip_pci, mean_fold_mae
from pci_ml.data import DataSplit
from pci_ml.ui import evaluation_table, param_table, show_report

tag = DataSplit.make_tag

ROOT_DIR = Path(__file__).resolve().parents[2]
models_dir = ROOT_DIR / "models"

# Hyperparameter tuning xgb
def best_xgb_candidate(candidates, features, target, seed, cv):
    best_mae, best_params, best_oof = float("inf"), None, None

    for params in candidates:
        model = XGBRegressor(**params, random_state=seed, verbosity=0)
        oof = clip_pci(cross_val_predict(model, features, target, cv=cv, n_jobs=-1))
        mae = mean_fold_mae(target, oof, cv)

        if mae < best_mae:
            best_mae, best_params, best_oof = mae, params, oof

    return best_mae, best_params, best_oof

# Save dan load model xgb
def xgb_path(filename, params):
    stem = Path(filename).stem
    return models_dir / f"{stem}_{tag(params)}" / f"{stem}_xgb.ubj"

def save_xgb(model, filename, params):
    path = xgb_path(filename, params)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
    print(f"Model saved to {path.relative_to(ROOT_DIR).as_posix()}")
    return path

def load_xgb(filename, params):
    model = XGBRegressor()
    model.load_model(xgb_path(filename, params))
    return model

def predict_xgb(model, features):
    return clip_pci(model.predict(features))

# XGB hyperparameter table
XGB_PARAM_EXPLANATIONS = {
    "n_estimators": "Jumlah pohon boosting.",
    "max_depth": "Kedalaman maksimum tiap pohon.",
    "learning_rate": "Laju pembelajaran tiap iterasi boosting.",
    "subsample": "Fraksi data latih yang dipakai untuk tiap pohon.",
    "colsample_bytree": "Fraksi fitur yang dipertimbangkan di tiap pohon.",
    "min_child_weight": "Bobot sampel minimum agar sebuah node boleh dipecah.",
    "gamma": "Penurunan loss minimum agar sebuah node boleh dipecah.",
    "reg_lambda": "Regularisasi L2 pada bobot daun.",
}


def xgb_param_table(best_params, search_space, explanations=XGB_PARAM_EXPLANATIONS):
    return param_table(best_params, search_space, explanations)


def xgb_evaluation_table(cv_score, test_score):
    return evaluation_table(cv_score, test_score)


def show_xgb_report(best_params, search_space, cv_score, test_score):
    show_report(best_params, search_space, cv_score, test_score, XGB_PARAM_EXPLANATIONS)
