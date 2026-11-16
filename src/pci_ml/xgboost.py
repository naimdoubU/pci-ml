from pathlib import Path
from sklearn.model_selection import cross_val_predict
from xgboost import XGBRegressor

from pci_ml.model import mean_fold_mae
from pci_ml.data import DataSplit
from pci_ml.ui import evaluation_table, param_table, show_report

tag = DataSplit.make_tag

ROOT_DIR = Path(__file__).resolve().parents[2]
models_dir = ROOT_DIR / "models"

# Toleransi selisih CV MAE agar kandidat yang hampir seri tidak dipilih karena noise.
XGB_TIE_TOL = 0.01

# Semakin besar skor sampling, semakin banyak data/fitur yang dipakai tiap pohon.
def _xgb_sample_score(params):
    return params.get("subsample", 1.0) * params.get("colsample_bytree", 1.0)

# Hyperparameter tuning xgb
def best_xgb_candidate(candidates, features, target, seed, cv):
    results = []

    for params in candidates:
        model = XGBRegressor(**params, random_state=seed, verbosity=0)
        oof = cross_val_predict(model, features, target, cv=cv, n_jobs=-1)
        mae = mean_fold_mae(target, oof, cv)
        results.append((mae, params, oof))

    if not results:
        return float("inf"), None, None

    best_mae = min(row[0] for row in results)
    tied = [row for row in results if row[0] <= best_mae + XGB_TIE_TOL]
    tied.sort(key=lambda row: (-_xgb_sample_score(row[1]), row[0]))
    return tied[0]

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
