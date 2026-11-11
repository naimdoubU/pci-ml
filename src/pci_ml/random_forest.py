import joblib
from pathlib import Path
from IPython.display import HTML, Markdown, display
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

from pci_ml.model import mean_fold_mae
from pci_ml.data import DataSplit
from pci_ml.ui import build_hyperparameter_table, format_table

tag = DataSplit.make_tag

ROOT_DIR = Path(__file__).resolve().parents[2]
models_dir = ROOT_DIR / "models"

# Hyperparameter tuning rf
def best_rf_candidate(candidates, features, target, seed, cv):
    best_mae, best_params, best_oof = float("inf"), None, None

    for params in candidates:
        forest = RandomForestRegressor(**params, random_state=seed)
        oof = cross_val_predict(forest, features, target, cv=cv, n_jobs=-1)
        mae = mean_fold_mae(target, oof, cv)

        if mae < best_mae:
            best_mae, best_params, best_oof = mae, params, oof

    return best_mae, best_params, best_oof

# Save dan load model rf
def rf_path(filename, params):
    stem = Path(filename).stem
    return models_dir / f"{stem}_{tag(params)}" / f"{stem}_rf.joblib"

def save_rf(model, filename, params):
    path = rf_path(filename, params)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path.relative_to(ROOT_DIR).as_posix()}")
    return path

def load_rf(filename, params):
    return joblib.load(rf_path(filename, params))

# RF hyperparameter table
RF_PARAM_EXPLANATIONS = {
    "n_estimators": "Jumlah pohon keputusan.",
    "max_depth": "Kedalaman maksimum tiap pohon.",
    "min_samples_split": "Jumlah sampel minimum agar sebuah node boleh dipecah.",
    "min_samples_leaf": "Jumlah sampel minimum di tiap daun.",
    "max_features": "Fraksi fitur yang dipertimbangkan di tiap pohon.",
    "max_samples": "Fraksi data latih yang dipakai untuk tiap pohon.",
}

_METRICS = ("mae", "rmse", "r2")


def _score_row(label, score):
    return [label, *(f"{score[m]:.4f}" for m in _METRICS)]


def rf_param_table(best_params, search_space, explanations=RF_PARAM_EXPLANATIONS):
    return build_hyperparameter_table(best_params, search_space.keys(), explanations)


def rf_evaluation_table(cv_score, test_score):
    return format_table(
        ["Evaluasi", "MAE", "RMSE", "R²"],
        [_score_row("Cross-Validation", cv_score), _score_row("Test", test_score)],
        aligns=["left", "center", "center", "center"],
    )


def show_rf_report(best_params, search_space, cv_score, test_score):
    display(Markdown("##### **Hyperparameter terbaik**"))
    display(HTML(rf_param_table(best_params, search_space)))
    display(Markdown("##### **Hasil evaluasi**"))
    display(HTML(rf_evaluation_table(cv_score, test_score)))