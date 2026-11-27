import json
from datetime import datetime
from pathlib import Path

import numpy as np

from pci_ml.data import DataSplit
from pci_ml.model import score_prediction
from pci_ml.paths import ROOT_DIR, test_predictions_path

models_dir = ROOT_DIR / "models"

# Ekstensi file model asli tiap algoritma
MODEL_SUFFIX = {"rf": ".joblib", "xgb": ".ubj", "ann": ".pt"}


# Folder artefak model untuk satu kombinasi dataset dan parameter split
def model_dir(filename, params):
    stem = Path(filename).stem
    return models_dir / f"{stem}_{DataSplit.make_tag(params)}"


# Path file model asli
def model_path(model, filename, params):
    return model_dir(filename, params) / f"{Path(filename).stem}_{model}{MODEL_SUFFIX[model]}"


# Path artefak input evaluasi: parameter tuning dan prediksi out-of-fold
def eval_path(model, filename, params):
    return model_dir(filename, params) / f"{Path(filename).stem}_{model}_eval.json"


# Path tabel nilai prediksi out-of-fold khusus data CV
def cv_predictions_path(model, filename, params):
    return model_dir(filename, params) / f"{Path(filename).stem}_{model}_cv_predictions.csv"


# Hitung metrik CV: skor gabungan OOF, rata-rata per fold, dan metrik tiap fold
def cv_metrics(actual, predicted, folds):
    actual, predicted = np.asarray(actual), np.asarray(predicted)

    fold_rows = []
    for i, (_, valid_idx) in enumerate(folds, start=1):
        valid_idx = np.asarray(valid_idx)
        score = score_prediction(actual[valid_idx], predicted[valid_idx])
        fold_rows.append({
            "fold": i,
            "n": int(len(valid_idx)),
            **{key: float(value) for key, value in score.items()},
        })

    metric_names = ("mae", "rmse", "r2")
    return {
        "cv_score": {key: float(value) for key, value in score_prediction(actual, predicted).items()},
        "cv_mean_fold": {
            metric: float(np.mean([row[metric] for row in fold_rows]))
            for metric in metric_names
        },
        "cv_folds": fold_rows,
    }


# Simpan seluruh input sel load/prediksi yang berasal dari proses tuning.
# Dipanggil di akhir sel tuning, setelah model final disimpan.
def save_eval_inputs(model, filename, params, best_params, search_space, best_mae, oof):
    path = eval_path(model, filename, params)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": 1,
        "model": model,
        "split_tag": DataSplit.make_tag(params),
        "best_params": best_params,
        "search_space": search_space,
        "best_mae": float(best_mae),
        "best_mae_note": (
            "Rata-rata MAE per fold saat tuning; berbeda dari outputs.cv_score.mae "
            "yang dihitung dari prediksi out-of-fold gabungan."
        ),
        "oof": [float(v) for v in oof],
        "oof_note": (
            "Prediksi out-of-fold dari proses tuning, bukan prediksi model final "
            "yang tersimpan di file model."
        ),
        "tuning_saved_at": datetime.now().strftime("%Y-%m-%d | %H:%M:%S"),
        "outputs": None,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Evaluasi CV  : {path.relative_to(ROOT_DIR).as_posix()}")


# Simpan prediksi out-of-fold per baris untuk data CV saja
def save_cv_predictions(model, filename, params, oof, actual, road_info, folds):
    path = cv_predictions_path(model, filename, params)
    path.parent.mkdir(parents=True, exist_ok=True)

    fold_ids = np.empty(len(actual), dtype=int)
    for i, (_, valid_idx) in enumerate(folds, start=1):
        fold_ids[np.asarray(valid_idx)] = i

    table = road_info.copy()
    table.insert(0, "fold", fold_ids)
    table.insert(1, "row_index", range(len(table)))
    table["actual"] = actual
    table["predicted"] = oof
    table["residual"] = table["actual"] - table["predicted"]
    table.to_csv(path, index=False)

    print(f"Prediksi OOF : {path.relative_to(ROOT_DIR).as_posix()}")
    return path


# Tambahkan hasil evaluasi CV ke JSON evaluasi tanpa menyentuh data test.
# Blok outputs.cv_* terpisah dari outputs.test_* yang nanti diisi stage prediksi.
def save_cv_outputs(model, filename, params, metrics, cv_predictions, plot_path):
    path = eval_path(model, filename, params)
    if not path.exists():
        raise FileNotFoundError(
            f"Input evaluasi tidak ditemukan: {path.relative_to(ROOT_DIR).as_posix()}. "
            "Jalankan sel tuning model terlebih dahulu."
        )

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    payload["format_version"] = 2
    outputs = payload.get("outputs") or {}
    outputs.update({
        "cv_score": metrics["cv_score"],
        "cv_mean_fold": metrics["cv_mean_fold"],
        "cv_folds": metrics["cv_folds"],
        "cv_predictions_path": cv_predictions.relative_to(ROOT_DIR).as_posix(),
        "plot_path": plot_path.relative_to(ROOT_DIR).as_posix(),
        "cv_saved_at": datetime.now().strftime("%Y-%m-%d | %H:%M:%S"),
    })
    payload["outputs"] = outputs

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# Simpan prediksi test per model ke outputs/<dataset>_<tag>/predictions/
def save_test_predictions(model, filename, params, predictions, actual, road_info):
    path = test_predictions_path(model, filename, params)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = road_info.copy()
    table.insert(0, "row_index", range(len(table)))
    table["actual"] = actual
    table["predicted"] = predictions
    table["residual"] = table["actual"] - table["predicted"]
    table.to_csv(path, index=False)

    # print(f"Prediksi test {model.upper():4}: {path.relative_to(ROOT_DIR).as_posix()}")
    return path


# Muat input evaluasi. hidden_layers ANN kembali menjadi list setelah JSON,
# jadi dikembalikan ke tuple agar laporan hyperparameter sama seperti saat tuning.
def load_eval(model, filename, params):
    path = eval_path(model, filename, params)
    if not path.exists():
        raise FileNotFoundError(
            f"Input evaluasi tidak ditemukan: {path.relative_to(ROOT_DIR).as_posix()}. "
            "Jalankan sel tuning model terlebih dahulu."
        )

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    hidden_layers = payload.get("best_params", {}).get("hidden_layers")
    if isinstance(hidden_layers, list):
        payload["best_params"]["hidden_layers"] = tuple(hidden_layers)

    model_file = model_path(model, filename, params)
    if model_file.exists() and path.stat().st_mtime < model_file.stat().st_mtime:
        print(
            f"PERINGATAN: {path.name} lebih lama dari {model_file.name}; "
            "jalankan ulang sel tuning agar input evaluasi sesuai model."
        )

    return payload