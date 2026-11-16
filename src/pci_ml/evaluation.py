import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from pci_ml.data import DataSplit

ROOT_DIR = Path(__file__).resolve().parents[2]
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


# Path tabel nilai prediksi per baris untuk data CV dan test
def predictions_path(model, filename, params):
    return model_dir(filename, params) / f"{Path(filename).stem}_{model}_predictions.csv"


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

    print(f"Input evaluasi disimpan ke {path.relative_to(ROOT_DIR).as_posix()}")
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


# Susun tabel nilai prediksi per baris: blok CV (out-of-fold) lalu blok test
def prediction_table(oof, test_predictions, train_pci, train_road_info,
                     test_pci, test_road_info):
    train = train_road_info.copy()
    train.insert(0, "split", "cv")
    train.insert(1, "row_index", range(len(train)))
    train["actual"] = train_pci
    train["predicted"] = oof
    train["residual"] = train["actual"] - train["predicted"]

    test = test_road_info.copy()
    test.insert(0, "split", "test")
    test.insert(1, "row_index", range(len(test)))
    test["actual"] = test_pci
    test["predicted"] = test_predictions
    test["residual"] = test["actual"] - test["predicted"]

    return pd.concat([train, test], ignore_index=True)


# Simpan output sel load/prediksi: metrik dan nilai prediksi test di JSON,
# serta tabel nilai prediksi per baris (CV dan test) di CSV.
def save_eval_outputs(model, filename, params, cv_score, test_score,
                      test_predictions, train_pci, train_road_info,
                      test_pci, test_road_info):
    path = eval_path(model, filename, params)
    if not path.exists():
        raise FileNotFoundError(
            f"Input evaluasi tidak ditemukan: {path.relative_to(ROOT_DIR).as_posix()}. "
            "Jalankan sel tuning model terlebih dahulu."
        )

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    payload["outputs"] = {
        "cv_score": {key: float(value) for key, value in cv_score.items()},
        "test_score": {key: float(value) for key, value in test_score.items()},
        "test_predictions": [float(v) for v in test_predictions],
        "saved_at": datetime.now().strftime("%Y-%m-%d | %H:%M:%S"),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_path = predictions_path(model, filename, params)
    prediction_table(
        payload["oof"], test_predictions, train_pci, train_road_info,
        test_pci, test_road_info,
    ).to_csv(csv_path, index=False)

    print(f"Output evaluasi disimpan ke {path.relative_to(ROOT_DIR).as_posix()}")
    print(f"Prediksi per baris disimpan ke {csv_path.relative_to(ROOT_DIR).as_posix()}")
    return path
