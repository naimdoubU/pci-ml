# Uji sensitivitas model terhadap variasi split data.
# Seed split valid dicari dalam rentang 0..n_seed-1, lalu sejumlah seed dipilih
# secara acak (di luar seed split yang sedang dipakai) dan tiap split diuji
# dengan melatih ulang model pada hyperparameter hasil tuning notebook.
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from xgboost import XGBRegressor

from pci_ml.ann import predict_ann, train_ann
from pci_ml.data import (
    DataSplit,
    ModelInput,
    PCI,
    ROAD_ID,
    clean_data,
    engineer_features,
    load_data,
    to_density,
)
from pci_ml.evaluation import eval_path
from pci_ml.model import make_folds, score_prediction
from pci_ml.ui import evaluation_table, format_table

ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "outputs" / "sensitivity"

DATASET = "pci-ml-26.xlsx"
SPLIT_PARAMS = {
    "n_seed": 200,
    "test_size": 0.2,
    "share_tol": 0.01,
    "pci_min": 0,
}

CV_SPLITS = 5
SEED = 42
ANN_EPOCHS = 200
ANN_PATIENCE = 15

MODEL_NAMES = ("rf", "xgb", "ann")
METRICS = ("test_mae", "test_rmse", "test_r2", "cv_mae", "cv_rmse", "cv_r2")

# Hyperparameter hasil tuning pada notebook
TUNED_PARAMS = {
    "rf": {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 4,
        "min_samples_leaf": 1,
        "max_features": 0.3,
        "max_samples": 0.9,
    },
    "xgb": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.4,
        "min_child_weight": 1,
        "gamma": 0.3,
        "reg_lambda": 1,
    },
    "ann": {
        "hidden_layers": (128, 64, 32),
        "activation": "tanh",
        "dropout": 0.1,
        "learning_rate": 0.003,
        "weight_decay": 0.0001,
        "batch_size": 32,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Uji sensitivitas model terhadap variasi split data."
    )
    parser.add_argument(
        "--model", default="all",
        help="Model yang diuji: rf, xgb, ann, all, atau daftar dipisah koma (rf,xgb).",
    )
    parser.add_argument(
        "--n-split-seeds", type=int, default=100,
        help="Jumlah split seed yang diuji, tersebar dari gap terbesar ke terkecil.",
    )
    parser.add_argument(
        "--n-seed", type=int, default=20000,
        help="Rentang seed yang dipindai (0..n_seed-1) untuk mencari split valid.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Model seed.",
    )
    parser.add_argument(
        "--cv-splits", type=int, default=CV_SPLITS,
        help="Jumlah fold GroupKFold.",
    )
    parser.add_argument(
        "--dataset", default=DATASET,
        help="Nama file dataset pada folder datasets.",
    )
    parser.add_argument(
        "--split-params", default=None,
        help="Override parameter split dalam format JSON.",
    )
    parser.add_argument(
        "--params", default=None,
        help="File JSON berisi override hyperparameter per model.",
    )
    parser.add_argument(
        "--out", default=str(OUT_DIR),
        help="Folder output untuk CSV, JSON, dan HTML.",
    )
    return parser.parse_args()


def parse_model_names(value):
    if value == "all":
        return list(MODEL_NAMES)

    names = [name.strip().lower() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in MODEL_NAMES]
    if unknown:
        raise SystemExit(f"Model tidak dikenal: {', '.join(unknown)}")
    if not names:
        raise SystemExit("Pilih minimal satu model.")
    return names


def load_params(path):
    if not path:
        return {}

    with open(path, encoding="utf-8") as file:
        return json.load(file)


# Ambil hyperparameter terbaik dari artefak evaluasi notebook. Jika artefak
# belum ada, pakai TUNED_PARAMS bawaan sebagai cadangan.
def load_tuned_params(name, dataset, split_params):
    path = eval_path(name, dataset, split_params)
    if path.exists():
        with open(path, encoding="utf-8") as file:
            return json.load(file)["best_params"]

    print(
        f"PERINGATAN: {path.relative_to(ROOT_DIR).as_posix()} tidak ditemukan; "
        "memakai TUNED_PARAMS bawaan."
    )
    return dict(TUNED_PARAMS[name])


def resolve_params(name, overrides, dataset, split_params):
    params = load_tuned_params(name, dataset, split_params)
    params.update(overrides.get(name, {}))

    if name == "ann":
        params["hidden_layers"] = tuple(params["hidden_layers"])
    return params


def load_split(dataset, split_params):
    model_input = ModelInput(dataset)
    tag = model_input.find(split_params)
    train_df, test_df, features, meta = model_input.load(tag)
    return train_df, test_df, features, meta, tag


def rebuild_data(dataset):
    raw = load_data(dataset)
    df, features = clean_data(raw)
    df, features = to_density(df, features)
    df, features = engineer_features(df, features)
    return df, features


def make_estimator(name, params, seed, n_jobs):
    if name == "rf":
        return RandomForestRegressor(**params, random_state=seed, n_jobs=n_jobs)
    if name == "xgb":
        return XGBRegressor(**params, random_state=seed, n_jobs=n_jobs, verbosity=0)
    raise ValueError(f"Model tidak dikenal: {name}")


def train_model(name, params, features, target, seed):
    if name == "ann":
        return train_ann(
            params, features, target, seed,
            epochs=ANN_EPOCHS, patience=ANN_PATIENCE,
        )

    model = make_estimator(name, params, seed, n_jobs=-1)
    model.fit(features, target)
    return model


def predict_model(name, model, features):
    if name == "ann":
        return predict_ann(model, features)
    return model.predict(features)


def cross_val_oof(name, params, features, target, folds, seed):
    if name == "ann":
        oof = np.zeros(len(target), dtype="float32")
        for train_idx, valid_idx in folds:
            model = train_model(
                name, params, features[train_idx], target[train_idx], seed
            )
            oof[valid_idx] = predict_model(name, model, features[valid_idx])
        return oof

    estimator = make_estimator(name, params, seed, n_jobs=1)
    return cross_val_predict(estimator, features, target, cv=folds, n_jobs=-1)


def safe_score(actual, predicted):
    score = score_prediction(actual, predicted)
    return {key: float(value) for key, value in score.items()}


def evaluate_split(name, params, train_df, test_df, features, seed, cv_splits):
    train_features = train_df[features].to_numpy(dtype="float32")
    train_pci = train_df[PCI].to_numpy(dtype="float32")
    test_features = test_df[features].to_numpy(dtype="float32")
    test_pci = test_df[PCI].to_numpy(dtype="float32")
    folds = make_folds(train_df[ROAD_ID].to_numpy(), cv_splits)

    started = time.perf_counter()
    model = train_model(name, params, train_features, train_pci, seed)
    elapsed = time.perf_counter() - started

    test_score = safe_score(test_pci, predict_model(name, model, test_features))
    oof = cross_val_oof(name, params, train_features, train_pci, folds, seed)
    cv_score = safe_score(train_pci, oof)

    return {
        "cv_mae": cv_score["mae"],
        "cv_rmse": cv_score["rmse"],
        "cv_r2": cv_score["r2"],
        "test_mae": test_score["mae"],
        "test_rmse": test_score["rmse"],
        "test_r2": test_score["r2"],
        "train_seconds": float(elapsed),
        "best_epoch": getattr(model, "best_epoch", None),
        "epochs_run": getattr(model, "epochs_run", None),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "roads_train": int(train_df[ROAD_ID].nunique()),
        "roads_test": int(test_df[ROAD_ID].nunique()),
    }


def valid_split_seeds(splitter):
    rows = []
    for seed in range(splitter.n_seed):
        result = splitter._try_seed(seed)
        if result is not None:
            rows.append((seed, float(result[0])))
    return rows


# Pilih split seed dari daftar seed valid, tanpa menyertakan seed split yang
# sedang dipakai. Seed diurutkan berdasarkan gap (besar ke kecil), lalu diambil
# merata sepanjang urutan tersebut agar variasi gap terwakili.
def pick_split_seeds(valid, count, exclude_seed):
    if count < 1:
        raise SystemExit("--n-split-seeds harus minimal 1.")

    candidates = [row for row in valid if row[0] != exclude_seed]
    if count > len(candidates):
        raise SystemExit(
            f"Hanya {len(candidates)} split seed valid di luar seed {exclude_seed}; "
            f"tidak cukup untuk {count} seed uji. Perbesar --n-seed."
        )

    candidates.sort(key=lambda row: row[1], reverse=True)
    index = np.linspace(0, len(candidates) - 1, count).round().astype(int)
    index = list(dict.fromkeys(index.tolist()))
    if len(index) < count:
        raise SystemExit(
            "Pemilihan seed tidak menghasilkan cukup indeks unik; "
            "kurangi --n-split-seeds atau perbesar --n-seed."
        )
    return [candidates[i] for i in index]


def split_seed_stability(name, params, splitter, features, split_seeds,
                         model_seed, cv_splits):
    rows = []
    for split_seed, gap in split_seeds:
        _, train_idx, test_idx = splitter._try_seed(split_seed)
        train_df = splitter.df.iloc[train_idx]
        test_df = splitter.df.iloc[test_idx]

        scores = evaluate_split(
            name, params, train_df, test_df, features, model_seed, cv_splits
        )
        rows.append({
            "model": name,
            "split_seed": int(split_seed),
            "split_gap": float(gap),
            "model_seed": int(model_seed),
            **scores,
        })
        print(
            f"  split {split_seed:<5} gap {gap:.4f} | CV MAE {scores['cv_mae']:.4f} | "
            f"Test MAE {scores['test_mae']:.4f} | R2 {scores['test_r2']:.4f} | "
            f"{scores['train_seconds']:.1f}s"
        )
    return rows


def summarize(rows):
    frame = pd.DataFrame(rows)
    summary = {}

    for metric in METRICS:
        values = frame[metric].dropna()
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "range": float(values.max() - values.min()),
        }
    return summary


def fmt(value, digits=4):
    if value is None:
        return "–"
    return f"{value:.{digits}f}"


def summary_table(summary):
    rows = []
    for metric in METRICS:
        stats = summary[metric]
        rows.append([
            metric.upper().replace("_", " "),
            fmt(stats["mean"]),
            fmt(stats["std"]),
            fmt(stats["min"]),
            fmt(stats["max"]),
            fmt(stats["range"]),
        ])
    return format_table(
        ["Metrik", "Mean", "Std", "Min", "Max", "Range"],
        rows,
        aligns=["left", "center", "center", "center", "center", "center"],
    )


def seed_table(rows, is_ann):
    headers = ["Split seed", "Gap", "CV MAE", "CV RMSE", "CV R2",
               "Test MAE", "Test RMSE", "Test R2", "Train (s)"]
    if is_ann:
        headers.extend(["Best epoch", "Epochs run"])

    body = []
    for row in rows:
        values = [row["split_seed"], fmt(row["split_gap"])]
        values.extend([
            fmt(row["cv_mae"]),
            fmt(row["cv_rmse"]),
            fmt(row["cv_r2"]),
            fmt(row["test_mae"]),
            fmt(row["test_rmse"]),
            fmt(row["test_r2"]),
            fmt(row["train_seconds"], 2),
        ])
        if is_ann:
            values.extend([row["best_epoch"] or "–", row["epochs_run"] or "–"])
        body.append(values)

    aligns = ["center"] * len(headers)
    aligns[0] = "left"
    return format_table(headers, body, aligns=aligns)


def baseline_table(scores):
    cv_score = {metric: scores[f"cv_{metric}"] for metric in ("mae", "rmse", "r2")}
    test_score = {metric: scores[f"test_{metric}"] for metric in ("mae", "rmse", "r2")}
    return evaluation_table(cv_score, test_score)


def params_table(params):
    rows = [[key, str(value)] for key, value in params.items()]
    return format_table(
        ["Hyperparameter", "Nilai"], rows, aligns=["left", "center"]
    )


def build_html(run, params_by_model, baseline, split_seed, summaries):
    parts = [
        "<!DOCTYPE html>",
        "<html lang='id'><head><meta charset='utf-8'>",
        "<title>Sensitivity / Stability Report</title>",
        "<style>",
        "body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; "
        "margin: 24px; color: #222; }",
        "h1 { font-size: 20px; }",
        "h2 { font-size: 16px; margin-top: 32px; border-bottom: 1px solid #ddd; "
        "padding-bottom: 4px; }",
        "h3 { font-size: 13px; margin-top: 20px; }",
        "p { font-size: 12px; color: #555; }",
        "</style></head><body>",
        "<h1>Sensitivity / Stability Report</h1>",
        "<p>Std dan range pada tabel ringkasan menunjukkan sensitivitas model "
        "terhadap variasi split data.</p>",
    ]

    meta_rows = [
        ["Dataset", run["dataset"]],
        ["Split tag", run["split_tag"]],
        ["Split seed saat ini", run["split_seed"]],
        ["Split gap saat ini", fmt(run["split_gap"])],
        ["Model seed", run["model_seed"]],
        ["CV splits", run["cv_splits"]],
        ["Seed split yang diuji", ", ".join(str(seed) for seed in run["tested_split_seeds"])],
        ["Model", ", ".join(run["models"])],
        ["Dibuat", run["created_at"]],
    ]
    parts.append(format_table(
        ["Parameter", "Nilai"], meta_rows, aligns=["left", "left"]
    ))

    for name in run["models"]:
        parts.append(f"<h2>{name.upper()}</h2>")

        parts.append("<h3>Hyperparameter</h3>")
        parts.append(params_table(params_by_model[name]))

        parts.append("<h3>Baseline (split saat ini)</h3>")
        parts.append(baseline_table(baseline[name]))

        parts.append("<h3>Stabilitas split seed</h3>")
        parts.append(summary_table(summaries[name]))
        parts.append(seed_table(split_seed[name], name == "ann"))

    parts.append("</body></html>")
    return "\n".join(parts)


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def write_summary(path, summaries):
    rows = []
    for name, summary in summaries.items():
        for metric in METRICS:
            stats = summary[metric]
            rows.append({
                "model": name,
                "metric": metric,
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
                "range": stats["range"],
            })
    write_csv(path, rows)


def write_baseline(path, baseline):
    rows = []
    for name, scores in baseline.items():
        rows.append({"model": name, **scores})
    write_csv(path, rows)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main():
    args = parse_args()
    models = parse_model_names(args.model)

    split_params = dict(SPLIT_PARAMS)
    if args.split_params:
        split_params.update(json.loads(args.split_params))

    overrides = load_params(args.params)
    params_by_model = {
        name: resolve_params(name, overrides, args.dataset, split_params)
        for name in models
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset : {args.dataset}")
    print(f"Split   : {split_params}")
    print(f"Model   : {', '.join(models)}")

    train_df, test_df, features, meta, tag = load_split(args.dataset, split_params)
    split_seed = int(meta["seed"])
    split_gap = float(meta["gap"])
    print(f"Checkpoint: {tag} | seed {split_seed} | gap {split_gap:.4f}")

    baseline = {}
    for name in models:
        print(f"\n[{name}] baseline ...")
        baseline[name] = evaluate_split(
            name, params_by_model[name], train_df, test_df, features,
            args.seed, args.cv_splits,
        )
        print(
            f"  CV MAE {baseline[name]['cv_mae']:.4f} | "
            f"Test MAE {baseline[name]['test_mae']:.4f} | "
            f"R2 {baseline[name]['test_r2']:.4f}"
        )

    df, rebuilt_features = rebuild_data(args.dataset)
    if list(rebuilt_features) != list(features):
        raise ValueError("Fitur hasil rebuild tidak cocok dengan checkpoint split.")

    splitter = DataSplit(df, rebuilt_features, args.dataset, split_params)
    splitter.n_seed = int(args.n_seed)

    valid = valid_split_seeds(splitter)
    print(
        f"\nSplit seed valid: {len(valid)} dari {splitter.n_seed} seed "
        f"(di luar seed saat ini: {len([row for row in valid if row[0] != split_seed])})"
    )

    split_seeds = pick_split_seeds(valid, args.n_split_seeds, split_seed)
    print(
        f"Split seed diuji (gap besar -> kecil): "
        f"{', '.join(str(seed) for seed, _ in split_seeds)}"
    )

    split_seed_rows = {}
    summaries = {}
    for name in models:
        print(f"\n[{name}] stabilitas split seed ...")
        rows = split_seed_stability(
            name, params_by_model[name], splitter, features,
            split_seeds, args.seed, args.cv_splits,
        )
        split_seed_rows[name] = rows
        summaries[name] = summarize(rows)
        write_csv(out_dir / f"split_seed_{name}.csv", rows)

    run = {
        "dataset": args.dataset,
        "split_tag": tag,
        "split_seed": split_seed,
        "split_gap": split_gap,
        "model_seed": int(args.seed),
        "cv_splits": int(args.cv_splits),
        "n_seed": int(args.n_seed),
        "tested_split_seeds": [int(seed) for seed, _ in split_seeds],
        "models": models,
        "created_at": datetime.now().strftime("%Y-%m-%d | %H:%M:%S"),
    }

    payload = {
        "run": run,
        "params": {name: params_by_model[name] for name in models},
        "baseline": baseline,
        "split_seed": {
            name: {
                "summary": summaries[name],
                "rows": split_seed_rows[name],
            }
            for name in models
        },
    }

    write_summary(out_dir / "summary.csv", summaries)
    write_baseline(out_dir / "baseline.csv", baseline)
    write_json(out_dir / "sensitivity.json", payload)
    (out_dir / "report.html").write_text(
        build_html(run, params_by_model, baseline, split_seed_rows, summaries),
        encoding="utf-8",
    )

    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
