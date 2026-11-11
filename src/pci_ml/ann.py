import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import gc
from pathlib import Path

import keras
import numpy as np

from pci_ml.model import clip_pci, mean_fold_mae
from pci_ml.data import DataSplit
from pci_ml.ui import evaluation_table, param_table, show_report

tag = DataSplit.make_tag

ROOT_DIR = Path(__file__).resolve().parents[2]
models_dir = ROOT_DIR / "models"

# Batas bawah variansi fitur agar normalisasi tidak meledak pada fitur langka
MIN_VARIANCE = 1e-2

# Bangun model ANN
def build_ann_model(params, features, seed):
    features = np.asarray(features)
    keras.utils.set_random_seed(seed)

    mean = features.mean(axis=0)
    variance = np.maximum(features.var(axis=0), MIN_VARIANCE)

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(features.shape[1],)))
    model.add(keras.layers.Normalization(axis=-1, mean=mean, variance=variance))

    for units in params["hidden_layers"]:
        model.add(keras.layers.Dense(units, activation=params["activation"]))
        if params["dropout"] > 0:
            model.add(keras.layers.Dropout(params["dropout"]))

    model.add(keras.layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=params["learning_rate"],
            clipnorm=1.0,
        ),
        loss="mse",
        metrics=["mae"],
    )
    return model

def _callbacks(patience):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
    ]

# Hyperparameter tuning ann
def best_ann_candidate(candidates, features, target, seed, cv, epochs=200, patience=15):
    features = np.asarray(features)
    target = np.asarray(target)
    best_mae, best_params, best_oof = float("inf"), None, None

    for params in candidates:
        oof = np.zeros(len(target), dtype="float32")

        for train_idx, valid_idx in cv:
            model = build_ann_model(params, features[train_idx], seed)
            model.fit(
                features[train_idx],
                target[train_idx],
                validation_data=(features[valid_idx], target[valid_idx]),
                epochs=epochs,
                batch_size=params["batch_size"],
                callbacks=_callbacks(patience),
                verbose=0,
            )
            oof[valid_idx] = clip_pci(model(features[valid_idx], training=False).numpy().ravel())

        mae = mean_fold_mae(target, oof, cv)
        if mae < best_mae:
            best_mae, best_params, best_oof = mae, params, oof

        keras.backend.clear_session()
        gc.collect()

    return best_mae, best_params, best_oof

def train_ann(params, features, target, seed, epochs=200, patience=15, validation_split=0.1):
    features = np.asarray(features)
    target = np.asarray(target)

    # Acak indeks agar validation set tidak bias oleh urutan data/road,
    # lalu sisihkan sebagian data khusus untuk early stopping
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    n_valid = max(1, int(round(len(features) * validation_split)))
    valid_idx, train_idx = order[:n_valid], order[n_valid:]

    model = build_ann_model(params, features[train_idx], seed)
    model.fit(
        features[train_idx],
        target[train_idx],
        validation_data=(features[valid_idx], target[valid_idx]),
        epochs=epochs,
        batch_size=params["batch_size"],
        callbacks=_callbacks(patience),
        verbose=0,
    )
    return model

# Save dan load model ann
def ann_path(filename, params):
    stem = Path(filename).stem
    return models_dir / f"{stem}_{tag(params)}" / f"{stem}_ann.keras"

def save_ann(model, filename, params):
    path = ann_path(filename, params)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    print(f"Model saved to {path.relative_to(ROOT_DIR).as_posix()}")
    return path

def load_ann(filename, params):
    return keras.models.load_model(ann_path(filename, params))

def predict_ann(model, features):
    return clip_pci(model(np.asarray(features, dtype="float32"), training=False).numpy().ravel())

# ANN hyperparameter table
ANN_PARAM_EXPLANATIONS = {
    "hidden_layers": "Jumlah dan ukuran neuron di tiap hidden layer.",
    "activation": "Fungsi aktivasi hidden layer.",
    "dropout": "Fraksi neuron yang dinonaktifkan saat training.",
    "learning_rate": "Laju pembelajaran optimizer Adam.",
    "batch_size": "Jumlah sampel per batch training.",
}


def ann_param_table(best_params, search_space, explanations=ANN_PARAM_EXPLANATIONS):
    return param_table(best_params, search_space, explanations)


def ann_evaluation_table(cv_score, test_score):
    return evaluation_table(cv_score, test_score)


def show_ann_report(best_params, search_space, cv_score, test_score):
    show_report(best_params, search_space, cv_score, test_score, ANN_PARAM_EXPLANATIONS)
