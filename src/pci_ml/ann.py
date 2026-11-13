import copy
import gc
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pci_ml.model import mean_fold_mae
from pci_ml.data import DataSplit
from pci_ml.ui import evaluation_table, param_table, show_report

tag = DataSplit.make_tag

ROOT_DIR = Path(__file__).resolve().parents[2]
models_dir = ROOT_DIR / "models"

# Model ANN hanya dilatih dan dijalankan di CPU
DEVICE = torch.device("cpu")

# Batas bawah variansi fitur agar normalisasi tidak meledak pada fitur langka
MIN_VARIANCE = 1e-2

# Bagi target PCI ke rata-rata dan standar deviasi data latih
# agar skala output jaringan tetap stabil; prediksi dikembalikan ke skala 0-100.
def _target_stats(target):
    target = np.asarray(target, dtype="float32")
    mean = float(target.mean())
    scale = float(target.std())
    return mean, scale if scale > 1e-6 else 1.0

# ANN: standardisasi fitur dan target ikut tersimpan bersama bobot model
class ANNRegressor(nn.Module):
    def __init__(self, params, mean, variance, target_mean, target_scale):
        super().__init__()

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(np.sqrt(variance), dtype=torch.float32))
        self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=torch.float32))
        self.register_buffer("target_scale", torch.as_tensor(target_scale, dtype=torch.float32))

        # Simpan arsitektur agar model bisa dimuat tanpa mengulang hyperparameter
        self.model_params = {
            "hidden_layers": list(params["hidden_layers"]),
            "activation": params["activation"],
            "dropout": float(params["dropout"]),
        }

        activation = {"relu": nn.ReLU, "tanh": nn.Tanh}[params["activation"]]
        layers = []
        n_input = len(mean)
        for units in params["hidden_layers"]:
            layers.append(nn.Linear(n_input, units))
            layers.append(activation())
            if params["dropout"] > 0:
                layers.append(nn.Dropout(params["dropout"]))
            n_input = units
        layers.append(nn.Linear(n_input, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network((x - self.mean) / self.scale).squeeze(-1)

# Bangun model ANN dari statistik data latih
def build_ann_model(params, features, target, seed):
    features = np.asarray(features, dtype="float32")
    mean = features.mean(axis=0)
    variance = np.maximum(features.var(axis=0), MIN_VARIANCE)
    target_mean, target_scale = _target_stats(target)

    torch.manual_seed(seed)
    return ANNRegressor(params, mean, variance, target_mean, target_scale).to(DEVICE)

# Satu epoch training: data diacak per batch
def _train_epoch(model, optimizer, x, y, batch_size, generator):
    model.train()
    order = torch.randperm(len(x), generator=generator)

    total_loss = 0.0
    for start in range(0, len(x), batch_size):
        idx = order[start:start + batch_size]
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x[idx]), y[idx])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(idx)

    return total_loss / len(x)

@torch.no_grad()
def _valid_loss(model, x, y):
    model.eval()
    return nn.functional.mse_loss(model(x), y).item()

# Latih model dengan early stopping, lalu kembalikan bobot terbaik
def fit_ann(model, params, features, target, seed, epochs, patience,
            valid_features=None, valid_target=None):
    features = np.asarray(features, dtype="float32")
    target = np.asarray(target, dtype="float32")

    x = torch.as_tensor(features, dtype=torch.float32, device=DEVICE)
    y = (torch.as_tensor(target, dtype=torch.float32, device=DEVICE) - model.target_mean) / model.target_scale
    generator = torch.Generator().manual_seed(seed)

    # Data validasi terpisah untuk early stopping; jika tidak ada, pakai data latih
    if valid_features is None:
        x_valid, y_valid = x, y
    else:
        x_valid = torch.as_tensor(np.asarray(valid_features, dtype="float32"),
                                  dtype=torch.float32, device=DEVICE)
        y_valid = (torch.as_tensor(np.asarray(valid_target, dtype="float32"),
                                   dtype=torch.float32, device=DEVICE) - model.target_mean) / model.target_scale

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    best_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0
    epochs_run = 0
    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        _train_epoch(model, optimizer, x, y, params["batch_size"], generator)
        valid_loss = _valid_loss(model, x_valid, y_valid)

        if valid_loss < best_loss - 1e-6:
            best_loss, best_state, best_epoch, wait = valid_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.epochs_run = epochs_run
    model.best_epoch = best_epoch
    return model

# Hyperparameter tuning ann
def best_ann_candidate(candidates, features, target, seed, cv, epochs=200, patience=15):
    features = np.asarray(features)
    target = np.asarray(target)
    best_mae, best_params, best_oof = float("inf"), None, None

    for params in candidates:
        oof = np.zeros(len(target), dtype="float32")

        for train_idx, valid_idx in cv:
            model = train_ann(
                params,
                features[train_idx],
                target[train_idx],
                seed,
                epochs=epochs,
                patience=patience,
            )
            oof[valid_idx] = predict_ann(model, features[valid_idx])

        mae = mean_fold_mae(target, oof, cv)
        if mae < best_mae:
            best_mae, best_params, best_oof = mae, params, oof

        del model
        gc.collect()

    return best_mae, best_params, best_oof

def train_ann(params, features, target, seed, epochs=200, patience=15, validation_split=0.1):
    features = np.asarray(features, dtype="float32")
    target = np.asarray(target, dtype="float32")

    # Acak indeks agar validation set tidak bias oleh urutan data/road,
    # lalu sisihkan sebagian data khusus untuk early stopping
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    n_valid = max(1, int(round(len(features) * validation_split)))
    valid_idx, train_idx = order[:n_valid], order[n_valid:]

    model = build_ann_model(params, features[train_idx], target[train_idx], seed)
    model = fit_ann(
        model,
        params,
        features[train_idx],
        target[train_idx],
        seed,
        epochs,
        patience,
        valid_features=features[valid_idx],
        valid_target=target[valid_idx],
    )

    # Simpan konfigurasi training agar model bisa direproduksi dari file saja
    model.training_params = {
        "learning_rate": float(params["learning_rate"]),
        "weight_decay": float(params["weight_decay"]),
        "batch_size": int(params["batch_size"]),
        "max_epochs": int(epochs),
        "patience": int(patience),
        "validation_split": float(validation_split),
        "epochs_run": int(getattr(model, "epochs_run", 0)),
        "best_epoch": int(getattr(model, "best_epoch", 0)),
    }
    model.seed = int(seed)
    return model

# Save dan load model ann
def ann_path(filename, params):
    stem = Path(filename).stem
    return models_dir / f"{stem}_{tag(params)}" / f"{stem}_ann.pt"

def save_ann(model, filename, params, features=None):
    path = ann_path(filename, params)
    path.parent.mkdir(parents=True, exist_ok=True)

    feature_names = list(features) if features is not None else []
    if feature_names and len(feature_names) != len(model.mean):
        raise ValueError(
            f"Jumlah fitur ({len(feature_names)}) tidak cocok dengan model ({len(model.mean)})."
        )

    checkpoint = {
        "format_version": 1,
        "model_params": model.model_params,
        "training_params": getattr(model, "training_params", {}),
        "feature_names": feature_names,
        "seed": getattr(model, "seed", None),
        "split_tag": tag(params),
        "torch_version": str(torch.__version__),
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path.relative_to(ROOT_DIR).as_posix()}")
    return path

def load_ann(filename, params):
    checkpoint = torch.load(ann_path(filename, params), map_location=DEVICE, weights_only=True)
    state_dict = checkpoint["state_dict"]

    model = ANNRegressor(
        checkpoint["model_params"],
        mean=state_dict["mean"].numpy(),
        variance=state_dict["scale"].numpy() ** 2,
        target_mean=state_dict["target_mean"].numpy(),
        target_scale=state_dict["target_scale"].numpy(),
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # Simpan metadata agar bisa diperiksa setelah model dimuat
    for key in ("format_version", "training_params", "feature_names",
                "seed", "split_tag", "torch_version"):
        if key in checkpoint:
            setattr(model, key, checkpoint[key])
    return model

@torch.no_grad()
def predict_ann(model, features):
    model.eval()
    x = torch.as_tensor(np.asarray(features, dtype="float32"), dtype=torch.float32, device=DEVICE)
    predicted = model(x) * model.target_scale + model.target_mean
    return np.clip(predicted.cpu().numpy(), 0.0, 100.0)

# ANN hyperparameter table
ANN_PARAM_EXPLANATIONS = {
    "hidden_layers": "Jumlah dan ukuran neuron di tiap hidden layer.",
    "activation": "Fungsi aktivasi hidden layer.",
    "dropout": "Fraksi neuron yang dinonaktifkan saat training.",
    "learning_rate": "Laju pembelajaran optimizer Adam.",
    "weight_decay": "Regularisasi L2 pada bobot jaringan.",
    "batch_size": "Jumlah sampel per batch training.",
}


def ann_param_table(best_params, search_space, explanations=ANN_PARAM_EXPLANATIONS):
    return param_table(best_params, search_space, explanations)


def ann_evaluation_table(cv_score, test_score):
    return evaluation_table(cv_score, test_score)


def show_ann_report(best_params, search_space, cv_score, test_score):
    show_report(best_params, search_space, cv_score, test_score, ANN_PARAM_EXPLANATIONS)
