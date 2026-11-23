import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", message="IProgress not found.*")
import shap
from IPython.display import HTML, Markdown, display

from pci_ml.ann import load_ann, predict_ann
from pci_ml.data import DataSplit, ENGINEERED_FEATURES, LEVEL_SUFFIXES
from pci_ml.evaluation import ROOT_DIR
from pci_ml.plots import figure_path
from pci_ml.random_forest import load_rf, predict_rf
from pci_ml.ui import format_table
from pci_ml.xgboost import load_xgb, predict_xgb

SUFFIXES = tuple(LEVEL_SUFFIXES.values())
MODELS = {
    "rf": ("Random Forest", load_rf, predict_rf),
    "xgb": ("XGBoost", load_xgb, predict_xgb),
    "ann": ("ANN", load_ann, predict_ann),
}
ALIASES = {"randomforest": "rf", "xgboost": "xgb"}
TYPE_EXPORT_COLUMNS = [
    "rank",
    "distress_type",
    "n_severity_columns",
    "n_present",
    "presence_rate",
    "mean_abs_shap",
    "share_of_total_contribution",
    "direction",
]
ENGINEERED_EXPORT_COLUMNS = [
    "rank",
    "feature",
    "mean_abs_shap",
    "share_of_all_features",
    "direction",
]


class _AnnPciModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x) * self.model.target_scale + self.model.target_mean
        return output.reshape(-1, 1)


def resolve_model(model_name):
    key = "".join(ch for ch in str(model_name).lower() if ch.isalnum())
    key = ALIASES.get(key, key)
    if key not in MODELS:
        raise ValueError(f"Model tidak dikenal: {model_name}")
    return key


def _distress_groups(feature_names):
    groups = {}
    for index, name in enumerate(feature_names):
        if name in ENGINEERED_FEATURES:
            continue
        for suffix in SUFFIXES:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        groups.setdefault(name, []).append(index)
    return groups


def _direction(signed_shap):
    if np.isnan(signed_shap):
        return "n/a"
    return "positif" if signed_shap > 0 else "negatif" if signed_shap < 0 else "netral"


def _normalise_values(values):
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError("SHAP multikeluaran tidak didukung untuk regresi PCI.")
        values = values[0]

    values = np.asarray(values, dtype="float64")
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"Bentuk nilai SHAP tidak didukung: {values.shape}")
    return values


def _make_explainer(model_key, model, baseline):
    if model_key in ("rf", "xgb"):
        explainer = shap.TreeExplainer(
            model, data=baseline, feature_perturbation="interventional"
        )
        return explainer, "TreeExplainer"

    wrapped = _AnnPciModel(model).eval()
    try:
        explainer = shap.DeepExplainer(
            wrapped, torch.as_tensor(baseline, dtype=torch.float32)
        )
        return explainer, "DeepExplainer"
    except Exception as exc:
        print(f"PERINGATAN: DeepExplainer gagal ({exc}); memakai KernelExplainer.")
        explainer = shap.KernelExplainer(
            lambda x: predict_ann(model, x), baseline
        )
        return explainer, "KernelExplainer"


def _shap_values(explainer, explainer_name, features):
    if explainer_name == "DeepExplainer":
        features = torch.as_tensor(features, dtype=torch.float32)
        return explainer.shap_values(features, check_additivity=False)
    if explainer_name == "KernelExplainer":
        return explainer.shap_values(features)
    return explainer.shap_values(features, check_additivity=False)


def _base_value(explainer):
    return float(np.asarray(explainer.expected_value).reshape(-1)[0])


def explain_model(best_model_name, dataset, params, test_features, feature_names):
    model_key = resolve_model(best_model_name)
    model_name, load_model, predict_model = MODELS[model_key]
    feature_names = list(feature_names)
    test_features = np.asarray(test_features, dtype="float32")
    baseline = np.zeros((1, len(feature_names)), dtype="float32")

    model = load_model(dataset, params)
    baseline_pci = float(np.asarray(predict_model(model, baseline)).reshape(-1)[0])
    explainer, explainer_name = _make_explainer(model_key, model, baseline)
    values = _normalise_values(_shap_values(explainer, explainer_name, test_features))
    base_value = _base_value(explainer)

    if values.shape != test_features.shape:
        raise ValueError(
            f"Bentuk nilai SHAP {values.shape} tidak cocok dengan data {test_features.shape}."
        )

    predicted = np.asarray(predict_model(model, test_features), dtype="float64").reshape(-1)
    return {
        "model": model_name,
        "model_key": model_key,
        "explainer": explainer_name,
        "baseline_pci": baseline_pci,
        "base_value": base_value,
        "feature_names": feature_names,
        "values": values,
        "predicted": predicted,
    }


def distress_type_table(values, features, feature_names):
    rows = []
    for name, indices in _distress_groups(feature_names).items():
        grouped = values[:, indices].sum(axis=1)
        present = features[:, indices].sum(axis=1) > 0
        mean_signed = float(grouped.mean())
        rows.append({
            "distress_type": name,
            "n_severity_columns": len(indices),
            "n_present": int(present.sum()),
            "presence_rate": float(present.mean()),
            "mean_abs_shap": float(np.abs(grouped).mean()),
            "mean_signed_shap": mean_signed,
            "mean_pci_reduction_present": (
                -float(grouped[present].mean()) if present.any() else np.nan
            ),
            "mean_pci_reduction_all": -mean_signed,
            "direction": "n/a" if not present.any() else _direction(mean_signed),
        })

    table = pd.DataFrame(rows)
    total_abs = float(table["mean_abs_shap"].sum())
    table["share_of_total_contribution"] = (
        table["mean_abs_shap"] / total_abs if total_abs else 0.0
    )

    reduction = table["mean_pci_reduction_present"].clip(lower=0).fillna(0)
    total_reduction = float(reduction.sum())
    table["share_of_total_reduction"] = reduction / total_reduction if total_reduction else 0.0

    table = table.sort_values(
        ["mean_abs_shap", "distress_type"], ascending=[False, True]
    ).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def engineered_table(values, feature_names):
    mean_abs_all = np.abs(values).mean(axis=0)
    mean_signed_all = values.mean(axis=0)
    total = float(mean_abs_all.sum())
    indices = [i for i, name in enumerate(feature_names) if name in ENGINEERED_FEATURES]

    table = pd.DataFrame({
        "feature": [feature_names[i] for i in indices],
        "mean_abs_shap": mean_abs_all[indices],
        "mean_signed_shap": mean_signed_all[indices],
    })
    table["share_of_all_features"] = table["mean_abs_shap"] / total if total else 0.0
    table["direction"] = table["mean_signed_shap"].map(_direction)

    table = table.sort_values(
        ["mean_abs_shap", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def build_shap_values_table(result, road_info=None, actual=None):
    values = result["values"]
    n_rows = len(values)
    rows = {}

    if road_info is not None:
        info = road_info.reset_index(drop=True)
        if len(info) != n_rows:
            raise ValueError("Jumlah road_info tidak cocok dengan jumlah baris test.")
        rows.update({column: info[column].to_numpy() for column in info.columns})

    rows["predicted"] = result["predicted"]
    if actual is not None:
        actual = np.asarray(actual, dtype="float64").reshape(-1)
        if len(actual) != n_rows:
            raise ValueError("Jumlah actual tidak cocok dengan jumlah baris test.")
        rows["actual"] = actual

    rows["base_value"] = np.full(n_rows, result["base_value"])
    rows.update({
        f"shap_{name}": values[:, index]
        for index, name in enumerate(result["feature_names"])
    })
    rows["sum_shap"] = values.sum(axis=1)
    rows["reconstructed_prediction"] = result["base_value"] + rows["sum_shap"]
    rows["reconstruction_error"] = result["predicted"] - rows["reconstructed_prediction"]
    return pd.DataFrame(rows)


def _display_table(title, headers, rows, aligns):
    display(HTML(
        f"<div style='font-weight: 600; margin: 4px 0;'>{title}</div>"
        + format_table(headers, rows, aligns=aligns)
    ))


def show_shap_summary(result):
    display(Markdown(
        f"**Model:** {result['model']} | "
        f"**Explainer:** {result['explainer']} | "
        f"**Baseline tanpa kerusakan:** {result['baseline_pci']:.2f} PCI | "
        f"**Sampel test:** {len(result['values'])} baris"
    ))


def show_shap_tables(type_ranking, engineered):
    rows = [
        [int(row.rank), row.distress_type, f"{row.mean_abs_shap:.4f}",
         f"{row.share_of_total_contribution:.2%}", row.direction]
        for row in type_ranking.itertuples()
    ]
    _display_table(
        "Kontribusi 19 jenis kerusakan",
        ["Peringkat", "Jenis Kerusakan", "Mean SHAP", "Kontribusi", "Arah"],
        rows,
        ["center", "left", "center", "center", "left"],
    )

    rows = [
        [int(row.rank), row.feature, f"{row.mean_abs_shap:.4f}",
         f"{row.share_of_all_features:.2%}", row.direction]
        for row in engineered.itertuples()
    ]
    _display_table(
        "Fitur agregat hasil rekayasa (dilaporkan terpisah)",
        ["Peringkat", "Fitur", "Mean SHAP", "Kontribusi", "Arah"],
        rows,
        ["center", "left", "center", "center", "left"],
    )


def _finish_figure(fig, save_path, show):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_distress_type_importance(table, model_name, baseline_pci, save_path=None, show=True):
    data = table.iloc[::-1]
    values = data["mean_abs_shap"]
    labels = [f"{int(rank)}. {name}" for rank, name in zip(data["rank"], data["distress_type"])]

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(labels, values, color="#3b7ea1")
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3)
    ax.set_xlim(0, values.max() * 1.18)
    ax.set(
        xlabel="Mean |SHAP| (PCI points)",
        ylabel="Distress type",
        title=(
            f"SHAP Kontribusi Jenis Kerusakan — {model_name}\n"
            f"Baseline tanpa kerusakan: {baseline_pci:.2f} PCI"
        ),
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _finish_figure(fig, save_path, show)
    return fig


def save_shap_outputs(result, type_ranking, values_table, dataset, params, engineered=None):
    stem = Path(dataset).stem
    shap_dir = ROOT_DIR / "outputs" / "shap" / DataSplit.make_tag(params)
    model_key = result["model_key"]
    paths = {
        "values": shap_dir / f"{stem}_{model_key}_shap_values.csv",
        "type_ranking": shap_dir / f"{stem}_{model_key}_shap_distress_type_ranking.csv",
        "meta": shap_dir / f"{stem}_{model_key}_shap_meta.json",
        "type_plot": figure_path(f"06_shap_distress_type_importance_{model_key}.png", params),
    }
    if engineered is not None:
        paths["engineered_ranking"] = (
            shap_dir / f"{stem}_{model_key}_shap_engineered_ranking.csv"
        )

    meta = {
        "format_version": 3,
        "model": result["model"],
        "model_key": result["model_key"],
        "split_tag": DataSplit.make_tag(params),
        "explainer": result["explainer"],
        "shap_version": shap.__version__,
        "baseline": "semua fitur distress = 0; TOTAL_DENSITY/TOTAL_L/TOTAL_M/TOTAL_H = 0",
        "baseline_pci": result["baseline_pci"],
        "base_value": result["base_value"],
        "ranking_basis": "mean_abs_shap",
        "direction_note": (
            "arah dihitung dari rata-rata SHAP bertanda; "
            "nilai bertanda tidak disimpan di artefak tabel"
        ),
        "presence_definition": "total nilai mentah distress type > 0",
        "n_explained": int(len(result["values"])),
        "n_features": int(len(result["feature_names"])),
        "n_distress_types": int(len(type_ranking)),
        "engineered_features": list(ENGINEERED_FEATURES),
        "engineered_feature_note": (
            "Fitur agregat tidak dimasukkan ke ranking distress karena "
            "merupakan penjumlahan fitur mentah; pengaruhnya dilaporkan terpisah."
        ),
        "top_distress_types": [
            {
                "rank": int(row.rank),
                "distress_type": row.distress_type,
                "mean_abs_shap": float(row.mean_abs_shap),
                "share_of_total_contribution": float(row.share_of_total_contribution),
                "direction": row.direction,
            }
            for row in type_ranking.head(10).itertuples()
        ],
    }
    if engineered is not None:
        meta["top_engineered_features"] = [
            {
                "rank": int(row.rank),
                "feature": row.feature,
                "mean_abs_shap": float(row.mean_abs_shap),
                "share_of_all_features": float(row.share_of_all_features),
                "direction": row.direction,
            }
            for row in engineered.itertuples()
        ]
    meta["paths"] = {
        name: str(path.relative_to(ROOT_DIR).as_posix())
        for name, path in paths.items()
    }
    meta["created_at"] = datetime.now().strftime("%Y-%m-%d | %H:%M:%S")

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    values_table.to_csv(paths["values"], index=False)
    type_ranking[TYPE_EXPORT_COLUMNS].to_csv(paths["type_ranking"], index=False)
    if engineered is not None:
        engineered[ENGINEERED_EXPORT_COLUMNS].to_csv(
            paths["engineered_ranking"], index=False
        )
    with open(paths["meta"], "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False, default=str)
    return paths
