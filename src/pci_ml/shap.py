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
MODEL_REGISTRY = {
    "rf": ("Random Forest", load_rf, predict_rf),
    "xgb": ("XGBoost", load_xgb, predict_xgb),
    "ann": ("ANN", load_ann, predict_ann),
}
MODEL_ALIASES = {
    "rf": "rf", "randomforest": "rf",
    "xgb": "xgb", "xgboost": "xgb",
    "ann": "ann",
}


class _AnnPciModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x) * self.model.target_scale + self.model.target_mean
        return output.reshape(-1, 1)


def resolve_model(model_name):
    key = "".join(ch for ch in str(model_name).lower() if ch.isalnum())
    if key not in MODEL_ALIASES:
        raise ValueError(f"Model tidak dikenal: {model_name}")
    return MODEL_ALIASES[key]


def _distress_type(feature_name):
    if feature_name in ENGINEERED_FEATURES:
        return None
    for suffix in SUFFIXES:
        if feature_name.endswith(suffix):
            return feature_name[: -len(suffix)]
    return feature_name


def _severity(feature_name):
    for suffix in SUFFIXES:
        if feature_name.endswith(suffix):
            return suffix[1:]
    return "AGG"


def _direction(signed_shap):
    if np.isnan(signed_shap):
        return "n/a"
    if signed_shap > 0:
        return "positif"
    if signed_shap < 0:
        return "negatif"
    return "netral"


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


def _predict_ann_raw(model, features):
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(features, dtype="float32"), dtype=torch.float32)
        predicted = model(x) * model.target_scale + model.target_mean
    return predicted.cpu().numpy()


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
            lambda x: _predict_ann_raw(model, x), baseline
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


def distress_type_table(values, features, feature_names):
    groups = {}
    for index, name in enumerate(feature_names):
        if name not in ENGINEERED_FEATURES:
            groups.setdefault(_distress_type(name), []).append(index)

    rows = []
    for name, indices in groups.items():
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
            "direction": (
                "n/a" if not present.any() else _direction(mean_signed)
            ),
        })

    table = pd.DataFrame(rows)
    total_abs = float(table["mean_abs_shap"].sum())
    table["share_of_total_contribution"] = (
        table["mean_abs_shap"] / total_abs if total_abs else 0.0
    )

    total_reduction = float(
        table["mean_pci_reduction_present"].clip(lower=0).fillna(0).sum()
    )
    table["share_of_total_reduction"] = (
        table["mean_pci_reduction_present"].clip(lower=0).fillna(0) / total_reduction
        if total_reduction else 0.0
    )

    table = table.sort_values(
        ["mean_abs_shap", "distress_type"], ascending=[False, True]
    ).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def feature_table(values, feature_names):
    total = float(np.abs(values).mean(axis=0).sum())
    rows = []
    for index, name in enumerate(feature_names):
        mean_signed = float(values[:, index].mean())
        mean_abs = float(np.abs(values[:, index]).mean())
        rows.append({
            "feature": name,
            "feature_group": "engineered" if name in ENGINEERED_FEATURES else "distress",
            "distress_type": _distress_type(name) or "TOTAL",
            "severity": _severity(name),
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
            "mean_pci_reduction_all": -mean_signed,
            "share_of_all_features": mean_abs / total if total else 0.0,
            "direction": _direction(mean_signed),
        })

    table = pd.DataFrame(rows).sort_values(
        ["mean_abs_shap", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def engineered_table(table):
    data = table[table["feature_group"] == "engineered"].copy()
    return data.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def _display_distress_types(data):
    rows = [
        [int(row.rank), row.distress_type, f"{row.mean_abs_shap:.4f}",
         f"{row.share_of_total_contribution:.2%}", f"{row.mean_signed_shap:.4f}",
         row.direction]
        for row in data.itertuples()
    ]
    table = format_table(
        ["Peringkat", "Jenis Kerusakan", "Mean |SHAP|", "Kontribusi", "Mean SHAP", "Arah"],
        rows,
        aligns=["center", "left", "center", "center", "center", "left"],
    )
    display(HTML(
        "<div style='font-weight: 600; margin: 4px 0;'>"
        "Kontribusi 19 jenis kerusakan</div>" + table
    ))


def _display_engineered(data):
    rows = [
        [row.feature, f"{row.mean_abs_shap:.4f}", f"{row.mean_signed_shap:.4f}",
         f"{row.mean_pci_reduction_all:.4f}", f"{row.share_of_all_features:.2%}",
         row.direction]
        for row in data.itertuples()
    ]
    table = format_table(
        ["Fitur", "Mean |SHAP|", "Mean SHAP", "Penurunan PCI", "Share semua fitur", "Arah"],
        rows,
        aligns=["left", "center", "center", "center", "center", "left"],
    )
    display(HTML(
        "<div style='font-weight: 600; margin: 4px 0;'>"
        "Fitur agregat hasil rekayasa (dilaporkan terpisah)</div>" + table
    ))


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


def plot_distress_type_reduction(table, model_name, baseline_pci, save_path=None, show=True):
    ordered = table.sort_values(
        "mean_pci_reduction_present", ascending=False, na_position="last"
    ).reset_index(drop=True)
    labels = [f"{rank}. {name}" for rank, name in zip(ordered.index + 1, ordered["distress_type"])]
    data = ordered.iloc[::-1]
    values = data["mean_pci_reduction_present"].fillna(0.0)
    colors = ["#c0504d" if value > 0 else "#3b7ea1" for value in values]

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(labels[::-1], values, color=colors)
    ax.bar_label(
        bars,
        labels=["n/a" if np.isnan(value) else f"{value:.2f}"
                for value in data["mean_pci_reduction_present"]],
        padding=3,
    )
    ax.axvline(0, color="#333333", linewidth=1)
    span = max(values.max() - min(values.min(), 0), 1.0)
    ax.set_xlim(min(values.min(), 0) - 0.18 * span, values.max() + 0.20 * span)
    ax.set(
        xlabel="Mean penurunan PCI saat kerusakan muncul (PCI points)",
        ylabel="Distress type",
        title=(
            f"SHAP Penurunan PCI per Distress Type — {model_name}\n"
            f"Baseline tanpa kerusakan: {baseline_pci:.2f} PCI"
        ),
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _finish_figure(fig, save_path, show)
    return fig


def plot_shap_beeswarm(explanation, model_name, max_display=15, save_path=None, show=True):
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    fig.suptitle(f"SHAP Feature Beeswarm — {model_name}", y=1.02, fontsize=14)
    _finish_figure(fig, save_path, show)
    return fig


def _paths(filename, params, model_key):
    stem = Path(filename).stem
    shap_dir = ROOT_DIR / "outputs" / "shap" / DataSplit.make_tag(params)
    return {
        "values": shap_dir / f"{stem}_{model_key}_shap_values.csv",
        "type_ranking": shap_dir / f"{stem}_{model_key}_shap_distress_type_ranking.csv",
        "feature_ranking": shap_dir / f"{stem}_{model_key}_shap_feature_ranking.csv",
        "meta": shap_dir / f"{stem}_{model_key}_shap_meta.json",
        "type_plot": figure_path(f"06_shap_distress_type_importance_{model_key}.png", params),
        "beeswarm_plot": figure_path(f"07_shap_feature_beeswarm_{model_key}.png", params),
        "reduction_plot": figure_path(f"08_shap_pci_reduction_{model_key}.png", params),
    }


def _save(values_table, type_table, features_table, meta, paths):
    for path in (paths["values"], paths["type_ranking"],
                 paths["feature_ranking"], paths["meta"]):
        path.parent.mkdir(parents=True, exist_ok=True)

    values_table.to_csv(paths["values"], index=False)
    type_table.to_csv(paths["type_ranking"], index=False)
    features_table.to_csv(paths["feature_ranking"], index=False)
    with open(paths["meta"], "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False, default=str)


def explain_best_model(best_model_name, dataset, params, test_features, feature_names,
                       road_info=None, actual=None, max_display=15, save=True, show=True):
    model_key = resolve_model(best_model_name)
    model_name, load_model, predict_model = MODEL_REGISTRY[model_key]
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

    explanation = shap.Explanation(
        values=values,
        base_values=np.full(len(test_features), base_value),
        data=test_features,
        feature_names=feature_names,
    )

    predicted = np.asarray(predict_model(model, test_features), dtype="float64").reshape(-1)
    type_ranking = distress_type_table(values, test_features, feature_names)
    features_table = feature_table(values, feature_names)
    engineered = engineered_table(features_table)

    rows = {}
    if road_info is not None:
        info = road_info.reset_index(drop=True)
        if len(info) != len(test_features):
            raise ValueError("Jumlah road_info tidak cocok dengan jumlah baris test.")
        rows.update({column: info[column].to_numpy() for column in info.columns})

    rows["predicted"] = predicted
    if actual is not None:
        actual = np.asarray(actual, dtype="float64").reshape(-1)
        if len(actual) != len(test_features):
            raise ValueError("Jumlah actual tidak cocok dengan jumlah baris test.")
        rows["actual"] = actual

    rows["base_value"] = np.full(len(test_features), base_value)
    rows.update({f"shap_{name}": values[:, index]
                 for index, name in enumerate(feature_names)})
    rows["sum_shap"] = values.sum(axis=1)
    rows["reconstructed_prediction"] = base_value + rows["sum_shap"]
    rows["reconstruction_error"] = predicted - rows["reconstructed_prediction"]
    values_table = pd.DataFrame(rows)
    paths = _paths(dataset, params, model_key)

    result = {
        "model": model_name,
        "model_key": model_key,
        "explainer": explainer_name,
        "baseline_pci": baseline_pci,
        "base_value": base_value,
        "explanation": explanation,
        "distress_type_ranking": type_ranking,
        "feature_ranking": features_table,
        "engineered_features": engineered,
        "shap_values": values_table,
        "paths": paths,
    }

    if save:
        meta = {
            "format_version": 2,
            "model": model_name,
            "model_key": model_key,
            "split_tag": DataSplit.make_tag(params),
            "explainer": explainer_name,
            "shap_version": shap.__version__,
            "baseline": "semua fitur distress = 0; TOTAL_DENSITY/TOTAL_L/TOTAL_M/TOTAL_H = 0",
            "baseline_pci": baseline_pci,
            "base_value": base_value,
            "ranking_basis": "mean_abs_shap",
            "direction_note": "arah tanda mean_signed_shap: positif atau negatif",
            "reduction_note": "mean_pci_reduction_present adalah analisis pendukung",
            "presence_definition": "total nilai mentah distress type > 0",
            "n_explained": int(len(test_features)),
            "n_features": int(len(feature_names)),
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
                    "mean_signed_shap": float(row.mean_signed_shap),
                    "direction": row.direction,
                }
                for row in type_ranking.head(10).itertuples()
            ],
            "paths": {name: str(path.relative_to(ROOT_DIR).as_posix())
                      for name, path in paths.items()},
            "created_at": datetime.now().strftime("%Y-%m-%d | %H:%M:%S"),
        }
        _save(values_table, type_ranking, features_table, meta, paths)
        result["meta"] = meta

    if show:
        display(Markdown(
            f"**Model:** {model_name} | **Explainer:** {explainer_name} | "
            f"**Baseline tanpa kerusakan:** {baseline_pci:.2f} PCI | "
            f"**Sampel test:** {len(test_features)} baris"
        ))

    if show or save:
        plot_distress_type_importance(
            type_ranking, model_name, baseline_pci,
            save_path=paths["type_plot"] if save else None, show=show,
        )

    if show:
        _display_distress_types(type_ranking)
        _display_engineered(engineered)

    if save:
        plot_shap_beeswarm(
            explanation, model_name, max_display=max_display,
            save_path=paths["beeswarm_plot"], show=False,
        )

    if save:
        plot_distress_type_reduction(
            type_ranking, model_name, baseline_pci,
            save_path=paths["reduction_plot"], show=False,
        )

    return result
