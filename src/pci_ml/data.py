import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

data_dir = Path(__file__).resolve().parents[2] / "datasets"
PCI, ROAD_ID, AREA = "PCI", "ROAD_ID", "SEGMENT_AREA"
METADATA = ["ROAD_NAME", "SEGMENT_ID"]

SEVERITY_WEIGHTS = {"_L": 1.0, "_M": 2.0, "_H": 3.0}
AGGREGATE_FEATURES = ["TOTAL_DENSITY", "SEVERITY_WEIGHTED"]


def list_datasets():
    return sorted (
        f for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f))
        and f.lower().endswith((".xlsx", ".xls"))
        and not f.startswith("~$")
    )

def load_data(filename):
    path = os.path.join(data_dir, filename)
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)

def clean_data(df):
    excluded = {PCI, ROAD_ID, AREA, *METADATA}
    features = [c for c in df.columns if c not in excluded]

    df = df.drop_duplicates().copy()
    df[PCI] = df[PCI].clip(lower=0, upper=100)
    df[features] = df[features].div(df[AREA], axis=0).clip(lower=0, upper=1)
    df = df.drop(columns=[AREA])
    return df, features


def add_aggregate_features(df, features):
    """Append total and severity-weighted density to the feature set.

    ``TOTAL_DENSITY`` is the row-wise sum of all density features.
    ``SEVERITY_WEIGHTED`` weights each feature by its severity suffix
    (``_L``=1, ``_M``=2, ``_H``=3). Both columns are inserted before
    ``PCI`` so it stays the last column. The transform is stateless, so
    the same columns can be derived for any split.
    """
    df = df.copy()
    density = df[features].astype("float32")
    severity = np.zeros(len(df), dtype="float32")
    for column in features:
        for suffix, weight in SEVERITY_WEIGHTS.items():
            if column.endswith(suffix):
                severity += density[column].to_numpy(dtype="float32") * weight
                break

    df["TOTAL_DENSITY"] = density.sum(axis=1).astype("float32")
    df["SEVERITY_WEIGHTED"] = severity

    # Keep PCI as the last column of the cleaned frame.
    ordered = [name for name in df.columns
               if name not in AGGREGATE_FEATURES and name != PCI]
    df = df[[*ordered, *AGGREGATE_FEATURES, PCI]]
    return df, [*features, *AGGREGATE_FEATURES]


def seed_search(df, features, params, progress):
    n_seed = int(params["n_seed"])
    test_size = params["test_size"]
    tol = params["share_tol"]
    pci_min = params["pci_min"]

    observed = [c for c in features if (df[c] != 0).any()]
    nonzero = (df[observed] != 0).to_numpy()
    pci_values = df[PCI].to_numpy()

    best_gap, best_seed, best_idx = None, None, None

    for seed in range(n_seed):
        try:
            progress(seed, n_seed, best_gap)

            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_idx, test_idx = next(gss.split(df, groups=df[ROAD_ID]))

            share = len(test_idx) / len(df)
            if not (test_size - tol) <= share <= (test_size + tol):
                continue

            if (~nonzero[train_idx].any(axis=0) & nonzero[test_idx].any(axis=0)).any():
                continue

            y_train, y_test = pci_values[train_idx], pci_values[test_idx]
            if y_test.min() > pci_min:
                continue

            gap = abs(y_train.mean() - y_test.mean()) + abs(y_train.std(ddof=1) - y_test.std(ddof=1))

            if best_gap is None or gap < best_gap:
                best_gap, best_seed, best_idx = gap, seed, (train_idx, test_idx)
        finally:
            if seed == n_seed - 1:
                progress(n_seed, n_seed, best_gap)

    if best_seed is None:
        raise ValueError("No seed passed the criteria.")
    return best_idx[0], best_idx[1], best_seed, float(best_gap)

def _tag(params):
    return (f"n{params['n_seed']}_t{params['test_size']}"
            f"_s{params['share_tol']}_p{params['pci_min']}")

def save_checkpoint(train_df, test_df, features, filename, params, seed, gap,
                    out_root=data_dir):
    out_dir = Path(out_root) / Path(filename).stem / _tag(params)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    meta = {
        "source_file": filename,
        "params": params,
        "seed": int(seed),
        "gap": float(gap),
        "features": features,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "created_at": datetime.now().strftime("%Y-%m-%d | %H:%M:%S"),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_dir

def split_data(df, features, filename, params, progress):
    train_idx, test_idx, seed, gap = seed_search(df, features, params, progress=progress)
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    out_dir = save_checkpoint(train_df, test_df, features, filename, params, seed, gap)

    summary = (
        f"Seed used: {seed} (Gap: {gap:.4f})\n"
        f"Train: {len(train_df)} rows ({len(train_df) / len(df):.2%}) | "
        f"Test: {len(test_df)} rows ({len(test_df) / len(df):.2%})\n"
        f"Roads: {train_df[ROAD_ID].nunique()} train | {test_df[ROAD_ID].nunique()} test\n"
        f'Saved to: "{out_dir}"'
    )
    return {"train": train_df, "test": test_df, "features": features,
            "seed": seed, "gap": gap, "summary": summary}

def list_checkpoints(filename, out_root=data_dir):
    base = Path(out_root) / Path(filename).stem
    if not base.exists():
        return []
    return [p.name for p in base.iterdir() if p.is_dir() and (p / "meta.json").exists()]

def load_checkpoint(filename, tag, out_root=data_dir):
    out_dir = Path(out_root) / Path(filename).stem / tag
    train_df = pd.read_csv(out_dir / "train.csv")
    test_df = pd.read_csv(out_dir / "test.csv")
    with open(out_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return train_df, test_df, meta["features"], meta

def latest_checkpoint(filename, out_root=data_dir):
    tags = list_checkpoints(filename, out_root)
    if not tags:
        raise FileNotFoundError("run the split first")
    base = Path(out_root) / Path(filename).stem

    def created_at(tag):
        with open(base / tag / "meta.json", encoding="utf-8") as f:
            return json.load(f)["created_at"]

    return max(tags, key=created_at)
