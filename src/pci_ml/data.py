import os
import json
from datetime import datetime
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
from tqdm.auto import tqdm

data_dir = Path(__file__).resolve().parents[2] / "datasets"
PCI, ROAD_ID, AREA = "PCI", "ROAD_ID", "SEGMENT_AREA"
METADATA = ["ROAD_NAME", "SEGMENT_ID"]

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

def seed_search(df, features, params):
    n_seed = int(params["n_seed"])
    test_size = params["test_size"]
    tol = params["share_tol"]
    pci_min = params["pci_min"]

    observed = [c for c in features if (df[c] != 0).any()]
    nonzero = (df[observed] != 0).to_numpy()
    pci_values = df[PCI].to_numpy()

    best_gap, best_seed, best_idx = None, None, None

    # Wrap range with tqdm and store it as pbar so we can update the suffix
    pbar = tqdm(range(n_seed), desc="Mencari Seeds")
    
    for seed in pbar:
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
            
            # This updates the progress bar to show the latest best gap
            pbar.set_postfix({"Best Gap": f"{best_gap:.4f}"})
        
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
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:   # written last
        json.dump(meta, f, indent=2)
    return out_dir

def split_data(df, features, filename, params):
    train_idx, test_idx, seed, gap = seed_search(df, features, params)
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    out_dir = save_checkpoint(train_df, test_df, features, filename, params, seed, gap)

    summary = (
        f"Seed used: {seed} (Gap: {gap:.4f})\n"
        f"Train: {len(train_df)} rows ({len(train_df) / len(df):.2%}) | "
        f"Test: {len(test_df)} rows ({len(test_df) / len(df):.2%})\n"
        f"Roads: {train_df[ROAD_ID].nunique()} train | {test_df[ROAD_ID].nunique()} test\n"
        f"Saved to: {out_dir}"
    )
    return {"train": train_df, "test": test_df, "features": features,
            "seed": seed, "gap": gap, "summary": summary}

def list_checkpoints(filename, out_root=data_dir):
    base = Path(out_root) / Path(filename).stem
    if not base.exists():
        return []
    return [p.name for p in base.iterdir() if p.is_dir() and (p / "meta.json").exists()]

def checkpoint_options(filename, out_root=data_dir):
    base = Path(out_root) / Path(filename).stem
    rows = []
    for tag in tqdm(list_checkpoints(filename, out_root), desc="Loading Checkpoints", leave=False):
        with open(base / tag / "meta.json", encoding="utf-8") as f:
            m = json.load(f)
        p = m["params"]
        label = (f"{m['created_at']} | seed {m['seed']} | gap {m['gap']:.4f} | "
                 f"test {p['test_size']} | tol {p['share_tol']} | pci_min {p['pci_min']}")
        rows.append((m["created_at"], label, tag))
    rows.sort(reverse=True)                      # newest first
    return [(label, tag) for _, label, tag in rows]

def load_checkpoint(filename, tag, out_root=data_dir):
    out_dir = Path(out_root) / Path(filename).stem / tag
    train_df = pd.read_csv(out_dir / "train.csv")
    test_df = pd.read_csv(out_dir / "test.csv")
    with open(out_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return train_df, test_df, meta["features"], meta
