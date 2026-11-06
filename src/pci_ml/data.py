import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

data_dir = Path(__file__).resolve().parents[2] / "datasets"

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
    pci = "PCI"
    road_id = "ROAD_ID"
    area = "SEGMENT_AREA"
    metadata = ["ROAD_NAME", "SEGMENT_ID"]

    excluded = {pci, road_id, area, *metadata}
    features = [c for c in df.columns if c not in excluded]

    df = df.drop_duplicates().copy()
    df[pci] = df[pci].clip(lower=0, upper=100)
    df[features] = df[features].div(df[area], axis=0).clip(lower=0, upper=1)
    df = df.drop(columns=[area])

    return df, features


def seed_search(df, features, params):
    seeds = int(params['n_seed'])
    test = params['test_size']
    tol = params['share_tol']
    pcimin = params['pci_min']
    pci = "PCI"
    road_id = "ROAD_ID"

    observed = [c for c in features if (df[c] != 0).any()]
    nonzero = (df[observed] != 0).to_numpy()
    pci_values = df[pci].to_numpy()

    best_gap, best_seed, best_idx = None, None, None

    print(f"\nSearching {seeds} seeds...")
    for seed in range(seeds):
        if seed == 0 or (seed + 1) % 10 == 0 or (seed + 1) == seeds:
            gap_str = f"{best_gap:.4f}" if best_gap is not None else "None"
            print(f"Progress: {seed + 1}/{seeds} seeds checked... (Best gap: {gap_str})", end="\r", flush=True)

        gss = GroupShuffleSplit(n_splits=1, test_size=test, random_state=seed)
        train_idx, test_idx = next(gss.split(df, groups=df[road_id]))

        share = len(test_idx) / len(df)
        if not (test - tol) <= share <= (test + tol):
            continue

        train_nz = nonzero[train_idx].any(axis=0)
        test_nz = nonzero[test_idx].any(axis=0)
        if (~train_nz & test_nz).any():
            continue

        y_train, y_test = pci_values[train_idx], pci_values[test_idx]
        if y_test.min() > pcimin:
            continue

        gap = abs (y_train.mean() - y_test.mean()) + abs(y_train.std(ddof=1) - y_test.std(ddof=1))
        if best_gap is None or gap < best_gap:
            best_gap, best_seed, best_idx = gap, seed, (train_idx, test_idx)

    print()
    if best_seed is None:
        raise SystemExit("Tidak ada seed yang memenuhi kriteria.")

    return best_idx[0], best_idx[1], best_seed, best_gap

def split_data(df, features, filename, params, road_id="ROAD_ID", out_root="./datasets"):

    train_idx, test_idx, seed, gap = seed_search(df, features, params)

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    name = Path(filename).stem
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / f"{name}-train.csv", index=False)
    test_df.to_csv(out_dir / f"{name}-test.csv", index=False)

    summary = (
        f"Seed used: {seed} (Gap: {gap:.4f})\n"
        f"Train: {len(train_df)} rows ({len(train_df) / len(df):.2%}) | "
        f"Test: {len(test_df)} rows ({len(test_df) / len(df):.2%})\n"
        f"Roads: {train_df[road_id].nunique()} train | {test_df[road_id].nunique()} test\n"
        f"Saved to: {out_dir}"
    )

    return {"train": train_df, "test": test_df, "features": features,
            "seed": seed, "gap": gap, "summary": summary}

