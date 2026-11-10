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

LEVEL_SUFFIXES = {"TOTAL_L": "_L", "TOTAL_M": "_M", "TOTAL_H": "_H"}
ENGINEERED_FEATURES = ["TOTAL_DENSITY", *LEVEL_SUFFIXES]

def load_data(filename):
    path = os.path.join(data_dir, filename)
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)

def clean_data(df):
    excluded = {PCI, ROAD_ID, AREA, *METADATA}
    features = [c for c in df.columns if c not in excluded]

    df = df.drop_duplicates().dropna(subset=[PCI, AREA]).copy()
    df = df[df[AREA] > 0].copy()

    df[PCI] = df[PCI].clip(lower=0, upper=100)
    df[features] = df[features].fillna(0)
    return df, features

def to_density(df, features):
    df = df.copy()
    df[features] = df[features].div(df[AREA], axis=0).clip(lower=0, upper=1)
    df = df.drop(columns=[AREA])
    return df, features

def engineer_features(df, features):
    # Feature engineering: total densitas dan total densitas per tingkat keparahan (L/M/H), tanpa bobot
    df = df.copy()
    density = df[features].astype("float32")

    df["TOTAL_DENSITY"] = density.sum(axis=1).astype("float32")
    for name, suffix in LEVEL_SUFFIXES.items():
        cols = [c for c in features if c.endswith(suffix)]
        if not cols:
            raise ValueError(f"Tidak ada kolom berakhiran '{suffix}' untuk {name}")
        df[name] = density[cols].sum(axis=1).astype("float32")

    ordered = [c for c in df.columns if c not in ENGINEERED_FEATURES and c != PCI]
    df = df[[*ordered, *ENGINEERED_FEATURES, PCI]]
    return df, [*features, *ENGINEERED_FEATURES]

class DataSplit:
    def __init__(self, df, features, filename, params):
        self.df = df
        self.features = features
        self.filename = filename
        self.params = params

        self.n_seed = int(params["n_seed"])
        self.test_size = params["test_size"]
        self.tol = params["share_tol"]
        self.pci_min = params["pci_min"]

        # variabel untuk kriteria seed: tabel boolean untuk fitur = 0; nilai_pci
        self._nonzero= (df[features] != 0).to_numpy()
        self._pci = df[PCI].to_numpy()

    # Search seed
    def _try_seed(self, seed):
        """Return (gap, train_idx, test_idx), or None if the seed fails a rule."""
        gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(self.df, groups=self.df[ROAD_ID]))

        # Kriteria 1: test, train = 80%, 20% (dengan toleransi 1%)
        share = len(test_idx) / len(self.df)
        if not (self.test_size - self.tol) <= share <= (self.test_size + self.tol):
            return None

        # Kriteria 2: tidak ada fitur yang hanya muncul di test
        only_in_test = ~self._nonzero[train_idx].any(axis=0) & self._nonzero[test_idx].any(axis=0)
        if only_in_test.any():
            return None

        # Kriteria 3: nilai target ekstrim, test harus punya jalan dengan PCI = 0
        y_train, y_test = self._pci[train_idx], self._pci[test_idx]
        if y_test.min() > self.pci_min:
            return None

        # Hitung kemiripan distribusi jika lolos 3 kriteria
        gap = (abs(y_train.mean() - y_test.mean())
               + abs(y_train.std(ddof=1) - y_test.std(ddof=1)))
        return gap, train_idx, test_idx

    def search(self):
        best = None   # (gap, seed, train_idx, test_idx)

        for seed in range(self.n_seed):
            result = self._try_seed(seed)
            if result and (best is None or result[0] < best[0]):
                best = (result[0], seed, result[1], result[2])

        if best is None:
            raise ValueError("No seed passed the criteria.")

        gap, seed, train_idx, test_idx = best
        return train_idx, test_idx, seed, float(gap)

    # Save data yang dipisah
    @staticmethod
    def make_tag(params):
        return f"n{params['n_seed']}_t{params['test_size']}_s{params['share_tol']}_p{params['pci_min']}"

    def _out_dir(self, out_root):
        return Path(out_root) / Path(self.filename).stem / self.make_tag(self.params)

    def save(self, train_df, test_df, seed, gap, out_root=None):
        out_dir = self._out_dir(out_root or data_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(out_dir / "train.csv", index=False)
        test_df.to_csv(out_dir / "test.csv", index=False)

        meta = {
            "source_file": self.filename,
            "params": self.params,
            "seed": int(seed),
            "gap": float(gap),
            "features": self.features,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "created_at": datetime.now().strftime("%Y-%m-%d | %H:%M:%S"),
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return out_dir

    # Mulai fungsi splitter
    def run(self):
        train_idx, test_idx, seed, gap = self.search()
        train_df = self.df.iloc[train_idx].copy()
        test_df = self.df.iloc[test_idx].copy()
        out_dir = self.save(train_df, test_df, seed, gap)

        summary = (
            f"Seed used: {seed} (Gap: {gap:.4f})\n"
            f"Train: {len(train_df)} rows ({len(train_df) / len(self.df):.2%}) | "
            f"Test: {len(test_df)} rows ({len(test_df) / len(self.df):.2%})\n"
            f"Roads: {train_df[ROAD_ID].nunique()} train | {test_df[ROAD_ID].nunique()} test\n"
            f'Saved to: "{out_dir}"'
        )
        return {"train": train_df, "test": test_df, "features": self.features,
                "seed": seed, "gap": gap, "summary": summary}

class ModelInput:
    def __init__(self, filename):
        self.base = Path(data_dir) / Path(filename).stem

    def _dir(self, tag):
        return self.base / tag

    def _read_meta(self, tag):
        with open(self._dir(tag) / "meta.json", encoding="utf-8") as f:
            return json.load(f)

    def list(self):
        if not self.base.exists():
            return []
        return [p.name for p in self.base.iterdir()
                if p.is_dir() and (p / "meta.json").exists()]

    def find(self, params):
        tag = DataSplit.make_tag(params)
        available = self.list()
        if tag in available:
            return tag

        print(f"PERINGATAN: tidak ada checkpoint untuk params {params}")
        print(f"Tag yang dicari : {tag}")
        print(f"Tag tersedia    : {available or 'tidak ada'}")
        return None

    def load(self, tag):
        if tag is None:
            raise FileNotFoundError("Checkpoint tidak ditemukan. Jalankan sel split terlebih dahulu.")
        out_dir = self._dir(tag)
        train_df = pd.read_csv(out_dir / "train.csv")
        test_df = pd.read_csv(out_dir / "test.csv")
        meta = self._read_meta(tag)
        return train_df, test_df, meta["features"], meta

    def load_params(self, params):
        tag = self.find(params)
        return (tag, *self.load(tag))
