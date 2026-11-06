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

