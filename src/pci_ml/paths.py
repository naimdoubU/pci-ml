from pathlib import Path

from pci_ml.data import DataSplit

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT_DIR / "outputs"


def dataset_stem(filename):
    return Path(filename).stem


def split_tag(params):
    return DataSplit.make_tag(params)


def run_output_dir(filename, params):
    return OUTPUTS_DIR / f"{dataset_stem(filename)}_{split_tag(params)}"


def output_dir(kind, filename, params):
    return run_output_dir(filename, params) / kind


def figure_path(name, filename, params):
    return output_dir("figures", filename, params) / name


def test_predictions_dir(filename, params):
    return output_dir("predictions", filename, params)


def test_predictions_path(model, filename, params):
    return test_predictions_dir(filename, params) / f"{dataset_stem(filename)}_{model}_test_predictions.csv"


def shap_output_dir(filename, params):
    return output_dir("shap", filename, params)
