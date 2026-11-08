"""Control panel for the PCI ML notebook."""

import pandas as pd
import ipywidgets as wg
from IPython.display import clear_output, display


def format_adaptive_df(df_input):
    """Return a compact HTML table with the first and last columns."""
    if df_input is None:
        return ""
    df_head = df_input.head()
    left_count = 5
    right_count = 2
    if df_head.shape[1] > (left_count + right_count):
        left_cols = df_head.iloc[:, :left_count]
        right_cols = df_head.iloc[:, -right_count:]
        middle_col = pd.DataFrame({"…": ["…"] * len(df_head)}, index=df_head.index)
        df_display = pd.concat([left_cols, middle_col, right_cols], axis=1)
    else:
        df_display = df_head
    markdown_table = df_display.to_markdown(index=False)
    return f'<div style="font-size: 11px;">\n\n{markdown_table}\n\n</div>'


def split_progress(done, total, best_gap):
    """Print split-search progress without flooding the notebook."""
    if done % 100 != 0 and done != total:
        return
    gap = "-" if best_gap is None else f"{best_gap:.4f}"
    clear_output(wait=True)
    print(f"Mencari Seeds: {done}/{total} | Best Gap: {gap}", flush=True)


def _resolve_file(datasets, defaults):
    return defaults["file"] or (datasets[0] if datasets else None)


class ControlPanel:
    """Holds the dataset dropdown and the split parameters."""

    def __init__(self, datasets, defaults, on_file_change=None):
        style = {"description_width": "110px"}
        box = wg.Layout(width="180px", margin="0 8px 0 0")

        self.file = wg.Dropdown(
            options=datasets,
            value=_resolve_file(datasets, defaults),
            description="Pilih file :",
            style=style,
            layout=wg.Layout(width="400px", margin="0 0 6px 0"),
        )
        self.n_seed = wg.IntText(value=defaults["n_seed"], description="Jumlah seed:",
                                 style=style, layout=box)
        self.test_size = wg.FloatText(value=defaults["test_size"], description="Test size:",
                                      style=style, layout=box)
        self.share_tol = wg.FloatText(value=defaults["share_tol"], description="Tolerance:",
                                      style=style, layout=box)
        self.pci_min = wg.FloatText(value=defaults["pci_min"], description="PCI minimum:",
                                    style=style, layout=box)

        self._on_file_change = on_file_change
        self.file.observe(self._file_changed, names="value")

        self.widget = wg.VBox([
            self.file,
            wg.HBox([self.n_seed, self.test_size, self.share_tol, self.pci_min],
                    layout=wg.Layout(align_items="flex-start", flex_flow="row wrap")),
        ], layout=wg.Layout(align_items="flex-start"))

    def _file_changed(self, change):
        if self._on_file_change is not None:
            self._on_file_change(change["new"])

    def values(self):
        """Return a plain snapshot of the current panel values."""
        return {
            "file": self.file.value,
            "n_seed": int(self.n_seed.value),
            "test_size": float(self.test_size.value),
            "share_tol": float(self.share_tol.value),
            "pci_min": float(self.pci_min.value),
        }


class StaticPanel:
    """Read-only panel used when export mode is on."""

    def __init__(self, values):
        self._values = values

    def values(self):
        return dict(self._values)


class ParamPanel:
    """Holds a group of numeric parameters."""

    def __init__(self, defaults, specs):
        style = {"description_width": "110px"}
        box = wg.Layout(width="180px", margin="0 8px 0 0")
        self._widgets = {}

        for key, spec in specs.items():
            widget_cls = wg.IntText if spec["type"] == "int" else wg.FloatText
            self._widgets[key] = widget_cls(
                value=defaults[key],
                description=spec["description"],
                style=style,
                layout=box,
            )

        self.widget = wg.VBox([
            wg.HBox(list(self._widgets.values()),
                    layout=wg.Layout(align_items="flex-start", flex_flow="row wrap")),
        ], layout=wg.Layout(align_items="flex-start"))

    def values(self):
        """Return a plain snapshot of the current panel values."""
        result = {}
        for key, widget in self._widgets.items():
            result[key] = int(widget.value) if isinstance(widget, wg.IntText) else float(widget.value)
        return result


def build_param_panel(defaults, specs, title, export=False):
    """Build and display a numeric parameter panel; return it for snapshotting."""
    if export:
        values = {}
        for key, spec in specs.items():
            values[key] = int(defaults[key]) if spec["type"] == "int" else float(defaults[key])
        print(title)
        for key, spec in specs.items():
            print(f"- {spec['label']:<18}: {values[key]}")
        return StaticPanel(values)

    panel = ParamPanel(defaults, specs)
    display(panel.widget)
    return panel


def build_panel(datasets, defaults, on_file_change=None, export=False):
    """Build and display the control panel; return it for snapshotting.

    In export mode the widgets are replaced by printed values taken from
    ``defaults``. The caller triggers the initial preview after creating
    its output handle, so the panel appears above that preview.
    """
    if export:
        values = {
            "file": _resolve_file(datasets, defaults),
            "n_seed": int(defaults["n_seed"]),
            "test_size": float(defaults["test_size"]),
            "share_tol": float(defaults["share_tol"]),
            "pci_min": float(defaults["pci_min"]),
        }
        print(f"File terpilih untuk analisis: {values['file']}")
        print("Parameter Pemisahan Data:")
        print(f"- Jumlah seed     : {values['n_seed']}")
        print(f"- Test size       : {values['test_size']}")
        print(f"- Share tolerance : {values['share_tol']}")
        print(f"- PCI minimum     : {values['pci_min']}")
        return StaticPanel(values)

    panel = ControlPanel(datasets, defaults, on_file_change=on_file_change)
    display(panel.widget)
    return panel


def build_hyperparameter_table(params, keys, explanations):
    """Return an HTML table describing selected hyperparameters."""
    def format_value(value):
        return "tanpa batas" if value is None else value

    rows_html = ""
    for key in keys:
        value = format_value(params[key])
        description = explanations[key]
        rows_html += (
            "    <tr style='border-bottom: 1px solid #eee;'>\n"
            f"      <td style='padding: 6px 10px; font-family: monospace; white-space: nowrap;'>{key}</td>\n"
            f"      <td style='padding: 6px 10px; text-align: center;'>{value}</td>\n"
            f"      <td style='padding: 6px 10px; max-width: 400px; word-wrap: break-word;'>{description}</td>\n"
            "    </tr>\n"
        )

    return (
        "<table style='font-size: 11px; border-collapse: collapse; width: 100%;'>\n"
        "  <thead>\n"
        "    <tr style='border-bottom: 2px solid #ddd;'>\n"
        "      <th style='padding: 6px 10px; text-align: left; white-space: nowrap;'>Hyperparameter</th>\n"
        "      <th style='padding: 6px 10px; text-align: center; white-space: nowrap;'>Nilai</th>\n"
        "      <th style='padding: 6px 10px; text-align: left;'>Keterangan</th>\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        + rows_html
        + "  </tbody>\n"
        "</table>"
    )
