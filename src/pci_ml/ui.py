"""Control panel for the PCI ML notebook."""

import html

import pandas as pd
import ipywidgets as wg
from IPython.display import clear_output, display

TABLE_STYLE = "font-size: 11px; border-collapse: collapse; width: 100%;"
HEADER_STYLE = "padding: 6px 10px;"
CELL_STYLE = "padding: 6px 10px;"


def build_table_rows(rows, aligns=None, monospace=None, nowrap=None, wrap=None):
    """Return HTML body rows for a table.

    ``rows`` holds one sequence per row. ``aligns`` gives the text
    alignment of each column, while ``monospace``, ``nowrap``, and
    ``wrap`` list column indices that need extra cell styling.
    """
    if not rows:
        return ""

    columns = len(rows[0])
    aligns = aligns or ["left"] * columns
    monospace = set(monospace or ())
    nowrap = set(nowrap or ())
    wrap = set(wrap or ())

    body = ""
    for row in rows:
        cells = ""
        for index, (value, align) in enumerate(zip(row, aligns)):
            style = f"{CELL_STYLE} text-align: {align};"
            if index in monospace:
                style += " font-family: monospace;"
            if index in nowrap:
                style += " white-space: nowrap;"
            if index in wrap:
                style += " max-width: 400px; word-wrap: break-word;"
            cells += f"      <td style='{style}'>{html.escape(str(value))}</td>\n"
        body += (
            "    <tr style='border-bottom: 1px solid #eee;'>\n"
            + cells
            + "    </tr>\n"
        )
    return body


def format_table(headers, rows, aligns=None, monospace=None, nowrap=None, wrap=None):
    """Return an HTML table using the shared notebook style."""
    columns = len(headers)
    aligns = aligns or ["left"] * columns
    nowrap = set(nowrap or ())

    header_cells = ""
    for index, (header, align) in enumerate(zip(headers, aligns)):
        style = f"{HEADER_STYLE} text-align: {align};"
        if index in nowrap:
            style += " white-space: nowrap;"
        header_cells += (
            f"      <th style='{style}'>{html.escape(str(header))}</th>\n"
        )

    return (
        f"<table style='{TABLE_STYLE}'>\n"
        "  <thead>\n"
        "    <tr style='border-bottom: 2px solid #ddd;'>\n"
        + header_cells
        + "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        + build_table_rows(rows, aligns, monospace, nowrap, wrap)
        + "  </tbody>\n"
        "</table>"
    )


def _format_cell(value):
    if pd.isna(value):
        return "–"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def format_dataframe_table(df_input, n_rows=5, left_count=5, right_count=2):
    """Return a compact HTML table with the first and last columns."""
    if df_input is None:
        return ""

    df_head = df_input.head(n_rows)
    if df_head.shape[1] > (left_count + right_count):
        left_cols = df_head.iloc[:, :left_count]
        right_cols = df_head.iloc[:, -right_count:]
        middle_col = pd.DataFrame({"…": ["…"] * len(df_head)}, index=df_head.index)
        df_display = pd.concat([left_cols, middle_col, right_cols], axis=1)
    else:
        df_display = df_head

    headers = [str(name) for name in df_display.columns]
    rows = df_display.map(_format_cell).values.tolist()

    aligns = []
    for name in df_display.columns:
        if name == "…":
            aligns.append("center")
        elif pd.api.types.is_numeric_dtype(df_display[name]):
            aligns.append("right")
        else:
            aligns.append("left")

    return format_table(headers, rows, aligns, nowrap=range(len(headers)))


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
    rows = []
    for key in keys:
        value = "tanpa batas" if params[key] is None else params[key]
        rows.append([key, value, explanations[key]])

    return format_table(
        ["Hyperparameter", "Nilai", "Keterangan"],
        rows,
        aligns=["left", "center", "left"],
        monospace=[0],
        nowrap=[0, 1],
        wrap=[2],
    )
