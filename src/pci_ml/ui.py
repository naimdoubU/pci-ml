"""Control panel for the PCI ML notebook."""

import ipywidgets as wg
from IPython.display import display


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
