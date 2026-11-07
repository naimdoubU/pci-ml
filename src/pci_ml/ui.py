"""Control panel for the PCI ML notebook."""

import ipywidgets as wg
from IPython.display import display


class ControlPanel:
    """Holds the dataset dropdown and the split parameters."""

    def __init__(self, datasets, defaults, on_file_change=None):
        style = {"description_width": "100px"}
        box = wg.Layout(width="180px", margin="0 15px")

        self.file = wg.Dropdown(
            options=datasets,
            value=defaults["file"] or (datasets[0] if datasets else None),
            description="Pilih file :",
            layout=wg.Layout(width="400px"),
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
                    layout=wg.Layout(justify_content="center", flex_flow="row wrap")),
        ])

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


def build_panel(datasets, defaults, on_file_change=None):
    """Build and display the control panel; return it for snapshotting."""
    panel = ControlPanel(datasets, defaults, on_file_change=on_file_change)
    display(panel.widget)
    if on_file_change is not None:
        on_file_change(panel.file.value)
    return panel
