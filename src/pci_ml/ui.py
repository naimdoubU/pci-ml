"""Tabel dan progress helper untuk notebook PCI ML."""

import html

import pandas as pd
from IPython.display import clear_output

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
