import html
from IPython.display import HTML, display
import pandas as pd

HEADER_STYLE = "padding: 5px 5px;"
TABLE_STYLE = "font-size: 10px; border-collapse: collapse; width: 100%; table-layout: fixed;"
CELL_STYLE = "padding: 2px 3px; word-break: break-word;"

# Fungsi dasar formatting tabel
def format_table(headers, rows, aligns=None, center=False):
    if center:
        aligns = ["center"] * len(headers)
    else:
        aligns = aligns or ["left"] * len(headers)

    def cell(tag, value, align):
        style = f"{CELL_STYLE} text-align: {align};"
        return f"<{tag} style='{style}'>{html.escape(str(value))}</{tag}>"

    head = "".join(cell("th", h, a) for h, a in zip(headers, aligns))
    body = "".join("<tr>" + "".join(cell("td", v, a) for v, a in zip(row, aligns)) + "</tr>" 
                   for row in rows)

    return (
        f"<table style='{TABLE_STYLE}'>"
        f"<thead><tr style='border-bottom: 2px solid #ddd;'>{head}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )

# Preview tabel dataset
def df_preview(df_input, n_rows=19, left_count=6, right_count=1):
    if df_input is None:
        return ""

    more_rows = len(df_input) > n_rows
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

    if more_rows:
        rows.append(["..."] * len(headers))

    return format_table(headers, rows, center=True)

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

def _format_cell(value):
    if pd.isna(value):
        return "–"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)

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
    )
