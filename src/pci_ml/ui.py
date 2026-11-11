import html
from IPython.display import HTML, Markdown, display
import pandas as pd

TABLE_STYLE = "font-size: 10px; border-collapse: collapse; width: 100%; table-layout: fixed;"
CELL_STYLE = "padding: 2px 3px; word-break: break-word;"


def _cell(tag, value, align):
    return f"<{tag} style='{CELL_STYLE} text-align: {align};'>{html.escape(str(value))}</{tag}>"


# Fungsi dasar formatting tabel
def format_table(headers, rows, aligns=None, center=False):
    aligns = ["center"] * len(headers) if center else (aligns or ["left"] * len(headers))

    head = "".join(_cell("th", h, a) for h, a in zip(headers, aligns))
    body = "".join(
        "<tr>" + "".join(_cell("td", v, a) for v, a in zip(row, aligns)) + "</tr>"
        for row in rows
    )

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
        middle_col = pd.DataFrame({"…": "…"}, index=df_head.index)
        df_display = pd.concat([left_cols, middle_col, right_cols], axis=1)
    else:
        df_display = df_head

    headers = [str(name) for name in df_display.columns]
    rows = df_display.map(_format_cell).values.tolist()

    if more_rows:
        rows.append(["..."] * len(headers))

    return format_table(headers, rows, center=True)


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
        value = params[key]
        rows.append([key, "tanpa batas" if value is None else value, explanations[key]])

    return format_table(
        ["Hyperparameter", "Nilai", "Keterangan"],
        rows,
        aligns=["left", "center", "left"],
    )


def param_table(best_params, search_space, explanations):
    return build_hyperparameter_table(best_params, search_space.keys(), explanations)


_METRICS = ("mae", "rmse", "r2")


def _score_row(label, score):
    return [label, *(f"{score[m]:.4f}" for m in _METRICS)]


def evaluation_table(cv_score, test_score):
    return format_table(
        ["Evaluasi", "MAE", "RMSE", "R²"],
        [_score_row("Cross-Validation", cv_score), _score_row("Test", test_score)],
        aligns=["left", "center", "center", "center"],
    )


def show_report(best_params, search_space, cv_score, test_score, explanations):
    display(Markdown("##### **Hyperparameter terbaik**"))
    display(HTML(param_table(best_params, search_space, explanations)))
    display(Markdown("##### **Hasil evaluasi**"))
    display(HTML(evaluation_table(cv_score, test_score)))
