import html
from IPython.display import HTML, Markdown, display
import pandas as pd

TABLE_STYLE = "font-size: 10px; border-collapse: collapse; width: 100%; table-layout: fixed;"
CELL_STYLE = "padding: 2px 3px; word-break: break-word;"


def _cell(tag, value, align, bold=False):
    text = html.escape(str(value))
    if bold:
        text = f"<strong>{text}</strong>"
    return f"<{tag} style='{CELL_STYLE} text-align: {align};'>{text}</{tag}>"


# Fungsi dasar formatting tabel
def format_table(headers, rows, aligns=None, center=False, bold_rows=None):
    aligns = ["center"] * len(headers) if center else (aligns or ["left"] * len(headers))
    bold_rows = set(bold_rows or [])

    head = "".join(_cell("th", h, a) for h, a in zip(headers, aligns))
    body = "".join(
        "<tr>" + "".join(_cell("td", v, a, i in bold_rows) for v, a in zip(row, aligns)) + "</tr>"
        for i, row in enumerate(rows)
    )

    return (
        f"<table style='{TABLE_STYLE}'>"
        f"<thead><tr style='border-bottom: 2px solid #ddd;'>{head}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


# Preview tabel dataset
def df_preview(df_input, n_rows=5, left_count=6, right_count=1):
    if df_input is None:
        return ""

    more_rows = len(df_input) > 2 * n_rows
    df_head = pd.concat([df_input.head(n_rows), df_input.tail(n_rows)]) if more_rows else df_input

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
        rows.insert(n_rows, ["..."] * len(headers))

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


# Tabel perbandingan model: metrik CV dan test tiap model, sudah terurut
# berdasarkan CV MAE. Input berupa list (nama, scores_dict) dengan
# scores_dict = {"cv_score", "cv_folds", "test_score"}.
def model_comparison_table(ranked):
    rows = [
        [
            name,
            f"{scores['cv_score']['mae']:.4f}",
            f"{scores['cv_score']['rmse']:.4f}",
            f"{scores['cv_score']['r2']:.4f}",
            f"{scores['test_score']['mae']:.4f}",
            f"{scores['test_score']['rmse']:.4f}",
            f"{scores['test_score']['r2']:.4f}",
        ]
        for name, scores in ranked
    ]
    return format_table(
        ["Model", "CV MAE", "CV RMSE", "CV R²", "Test MAE", "Test RMSE", "Test R²"],
        rows,
        aligns=["left", "center", "center", "center", "center", "center", "center"],
        bold_rows=[0],
    )


# Ringkasan model terbaik: model peringkat pertama berdasarkan CV MAE dengan
# 1-SE rule, tie-break CV RMSE, dan selisih CV MAE terhadap peringkat kedua.
def model_selection_summary(ranked, se, band, tie_break):
    best_name, best_scores = ranked[0]
    margin = ranked[1][1]["cv_score"]["mae"] - best_scores["cv_score"]["mae"]
    tie_text = "tie-break CV RMSE dipakai" if tie_break else "tidak ada tie-break"

    return (
        f"**Model terbaik: {best_name}**\n\n"
        f"Kriteria seleksi: CV MAE, 1-SE rule (SE = {se:.4f}; batas {band:.4f}), "
        f"{tie_text}. "
        f"Selisih CV MAE terhadap peringkat kedua: {margin:.4f}."
    )


# Tampilkan perbandingan model: ringkasan model terbaik lalu tabel metrik.
def show_model_comparison(ranked, se, band, tie_break):
    display(Markdown(model_selection_summary(ranked, se, band, tie_break)))
    display(HTML(model_comparison_table(ranked)))
