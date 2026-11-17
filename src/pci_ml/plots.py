"""
Plot utilities for PCI-ML project.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import io
import base64
from IPython.display import display, HTML


# ─── Color palettes ────────────────────────────────────────────────

def get_train_colors():
    """Return the standard train color palette (3 shades of blue + no severity)."""
    pastel = sns.color_palette("pastel")
    return {
        'L': sns.color_palette("Blues", n_colors=5)[1],
        'M': sns.color_palette("Blues", n_colors=5)[2],
        'H': sns.color_palette("Blues", n_colors=5)[3],
        'No Severity': pastel[9],
    }


def get_test_colors():
    """Return the standard test color palette (3 shades of orange + no severity)."""
    pastel = sns.color_palette("pastel")
    return {
        'L': sns.color_palette("Oranges", n_colors=5)[1],
        'M': sns.color_palette("Oranges", n_colors=5)[2],
        'H': sns.color_palette("Oranges", n_colors=5)[3],
        'No Severity': pastel[8],
    }


def get_severity_order():
    """Return the standard severity order for stacking."""
    return ['L', 'M', 'H', 'No Severity']


# ─── Figure helpers ────────────────────────────────────────────────

def setup_figure(figsize=(30/2.54, 25/2.54), style='white'):
    """Create figure and axes with standard theme."""
    sns.set_theme(style=style)
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def finalize_figure(fig, ax, dpi=150):
    """Save figure to base64 PNG and close."""
    sns.despine(left=True, bottom=True)
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return base64.b64encode(buf.getvalue()).decode('utf-8')


def show_base64_image(img_base64):
    """Display base64 encoded image centered in notebook."""
    display(HTML(f'<div style="text-align: center;"><img src="data:image/png;base64,{img_base64}"/></div>'))


# ─── Y-axis spacing helpers ────────────────────────────────────────

def compute_y_positions(n_items, bar_height=0.36, offset=0.20, inter_feature_gap=0.10):
    """
    Compute y positions for grouped horizontal bars.

    Parameters
    ----------
    n_items : int
        Number of items (features) on y-axis.
    bar_height : float
        Height of each bar.
    offset : float
        Offset from center for train/test bars.
    inter_feature_gap : float
        Gap between adjacent features.

    Returns
    -------
    y : np.ndarray
        Center positions for each item.
    y_step : float
        Step size between item centers.
    """
    y_step = bar_height + offset + inter_feature_gap
    y = np.arange(n_items) * y_step
    return y, y_step


def set_y_axis(ax, y, labels, y_step, fontsize=10, fontweight='bold', grid_color='#e0e0e0', grid_width=0.8):
    """Configure y-axis ticks, labels, and horizontal grid lines."""
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fontsize, fontweight=fontweight)
    ax.set_ylim(-0.5, (len(y) - 1) * y_step + 0.5)

    for pos in y[:-1] + y_step / 2:
        ax.axhline(pos, color=grid_color, linestyle='--', linewidth=grid_width)


# ─── Data preparation ──────────────────────────────────────────────

def prepare_severity_counts(df, cols):
    """
    Aggregate non-zero counts by distress type and severity.

    Parameters
    ----------
    df : pd.DataFrame
        Data with distress columns.
    cols : list
        Column names to process.

    Returns
    -------
    pd.DataFrame
        Pivot table with index=Distress, columns=['L','M','H','No Severity'], values=Count.
    """
    records = []
    for col in cols:
        if col.endswith(('_L', '_M', '_H')):
            distress, severity = col.rsplit('_', 1)
        else:
            distress, severity = col, 'No Severity'
        records.append({
            'Distress': distress,
            'Severity': severity,
            'Count': int((df[col] > 0).sum())
        })
    piv = pd.DataFrame(records).pivot(index='Distress', columns='Severity', values='Count').fillna(0).astype(int)
    return piv.reindex(columns=['L', 'M', 'H', 'No Severity'], fill_value=0)


def filter_distress_cols(features):
    """Remove summary total columns from feature list."""
    summary_cols = {'TOTAL_DENSITY', 'TOTAL_L', 'TOTAL_M', 'TOTAL_H'}
    return [c for c in features if c not in summary_cols]


def sort_distress_by_total(train_piv, test_piv, ascending=True):
    """Sort distress types by combined train+test total count."""
    order = (train_piv.sum(axis=1) + test_piv.sum(axis=1)).sort_values(ascending=ascending).index
    return train_piv.loc[order], test_piv.loc[order], order


# ─── Annotation helpers ────────────────────────────────────────────

def annotate_bar_counts(ax, piv, y_pos, x_offset=2.0, fontsize=10, color='#222222'):
    """Annotate each bar with severity counts or total."""
    for idx, (distress, row) in enumerate(piv.iterrows()):
        total = row.sum()
        if distress == 'POLISH_AGG':
            label = f"Total = {total}"
        else:
            label = f"L = {row['L']}; M = {row['M']}; H = {row['H']}"
        ax.text(total + x_offset, y_pos[idx], label, va='center', ha='left', fontsize=fontsize, color=color)


# ─── Legend helpers ────────────────────────────────────────────────

def build_split_severity_legend(train_colors, test_colors):
    """Build standard legend handles for train/test split with severity."""
    handles = []
    for split_name, colors, prefix in [
        ('Train', train_colors, 'Train'),
        ('Test', test_colors, 'Test'),
    ]:
        for sev in ['L', 'M', 'H', 'No Severity']:
            label_map = {'L': 'Low (L)', 'M': 'Medium (M)', 'H': 'High (H)', 'No Severity': 'No Severity'}
            handles.append(Patch(facecolor=colors[sev], label=f'{prefix} - {label_map[sev]}'))
    return handles


def add_standard_legend(ax, handles, loc='lower right', ncol=2):
    """Add legend with standard styling."""
    ax.legend(
        handles=handles,
        title='Split & Severity',
        loc=loc,
        ncol=ncol,
        frameon=True,
        facecolor='white',
        edgecolor='#cccccc',
        framealpha=0.95
    )


# ─── Main plot function ────────────────────────────────────────────

def plot_distress_distribution(train_df, test_df, features, figsize=(30/2.54, 25/2.54)):
    """
    Create grouped + stacked horizontal bar plot showing distress distribution
    for train and test splits.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with distress columns and PCI.
    test_df : pd.DataFrame
        Test data with distress columns and PCI.
    features : list
        List of feature column names (59 features after engineering).
    figsize : tuple, optional
        Figure size in inches (width, height).

    Returns
    -------
    str
        Base64 encoded PNG image for display in notebook.
    """
    # Prepare data
    distress_cols = filter_distress_cols(features)
    train_piv = prepare_severity_counts(train_df, distress_cols)
    test_piv = prepare_severity_counts(test_df, distress_cols)
    train_piv, test_piv, order = sort_distress_by_total(train_piv, test_piv)

    # Colors and spacing
    train_colors = get_train_colors()
    test_colors = get_test_colors()
    severity_order = get_severity_order()

    bar_height = 0.36
    offset = 0.20
    inter_feature_gap = 0.10
    y, y_step = compute_y_positions(len(order), bar_height, offset, inter_feature_gap)

    # Create figure
    fig, ax = setup_figure(figsize)

    # Plot train and test bars
    for piv, y_pos, color_map in [
        (train_piv, y + offset, train_colors),
        (test_piv,  y - offset, test_colors),
    ]:
        left = np.zeros(len(order))
        for sev in severity_order:
            vals = piv[sev].to_numpy()
            ax.barh(
                y_pos,
                vals,
                left=left,
                height=bar_height,
                color=color_map[sev],
                edgecolor='white',
                linewidth=0.7
            )
            left += vals

        annotate_bar_counts(ax, piv, y_pos)

    # Axes and legend
    max_total = max(train_piv.sum(axis=1).max(), test_piv.sum(axis=1).max())
    ax.set_xlim(0, max_total * 1.22)
    ax.margins(x=0, y=0)

    set_y_axis(ax, y, order, y_step)

    legend_handles = build_split_severity_legend(train_colors, test_colors)
    add_standard_legend(ax, legend_handles)

    ax.set_xlabel('Jumlah Kerusakan', fontsize=11)
    ax.set_ylabel('Jenis Kerusakan', fontsize=11)
    ax.set_title('Distribusi Jenis Kerusakan', fontsize=14, pad=14)

    return finalize_figure(fig, ax)


def show_distress_distribution(train_df, test_df, features, figsize=(30/2.54, 25/2.54)):
    """
    Display the distress distribution plot in a notebook.
    """
    img_base64 = plot_distress_distribution(train_df, test_df, features, figsize)
    show_base64_image(img_base64)