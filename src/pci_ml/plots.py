from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pci_ml.data import DataSplit
from pci_ml.model import score_prediction

ROOT_DIR = Path(__file__).resolve().parents[2]

# Path gambar output untuk satu kombinasi dataset dan parameter split
def figure_path(name, params):
  return ROOT_DIR / 'outputs' / 'figures' / DataSplit.make_tag(params) / name

# Plot distribusi kerusakan pada data train dan test
def plot_data_dist(train_df, test_df, features, save_path=None):
  sevs = ['L', 'M', 'H', 'No Severity']
  labels = ['Low (L)', 'Medium (M)', 'High (H)', 'No Severity']
  colors = ['#a1c9f4', '#6ca6cd', '#3b7ea1', sns.color_palette('pastel')[1]]
  cols = [c for c in features if not c.startswith('TOTAL_')]

  def get_pivot(df):
    vals = (df[cols] > 0).sum()
    feat = vals.index.str.replace(r'_[LMH]$', '', regex=True)
    sev = vals.index.str.extract(r'_([LMH])$')[0].fillna('No Severity')
    return (vals.set_axis(pd.MultiIndex.from_arrays([feat, sev]))
            .unstack(fill_value=0).reindex(columns=sevs, fill_value=0))

  train, test = get_pivot(train_df), get_pivot(test_df)
  order = train.sum(axis=1).sort_values(ascending=False).index
  x = np.arange(len(order))

  sns.set_theme(style='whitegrid')
  fig, ax = plt.subplots(figsize=(16, 6.5))

  for df, shift, edge, hatch in [(train.loc[order], -0.205, 'white', None),
                                 (test.loc[order], 0.205, '#444444', '///')]:
    bottom = 0
    for s, lbl, c in zip(sevs, labels, colors):
      ax.bar(x + shift, df[s], 0.38, bottom=bottom, color=c, edgecolor=edge,
             linewidth=0.6, hatch=hatch, label=lbl if shift < 0 else None)
      bottom = bottom + df[s].values

  ax.legend(loc='upper right', handles=ax.get_legend_handles_labels()[0] + [
      mpatches.Patch(facecolor='#d3d3d3', edgecolor='white', label='Train (Kiri)'),
      mpatches.Patch(facecolor='#d3d3d3', edgecolor='#444444', hatch='///',
                     label='Test (Kanan)')])

  ax.set_xticks(x, order, rotation=45, ha='right')
  ax.set(xlabel='Jenis Kerusakan', ylabel='Jumlah Segmen',
         title='Distribusi Jenis Kerusakan (Train vs Test)')
  plt.tight_layout()

  if save_path is not None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')

  plt.show()

# Plot performa cross-validation out-of-fold: metrik per fold dan scatter OOF
def plot_cv_oof(actual, predicted, folds, model_name='Random Forest', save_path=None):
  actual, predicted = np.asarray(actual), np.asarray(predicted)
  fold_ids = np.empty(len(actual), dtype=int)
  records = []

  for i, (_, valid_idx) in enumerate(folds, start=1):
    valid_idx = np.asarray(valid_idx)
    fold_ids[valid_idx] = i
    score = score_prediction(actual[valid_idx], predicted[valid_idx])
    records.append({'fold': f'Fold {i}', 'n': len(valid_idx), **score})

  metrics = pd.DataFrame(records)
  overall = score_prediction(actual, predicted)
  fold_order = metrics['fold'].tolist()

  sns.set_theme(style='whitegrid')
  palette = sns.color_palette('deep', n_colors=len(fold_order))
  fig, (ax_fold, ax_oof) = plt.subplots(1, 2, figsize=(16, 6.5))

  # Panel kiri: RMSE dan MAE tiap fold beserta rata-ratanya
  metric_long = metrics.melt(
      id_vars=['fold', 'n'], value_vars=['rmse', 'mae'],
      var_name='Metric', value_name='Value')
  metric_long['Metric'] = metric_long['Metric'].str.upper()
  metric_colors = {'RMSE': '#3b7ea1', 'MAE': sns.color_palette('pastel')[1]}

  sns.barplot(data=metric_long, x='fold', y='Value', hue='Metric',
              order=fold_order, hue_order=['RMSE', 'MAE'],
              palette=metric_colors, errorbar=None, edgecolor='white',
              linewidth=0.6, ax=ax_fold)

  def format_value(value):
    return f'{value:.2f}'.rstrip('0').rstrip('.')

  for container in ax_fold.containers:
    ax_fold.bar_label(container,
                      labels=[format_value(v) for v in container.datavalues],
                      padding=2, fontsize=9)

  mean_rmse = metrics['rmse'].mean()
  mean_mae = metrics['mae'].mean()
  ax_fold.axhline(mean_rmse, color=metric_colors['RMSE'], linestyle='--',
                  linewidth=1.5, label='Mean Fold RMSE')
  ax_fold.axhline(mean_mae, color=metric_colors['MAE'], linestyle='--',
                  linewidth=1.5, label='Mean Fold MAE')
  ax_fold.set_xticks(range(len(metrics)),
                     [f'{row.fold}\n(n={row.n})' for row in metrics.itertuples()])
  ax_fold.set(xlabel='', ylabel='Error Value',
              title=f'{model_name} — Performa RMSE & MAE per Fold')
  ax_fold.set_ylim(0, metric_long['Value'].max() * 1.18)
  ax_fold.legend(loc='upper right', title=None)

  # Panel kanan: prediksi OOF seluruh data, diwarnai per fold
  scatter = pd.DataFrame({
      'actual': actual,
      'predicted': predicted,
      'fold': [f'Fold {i}' for i in fold_ids],
  })
  sns.scatterplot(data=scatter, x='actual', y='predicted', hue='fold',
                  hue_order=fold_order, palette=palette, s=40, alpha=0.8,
                  edgecolor='white', linewidth=0.3, ax=ax_oof)

  high = max(actual.max(), predicted.max()) * 1.05
  ax_oof.plot([0, high], [0, high], color='black', linestyle='--',
              linewidth=1.2, label='Ideal (y = ŷ)')

  metric_text = (f'Model: {model_name}\n'
                 f'Overall OOF Metrics (N={len(actual)}):\n'
                 f'• RMSE: {overall["rmse"]:.3f}\n'
                 f'• MAE: {overall["mae"]:.3f}\n'
                 f'• R²: {overall["r2"]:.3f}')
  ax_oof.text(0.03, 0.97, metric_text, transform=ax_oof.transAxes,
              ha='left', va='top',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#f2f2f2',
                        edgecolor='#cccccc', alpha=0.95))

  ax_oof.set(xlim=(0, high), ylim=(0, high),
             xlabel='Nilai Aktual (y)', ylabel='Nilai Prediksi OOF (ŷ)',
             title='Prediksi Out-of-Fold Keseluruhan (OOF)')
  ax_oof.set_aspect('equal', adjustable='box')
  ax_oof.set_anchor('W')
  ax_oof.set_ylabel('Nilai Prediksi OOF (ŷ)', labelpad=0)
  ax_oof.tick_params(axis='y', pad=1, labelsize=9)
  ax_oof.legend(loc='lower right', title=None)

  fig.suptitle(f'{model_name} — GroupKFold Out-of-Fold (OOF) Performance',
               fontsize=16)
  fig.tight_layout(rect=(0, 0, 1, 0.95))
  fig.subplots_adjust(wspace=0.09)

  if save_path is not None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')

  plt.show()
