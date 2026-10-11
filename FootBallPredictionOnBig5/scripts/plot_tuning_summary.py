"""
Generate per-task model-comparison bar charts from the tuning summary CSVs.

Produces:
    outputs/reports/hyperparameter_tuning/summary/fig_prematch_acc_by_league.png
    outputs/reports/hyperparameter_tuning/summary/fig_prematch_f1_by_league.png
    outputs/reports/hyperparameter_tuning/summary/fig_halftime_acc_by_league.png
    outputs/reports/hyperparameter_tuning/summary/fig_halftime_f1_by_league.png
    outputs/reports/hyperparameter_tuning/summary/fig_prevshalf_uplift.png
"""

import sys  # noqa: F401
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_DIR = ROOT / 'outputs' / 'reports' / 'hyperparameter_tuning' / 'summary'

LEAGUES = ['england', 'spain', 'germany', 'italy', 'france']
LEAGUE_LABELS = {
    'england': 'Premier League',
    'spain':   'La Liga',
    'germany': 'Bundesliga',
    'italy':   'Serie A',
    'france':  'Ligue 1',
}
MODEL_ORDER = ['Logistic Regression', 'XGBoost', 'Random Forest',
               'SVM', 'Naive Bayes', 'FNN']
MODEL_SHORT = {
    'Logistic Regression': 'LR',
    'XGBoost':             'XGB',
    'Random Forest':       'RF',
    'SVM':                 'SVM',
    'Naive Bayes':         'NB',
    'FNN':                 'FNN',
}


def load_task(task):
    """Return (n_models, n_leagues) arrays of Acc and F1 in MODEL_ORDER x LEAGUES."""
    acc = np.full((len(MODEL_ORDER), len(LEAGUES)), np.nan)
    f1 = np.full((len(MODEL_ORDER), len(LEAGUES)), np.nan)
    for j, lg in enumerate(LEAGUES):
        df = pd.read_csv(SUMMARY_DIR / f'tuned_{task}_{lg}.csv')
        df = df.set_index('Model')
        for i, m in enumerate(MODEL_ORDER):
            if m in df.index:
                acc[i, j] = float(df.loc[m, 'Acc'])
                f1[i, j] = float(df.loc[m, 'F1'])
    return acc, f1


def _grouped_bar(ax, values, metric_label, task_label):
    n_models, n_leagues = values.shape
    x = np.arange(n_leagues)
    width = 0.13
    colors = plt.get_cmap('tab10')
    for i, m in enumerate(MODEL_ORDER):
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, values[i], width=width,
                      label=MODEL_SHORT[m], color=colors(i), edgecolor='black', linewidth=0.4)
        for b in bars:
            h = b.get_height()
            if not np.isnan(h):
                ax.text(b.get_x() + b.get_width() / 2, h + 0.005, f"{h:.2f}",
                        ha='center', va='bottom', fontsize=7, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels([LEAGUE_LABELS[lg] for lg in LEAGUES], rotation=15, ha='right')
    ax.set_ylabel(metric_label)
    ax.set_title(f'{task_label}: {metric_label} by model and league')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(ncol=3, fontsize=8, loc='upper center',
              bbox_to_anchor=(0.5, -0.22), frameon=False)


def save_bar(values, metric_label, task_label, filename):
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    _grouped_bar(ax, values, metric_label, task_label)
    fig.tight_layout()
    fig.savefig(SUMMARY_DIR / filename, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"saved -> {filename}")


def save_uplift(pre_acc, ht_acc):
    """Show HT - pre-match accuracy uplift per model, averaged over leagues."""
    mean_pre = np.nanmean(pre_acc, axis=1)
    mean_ht = np.nanmean(ht_acc, axis=1)
    uplift = mean_ht - mean_pre
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(MODEL_ORDER))
    b1 = ax.bar(x - 0.2, mean_pre, width=0.4, label='Pre-match', color='#3b78b5', edgecolor='black', linewidth=0.4)
    b2 = ax.bar(x + 0.2, mean_ht,  width=0.4, label='Half-time', color='#d8703e', edgecolor='black', linewidth=0.4)
    for i, u in enumerate(uplift):
        ax.text(i, max(mean_pre[i], mean_ht[i]) + 0.01,
                f'+{u:.2f}', ha='center', va='bottom', fontsize=9,
                color='green', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER])
    ax.set_ylabel('Accuracy (mean across 5 leagues)')
    ax.set_title('Half-time uplift over pre-match, by model')
    ax.set_ylim(0, 0.75)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend()
    fig.tight_layout()
    fig.savefig(SUMMARY_DIR / 'fig_prevshalf_uplift.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("saved -> fig_prevshalf_uplift.png")


def main():
    pre_acc, pre_f1 = load_task('prematch')
    ht_acc, ht_f1 = load_task('halftime')

    save_bar(pre_acc, 'Accuracy', 'Pre-match', 'fig_prematch_acc_by_league.png')
    save_bar(pre_f1, 'F1-macro',  'Pre-match', 'fig_prematch_f1_by_league.png')
    save_bar(ht_acc, 'Accuracy',  'Half-time', 'fig_halftime_acc_by_league.png')
    save_bar(ht_f1,  'F1-macro',  'Half-time', 'fig_halftime_f1_by_league.png')
    save_uplift(pre_acc, ht_acc)


if __name__ == '__main__':
    main()
