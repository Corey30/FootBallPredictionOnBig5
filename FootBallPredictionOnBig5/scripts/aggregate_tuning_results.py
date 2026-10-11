"""
Aggregate per-(league, task, model) tuning results into paper-Table-6
style summaries.

Reads:
    outputs/reports/hyperparameter_tuning/<league>_<task>_<model>__best_params.json

Writes:
    outputs/reports/hyperparameter_tuning/summary/
        all_results.csv                    (one row per (league, task, model))
        tuned_<task>_<league>.csv          (Table-6 layout per league/task)
        tuned_<task>_<league>.md           (markdown version of the above)

Table layout (matches Atta Mills et al. 2024, Table 6):
    Model | Normalizer | Sampler | Acc | F1 | Prec | Rec | Best params
"""

import json
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / 'outputs' / 'reports' / 'hyperparameter_tuning'
OUT_DIR = IN_DIR / 'summary'

MODEL_DISPLAY = {
    'logistic_regression': 'Logistic Regression',
    'xgboost':             'XGBoost',
    'random_forest':       'Random Forest',
    'svm':                 'SVM',
    'naive_bayes':         'Naive Bayes',
    'fnn':                 'FNN',
}

SAMPLER_DISPLAY = {
    'passthrough':        'None',
    'RandomOverSampler':  'ROS',
    'SVMSMOTE':           'SVM-SMOTE',
    'SMOTE':              'SMOTE-NN',
    'NearMiss':           'NM',
    'SMOTETomek':         'SMOTE-Tomek',
}

SCALER_DISPLAY = {
    'passthrough':    'None',
    'StandardScaler': 'Std.',
    'MinMaxScaler':   'Min-Max',
}

MODEL_ORDER = ['logistic_regression', 'xgboost', 'random_forest',
               'svm', 'naive_bayes', 'fnn']
LEAGUE_ORDER = ['England', 'Spain', 'Germany', 'Italy', 'France']


def collect_records():
    rows = []
    for path in sorted(IN_DIR.glob('*__best_params.json')):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        m = data.get('best_metrics', {})
        bp = data.get('best_params', {})
        rows.append({
            'league': data.get('league'),
            'task':   data.get('task'),
            'model':  data.get('model'),
            'normalizer_raw': bp.get('scaler'),
            'sampler_raw':    bp.get('sampler'),
            'normalizer': SCALER_DISPLAY.get(bp.get('scaler'), bp.get('scaler')),
            'sampler':    SAMPLER_DISPLAY.get(bp.get('sampler'), bp.get('sampler')),
            'accuracy':         m.get('accuracy'),
            'f1_macro':         m.get('f1_macro'),
            'precision_macro':  m.get('precision_macro'),
            'recall_macro':     m.get('recall_macro'),
            'best_params': {k: v for k, v in bp.items()
                            if k not in ('scaler', 'sampler')},
            'n_candidates':    data.get('n_candidates'),
            'elapsed_seconds': data.get('elapsed_seconds'),
            'refit_on':        data.get('refit_on'),
            'cv_splits':       data.get('cv_splits'),
            'samplers_searched': data.get('samplers_searched'),
            'scalers_searched':  data.get('scalers_searched'),
        })
    return rows


def fmt_score(x):
    return '' if x is None else f'{x:.2f}'


def render_league_task_table(records, league, task):
    sub = [r for r in records if r['league'] == league.lower()
           and r['task'] == task]
    by_model = {r['model']: r for r in sub}

    rows = []
    for m in MODEL_ORDER:
        r = by_model.get(m)
        if r is None:
            rows.append({
                'Model': MODEL_DISPLAY[m],
                'Normalizer': '-', 'Sampler': '-',
                'Acc': '-', 'F1': '-', 'Prec': '-', 'Rec': '-',
                'Best params': '(missing)',
            })
        else:
            params_str = ', '.join(
                f"{k.replace('clf__', '').replace('estimator__', '')}={v}"
                for k, v in r['best_params'].items()
            )
            rows.append({
                'Model': MODEL_DISPLAY[m],
                'Normalizer': r['normalizer'],
                'Sampler': r['sampler'],
                'Acc':  fmt_score(r['accuracy']),
                'F1':   fmt_score(r['f1_macro']),
                'Prec': fmt_score(r['precision_macro']),
                'Rec':  fmt_score(r['recall_macro']),
                'Best params': params_str,
            })
    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame) -> str:
    """Plain-Python markdown writer (avoids the optional tabulate dep)."""
    cols = list(df.columns)
    out = ['| ' + ' | '.join(cols) + ' |',
           '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for _, row in df.iterrows():
        out.append('| ' + ' | '.join(str(row[c]) for c in cols) + ' |')
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', default=str(IN_DIR))
    parser.add_argument('--out-dir', default=str(OUT_DIR))
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not any(in_dir.glob('*__best_params.json')):
        print(f"No best_params.json files found in {in_dir}.")
        return

    records = collect_records()
    df = pd.DataFrame(records)
    df = df.sort_values(['league', 'task', 'model']).reset_index(drop=True)
    df_path = out_dir / 'all_results.csv'
    df.drop(columns=['best_params']).to_csv(df_path, index=False)
    print(f"All results -> {df_path}  ({len(df)} rows)")

    for task in ['prematch', 'halftime']:
        for league in LEAGUE_ORDER:
            tbl = render_league_task_table(records, league, task)
            csv_path = out_dir / f"tuned_{task}_{league.lower()}.csv"
            md_path = out_dir / f"tuned_{task}_{league.lower()}.md"
            tbl.to_csv(csv_path, index=False)
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# Tuned {task} performance --- {league}\n\n")
                f.write(df_to_markdown(tbl))
                f.write("\n")
            print(f"  {task:9s} | {league:7s} -> {csv_path.name}")


if __name__ == '__main__':
    main()
