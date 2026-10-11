"""
Hyper-parameter tuning script (paper-aligned, project-tailored).

Selection metric: ACCURACY (matches Atta Mills et al. 2024,
J. Big Data 11:170, "the ultimate metric for evaluation is predicated
on accuracy"). The script reports Accuracy / F1-macro /
Precision-macro / Recall-macro at the best configuration.

Pipeline per fold:
    [optional scaler]  ->  [optional resampler]  ->  [classifier]
The scaler and resampler choices are part of the search space, so a
single run jointly tunes the model, the augmentation method, and the
normaliser.

Cross-validation: 5-fold *season-aware* time-series CV using
src.data.splitter.season_cv_split (no future season leaks into a
validation fold). Resampling is applied inside the imblearn Pipeline,
so each fold's validation set is never resampled.

Outputs:
    outputs/reports/hyperparameter_tuning/
        <league>_<task>_<model>__results.csv
        <league>_<task>_<model>__best_params.json

Usage:
    python scripts/tune_hyperparameters.py --model xgboost --task prematch \\
        --league England --search random --n-iter 60
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, PredefinedSplit,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import (
    SMOTE, SMOTEN, SVMSMOTE, RandomOverSampler,
)
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTETomek

from src.data.loader import load_data, preprocess_data, clean_data
from src.data.splitter import season_cv_split, TRAIN_SEASONS
from src.features.halftime_features import process_halftime_features
from src.features.prematch_features import process_prematch_features
from src.config.config import (
    MODEL_PARAMS, MODEL_DISPLAY_NAMES, RANDOM_SEED,
)


LEAGUE_FILE_MAPPING = {
    'England': 'England.csv',
    'Spain':   'Spain.csv',
    'Germany': 'Germany.csv',
    'Italy':   'Italy.csv',
    'France':  'France.csv',
}

DATA_DIR_DEFAULT = Path('d:/program/SoccerPre/merged_data/feature_engineer5')



# Sampler factory.  Names match paper Table 6 + a SMOTE-Tomek extra.

def make_sampler(name: str):
    if name == 'none':
        return 'passthrough'
    if name == 'random_oversample':
        return RandomOverSampler(random_state=RANDOM_SEED)
    if name == 'svm_smote':
        return SVMSMOTE(random_state=RANDOM_SEED, k_neighbors=5)
    if name == 'smote_nn':
        # SMOTEN expects categorical features; for our continuous
        # numeric features standard SMOTE is the closer match to the
        # paper's "SMOTE-NN" intent (synthesise from nearest neighbours).
        return SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
    if name == 'near_miss':
        return NearMiss(version=3)
    if name == 'smote_tomek':
        return SMOTETomek(random_state=RANDOM_SEED)
    raise ValueError(f"Unknown sampler: {name}")


def make_scaler(name: str):
    if name == 'none':
        return 'passthrough'
    if name == 'standard':
        return StandardScaler()
    if name == 'minmax':
        return MinMaxScaler()
    raise ValueError(f"Unknown scaler: {name}")



# Per-model classifier search spaces.  Keys are namespaced for the
# imblearn Pipeline ("clf__<param>").

CLF_SPACES = {
    'logistic_regression': {
        'clf__C': [0.05, 0.1, 0.3, 0.5, 1.0, 3.0],
        'clf__penalty': ['l2'],
        'clf__solver': ['lbfgs'],
        'clf__class_weight': [None, 'balanced'],
        'clf__max_iter': [3000],
    },
    'xgboost': {
        'clf__n_estimators': [200, 300, 500],
        'clf__max_depth': [4, 6, 8, 10],
        'clf__learning_rate': [0.03, 0.06, 0.1],
        'clf__subsample': [0.7, 0.85, 1.0],
        'clf__colsample_bytree': [0.7, 0.85, 1.0],
        'clf__reg_alpha': [0.0, 0.2, 0.9],
        'clf__reg_lambda': [0.5, 1.0, 1.5],
        'clf__min_child_weight': [1, 3, 7],
    },
    'random_forest': {
        'clf__n_estimators': [300, 600, 1000],
        'clf__max_depth': [None, 12, 24, 46],
        'clf__min_samples_leaf': [1, 2, 4, 9],
        'clf__min_samples_split': [2, 5, 10, 12],
        'clf__max_features': ['sqrt', 'log2'],
        'clf__class_weight': [None, 'balanced'],
    },
    'svm': {
        # CalibratedClassifierCV wraps LinearSVC at attribute "estimator".
        'clf__estimator__C': [0.3, 0.5, 0.7, 1.0, 2.0],
        'clf__estimator__class_weight': [None, 'balanced'],
        'clf__estimator__max_iter': [20000],
    },
    'naive_bayes': {
        'clf__var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
    },
    'fnn': {
        'clf__hidden_layer_sizes': [(64,), (128, 64), (128, 64, 64),
                                    (256, 128, 64)],
        'clf__activation': ['relu', 'tanh'],
        'clf__alpha': [1e-4, 1e-3, 1e-2, 0.1, 0.5],
        'clf__learning_rate_init': [3e-4, 6e-4, 1e-3, 3e-3],
        'clf__batch_size': [32, 64, 128],
        'clf__max_iter': [600],
        'clf__early_stopping': [True],
    },
}

# Default sampler grid per model (from paper Table 6 + adjacent options).
# 'all' below expands to the full set.
DEFAULT_SAMPLERS = {
    'logistic_regression': ['none', 'random_oversample',
                            'svm_smote', 'smote_nn',
                            'near_miss', 'smote_tomek'],
    'xgboost':             ['none', 'random_oversample',
                            'svm_smote', 'smote_nn',
                            'near_miss', 'smote_tomek'],
    'random_forest':       ['none', 'random_oversample',
                            'svm_smote', 'smote_nn',
                            'near_miss', 'smote_tomek'],
    'svm':                 ['none', 'random_oversample',
                            'svm_smote', 'smote_nn',
                            'near_miss', 'smote_tomek'],
    'naive_bayes':         ['none', 'random_oversample',
                            'svm_smote', 'smote_nn',
                            'near_miss', 'smote_tomek'],
    'fnn':                 ['none', 'random_oversample',
                            'svm_smote', 'smote_nn',
                            'near_miss', 'smote_tomek'],
}

# Default scaler grid per model — paper Table 6:
# Std for LR/XGB/RF/SVM/RNN/FNN; MinMax for NB.
DEFAULT_SCALERS = {
    'logistic_regression': ['standard'],
    'xgboost':             ['none'],
    'random_forest':       ['none'],
    'svm':                 ['standard'],
    'naive_bayes':         ['minmax'],
    'fnn':                 ['standard'],
}


def build_estimator(model_type: str):
    """Return a fresh classifier instance with identity defaults; the
    search space overrides relevant fields."""
    if model_type == 'logistic_regression':
        return LogisticRegression(random_state=RANDOM_SEED)
    if model_type == 'random_forest':
        return RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
    if model_type == 'naive_bayes':
        return GaussianNB()
    if model_type == 'fnn':
        return MLPClassifier(random_state=RANDOM_SEED)
    if model_type == 'svm':
        base = LinearSVC(random_state=RANDOM_SEED, dual='auto')
        return CalibratedClassifierCV(base, cv=3)
    if model_type == 'xgboost':
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost not installed. pip install xgboost") from exc
        return xgb.XGBClassifier(
            random_state=RANDOM_SEED,
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_jobs=1,
        )
    raise ValueError(f"Unknown model: {model_type}")


def build_param_grid(model_type, samplers, scalers):
    """Return a list of parameter dicts representing the joint search:
    classifier hyper-params x sampler x scaler.  RandomizedSearchCV
    accepts a list-of-dicts for combinatorial sampling.
    """
    base = CLF_SPACES[model_type]
    grids = []
    for sc in scalers:
        for sm in samplers:
            g = {
                'scaler': [make_scaler(sc)],
                'sampler': [make_sampler(sm)],
            }
            g.update(base)
            grids.append(g)
    return grids


def load_league_frame(league: str, data_dir: Path) -> pd.DataFrame:
    if league not in LEAGUE_FILE_MAPPING:
        raise ValueError(
            f"Unknown league {league}. Choices: {list(LEAGUE_FILE_MAPPING)}"
        )
    df = pd.read_csv(data_dir / LEAGUE_FILE_MAPPING[league], low_memory=False)
    df['league'] = league
    return df


def load_all_leagues(data_dir: Path) -> pd.DataFrame:
    frames = [load_league_frame(lg, data_dir) for lg in LEAGUE_FILE_MAPPING]
    return pd.concat(frames, ignore_index=True)


def build_features(df: pd.DataFrame, task: str):
    df = preprocess_data(df)
    df = clean_data(df)
    if task == 'halftime':
        return df, process_halftime_features
    if task == 'prematch':
        return df, process_prematch_features
    raise ValueError(f"Unknown task {task}; expected prematch | halftime")


def build_predefined_split(df_train: pd.DataFrame, n_splits: int = 5):
    """Build a sklearn PredefinedSplit using season-aware folds from
    src.data.splitter.season_cv_split. Each fold leaves out one of
    the most-recent training seasons as validation.
    """
    splits = season_cv_split(df_train, n_splits=n_splits)
    test_fold = np.full(len(df_train), -1, dtype=int)
    fold_descriptions = []
    df_indexed = df_train.reset_index(drop=True)
    for fold_id, (_train_df, _val_df, train_seasons, val_season) in enumerate(splits):
        val_mask = df_indexed['season'].isin([val_season]).to_numpy()
        test_fold[val_mask] = fold_id
        fold_descriptions.append({
            'fold': fold_id,
            'val_season': val_season,
            'train_seasons': list(train_seasons),
            'n_val': int(val_mask.sum()),
        })
    return test_fold, fold_descriptions


def main():
    parser = argparse.ArgumentParser(
        description=("Hyper-parameter tuning with 5-fold season-aware CV. "
                     "Joint search over classifier params x sampler x "
                     "scaler. Selection metric defaults to accuracy."),
    )
    parser.add_argument('--model', required=True,
                        choices=list(CLF_SPACES.keys()))
    parser.add_argument('--task', required=True,
                        choices=['prematch', 'halftime'])
    parser.add_argument('--league', default='England',
                        help='League name or "all" for pooled five leagues.')
    parser.add_argument('--data-dir', default=str(DATA_DIR_DEFAULT))
    parser.add_argument('--search', default='random',
                        choices=['grid', 'random'])
    parser.add_argument('--n-iter', type=int, default=60,
                        help='Number of samples for RandomizedSearchCV.')
    parser.add_argument('--scoring', default='accuracy',
                        help=('Selection metric (refit). Default "accuracy" '
                              'matches the reference paper. Multi-metric '
                              'scoring is always reported alongside.'))
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--samplers', default='all',
                        help=('Comma-separated subset of {none, '
                              'random_oversample, svm_smote, smote_nn, '
                              'near_miss, smote_tomek}, or "all" for the '
                              'paper-aligned default for this model.'))
    parser.add_argument('--scalers', default='default',
                        help=('Comma-separated subset of '
                              '{none, standard, minmax}, or "default" for '
                              'the paper-aligned default for this model.'))
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip if best_params.json already present.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (Path(__file__).resolve().parent.parent /
                   'outputs' / 'reports' / 'hyperparameter_tuning')
    out_dir.mkdir(parents=True, exist_ok=True)

    league_tag = 'all' if args.league.lower() == 'all' else args.league.lower()
    stem = f"{league_tag}_{args.task}_{args.model}"
    best_path = out_dir / f"{stem}__best_params.json"
    if args.skip_existing and best_path.exists():
        print(f"[SKIP] {best_path} already exists.")
        return

    if args.league.lower() == 'all':
        df = load_all_leagues(data_dir)
    else:
        df = load_league_frame(args.league, data_dir)

    df, build_X_y = build_features(df, args.task)

    df_train = df[df['season'].isin(TRAIN_SEASONS)].copy()
    if df_train.empty:
        raise RuntimeError("No rows fall in TRAIN_SEASONS; check splitter config.")

    X_train, y_train = build_X_y(df_train)

    if hasattr(X_train, 'index'):
        seasons_aligned = df_train.loc[X_train.index, 'season']
    else:
        seasons_aligned = df_train['season']
    df_for_split = pd.DataFrame({'season': seasons_aligned.values})

    test_fold, fold_descriptions = build_predefined_split(
        df_for_split, n_splits=args.cv_splits,
    )
    cv = PredefinedSplit(test_fold=test_fold)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)

    if args.samplers == 'all':
        samplers = DEFAULT_SAMPLERS[args.model]
    else:
        samplers = [s.strip() for s in args.samplers.split(',') if s.strip()]
    if args.scalers == 'default':
        scalers = DEFAULT_SCALERS[args.model]
    else:
        scalers = [s.strip() for s in args.scalers.split(',') if s.strip()]

    pipeline = ImbPipeline(steps=[
        ('scaler', 'passthrough'),
        ('sampler', 'passthrough'),
        ('clf', build_estimator(args.model)),
    ])

    param_grid = build_param_grid(args.model, samplers, scalers)

    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
    }
    refit_metric = args.scoring if args.scoring in scoring else 'accuracy'

    print("\n" + "=" * 70)
    print(f"Tuning  model={args.model}  task={args.task}  league={league_tag}")
    print(f"        search={args.search}  refit-on={refit_metric}")
    print(f"        cv-splits={args.cv_splits}")
    print(f"        scalers={scalers}  samplers={samplers}")
    print(f"        n_train={len(X_train)}  n_features={X_train.shape[1]}")
    print("=" * 70)
    for fd in fold_descriptions:
        print(f"  fold {fd['fold']}: val={fd['val_season']} "
              f"(n_val={fd['n_val']})  train={fd['train_seasons']}")

    if args.search == 'grid':
        search = GridSearchCV(
            pipeline, param_grid, scoring=scoring, refit=refit_metric,
            cv=cv, n_jobs=args.n_jobs, return_train_score=False,
            verbose=1, error_score='raise',
        )
    else:
        search = RandomizedSearchCV(
            pipeline, param_grid, n_iter=args.n_iter, scoring=scoring,
            refit=refit_metric, cv=cv, n_jobs=args.n_jobs,
            return_train_score=False, random_state=RANDOM_SEED,
            verbose=1, error_score='raise',
        )

    t0 = datetime.now()
    search.fit(X_train, y_encoded)
    elapsed = (datetime.now() - t0).total_seconds()

    results = pd.DataFrame(search.cv_results_)
    results_path = out_dir / f"{stem}__results.csv"
    results.to_csv(results_path, index=False)

    best_idx = int(search.best_index_)
    best_row = {
        m: float(results.loc[best_idx, f"mean_test_{m}"])
        for m in scoring
    }
    # Render best params (the values for sampler/scaler are objects;
    # keep their class names instead).
    best_params_serialised = {}
    for k, v in search.best_params_.items():
        if k in ('scaler', 'sampler'):
            best_params_serialised[k] = (
                'passthrough' if v == 'passthrough' else type(v).__name__
            )
        else:
            best_params_serialised[k] = v

    best = {
        'model': args.model,
        'task': args.task,
        'league': league_tag,
        'search': args.search,
        'refit_on': refit_metric,
        'cv_splits': args.cv_splits,
        'samplers_searched': samplers,
        'scalers_searched': scalers,
        'best_score': float(search.best_score_),
        'best_metrics': best_row,
        'best_params': best_params_serialised,
        'best_index': best_idx,
        'elapsed_seconds': elapsed,
        'n_candidates': len(results),
        'fold_descriptions': fold_descriptions,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }
    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump(best, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "-" * 70)
    print(f"Refit on {refit_metric}: {search.best_score_:.4f}")
    print(f"All metrics at best : {best_row}")
    print(f"Best params         : {best_params_serialised}")
    print(f"Candidates tried    : {len(results)}")
    print(f"Elapsed             : {elapsed:.1f}s")
    print(f"Results -> {results_path}")
    print(f"Best    -> {best_path}")


if __name__ == '__main__':
    main()
