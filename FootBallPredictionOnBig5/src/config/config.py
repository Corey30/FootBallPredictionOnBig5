import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / 'data'
MERGED_DATA_DIR = BASE_DIR / 'merged_data'
RAW_DATA_PATH = MERGED_DATA_DIR / 'feature_engineer5' / 'England.csv'
OUTPUT_DIR = BASE_DIR / 'outputs'
MODEL_DIR = OUTPUT_DIR / 'models'
REPORT_DIR = OUTPUT_DIR / 'reports'

HALFTIME_COLUMNS = [
    'home_team_goal_count_half_time',
    'away_team_goal_count_half_time',
    'total_goals_at_half_time',
    'ht_result',
    'home_team_first_half_cards',
    'away_team_first_half_cards',
    'half_time_lead',

]

IDENTIFIER_COLUMNS = [
    # Raw IDs, timestamps, and metadata

    'match_id',
    'timestamp',
    'date_GMT',
    'match_datetime',
    'match_date',
    'season',
    'status',

    # Text and categorical identifiers; usually exclude them from the model
    'stadium_name',
    'referee',
    'home_team_name',
    'away_team_name',
    'Game Week',
    'league',

    # Helper columns for merging and cleaning
    'row_id',
    '_main_row_id',
    'league_key',
    'home_team_key',
    'away_team_key',
    'date_offset_used',

    # Odds source strings; exclude unless explicitly encoded
    'odds_open_source',
    'odds_close_source',

    # Usually the actual post-match attendance; not recommended as a half-time feature
    'attendance',
]

TARGET_COLUMNS = [
    'target_match_result',
    'target_home_team_goal_count',
    'target_away_team_goal_count',
    'target_total_goal_count'
]

TARGET_COLUMN = 'target_match_result'

LABEL_ENCODING = {'H': 0, 'D': 1, 'A': 2}
LABEL_DECODING = {0: 'H', 1: 'D', 2: 'A'}

RANDOM_SEED = 42

EXCLUDE_SEASONS = ['2025-2026']

MODEL_DISPLAY_NAMES = {
    'random_forest': 'Random Forest',
    'logistic_regression': 'Logistic Regression',
    'xgboost': 'XGBoost',
    'naive_bayes': 'Naive Bayes',
    'svm': 'SVM',
    'fnn': 'Feedforward NN',
    'vanilla_rnn': 'Vanilla RNN',
    'voting_ensemble': 'Voting Ensemble',
    'stacking_ensemble': 'Stacking Ensemble',
}

MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 500,
        'max_depth': 6,
        'min_samples_split': 20,
        'min_samples_leaf': 8,
        'max_features': 0.6,
        'random_state': 42,
        'class_weight': 'balanced',
        'n_jobs': -1,
    },
    'logistic_regression': {
        'C': 0.5,
        'random_state': 42,
        'class_weight': {0: 1.0, 1: 1.5, 2: 1.0},
        'solver': 'lbfgs',
        'max_iter': 1500,
    },
    'xgboost': {
        'n_estimators': 300,
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'reg_lambda': 2.0,
        'reg_alpha': 0.3,
        'min_child_weight': 5,
        'gamma': 0.05,
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'early_stopping_rounds': 50,
    },
    'naive_bayes': {
        'var_smoothing': 1e-8,
    },
    'svm': {
        'C': 1.0,
        'max_iter': 10000,
        'random_state': 42,
        'class_weight': 'balanced',
        'dual': 'auto',
    },
    'fnn': {
        'hidden_layer_sizes': (64, 32),
        'activation': 'relu',
        'solver': 'adam',
        'learning_rate_init': 0.0005,
        'max_iter': 500,
        'random_state': 42,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 20,
        'alpha': 0.01,
        'batch_size': 64,
        'learning_rate': 'adaptive',
    },
    'vanilla_rnn': {
        'rnn_units': 48,
        'dense_units': 24,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'epochs': 80,
        'batch_size': 64,
    },
    'voting_ensemble': {
        'base_models': ['xgboost', 'random_forest'],
        'voting': 'soft',
        'weights': None,
    },
    'stacking_ensemble': {
        'base_models': ['random_forest', 'logistic_regression', 'svm'],
        'meta_model': 'logistic_regression',
        'cv': 5,
    },
}

GRID_SEARCH_PARAMS = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 8, 12],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']
    },
    'logistic_regression': {
        'C': [0.01, 0.1, 1, 3, 10],
        'class_weight': [None, 'balanced'],
        'solver': ['lbfgs']
    },
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    },
    'naive_bayes': {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    },
    'svm': {
        'C': [0.1, 1, 10],
        'class_weight': [None, 'balanced']
    },
    'fnn': {
        'hidden_layer_sizes': [(64,), (128, 64), (256, 128, 64)],
        'activation': ['relu', 'tanh'],
        'learning_rate_init': [0.001, 0.01]
    }
}


