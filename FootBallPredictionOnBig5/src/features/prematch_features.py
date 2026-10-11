import pandas as pd
import numpy as np
from src.config.config import (
    HALFTIME_COLUMNS, IDENTIFIER_COLUMNS, TARGET_COLUMNS, TARGET_COLUMN
)


def get_prematch_feature_columns(df):
    exclude_cols = set(HALFTIME_COLUMNS + IDENTIFIER_COLUMNS + TARGET_COLUMNS)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def build_prematch_features(df):
    df = df.copy()
    feature_cols = get_prematch_feature_columns(df)
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy() if TARGET_COLUMN in df.columns else None
    return X, y


def handle_missing_values(X, strategy='smart'):
    """
    智能缺失值处理策略

    注意：此函数假设数据泄露列已在数据清洗阶段被删除

    Args:
        X: 特征DataFrame（应该已经排除了IDENTIFIER_COLUMNS和TARGET_COLUMNS）
        strategy: 'smart' (智能分类处理), 'median', 'mean', 'zero'

    Returns:
        处理后的DataFrame
    """
    X = X.copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    if strategy == 'smart':

        # Create missingness indicators and process feature groups.
        important_features = []


        odds_patterns = ['odds_ft_', 'odds_open_', 'odds_close_', 'odds_btts_',
                        'implied_prob_', 'overround_', 'odds_move_']

        history_patterns = ['Total home win rate', 'Total away win rate',
                           'Total home draw rate', 'Total away draw rate',
                           'Total home loss rate', 'Total away loss rate']

        performance_patterns = ['ppg_differential', 'xg_power_index', 'home_ppg', 'away_ppg',
                               'Pre-Match PPG']

        goal_patterns = ['Home goals forward', 'Away goals forward',
                        'Home goal differential', 'Away goal differential',
                        'goal_diff_', 'xg_diff_']


        for col in X.columns:
            if (any(pattern in col for pattern in odds_patterns) or
                any(pattern in col for pattern in history_patterns) or
                any(pattern in col for pattern in performance_patterns) or
                any(pattern in col for pattern in goal_patterns)):
                important_features.append(col)


        for col in important_features:
            if X[col].isna().sum() > 0:
                X[f'{col}_is_missing'] = X[col].isna().astype(int)




        history_cols = []
        for col in X.columns:
            if any(keyword in col for keyword in [
                'Total home win rate', 'Total away win rate',
                'Total home draw rate', 'Total away draw rate',
                'Total home loss rate', 'Total away loss rate',
                'Home wins in the window', 'Away wins in the window',
                'Home goals forward', 'Away goals forward',
                'Home goals against', 'Away goals against',
                'Home goal differential', 'Away goal differential',
                'Home wins margin goal', 'Away wins margin goal',
                'Home losses margin goal', 'Away losses margin goal',
                'goals forward per match', 'goals against per match',
                'goal differential per match'
            ]):
                history_cols.append(col)

        for col in history_cols:
            if col in X.columns:
                X[col] = X[col].fillna(0)


        odds_cols = []
        for col in X.columns:
            if any(keyword in col for keyword in [
                'odds_ft_', 'odds_open_', 'odds_close_', 'odds_btts_',
                'implied_prob_', 'overround_'
            ]):
                odds_cols.append(col)

        for col in odds_cols:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())


        momentum_cols = []
        for col in X.columns:
            if any(keyword in col for keyword in [
                'home_recent_', 'away_recent_',
                'home_match_density', 'away_match_density',
                'form_pts_', 'mom_slope', 'mom_volatility',
                'venue_form_', 'odds_move_', 'implied_prob_shift_',
                'opp_adj_form_', '_ewm_', '_mom_'
            ]):
                momentum_cols.append(col)

        for col in momentum_cols:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].mean())


        rest_cols = []
        for col in X.columns:
            if any(keyword in col for keyword in [
                'rest_days', 'travel_', 'fatigue_'
            ]):
                rest_cols.append(col)

        for col in rest_cols:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())


        performance_cols = []
        for col in X.columns:
            if any(keyword in col for keyword in [
                'ppg_differential', 'xg_power_index', 'home_ppg', 'away_ppg',
                'Pre-Match PPG', 'xg_diff_'
            ]):
                performance_cols.append(col)

        for col in performance_cols:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())


        processed_cols = set(history_cols + odds_cols + momentum_cols + rest_cols + performance_cols)
        remaining_cols = [col for col in X.columns
                         if col not in processed_cols and not col.endswith('_is_missing')]

        for col in remaining_cols:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())


        X = X.fillna(0)

    elif strategy == 'median':
        X = X.fillna(X.median())
        X = X.fillna(0)
    elif strategy == 'mean':
        X = X.fillna(X.mean())
        X = X.fillna(0)
    elif strategy == 'zero':
        X = X.fillna(0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return X


def encode_labels(y):
    from src.config.config import LABEL_ENCODING
    return y.map(LABEL_ENCODING)


def decode_labels(y_encoded):
    from src.config.config import LABEL_DECODING
    if isinstance(y_encoded, pd.Series):
        return y_encoded.map(LABEL_DECODING)
    return pd.Series(y_encoded).map(LABEL_DECODING)


def get_feature_names(X):
    return X.columns.tolist()


def process_prematch_features(df):
    X, y = build_prematch_features(df)
    X = handle_missing_values(X)
    return X, y
