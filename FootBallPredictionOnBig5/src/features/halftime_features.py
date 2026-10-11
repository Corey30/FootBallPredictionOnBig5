import pandas as pd
import numpy as np
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.config import (
    HALFTIME_COLUMNS, IDENTIFIER_COLUMNS, TARGET_COLUMNS, TARGET_COLUMN,
    LABEL_ENCODING
)
from src.features.prematch_features import (
    handle_missing_values, get_feature_names
)


def get_halftime_feature_columns(df):
    """
    获取半场预测的特征列

    包括：
    1. 所有赛前特征（历史数据、排名、对战记录等）
    2. 半场实时数据（半场比分、黄牌等）

    排除：
    1. 标识符列（球队名、日期等）
    2. 目标列（全场结果）
    3. 下半场数据（数据泄露）
    """
    exclude_cols = set(IDENTIFIER_COLUMNS + TARGET_COLUMNS)


    feature_cols = [col for col in df.columns if col not in exclude_cols]


    # Ensure half-time features are included when present
    halftime_features_present = [col for col in HALFTIME_COLUMNS if col in df.columns]


    if not hasattr(get_halftime_feature_columns, '_printed'):
        print(f"\n[半场特征检查]")
        print(f"  定义的半场特征数: {len(HALFTIME_COLUMNS)}")
        print(f"  实际存在的半场特征数: {len(halftime_features_present)}")
        if halftime_features_present:
            print(f"  半场特征列表:")
            for col in halftime_features_present:
                print(f"    - {col}")
        else:
            print(f"  [!] 警告：数据中没有找到任何半场特征！")
        get_halftime_feature_columns._printed = True

    return feature_cols


def build_halftime_features(df):
    df = df.copy()
    feature_cols = get_halftime_feature_columns(df)
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy() if TARGET_COLUMN in df.columns else None
    return X, y


def encode_ht_result(X):
    """编码半场结果"""
    X = X.copy()
    if 'ht_result' in X.columns:
        X['ht_result_encoded'] = X['ht_result'].map(LABEL_ENCODING)
        X = X.drop(columns=['ht_result'])
        print(f"  [OK] 半场结果(ht_result)已编码为数值特征")
    return X


def process_halftime_features(df):
    X, y = build_halftime_features(df)
    X = encode_ht_result(X)
    X = handle_missing_values(X)
    return X, y
