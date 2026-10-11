import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config.config import RAW_DATA_PATH
from data.data_cleaning import DataCleaner


def load_data(filepath=None):
    """加载数据"""
    if filepath is None:
        filepath = RAW_DATA_PATH
    
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Cannot read file {filepath} with any supported encoding")


def preprocess_data(df):
    """数据预处理（已废弃，使用clean_data代替）"""
    df = df.copy()
    if 'date_GMT' in df.columns:
        df['date_GMT'] = pd.to_datetime(df['date_GMT'], format='%b %d %Y - %I:%M%p', errors='coerce')
    df = df.sort_values('date_GMT').reset_index(drop=True)
    return df


def clean_data(df,
               keep_only_complete_status=True,
               drop_columns_missing_rate_gt=0.6):
    """
    数据清洗（使用统一的DataCleaner）

    Args:
        df: 原始数据框
        keep_only_complete_status: 是否只保留已完成的比赛
        drop_columns_missing_rate_gt: 删除缺失率超过此阈值的列

    Returns:
        清洗后的数据框
    """
    cleaner = DataCleaner(
        keep_only_complete_status=keep_only_complete_status,
        drop_columns_missing_rate_gt=drop_columns_missing_rate_gt
    )

    return cleaner.clean_data(df)
