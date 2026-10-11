import pandas as pd

try:
    from src.config.config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO
except ImportError:
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.15, 0.15

TRAIN_SEASONS = [
    "2015-2016",
    "2016-2017",
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
]

VAL_SEASONS = [
    "2021-2022",
    "2022-2023",
]

TEST_SEASONS = [
    "2023-2024",
    "2024-2025",
    
]


def season_based_split(df):
    if 'season' not in df.columns:
        raise ValueError("DataFrame must have 'season' column for season-based split")
    
    train_df = df[df['season'].isin(TRAIN_SEASONS)].copy()
    val_df = df[df['season'].isin(VAL_SEASONS)].copy()
    test_df = df[df['season'].isin(TEST_SEASONS)].copy()
    
    return train_df, val_df, test_df


# Split data into time-ordered train, validation, and test sets
def time_series_split(df, date_col='date_GMT', train_ratio=None, val_ratio=None, test_ratio=None):
    if 'season' in df.columns:
        return season_based_split(df)
    
    if train_ratio is None:
        train_ratio = TRAIN_RATIO
    if val_ratio is None:
        val_ratio = VAL_RATIO
    if test_ratio is None:
        test_ratio = TEST_RATIO
    
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

def season_cv_split(df, n_splits=5):
    """
    赛季交叉验证划分
    
    每次留出一个赛季作为验证集，之前的所有赛季作为训练集
    保持时间顺序，避免数据泄露
    
    Args:
        df: 包含 'season' 列的 DataFrame
        n_splits: 交叉验证折数（最多使用最后 n_splits 个训练赛季）
    
    Returns:
        list of (train_df, val_df) tuples
    """
    if 'season' not in df.columns:
        raise ValueError("DataFrame must have 'season' column for season CV")
    
    all_seasons = sorted(df['season'].unique())
    train_seasons_list = [s for s in all_seasons if s in TRAIN_SEASONS]
    
    if len(train_seasons_list) < 2:
        raise ValueError(f"Need at least 2 training seasons for CV, got {len(train_seasons_list)}")
    
    n_splits = min(n_splits, len(train_seasons_list) - 1)
    
    splits = []
    for i in range(n_splits):
        val_season = train_seasons_list[-(i + 1)]
        train_seasons = train_seasons_list[:-(i + 1)]
        
        if len(train_seasons) == 0:
            continue
        
        train_df = df[df['season'].isin(train_seasons)].copy()
        val_df = df[df['season'] == val_season].copy()
        
        splits.append((train_df, val_df, train_seasons, val_season))
    
    return splits
