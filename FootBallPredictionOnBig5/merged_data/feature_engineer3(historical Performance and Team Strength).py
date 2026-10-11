"""
特征工程第3阶段（合并版）：历史统计 + 对战历史 + 球队状态 + 排名特征

- 所有特征都基于比赛前的历史数据，避免数据泄露
"""

import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path
from math import isclose
import warnings

warnings.filterwarnings("ignore")


BASE_DIR = Path(r"d:/program/SoccerPre/merged_data")

INPUT_DIR = BASE_DIR / "feature_enginner2"
OUTPUT_DIR = BASE_DIR / "feature_engineer3"
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_GLOB = "*feature*.csv"

WINDOW = 5
RESET_EACH_SEASON = False
DROP_HELPER_COLUMNS = True

VALIDATION_MODE = "sample"
VALIDATION_SAMPLE_N = 500
TEST_TOL = 1e-9

H2H_WINDOW = 10
STREAK_WINDOW = 10

# Elo 参数
ELO_INITIAL = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 60.0




REQUIRED_FEATURES = [
    "Away wins in the window",
    "Home wins in the window",
    "Total home win rate",
    "Total away win rate",
    "Total away loss rate",
    "Total away draw rate",
    "Total home draw rate",
    "Total home loss rate",
    "Away goal differential",
    "Away goals forward",
    "Home goal differential",
    "Away goals against",
    "Home goals forward",
    "Home goals against",
    "Home wins margin goal",
    "Away losses margin goal",
    "Away wins margin goal",
    "Home losses margin goal",
]

ENHANCED_TRAINING_FEATURES = [
    "Home goals forward per match",
    "Home goals against per match",
    "Home goal differential per match",
    "Away goals forward per match",
    "Away goals against per match",
    "Away goal differential per match",
]

H2H_FEATURES = [
    'h2h_matches_played',
    'h2h_home_wins',
    'h2h_draws',
    'h2h_away_wins',
    'h2h_home_win_rate',
    'h2h_recent_home_wins',
    'h2h_recent_draws',
    'h2h_recent_away_wins',
]

STREAK_FEATURES = [
    'home_current_streak',
    'away_current_streak',
    'home_unbeaten_streak',
    'away_unbeaten_streak',
    'home_winless_streak',
    'away_winless_streak',
]

RANKING_FEATURES = [
    'home_current_rank',
    'away_current_rank',
    'rank_difference',
    'home_points_before_match',
    'away_points_before_match',
    'points_difference',
]

ELO_FEATURES = [
    'home_elo_pre',
    'away_elo_pre',
    'elo_diff_pre',
]

FINAL_NEW_FEATURES = (
    REQUIRED_FEATURES +
    ENHANCED_TRAINING_FEATURES +
    H2H_FEATURES +
    STREAK_FEATURES +
    RANKING_FEATURES +
    ELO_FEATURES
)

HELPER_COLUMNS = [
    "_orig_idx",
    "_row_order",
    "_home_win", "_home_draw", "_home_loss",
    "_away_win", "_away_draw", "_away_loss",
    "_home_gf", "_home_ga",
    "_away_gf", "_away_ga",
    "_home_goal_diff",
    "_away_goal_diff",
    "_home_win_margin",
    "_away_win_margin",
    "_home_loss_margin",
    "_away_loss_margin",
    "_home_matches_before",
    "_away_matches_before",
    "_home_wins_before",
    "_home_draws_before",
    "_home_losses_before",
    "_away_wins_before",
    "_away_draws_before",
    "_away_losses_before",
]



def normalize_league_key(value) -> str:
    x = str(value).strip().lower()
    if "england" in x or "premier" in x or x == "epl":
        return "england"
    if "spain" in x or "laliga" in x or "la liga" in x or "span" in x:
        return "spain"
    if "germany" in x or "german" in x or "bundesliga" in x:
        return "germany"
    if "italy" in x or "serie" in x:
        return "italy"
    if "france" in x or "ligue" in x:
        return "france"
    return x


def infer_league_from_filename(path: Path) -> str:
    name = path.name.lower()
    if "england" in name:
        return "England"
    if "france" in name:
        return "France"
    if "germany" in name or "german" in name:
        return "Germany"
    if "italy" in name:
        return "Italy"
    if "spain" in name or "span" in name:
        return "Spain"
    return "Unknown"


def detect_col(df: pd.DataFrame, candidates: list):
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def parse_match_datetime(df: pd.DataFrame) -> pd.Series:
    if "match_datetime" in df.columns:
        return pd.to_datetime(df["match_datetime"], errors="coerce", utc=True)
    if "date_GMT" in df.columns:
        return pd.to_datetime(df["date_GMT"], errors="coerce", utc=True)
    if "date_gmt" in df.columns:
        return pd.to_datetime(df["date_gmt"], errors="coerce", utc=True)
    if "timestamp" in df.columns:
        return pd.to_datetime(
            pd.to_numeric(df["timestamp"], errors="coerce"),
            unit="s",
            errors="coerce",
            utc=True
        )
    raise ValueError("Cannot find datetime column")


def result_to_home_points(result: str) -> int:
    if result == "H":
        return 3
    if result == "D":
        return 1
    if result == "A":
        return 0
    return np.nan


def values_equal(a, b, tol=1e-9) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) != pd.isna(b):
        return False
    return isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def safe_mean(series: pd.Series):
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    return s.mean()



def load_and_prepare_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    df = df.copy()

    required_cols = [
        "match_id",
        "timestamp",
        "home_team_name",
        "away_team_name",
        "target_match_result",
        "target_home_team_goal_count",
        "target_away_team_goal_count",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{input_path.name} missing required columns: {missing}")

    if "league" not in df.columns:
        df["league"] = infer_league_from_filename(input_path)

    if "season" not in df.columns:
        raise ValueError(f"{input_path.name} missing season column")

    df["league_key"] = df["league"].map(normalize_league_key)
    df["match_datetime"] = parse_match_datetime(df)

    if df["match_datetime"].isna().any():
        bad_count = int(df["match_datetime"].isna().sum())
        raise ValueError(f"{input_path.name} has {bad_count} rows with unparseable dates")

    cols_to_drop = [c for c in FINAL_NEW_FEATURES + HELPER_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    df["_orig_idx"] = np.arange(len(df))

    sort_cols = ["league_key", "match_datetime"]
    if "timestamp" in df.columns:
        sort_cols.append("timestamp")
    sort_cols.append("_orig_idx")

    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["_row_order"] = np.arange(len(df))

    return df




def add_intermediate_match_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    home_goals = pd.to_numeric(out["target_home_team_goal_count"], errors="coerce")
    away_goals = pd.to_numeric(out["target_away_team_goal_count"], errors="coerce")

    out["_home_win"] = (out["target_match_result"] == "H").astype(int)
    out["_home_draw"] = (out["target_match_result"] == "D").astype(int)
    out["_home_loss"] = (out["target_match_result"] == "A").astype(int)

    out["_away_win"] = (out["target_match_result"] == "A").astype(int)
    out["_away_draw"] = (out["target_match_result"] == "D").astype(int)
    out["_away_loss"] = (out["target_match_result"] == "H").astype(int)

    out["_home_gf"] = home_goals
    out["_home_ga"] = away_goals
    out["_away_gf"] = away_goals
    out["_away_ga"] = home_goals

    out["_home_goal_diff"] = out["_home_gf"] - out["_home_ga"]
    out["_away_goal_diff"] = out["_away_gf"] - out["_away_ga"]

    out["_home_win_margin"] = np.where(
        out["_home_win"] == 1,
        out["_home_goal_diff"],
        np.nan
    )

    out["_away_win_margin"] = np.where(
        out["_away_win"] == 1,
        out["_away_goal_diff"],
        np.nan
    )

    out["_home_loss_margin"] = np.where(
        out["_home_loss"] == 1,
        -out["_home_goal_diff"],
        np.nan
    )

    out["_away_loss_margin"] = np.where(
        out["_away_loss"] == 1,
        -out["_away_goal_diff"],
        np.nan
    )

    return out


# =====================================================
# venue-history 特征
# =====================================================

def get_group_cols(team_col: str) -> list:
    cols = ["league_key", team_col]
    if RESET_EACH_SEASON:
        cols.insert(1, "season")
    return cols


def grouped_shift_rolling_sum(
    df: pd.DataFrame,
    group_cols: list,
    value_col: str,
    window: int
) -> pd.Series:
    return (
        df.groupby(group_cols, sort=False)[value_col]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
        .fillna(0)
    )


def grouped_shift_cumsum(
    df: pd.DataFrame,
    group_cols: list,
    value_col: str
) -> pd.Series:
    return (
        df.groupby(group_cols, sort=False)[value_col]
        .transform(lambda s: s.shift(1).cumsum())
        .fillna(0)
    )


def grouped_shift_expanding_mean(
    df: pd.DataFrame,
    group_cols: list,
    value_col: str
) -> pd.Series:
    return (
        df.groupby(group_cols, sort=False)[value_col]
        .transform(lambda s: s.shift(1).expanding().mean())
    )


def add_venue_history_features(
    df: pd.DataFrame,
    team_col: str,
    prefix: str,
    window: int = 5
) -> pd.DataFrame:
    out = df.copy()

    if prefix == "home":
        win_col = "_home_win"
        draw_col = "_home_draw"
        loss_col = "_home_loss"
        gf_col = "_home_gf"
        ga_col = "_home_ga"
        gd_col = "_home_goal_diff"
        win_margin_col = "_home_win_margin"
        loss_margin_col = "_home_loss_margin"
    elif prefix == "away":
        win_col = "_away_win"
        draw_col = "_away_draw"
        loss_col = "_away_loss"
        gf_col = "_away_gf"
        ga_col = "_away_ga"
        gd_col = "_away_goal_diff"
        win_margin_col = "_away_win_margin"
        loss_margin_col = "_away_loss_margin"
    else:
        raise ValueError("prefix must be 'home' or 'away'")

    group_cols = get_group_cols(team_col)
    g = out.groupby(group_cols, sort=False)

    out[f"_{prefix}_matches_before"] = g.cumcount()

    out[f"{prefix.capitalize()} wins in the window"] = grouped_shift_rolling_sum(
        out, group_cols, win_col, window
    )

    out[f"_{prefix}_wins_before"] = g[win_col].cumsum() - out[win_col]
    out[f"_{prefix}_draws_before"] = g[draw_col].cumsum() - out[draw_col]
    out[f"_{prefix}_losses_before"] = g[loss_col].cumsum() - out[loss_col]

    denom = out[f"_{prefix}_matches_before"].replace(0, np.nan)

    out[f"Total {prefix} win rate"] = out[f"_{prefix}_wins_before"] / denom
    out[f"Total {prefix} draw rate"] = out[f"_{prefix}_draws_before"] / denom
    out[f"Total {prefix} loss rate"] = out[f"_{prefix}_losses_before"] / denom

    out[f"{prefix.capitalize()} goals forward"] = grouped_shift_cumsum(out, group_cols, gf_col)
    out[f"{prefix.capitalize()} goals against"] = grouped_shift_cumsum(out, group_cols, ga_col)
    out[f"{prefix.capitalize()} goal differential"] = grouped_shift_cumsum(out, group_cols, gd_col)

    out[f"{prefix.capitalize()} goals forward per match"] = (
        out[f"{prefix.capitalize()} goals forward"] / denom
    )
    out[f"{prefix.capitalize()} goals against per match"] = (
        out[f"{prefix.capitalize()} goals against"] / denom
    )
    out[f"{prefix.capitalize()} goal differential per match"] = (
        out[f"{prefix.capitalize()} goal differential"] / denom
    )

    out[f"{prefix.capitalize()} wins margin goal"] = grouped_shift_expanding_mean(
        out, group_cols, win_margin_col
    )
    out[f"{prefix.capitalize()} losses margin goal"] = grouped_shift_expanding_mean(
        out, group_cols, loss_margin_col
    )

    return out


# =====================================================
# 对战历史特征（Head-to-Head）
# =====================================================

def add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加对战历史特征

    特征：
    - h2h_matches_played: 历史交锋场次
    - h2h_home_wins: 主队历史交锋胜场
    - h2h_draws: 历史交锋平局
    - h2h_away_wins: 客队历史交锋胜场
    - h2h_home_win_rate: 主队历史交锋胜率
    - h2h_recent_home_wins: 最近N次交锋主队胜场
    - h2h_recent_draws: 最近N次交锋平局
    - h2h_recent_away_wins: 最近N次交锋客队胜场
    """
    df = df.copy()

    df['h2h_matches_played'] = 0
    df['h2h_home_wins'] = 0
    df['h2h_draws'] = 0
    df['h2h_away_wins'] = 0
    df['h2h_home_win_rate'] = 0.0
    df['h2h_recent_home_wins'] = 0
    df['h2h_recent_draws'] = 0
    df['h2h_recent_away_wins'] = 0

    for league in df['league'].unique():
        league_df = df[df['league'] == league].copy()

        for idx in league_df.index:
            row = league_df.loc[idx]
            home_team = row['home_team_name']
            away_team = row['away_team_name']
            current_date = row['match_datetime']

            h2h_history = df[
                (df['league'] == league) &
                (df['match_datetime'] < current_date) &
                (
                    ((df['home_team_name'] == home_team) & (df['away_team_name'] == away_team)) |
                    ((df['home_team_name'] == away_team) & (df['away_team_name'] == home_team))
                )
            ].copy()

            if len(h2h_history) > 0:
                h2h_history['is_current_home_team_home'] = (
                    h2h_history['home_team_name'] == home_team
                )

                h2h_history['current_home_result'] = h2h_history.apply(
                    lambda x: x['target_match_result'] if x['is_current_home_team_home']
                    else ('A' if x['target_match_result'] == 'H'
                          else 'H' if x['target_match_result'] == 'A'
                          else 'D'),
                    axis=1
                )

                total_matches = len(h2h_history)
                home_wins = (h2h_history['current_home_result'] == 'H').sum()
                draws = (h2h_history['current_home_result'] == 'D').sum()
                away_wins = (h2h_history['current_home_result'] == 'A').sum()

                df.loc[idx, 'h2h_matches_played'] = total_matches
                df.loc[idx, 'h2h_home_wins'] = home_wins
                df.loc[idx, 'h2h_draws'] = draws
                df.loc[idx, 'h2h_away_wins'] = away_wins
                df.loc[idx, 'h2h_home_win_rate'] = home_wins / total_matches if total_matches > 0 else 0.0

                recent_h2h = h2h_history.tail(H2H_WINDOW)
                df.loc[idx, 'h2h_recent_home_wins'] = (recent_h2h['current_home_result'] == 'H').sum()
                df.loc[idx, 'h2h_recent_draws'] = (recent_h2h['current_home_result'] == 'D').sum()
                df.loc[idx, 'h2h_recent_away_wins'] = (recent_h2h['current_home_result'] == 'A').sum()

    return df


# =====================================================
# 球队状态特征
# =====================================================

def add_team_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加球队状态特征

    特征：
    - home_current_streak: 主队当前连胜/连败场次（正数=连胜，负数=连败）
    - away_current_streak: 客队当前连胜/连败场次
    - home_unbeaten_streak: 主队不败场次（连续未输）
    - away_unbeaten_streak: 客队不败场次
    - home_winless_streak: 主队不胜场次（连续未赢）
    - away_winless_streak: 客队不胜场次
    """
    df = df.copy()

    df['home_current_streak'] = 0
    df['away_current_streak'] = 0
    df['home_unbeaten_streak'] = 0
    df['away_unbeaten_streak'] = 0
    df['home_winless_streak'] = 0
    df['away_winless_streak'] = 0

    for (league, team) in df.groupby(['league', 'home_team_name']).groups.keys():
        team_matches = df[
            (df['league'] == league) &
            (df['home_team_name'] == team)
        ].sort_values('match_datetime')

        for idx in team_matches.index:
            row = team_matches.loc[idx]
            current_date = row['match_datetime']

            team_history = df[
                (df['league'] == league) &
                (df['home_team_name'] == team) &
                (df['match_datetime'] < current_date)
            ].sort_values('match_datetime').tail(STREAK_WINDOW)

            if len(team_history) > 0:
                results = team_history['target_match_result'].values
                streak = 0
                unbeaten = 0
                winless = 0

                for i in range(len(results) - 1, -1, -1):
                    if results[i] == 'H':
                        if streak >= 0:
                            streak += 1
                        else:
                            break
                        unbeaten += 1
                    elif results[i] == 'D':
                        if streak == 0:
                            unbeaten += 1
                            winless += 1
                        else:
                            break
                    else:
                        if streak <= 0:
                            streak -= 1
                            winless += 1
                        else:
                            break

                df.loc[idx, 'home_current_streak'] = streak
                df.loc[idx, 'home_unbeaten_streak'] = unbeaten
                df.loc[idx, 'home_winless_streak'] = winless

    for (league, team) in df.groupby(['league', 'away_team_name']).groups.keys():
        team_matches = df[
            (df['league'] == league) &
            (df['away_team_name'] == team)
        ].sort_values('match_datetime')

        for idx in team_matches.index:
            row = team_matches.loc[idx]
            current_date = row['match_datetime']

            team_history = df[
                (df['league'] == league) &
                (df['away_team_name'] == team) &
                (df['match_datetime'] < current_date)
            ].sort_values('match_datetime').tail(STREAK_WINDOW)

            if len(team_history) > 0:
                results = team_history['target_match_result'].values
                streak = 0
                unbeaten = 0
                winless = 0

                for i in range(len(results) - 1, -1, -1):
                    if results[i] == 'A':
                        if streak >= 0:
                            streak += 1
                        else:
                            break
                        unbeaten += 1
                    elif results[i] == 'D':
                        if streak == 0:
                            unbeaten += 1
                            winless += 1
                        else:
                            break
                    else:
                        if streak <= 0:
                            streak -= 1
                            winless += 1
                        else:
                            break

                df.loc[idx, 'away_current_streak'] = streak
                df.loc[idx, 'away_unbeaten_streak'] = unbeaten
                df.loc[idx, 'away_winless_streak'] = winless

    return df


# =====================================================
# 排名特征
# =====================================================

def add_ranking_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加排名特征

    特征：
    - home_current_rank: 主队当前排名
    - away_current_rank: 客队当前排名
    - rank_difference: 排名差距（主队排名 - 客队排名，负数表示主队排名更高）
    - home_points_before_match: 主队当前积分
    - away_points_before_match: 客队当前积分
    - points_difference: 积分差距
    """
    df = df.copy()

    df['home_current_rank'] = 0
    df['away_current_rank'] = 0
    df['rank_difference'] = 0
    df['home_points_before_match'] = 0
    df['away_points_before_match'] = 0
    df['points_difference'] = 0

    for (league, season), group in df.groupby(['league', 'season']):
        group = group.sort_values('match_datetime').reset_index(drop=True)

        teams = pd.concat([group['home_team_name'], group['away_team_name']]).unique()
        standings = {team: {'points': 0, 'played': 0} for team in teams}

        for idx in group.index:
            row = group.loc[idx]
            home_team = row['home_team_name']
            away_team = row['away_team_name']

            current_standings = sorted(
                standings.items(),
                key=lambda x: (x[1]['points'], -x[1]['played']),
                reverse=True
            )

            rank_dict = {team: rank + 1 for rank, (team, _) in enumerate(current_standings)}

            home_rank = rank_dict.get(home_team, len(teams))
            away_rank = rank_dict.get(away_team, len(teams))
            home_points = standings[home_team]['points']
            away_points = standings[away_team]['points']

            df.loc[row.name, 'home_current_rank'] = home_rank
            df.loc[row.name, 'away_current_rank'] = away_rank
            df.loc[row.name, 'rank_difference'] = home_rank - away_rank
            df.loc[row.name, 'home_points_before_match'] = home_points
            df.loc[row.name, 'away_points_before_match'] = away_points
            df.loc[row.name, 'points_difference'] = home_points - away_points

            result = row['target_match_result']
            if pd.notna(result):
                if result == 'H':
                    standings[home_team]['points'] += 3
                elif result == 'D':
                    standings[home_team]['points'] += 1
                    standings[away_team]['points'] += 1
                elif result == 'A':
                    standings[away_team]['points'] += 3

                standings[home_team]['played'] += 1
                standings[away_team]['played'] += 1

    return df


# =====================================================
# Elo 赛前评级
# =====================================================

def result_to_home_score_elo(result) -> float:
    if result == "H":
        return 1.0
    if result == "D":
        return 0.5
    if result == "A":
        return 0.0
    return np.nan


def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 Elo 评级特征
    
    - home_elo_pre: 主队赛前 Elo 评级
    - away_elo_pre: 客队赛前 Elo 评级
    - elo_diff_pre: 主客队 Elo 差值
    """
    out = df.copy()
    
    home_elo_list = []
    away_elo_list = []
    
    ratings_by_league = {}
    
    for _, row in out.iterrows():
        league_key = row.get("league_key", row.get("league", "unknown"))
        home_team = str(row["home_team_name"])
        away_team = str(row["away_team_name"])
        result = row.get("target_match_result", None)
        
        if league_key not in ratings_by_league:
            ratings_by_league[league_key] = {}
        
        ratings = ratings_by_league[league_key]
        
        h_elo = ratings.get(home_team, ELO_INITIAL)
        a_elo = ratings.get(away_team, ELO_INITIAL)
        
        home_elo_list.append(h_elo)
        away_elo_list.append(a_elo)
        
        actual_home = result_to_home_score_elo(result)
        
        if pd.isna(actual_home):
            continue
        
        expected_home = 1.0 / (
            1.0 + 10.0 ** (-((h_elo + ELO_HOME_ADVANTAGE) - a_elo) / 400.0)
        )
        
        delta = ELO_K * (actual_home - expected_home)
        
        ratings[home_team] = h_elo + delta
        ratings[away_team] = a_elo - delta
    
    out["home_elo_pre"] = home_elo_list
    out["away_elo_pre"] = away_elo_list
    out["elo_diff_pre"] = out["home_elo_pre"] - out["away_elo_pre"]
    
    return out


# =====================================================
# 构建特征
# =====================================================

def build_features_for_file(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = load_and_prepare_data(input_path)
    original_rows = len(df)

    df = add_intermediate_match_columns(df)

    print("  [1/7] Adding home venue history features...")
    df = add_venue_history_features(df=df, team_col="home_team_name", prefix="home", window=WINDOW)

    print("  [2/7] Adding away venue history features...")
    df = add_venue_history_features(df=df, team_col="away_team_name", prefix="away", window=WINDOW)

    print("  [3/7] Adding Head-to-Head features...")
    df = add_h2h_features(df)

    print("  [4/7] Adding team form features...")
    df = add_team_form_features(df)

    print("  [5/7] Adding ranking features...")
    df = add_ranking_features(df)

    print("  [6/7] Adding Elo features...")
    df = add_elo_features(df)

    missing = [c for c in FINAL_NEW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"{input_path.name} features not generated: {missing}")

    if len(df) != original_rows:
        raise RuntimeError(f"{input_path.name} row count changed: {original_rows} -> {len(df)}")

    validate_feature_basic(df, input_path.name)

    if VALIDATION_MODE in ["sample", "all"]:
        run_bruteforce_validation(df, input_path.name)

    if DROP_HELPER_COLUMNS:
        df = df.drop(columns=[c for c in HELPER_COLUMNS if c in df.columns], errors="ignore")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.shape[1]}")

    return df



def validate_feature_basic(df: pd.DataFrame, label: str):
    print("\n" + "=" * 70)
    print(f"BASIC FEATURE CHECK: {label}")
    print("=" * 70)

    print("Rows:", len(df))
    print("Columns:", df.shape[1])

    print("\nNew feature missing values:")
    print(df[FINAL_NEW_FEATURES].isna().sum().to_string())

    rate_cols = [
        "Total home win rate",
        "Total away win rate",
        "Total away loss rate",
        "Total away draw rate",
        "Total home draw rate",
        "Total home loss rate",
    ]

    bad_rates = {}
    for c in rate_cols:
        bad_rates[c] = int(
            df[c].notna().sum()
            - df[c].dropna().between(0, 1).sum()
        )

    print("\nRate out-of-range counts:")
    print(pd.Series(bad_rates).to_string())

    window_cols = [
        "Home wins in the window",
        "Away wins in the window",
    ]

    bad_windows = {}
    for c in window_cols:
        bad_windows[c] = int(
            df[c].notna().sum()
            - df[c].dropna().between(0, WINDOW).sum()
        )

    print("\nWindow out-of-range counts:")
    print(pd.Series(bad_windows).to_string())

    if "match_id" in df.columns:
        dup = int(df["match_id"].duplicated().sum())
        print("\nDuplicate match_id:", dup)
        if dup > 0:
            raise RuntimeError(f"{label} has duplicate match_id: {dup}")

    if any(v > 0 for v in bad_rates.values()):
        raise RuntimeError(f"{label} has rate out of [0,1]")

    if any(v > 0 for v in bad_windows.values()):
        raise RuntimeError(f"{label} has window wins out of [0,{WINDOW}]")



def calc_expected_features_bruteforce(df: pd.DataFrame, idx: int) -> dict:
    row = df.iloc[idx]

    prior = df[df["_row_order"] < row["_row_order"]].copy()
    prior = prior[prior["league_key"] == row["league_key"]]

    if RESET_EACH_SEASON:
        prior = prior[prior["season"] == row["season"]]

    home_team = row["home_team_name"]
    away_team = row["away_team_name"]

    home_hist = prior[prior["home_team_name"] == home_team].copy()
    away_hist = prior[prior["away_team_name"] == away_team].copy()

    home_last5 = home_hist.tail(WINDOW)
    away_last5 = away_hist.tail(WINDOW)

    exp = {}

    exp["Home wins in the window"] = float(home_last5["_home_win"].sum())
    exp["Away wins in the window"] = float(away_last5["_away_win"].sum())

    home_n = len(home_hist)
    away_n = len(away_hist)

    exp["Total home win rate"] = np.nan if home_n == 0 else home_hist["_home_win"].sum() / home_n
    exp["Total home draw rate"] = np.nan if home_n == 0 else home_hist["_home_draw"].sum() / home_n
    exp["Total home loss rate"] = np.nan if home_n == 0 else home_hist["_home_loss"].sum() / home_n

    exp["Total away win rate"] = np.nan if away_n == 0 else away_hist["_away_win"].sum() / away_n
    exp["Total away draw rate"] = np.nan if away_n == 0 else away_hist["_away_draw"].sum() / away_n
    exp["Total away loss rate"] = np.nan if away_n == 0 else away_hist["_away_loss"].sum() / away_n

    exp["Home goals forward"] = float(home_hist["_home_gf"].sum())
    exp["Home goals against"] = float(home_hist["_home_ga"].sum())
    exp["Home goal differential"] = float(home_hist["_home_goal_diff"].sum())

    exp["Away goals forward"] = float(away_hist["_away_gf"].sum())
    exp["Away goals against"] = float(away_hist["_away_ga"].sum())
    exp["Away goal differential"] = float(away_hist["_away_goal_diff"].sum())

    exp["Home goals forward per match"] = np.nan if home_n == 0 else home_hist["_home_gf"].sum() / home_n
    exp["Home goals against per match"] = np.nan if home_n == 0 else home_hist["_home_ga"].sum() / home_n
    exp["Home goal differential per match"] = np.nan if home_n == 0 else home_hist["_home_goal_diff"].sum() / home_n

    exp["Away goals forward per match"] = np.nan if away_n == 0 else away_hist["_away_gf"].sum() / away_n
    exp["Away goals against per match"] = np.nan if away_n == 0 else away_hist["_away_ga"].sum() / away_n
    exp["Away goal differential per match"] = np.nan if away_n == 0 else away_hist["_away_goal_diff"].sum() / away_n

    exp["Home wins margin goal"] = safe_mean(home_hist["_home_win_margin"])
    exp["Home losses margin goal"] = safe_mean(home_hist["_home_loss_margin"])
    exp["Away wins margin goal"] = safe_mean(away_hist["_away_win_margin"])
    exp["Away losses margin goal"] = safe_mean(away_hist["_away_loss_margin"])

    return exp


def run_bruteforce_validation(df: pd.DataFrame, label: str):
    print("\n" + "=" * 70)
    print(f"BRUTEFORCE VALIDATION: {label}")
    print("=" * 70)

    if VALIDATION_MODE == "all":
        idxs = list(range(len(df)))
    elif VALIDATION_MODE == "sample":
        n = min(VALIDATION_SAMPLE_N, len(df))
        rng = np.random.default_rng(42)
        idxs = sorted(rng.choice(len(df), size=n, replace=False).tolist())
    else:
        return

    failures = []

    for idx in idxs:
        expected = calc_expected_features_bruteforce(df, idx)

        for col in REQUIRED_FEATURES + ENHANCED_TRAINING_FEATURES:
            actual = df.iloc[idx][col]
            exp = expected[col]

            if not values_equal(actual, exp, tol=TEST_TOL):
                failures.append({
                    "row_idx": idx,
                    "match_id": df.iloc[idx].get("match_id", None),
                    "feature": col,
                    "actual": actual,
                    "expected": exp,
                })

    failure_df = pd.DataFrame(failures)

    print(f"Rows tested: {len(idxs)}")
    print(f"Cell mismatches: {len(failure_df)}")

    if len(failure_df) > 0:
        out_path = OUTPUT_DIR / f"{Path(label).stem}_venue_history_failures.csv"
        failure_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(failure_df.head(20).to_string(index=False))
        raise RuntimeError(f"{label} validation failed, details saved: {out_path}")

    print("PASS: venue-history features validated.")



def main():
    files = sorted(INPUT_DIR.glob(INPUT_GLOB))

    if not files:
        raise FileNotFoundError(f"No files found in {INPUT_DIR}: {INPUT_GLOB}")

    print("=" * 70)
    print("Feature Engineering Stage 3 (Merged Version)")
    print("  - Venue History Features")
    print("  - Head-to-Head Features")
    print("  - Team Form Features")
    print("  - Ranking Features")
    print("  - Elo Features")
    print("=" * 70)

    print("\nFound files:")
    for f in files:
        print("-", f)

    summary = []

    for input_path in files:
        output_path = OUTPUT_DIR / input_path.name.replace(
            "_feature_engineered_with_odds_FINAL.csv",
            "_feature_engineered_with_odds_venue_history_FINAL.csv"
        )

        df_out = build_features_for_file(input_path, output_path)

        league = df_out["league"].iloc[0] if "league" in df_out.columns else input_path.stem

        summary.append({
            "league": league,
            "rows": len(df_out),
            "columns": df_out.shape[1],
            "output_file": str(output_path),
        })

    summary_df = pd.DataFrame(summary)

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    summary_path = OUTPUT_DIR / "feature_engineering_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\nSaved summary:")
    print(summary_path)


if __name__ == "__main__":
    main()
