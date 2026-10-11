import pandas as pd
import numpy as np
from pathlib import Path
from math import isclose



BASE_DIR = Path(r"d:/program/SoccerPre/merged_data")

INPUT_DIR = BASE_DIR / "feature_engineer3"
OUTPUT_DIR = BASE_DIR / "feature_engineer4"
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_PATTERNS = [
    "*_feature_engineered_with_odds_venue_history_FINAL.csv"
   
]

WINDOW_FORM_SHORT = 3
WINDOW_FORM_MAIN = 5
WINDOW_FORM_LONG = 8
EWM_SPAN = 5

ELO_INITIAL = 1500.0

RESET_EACH_SEASON = False


USE_GOAL_DIFF_AS_XG_PROXY_IF_MISSING = True


RUN_CHECKS = True



SELECTED_NEW_FEATURES = [
    # form points 3 / 5 / 8
    "home_form_pts_3",
    "home_form_pts_5",
    "home_form_pts_8",
    "away_form_pts_3",
    "away_form_pts_5",
    "away_form_pts_8",
    "form_pts_diff_3",
    "form_pts_diff_5",
    "form_pts_diff_8",

    # goal / xG momentum
    "home_goal_diff_5",
    "away_goal_diff_5",
    "goal_diff_mom_5_diff",
    "home_xg_diff_5_proxy",
    "away_xg_diff_5_proxy",
    "xg_diff_mom_5_proxy_diff",

    # slope / volatility
    "home_mom_slope",
    "away_mom_slope",
    "mom_slope_diff",
    "home_mom_volatility_5",
    "away_mom_volatility_5",
    "mom_volatility_diff",

    # venue recent form
    "home_venue_form_pts_5",
    "away_venue_form_pts_5",
    "venue_form_pts_5_diff",

    # venue recent xG
    "home_venue_xg_diff_5_proxy",
    "away_venue_xg_diff_5_proxy",
    "venue_xg_diff_5_proxy_diff",

    # odds momentum
    "home_odds_move_for_ewm_5",
    "away_odds_move_for_ewm_5",
    "odds_move_for_ewm_diff_5",
    "home_implied_prob_shift_for_ewm_5",
    "away_implied_prob_shift_for_ewm_5",
    "implied_prob_shift_for_ewm_diff_5",
    "odds_move_abs_for_mom_diff_5",

    # opponent-adjusted form
    "home_opp_adj_form_5",
    "away_opp_adj_form_5",
    "opp_adj_form_diff_5",
    "home_opp_adj_form_5_elo",
    "away_opp_adj_form_5_elo",
    "opp_adj_form_diff_5_elo",
]



def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {path}")


def detect_col(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c

    lower_map = {str(c).lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    return None


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
            utc=True,
        )

    raise ValueError("找不到 match_datetime / date_GMT / date_gmt / timestamp 日期列。")


def result_to_home_points(result) -> float:
    if result == "H":
        return 3.0
    if result == "D":
        return 1.0
    if result == "A":
        return 0.0
    return np.nan


def result_to_away_points(result) -> float:
    if result == "A":
        return 3.0
    if result == "D":
        return 1.0
    if result == "H":
        return 0.0
    return np.nan


def slope_last_values(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) < 2:
        return np.nan

    x = np.arange(len(values), dtype=float)
    x_mean = x.mean()
    y_mean = values.mean()

    denom = ((x - x_mean) ** 2).sum()

    if denom == 0:
        return np.nan

    return float(((x - x_mean) * (values - y_mean)).sum() / denom)


def values_equal(a, b, tol=1e-9) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) != pd.isna(b):
        return False
    return isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)



def detect_required_columns(df: pd.DataFrame) -> dict:
    cols = {}

    cols["league"] = detect_col(df, ["league", "league_key"])
    cols["season"] = detect_col(df, ["season"])
    cols["timestamp"] = detect_col(df, ["timestamp"])
    cols["home_team"] = detect_col(df, ["home_team_name", "home_team", "home"])
    cols["away_team"] = detect_col(df, ["away_team_name", "away_team", "away"])

    cols["target_result"] = detect_col(df, ["target_match_result"])
    cols["home_goals"] = detect_col(df, ["target_home_team_goal_count", "home_team_goal_count"])
    cols["away_goals"] = detect_col(df, ["target_away_team_goal_count", "away_team_goal_count"])

    cols["home_ppg"] = detect_col(df, [
        "Pre-Match PPG (Home)",
        "pre_match_ppg_home",
        "pre_match_ppg_(home)",
        "home_pre_match_ppg",
    ])

    cols["away_ppg"] = detect_col(df, [
        "Pre-Match PPG (Away)",
        "pre_match_ppg_away",
        "pre_match_ppg_(away)",
        "away_pre_match_ppg",
    ])

    cols["home_xg"] = detect_col(df, [
        "team_a_xg",
        "home_team_xg",
        "home_xg",
        "home_team_expected_goals",
        "Home Team xG",
    ])

    cols["away_xg"] = detect_col(df, [
        "team_b_xg",
        "away_team_xg",
        "away_xg",
        "away_team_expected_goals",
        "Away Team xG",
    ])

    cols["odds_move_home"] = detect_col(df, ["odds_move_home"])
    cols["odds_move_away"] = detect_col(df, ["odds_move_away"])
    cols["odds_move_abs_home"] = detect_col(df, ["odds_move_abs_home"])
    cols["odds_move_abs_away"] = detect_col(df, ["odds_move_abs_away"])
    cols["implied_prob_shift_home"] = detect_col(df, ["implied_prob_shift_home"])
    cols["implied_prob_shift_away"] = detect_col(df, ["implied_prob_shift_away"])

    required = [
        "season",
        "home_team",
        "away_team",
        "target_result",
        "home_goals",
        "away_goals",
    ]

    missing = [k for k in required if cols[k] is None]

    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    return cols




def load_and_prepare_data(input_path: Path) -> tuple[pd.DataFrame, dict]:
    df = read_table(input_path).copy()

    cols = detect_required_columns(df)

    if cols["league"] is None:
        df["league"] = infer_league_from_filename(input_path)
        cols["league"] = "league"

    df["league_key"] = df[cols["league"]].map(normalize_league_key)
    df["match_datetime"] = parse_match_datetime(df)

    if df["match_datetime"].isna().any():
        bad = int(df["match_datetime"].isna().sum())
        raise ValueError(f"{input_path.name} 有 {bad} 行 match_datetime 无法解析。")

    df = df.drop(columns=[c for c in SELECTED_NEW_FEATURES if c in df.columns], errors="ignore")

    df["_orig_idx_fe4"] = np.arange(len(df))

    sort_cols = ["league_key", "match_datetime"]
    if cols["timestamp"] is not None:
        sort_cols.append(cols["timestamp"])
    sort_cols.append("_orig_idx_fe4")

    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["_row_id_fe4"] = np.arange(len(df))

    return df, cols



def build_team_long(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    result_col = cols["target_result"]
    home_col = cols["home_team"]
    away_col = cols["away_team"]

    home_goals = pd.to_numeric(df[cols["home_goals"]], errors="coerce")
    away_goals = pd.to_numeric(df[cols["away_goals"]], errors="coerce")

    home_points = df[result_col].map(result_to_home_points)
    away_points = df[result_col].map(result_to_away_points)

    home_goal_diff = home_goals - away_goals
    away_goal_diff = away_goals - home_goals

    if cols["home_xg"] is not None and cols["away_xg"] is not None:
        home_xg = pd.to_numeric(df[cols["home_xg"]], errors="coerce")
        away_xg = pd.to_numeric(df[cols["away_xg"]], errors="coerce")

        home_xg_diff = home_xg - away_xg
        away_xg_diff = away_xg - home_xg
    else:
        if USE_GOAL_DIFF_AS_XG_PROXY_IF_MISSING:
            print("[WARN] 没找到实际 xG 字段，xg_diff_5_proxy 将使用 goal_diff 作为 proxy。")
            home_xg_diff = home_goal_diff
            away_xg_diff = away_goal_diff
        else:
            home_xg_diff = pd.Series(np.nan, index=df.index)
            away_xg_diff = pd.Series(np.nan, index=df.index)

    def get_series_or_nan(col_name):
        if col_name is None:
            return pd.Series(np.nan, index=df.index)
        return pd.to_numeric(df[col_name], errors="coerce")

    odds_move_home = get_series_or_nan(cols["odds_move_home"])
    odds_move_away = get_series_or_nan(cols["odds_move_away"])
    odds_move_abs_home = get_series_or_nan(cols["odds_move_abs_home"])
    odds_move_abs_away = get_series_or_nan(cols["odds_move_abs_away"])
    implied_shift_home = get_series_or_nan(cols["implied_prob_shift_home"])
    implied_shift_away = get_series_or_nan(cols["implied_prob_shift_away"])

    if cols["home_ppg"] is not None and cols["away_ppg"] is not None:
        home_ppg = pd.to_numeric(df[cols["home_ppg"]], errors="coerce")
        away_ppg = pd.to_numeric(df[cols["away_ppg"]], errors="coerce")
    else:
        print("[WARN] 没找到 Pre-Match PPG 字段，非 Elo opponent-adjusted form 将退化为普通 points form。")
        home_ppg = pd.Series(1.5, index=df.index)
        away_ppg = pd.Series(1.5, index=df.index)

    home_long = pd.DataFrame({
        "row_id": df["_row_id_fe4"],
        "league_key": df["league_key"],
        "season": df[cols["season"]].astype(str),
        "match_datetime": df["match_datetime"],
        "team": df[home_col].astype(str),
        "opponent": df[away_col].astype(str),
        "side": "home",
        "points": home_points,
        "goal_diff": home_goal_diff,
        "xg_diff_proxy": home_xg_diff,
        "odds_move_for_team": odds_move_home,
        "odds_move_abs_for_team": odds_move_abs_home,
        "implied_prob_shift_for_team": implied_shift_home,
        "opponent_ppg_pre": away_ppg,
        "opponent_elo_pre": df["away_elo_pre"],
    })

    away_long = pd.DataFrame({
        "row_id": df["_row_id_fe4"],
        "league_key": df["league_key"],
        "season": df[cols["season"]].astype(str),
        "match_datetime": df["match_datetime"],
        "team": df[away_col].astype(str),
        "opponent": df[home_col].astype(str),
        "side": "away",
        "points": away_points,
        "goal_diff": away_goal_diff,
        "xg_diff_proxy": away_xg_diff,
        "odds_move_for_team": odds_move_away,
        "odds_move_abs_for_team": odds_move_abs_away,
        "implied_prob_shift_for_team": implied_shift_away,
        "opponent_ppg_pre": home_ppg,
        "opponent_elo_pre": df["home_elo_pre"],
    })

    team_long = pd.concat([home_long, away_long], ignore_index=True)

    team_long = team_long.sort_values(
        ["league_key", "team", "match_datetime", "row_id"]
    ).reset_index(drop=True)

    ppg_factor = pd.to_numeric(team_long["opponent_ppg_pre"], errors="coerce") / 1.5
    ppg_factor = ppg_factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    team_long["opp_adj_points"] = team_long["points"] * ppg_factor

    elo_factor = pd.to_numeric(team_long["opponent_elo_pre"], errors="coerce") / ELO_INITIAL
    elo_factor = elo_factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    team_long["opp_adj_points_elo"] = team_long["points"] * elo_factor

    return team_long



def get_group_cols(base_cols: list[str]) -> list[str]:
    if RESET_EACH_SEASON:
        return ["league_key", "season"] + base_cols
    return ["league_key"] + base_cols


def prev_rolling_mean(s: pd.Series, window: int) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=1).mean()


def prev_rolling_std(s: pd.Series, window: int) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=2).std(ddof=0)


def prev_rolling_slope(s: pd.Series, window: int) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=2).apply(
        lambda x: slope_last_values(x),
        raw=True,
    )


def prev_ewm_mean(s: pd.Series, span: int) -> pd.Series:
    return s.shift(1).ewm(
        span=span,
        min_periods=1,
        adjust=False,
        ignore_na=True,
    ).mean()


def add_team_history_features(team_long: pd.DataFrame) -> pd.DataFrame:
    out = team_long.copy()

    overall_group_cols = get_group_cols(["team"])
    venue_group_cols = get_group_cols(["team", "side"])

    g_overall = out.groupby(overall_group_cols, sort=False)
    g_venue = out.groupby(venue_group_cols, sort=False)

    # 多窗口 form
    for w in [3, 5, 8]:
        out[f"form_pts_{w}"] = g_overall["points"].transform(
            lambda s, window=w: prev_rolling_mean(s, window)
        )

    # goal / xG momentum
    out["goal_diff_5"] = g_overall["goal_diff"].transform(
        lambda s: prev_rolling_mean(s, WINDOW_FORM_MAIN)
    )

    out["xg_diff_5_proxy"] = g_overall["xg_diff_proxy"].transform(
        lambda s: prev_rolling_mean(s, WINDOW_FORM_MAIN)
    )

    # slope / volatility based on points
    out["mom_slope"] = g_overall["points"].transform(
        lambda s: prev_rolling_slope(s, WINDOW_FORM_MAIN)
    )

    out["mom_volatility_5"] = g_overall["points"].transform(
        lambda s: prev_rolling_std(s, WINDOW_FORM_MAIN)
    )

    # venue recent form
    out["venue_form_pts_5"] = g_venue["points"].transform(
        lambda s: prev_rolling_mean(s, WINDOW_FORM_MAIN)
    )

    out["venue_xg_diff_5_proxy"] = g_venue["xg_diff_proxy"].transform(
        lambda s: prev_rolling_mean(s, WINDOW_FORM_MAIN)
    )

    # odds momentum
    out["odds_move_for_ewm_5"] = g_overall["odds_move_for_team"].transform(
        lambda s: prev_ewm_mean(s, EWM_SPAN)
    )

    out["implied_prob_shift_for_ewm_5"] = g_overall["implied_prob_shift_for_team"].transform(
        lambda s: prev_ewm_mean(s, EWM_SPAN)
    )

    out["odds_move_abs_for_mom_5"] = g_overall["odds_move_abs_for_team"].transform(
        lambda s: prev_rolling_mean(s, WINDOW_FORM_MAIN)
    )

    # opponent-adjusted form
    out["opp_adj_form_5"] = g_overall["opp_adj_points"].transform(
        lambda s: prev_rolling_mean(s, WINDOW_FORM_MAIN)
    )

    out["opp_adj_form_5_elo"] = g_overall["opp_adj_points_elo"].transform(
        lambda s: prev_rolling_mean(s, WINDOW_FORM_MAIN)
    )

    return out



def merge_back_to_matches(df: pd.DataFrame, team_long: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    feature_cols_long = [
        "row_id",

        "form_pts_3",
        "form_pts_5",
        "form_pts_8",

        "goal_diff_5",
        "xg_diff_5_proxy",

        "mom_slope",
        "mom_volatility_5",

        "venue_form_pts_5",
        "venue_xg_diff_5_proxy",

        "odds_move_for_ewm_5",
        "implied_prob_shift_for_ewm_5",
        "odds_move_abs_for_mom_5",

        "opp_adj_form_5",
        "opp_adj_form_5_elo",
    ]

    home_features = (
        team_long[team_long["side"] == "home"][feature_cols_long]
        .set_index("row_id")
        .rename(columns={
            "form_pts_3": "home_form_pts_3",
            "form_pts_5": "home_form_pts_5",
            "form_pts_8": "home_form_pts_8",

            "goal_diff_5": "home_goal_diff_5",
            "xg_diff_5_proxy": "home_xg_diff_5_proxy",

            "mom_slope": "home_mom_slope",
            "mom_volatility_5": "home_mom_volatility_5",

            "venue_form_pts_5": "home_venue_form_pts_5",
            "venue_xg_diff_5_proxy": "home_venue_xg_diff_5_proxy",

            "odds_move_for_ewm_5": "home_odds_move_for_ewm_5",
            "implied_prob_shift_for_ewm_5": "home_implied_prob_shift_for_ewm_5",
            "odds_move_abs_for_mom_5": "_home_odds_move_abs_for_mom_5_tmp",

            "opp_adj_form_5": "home_opp_adj_form_5",
            "opp_adj_form_5_elo": "home_opp_adj_form_5_elo",
        })
    )

    away_features = (
        team_long[team_long["side"] == "away"][feature_cols_long]
        .set_index("row_id")
        .rename(columns={
            "form_pts_3": "away_form_pts_3",
            "form_pts_5": "away_form_pts_5",
            "form_pts_8": "away_form_pts_8",

            "goal_diff_5": "away_goal_diff_5",
            "xg_diff_5_proxy": "away_xg_diff_5_proxy",

            "mom_slope": "away_mom_slope",
            "mom_volatility_5": "away_mom_volatility_5",

            "venue_form_pts_5": "away_venue_form_pts_5",
            "venue_xg_diff_5_proxy": "away_venue_xg_diff_5_proxy",

            "odds_move_for_ewm_5": "away_odds_move_for_ewm_5",
            "implied_prob_shift_for_ewm_5": "away_implied_prob_shift_for_ewm_5",
            "odds_move_abs_for_mom_5": "_away_odds_move_abs_for_mom_5_tmp",

            "opp_adj_form_5": "away_opp_adj_form_5",
            "opp_adj_form_5_elo": "away_opp_adj_form_5_elo",
        })
    )

    if len(home_features) != len(out):
        raise RuntimeError(f"home_features row count mismatch: {len(home_features)} vs {len(out)}")

    if len(away_features) != len(out):
        raise RuntimeError(f"away_features row count mismatch: {len(away_features)} vs {len(out)}")

    if not home_features.index.is_unique:
        raise RuntimeError("home_features row_id is not unique")

    if not away_features.index.is_unique:
        raise RuntimeError("away_features row_id is not unique")

    out = out.set_index("_row_id_fe4", drop=False)
    out = out.join(home_features, how="left")
    out = out.join(away_features, how="left")
    out = out.reset_index(drop=True)

    # diff features
    for w in [3, 5, 8]:
        out[f"form_pts_diff_{w}"] = out[f"home_form_pts_{w}"] - out[f"away_form_pts_{w}"]

    out["goal_diff_mom_5_diff"] = out["home_goal_diff_5"] - out["away_goal_diff_5"]
    out["xg_diff_mom_5_proxy_diff"] = out["home_xg_diff_5_proxy"] - out["away_xg_diff_5_proxy"]

    out["mom_slope_diff"] = out["home_mom_slope"] - out["away_mom_slope"]
    out["mom_volatility_diff"] = out["home_mom_volatility_5"] - out["away_mom_volatility_5"]

    out["venue_form_pts_5_diff"] = out["home_venue_form_pts_5"] - out["away_venue_form_pts_5"]
    out["venue_xg_diff_5_proxy_diff"] = out["home_venue_xg_diff_5_proxy"] - out["away_venue_xg_diff_5_proxy"]

    out["odds_move_for_ewm_diff_5"] = out["home_odds_move_for_ewm_5"] - out["away_odds_move_for_ewm_5"]
    out["implied_prob_shift_for_ewm_diff_5"] = (
        out["home_implied_prob_shift_for_ewm_5"]
        - out["away_implied_prob_shift_for_ewm_5"]
    )

    out["odds_move_abs_for_mom_diff_5"] = (
        out["_home_odds_move_abs_for_mom_5_tmp"]
        - out["_away_odds_move_abs_for_mom_5_tmp"]
    )

    out["opp_adj_form_diff_5"] = out["home_opp_adj_form_5"] - out["away_opp_adj_form_5"]
    out["opp_adj_form_diff_5_elo"] = out["home_opp_adj_form_5_elo"] - out["away_opp_adj_form_5_elo"]

    out = out.drop(
        columns=[
            "_home_odds_move_abs_for_mom_5_tmp",
            "_away_odds_move_abs_for_mom_5_tmp",
        ],
        errors="ignore",
    )

    return out



def build_selected_features_for_file(input_path: Path, output_path: Path) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print(f"PROCESSING: {input_path.name}")
    print("=" * 80)

    df, cols = load_and_prepare_data(input_path)

    original_rows = len(df)
    original_cols = df.shape[1]

    team_long = build_team_long(df, cols)
    team_long = add_team_history_features(team_long)
    df = merge_back_to_matches(df, team_long)

    missing_features = [c for c in SELECTED_NEW_FEATURES if c not in df.columns]
    if missing_features:
        raise RuntimeError(f"以下特征没有生成: {missing_features}")

    if len(df) != original_rows:
        raise RuntimeError(f"行数变化: before={original_rows}, after={len(df)}")

    validate_selected_features(df, input_path.name)

    # 清理临时列
    df = df.drop(columns=["_orig_idx_fe4", "_row_id_fe4"], errors="ignore")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns before: {original_cols}")
    print(f"Columns after:  {df.shape[1]}")
    print(f"New selected features added: {len(SELECTED_NEW_FEATURES)}")

    return df


def validate_selected_features(df: pd.DataFrame, label: str):
    if not RUN_CHECKS:
        return

    print("\n" + "-" * 80)
    print(f"CHECK: {label}")
    print("-" * 80)

    print("Rows:", len(df))
    print("Columns:", df.shape[1])

    print("\nSelected new feature missing values:")
    print(df[SELECTED_NEW_FEATURES].isna().sum().to_string())

    if "match_id" in df.columns:
        dup = int(df["match_id"].duplicated().sum())
        print("\nDuplicate match_id:", dup)
        if dup > 0:
            raise RuntimeError(f"{label} duplicate match_id: {dup}")

    # diff 公式检查
    formula_checks = {}

    for w in [3, 5, 8]:
        expected = df[f"home_form_pts_{w}"] - df[f"away_form_pts_{w}"]
        diff = (df[f"form_pts_diff_{w}"] - expected).abs()
        formula_checks[f"form_pts_diff_{w}_mismatch"] = int((diff > 1e-9).sum())

    check_pairs = [
        ("goal_diff_mom_5_diff", "home_goal_diff_5", "away_goal_diff_5"),
        ("xg_diff_mom_5_proxy_diff", "home_xg_diff_5_proxy", "away_xg_diff_5_proxy"),
        ("mom_slope_diff", "home_mom_slope", "away_mom_slope"),
        ("mom_volatility_diff", "home_mom_volatility_5", "away_mom_volatility_5"),
        ("venue_form_pts_5_diff", "home_venue_form_pts_5", "away_venue_form_pts_5"),
        ("venue_xg_diff_5_proxy_diff", "home_venue_xg_diff_5_proxy", "away_venue_xg_diff_5_proxy"),
        ("odds_move_for_ewm_diff_5", "home_odds_move_for_ewm_5", "away_odds_move_for_ewm_5"),
        ("implied_prob_shift_for_ewm_diff_5", "home_implied_prob_shift_for_ewm_5", "away_implied_prob_shift_for_ewm_5"),
        ("opp_adj_form_diff_5", "home_opp_adj_form_5", "away_opp_adj_form_5"),
        ("opp_adj_form_diff_5_elo", "home_opp_adj_form_5_elo", "away_opp_adj_form_5_elo"),
    ]

    for diff_col, home_col, away_col in check_pairs:
        expected = df[home_col] - df[away_col]
        diff = (df[diff_col] - expected).abs()
        formula_checks[f"{diff_col}_mismatch"] = int((diff > 1e-9).sum())

    print("\nFormula mismatch counts:")
    print(pd.Series(formula_checks).to_string())

    if any(v > 0 for v in formula_checks.values()):
        raise RuntimeError(f"{label} diff formula check failed.")

    # 合理范围检查
    point_cols = [
        "home_form_pts_3", "home_form_pts_5", "home_form_pts_8",
        "away_form_pts_3", "away_form_pts_5", "away_form_pts_8",
        "home_venue_form_pts_5", "away_venue_form_pts_5",
    ]

    bad_ranges = {}
    for c in point_cols:
        s = df[c].dropna()
        bad_ranges[c] = int((~s.between(0, 3)).sum())

    print("\nPoint feature out-of-range counts:")
    print(pd.Series(bad_ranges).to_string())

    if any(v > 0 for v in bad_ranges.values()):
        raise RuntimeError(f"{label} points-form feature out of [0,3].")



def main():
    print("INPUT_DIR:", INPUT_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"输入目录不存在: {INPUT_DIR}")

    files = []

    for pattern in INPUT_PATTERNS:
        files.extend(INPUT_DIR.glob(pattern))

    files = sorted(set(files))

    if not files:
        print("\nAll files in INPUT_DIR:")
        for f in INPUT_DIR.iterdir():
            print("-", repr(f.name))

        raise FileNotFoundError(f"没有在 {INPUT_DIR} 找到输入文件。")

    print("\nMatched input files:")
    for f in files:
        print("-", f.name)

    summary = []

    for input_path in files:
        output_path = OUTPUT_DIR / input_path.name.replace(
            ".csv",
            "_selected_advanced_FINAL.csv"
        )

        df_out = build_selected_features_for_file(input_path, output_path)

        league = df_out["league"].iloc[0] if "league" in df_out.columns else input_path.stem

        summary.append({
            "league": league,
            "rows": len(df_out),
            "columns": df_out.shape[1],
            "output_file": str(output_path),
        })

    summary_df = pd.DataFrame(summary)

    summary_path = OUTPUT_DIR / "selected_advanced_feature_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("ALL DONE")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("\nSaved summary:")
    print(summary_path)


if __name__ == "__main__":
    main()