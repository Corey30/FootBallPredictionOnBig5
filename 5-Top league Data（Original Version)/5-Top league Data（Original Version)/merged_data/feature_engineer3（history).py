import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path
from math import isclose


# =====================================================
# 0. 配置
# =====================================================

from pathlib import Path

BASE_DIR = Path(
    r"E:\soccerf\\5-Top league Data（Original Version)\5-Top league Data（Original Version)\merged_data"
)

INPUT_DIR = BASE_DIR / "feature_enginner2"
OUTPUT_DIR = BASE_DIR / "feature_engineer3"
OUTPUT_DIR.mkdir(exist_ok=True)

# 你的文件名看起来不是完全固定，所以用宽一点的匹配
INPUT_GLOB = "*feature*.csv"

WINDOW = 5

# False = 跨赛季累计历史，和你原始 span3 代码定义一致
# True  = 每个赛季重新开始累计
RESET_EACH_SEASON = False

# 是否删除辅助列
DROP_HELPER_COLUMNS = True

# 验证模式：
# "none"   = 不做暴力验证
# "sample" = 抽样验证
# "all"    = 全量逐行验证，最严格但较慢
VALIDATION_MODE = "sample"
VALIDATION_SAMPLE_N = 500
TEST_TOL = 1e-9


# =====================================================
# 1. 新增特征清单
# =====================================================

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

FINAL_NEW_FEATURES = REQUIRED_FEATURES + ENHANCED_TRAINING_FEATURES

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


# =====================================================
# 2. 工具函数
# =====================================================

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


def detect_col(df: pd.DataFrame, candidates: list[str]):
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

    raise ValueError("找不到 match_datetime / date_GMT / date_gmt / timestamp 日期列。")


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


# =====================================================
# 3. 读取和预处理
# =====================================================

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
        raise ValueError(f"{input_path.name} 缺少必要列: {missing}")

    if "league" not in df.columns:
        df["league"] = infer_league_from_filename(input_path)

    if "season" not in df.columns:
        raise ValueError(f"{input_path.name} 缺少 season 列。")

    df["league_key"] = df["league"].map(normalize_league_key)
    df["match_datetime"] = parse_match_datetime(df)

    if df["match_datetime"].isna().any():
        bad_count = int(df["match_datetime"].isna().sum())
        raise ValueError(f"{input_path.name} 有 {bad_count} 行日期无法解析。")

    # 删除旧版同名特征，避免重复运行时污染
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


# =====================================================
# 4. 构造中间列
# =====================================================

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
# 5. venue-history 特征
# =====================================================

def get_group_cols(team_col: str) -> list[str]:
    # 核心修复：五大联赛必须至少按 league + team 分组，避免跨联赛污染
    cols = ["league_key", team_col]

    if RESET_EACH_SEASON:
        cols.insert(1, "season")

    return cols


def grouped_shift_rolling_sum(
    df: pd.DataFrame,
    group_cols: list[str],
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
    group_cols: list[str],
    value_col: str
) -> pd.Series:
    return (
        df.groupby(group_cols, sort=False)[value_col]
        .transform(lambda s: s.shift(1).cumsum())
        .fillna(0)
    )


def grouped_shift_expanding_mean(
    df: pd.DataFrame,
    group_cols: list[str],
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

    # 历史同场地比赛场次，不含当前场
    out[f"_{prefix}_matches_before"] = g.cumcount()

    # 最近 window 个同场地比赛胜场数
    out[f"{prefix.capitalize()} wins in the window"] = grouped_shift_rolling_sum(
        out,
        group_cols,
        win_col,
        window
    )

    # 历史累计胜平负场数，不含当前场
    out[f"_{prefix}_wins_before"] = g[win_col].cumsum() - out[win_col]
    out[f"_{prefix}_draws_before"] = g[draw_col].cumsum() - out[draw_col]
    out[f"_{prefix}_losses_before"] = g[loss_col].cumsum() - out[loss_col]

    denom = out[f"_{prefix}_matches_before"].replace(0, np.nan)

    # 胜平负率
    out[f"Total {prefix} win rate"] = out[f"_{prefix}_wins_before"] / denom
    out[f"Total {prefix} draw rate"] = out[f"_{prefix}_draws_before"] / denom
    out[f"Total {prefix} loss rate"] = out[f"_{prefix}_losses_before"] / denom

    # 累计进球 / 失球 / 净胜球
    out[f"{prefix.capitalize()} goals forward"] = grouped_shift_cumsum(
        out,
        group_cols,
        gf_col
    )

    out[f"{prefix.capitalize()} goals against"] = grouped_shift_cumsum(
        out,
        group_cols,
        ga_col
    )

    out[f"{prefix.capitalize()} goal differential"] = grouped_shift_cumsum(
        out,
        group_cols,
        gd_col
    )

    # 场均
    out[f"{prefix.capitalize()} goals forward per match"] = (
        out[f"{prefix.capitalize()} goals forward"] / denom
    )

    out[f"{prefix.capitalize()} goals against per match"] = (
        out[f"{prefix.capitalize()} goals against"] / denom
    )

    out[f"{prefix.capitalize()} goal differential per match"] = (
        out[f"{prefix.capitalize()} goal differential"] / denom
    )

    # 平均赢球净胜幅度 / 输球净负幅度
    out[f"{prefix.capitalize()} wins margin goal"] = grouped_shift_expanding_mean(
        out,
        group_cols,
        win_margin_col
    )

    out[f"{prefix.capitalize()} losses margin goal"] = grouped_shift_expanding_mean(
        out,
        group_cols,
        loss_margin_col
    )

    return out


# =====================================================
# 6. 构建特征
# =====================================================

def build_features_for_file(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = load_and_prepare_data(input_path)
    original_rows = len(df)

    df = add_intermediate_match_columns(df)

    df = add_venue_history_features(
        df=df,
        team_col="home_team_name",
        prefix="home",
        window=WINDOW
    )

    df = add_venue_history_features(
        df=df,
        team_col="away_team_name",
        prefix="away",
        window=WINDOW
    )

    missing = [c for c in FINAL_NEW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"{input_path.name} 以下特征未成功生成: {missing}")

    if len(df) != original_rows:
        raise RuntimeError(f"{input_path.name} 行数变化，原始 {original_rows}, 当前 {len(df)}")

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


# =====================================================
# 7. 基础验证
# =====================================================

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
            raise RuntimeError(f"{label} 出现重复 match_id: {dup}")

    if any(v > 0 for v in bad_rates.values()):
        raise RuntimeError(f"{label} 有 rate 超出 [0,1]")

    if any(v > 0 for v in bad_windows.values()):
        raise RuntimeError(f"{label} 有 window wins 超出 [0,{WINDOW}]")


# =====================================================
# 8. 暴力验证
# =====================================================

def calc_expected_features_bruteforce(df: pd.DataFrame, idx: int) -> dict:
    row = df.iloc[idx]

    prior = df[df["_row_order"] < row["_row_order"]].copy()

    # 五大联赛修复：验证时也必须限制同 league
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

        for col in FINAL_NEW_FEATURES:
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
        raise RuntimeError(f"{label} 暴力验证失败，详情已保存: {out_path}")

    print("PASS: venue-history features validated.")


# =====================================================
# 9. 主程序
# =====================================================

def main():
    files = sorted(INPUT_DIR.glob(INPUT_GLOB))

    if not files:
        raise FileNotFoundError(f"没有在 {INPUT_DIR} 找到匹配文件: {INPUT_GLOB}")

    print("Found files:")
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

    summary_path = OUTPUT_DIR / "venue_history_feature_engineering_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\nSaved summary:")
    print(summary_path)


if __name__ == "__main__":
    main()