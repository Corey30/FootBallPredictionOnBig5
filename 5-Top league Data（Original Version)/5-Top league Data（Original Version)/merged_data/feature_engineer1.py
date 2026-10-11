import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings

warnings.filterwarnings("ignore")


# =====================================================
# 1. 输入文件路径
# =====================================================

INPUT_FILE = Path(
    r"E:\soccerf\5-Top league Data（Original Version)\5-Top league Data（Original Version)\merged_data\Spain_merged_2015_2025.csv"
)

OUTPUT_FILE = INPUT_FILE.with_name(INPUT_FILE.stem + "_feature_engineered.csv")


# =====================================================
# 2. 参数设置
# =====================================================

RECENT_WINDOW = 5     # 近期牌数 / 犯规数：过去 5 场平均
DENSITY_DAYS = 14     # 赛程密度：过去 14 天内踢过几场


# =====================================================
# 3. 工具函数
# =====================================================

def require_columns(df: pd.DataFrame, cols: list):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def make_safe_id_text(x):
    x = str(x).strip().lower()
    x = re.sub(r"\s+", "_", x)
    x = re.sub(r"[^a-z0-9_]+", "", x)
    return x


def result_label(home_goals, away_goals):
    if pd.isna(home_goals) or pd.isna(away_goals):
        return np.nan
    if home_goals > away_goals:
        return "H"
    elif home_goals < away_goals:
        return "A"
    else:
        return "D"


def get_match_datetime(df: pd.DataFrame) -> pd.Series:
    """
    优先使用 timestamp。
    你的原始数据 timestamp 是 Unix 秒级时间戳。
    """
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        return pd.to_datetime(ts, unit="s", errors="coerce")

    if "date_GMT" in df.columns:
        return pd.to_datetime(df["date_GMT"], errors="coerce")

    if "date_gmt" in df.columns:
        return pd.to_datetime(df["date_gmt"], errors="coerce")

    if "match_datetime" in df.columns:
        return pd.to_datetime(df["match_datetime"], errors="coerce")

    raise ValueError("Cannot find timestamp / date_GMT / date_gmt / match_datetime column.")


def find_col(df: pd.DataFrame, candidates: list):
    """
    兼容大小写或不同命名。
    优先精确匹配，其次 lower 后匹配。
    """
    for c in candidates:
        if c in df.columns:
            return c

    lower_map = {str(c).lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    return None


# =====================================================
# 4. 读取数据
# =====================================================

df = pd.read_csv(INPUT_FILE, low_memory=False)
df = df.copy()

# 去掉之前错误生成的 feature，防止重复运行时污染
old_feature_cols = [
    "match_id",
    "match_datetime",
    "home_recent_cards",
    "away_recent_cards",
    "home_recent_fouls",
    "away_recent_fouls",
    "home_match_density",
    "away_match_density",
    "ppg_differential",
    "xg_power_index",
    "ht_result",
    "half_time_lead",
    "target_match_result",
    "target_home_team_goal_count",
    "target_away_team_goal_count",
    "target_total_goal_count",
]

df = df.drop(columns=[c for c in old_feature_cols if c in df.columns], errors="ignore")


# =====================================================
# 5. 自动识别关键列
# =====================================================

timestamp_col = find_col(df, ["timestamp"])
season_col = find_col(df, ["season"])

home_team_col = find_col(df, ["home_team_name", "home_team", "home"])
away_team_col = find_col(df, ["away_team_name", "away_team", "away"])

home_ppg_col = find_col(df, ["Pre-Match PPG (Home)", "pre_match_ppg_home"])
away_ppg_col = find_col(df, ["Pre-Match PPG (Away)", "pre_match_ppg_away"])

home_prematch_xg_col = find_col(df, ["Home Team Pre-Match xG", "home_team_pre_match_xg"])
away_prematch_xg_col = find_col(df, ["Away Team Pre-Match xG", "away_team_pre_match_xg"])

home_goal_col = find_col(df, ["home_team_goal_count"])
away_goal_col = find_col(df, ["away_team_goal_count"])
total_goal_col = find_col(df, ["total_goal_count"])

home_ht_goal_col = find_col(df, ["home_team_goal_count_half_time"])
away_ht_goal_col = find_col(df, ["away_team_goal_count_half_time"])

home_yellow_col = find_col(df, ["home_team_yellow_cards"])
home_red_col = find_col(df, ["home_team_red_cards"])
away_yellow_col = find_col(df, ["away_team_yellow_cards"])
away_red_col = find_col(df, ["away_team_red_cards"])

home_fouls_col = find_col(df, ["home_team_fouls"])
away_fouls_col = find_col(df, ["away_team_fouls"])


required_detected = {
    "timestamp": timestamp_col,
    "season": season_col,
    "home_team": home_team_col,
    "away_team": away_team_col,
    "home_ppg": home_ppg_col,
    "away_ppg": away_ppg_col,
    "home_prematch_xg": home_prematch_xg_col,
    "away_prematch_xg": away_prematch_xg_col,
    "home_goal": home_goal_col,
    "away_goal": away_goal_col,
    "total_goal": total_goal_col,
    "home_ht_goal": home_ht_goal_col,
    "away_ht_goal": away_ht_goal_col,
    "home_yellow": home_yellow_col,
    "home_red": home_red_col,
    "away_yellow": away_yellow_col,
    "away_red": away_red_col,
    "home_fouls": home_fouls_col,
    "away_fouls": away_fouls_col,
}

missing = [k for k, v in required_detected.items() if v is None]
if missing:
    raise ValueError(f"Cannot detect required columns: {missing}")

print("Detected columns:")
for k, v in required_detected.items():
    print(f"{k}: {v}")


# =====================================================
# 6. 创建比赛时间，并排序
# =====================================================

df["match_datetime"] = get_match_datetime(df)

if df["match_datetime"].isna().any():
    bad_rows = df[df["match_datetime"].isna()].index.tolist()[:10]
    raise ValueError(f"Some rows have invalid match_datetime. Example row indexes: {bad_rows}")

df = df.sort_values(
    [season_col, "match_datetime", home_team_col, away_team_col]
).reset_index(drop=True)

# 固定 row_id，后面所有 features 都靠它回填，避免错位
df["row_id"] = df.index


# =====================================================
# 7. 基础衍生字段
# =====================================================

df["match_id"] = (
    df[timestamp_col].astype(str)
    + "_"
    + df[home_team_col].map(make_safe_id_text)
    + "_"
    + df[away_team_col].map(make_safe_id_text)
)

df["ppg_differential"] = (
    pd.to_numeric(df[home_ppg_col], errors="coerce")
    - pd.to_numeric(df[away_ppg_col], errors="coerce")
)

df["xg_power_index"] = (
    pd.to_numeric(df[home_prematch_xg_col], errors="coerce")
    - pd.to_numeric(df[away_prematch_xg_col], errors="coerce")
)

df["ht_result"] = [
    result_label(h, a)
    for h, a in zip(
        pd.to_numeric(df[home_ht_goal_col], errors="coerce"),
        pd.to_numeric(df[away_ht_goal_col], errors="coerce")
    )
]

df["half_time_lead"] = (
    pd.to_numeric(df[home_ht_goal_col], errors="coerce")
    - pd.to_numeric(df[away_ht_goal_col], errors="coerce")
)

df["target_match_result"] = [
    result_label(h, a)
    for h, a in zip(
        pd.to_numeric(df[home_goal_col], errors="coerce"),
        pd.to_numeric(df[away_goal_col], errors="coerce")
    )
]

df["target_home_team_goal_count"] = pd.to_numeric(df[home_goal_col], errors="coerce")
df["target_away_team_goal_count"] = pd.to_numeric(df[away_goal_col], errors="coerce")
df["target_total_goal_count"] = pd.to_numeric(df[total_goal_col], errors="coerce")


# =====================================================
# 8. 构建球队级 long table
# =====================================================
# 每场比赛拆成两行：
# 主队一行，客队一行。
# recent_cards / recent_fouls / match_density 都只基于当前比赛之前的数据。

home_cards = (
    pd.to_numeric(df[home_yellow_col], errors="coerce").fillna(0)
    + pd.to_numeric(df[home_red_col], errors="coerce").fillna(0)
)

away_cards = (
    pd.to_numeric(df[away_yellow_col], errors="coerce").fillna(0)
    + pd.to_numeric(df[away_red_col], errors="coerce").fillna(0)
)

home_long = pd.DataFrame({
    "row_id": df["row_id"],
    "season": df[season_col],
    "match_datetime": df["match_datetime"],
    "team": df[home_team_col],
    "is_home": 1,
    "cards": home_cards,
    "fouls": pd.to_numeric(df[home_fouls_col], errors="coerce"),
})

away_long = pd.DataFrame({
    "row_id": df["row_id"],
    "season": df[season_col],
    "match_datetime": df["match_datetime"],
    "team": df[away_team_col],
    "is_home": 0,
    "cards": away_cards,
    "fouls": pd.to_numeric(df[away_fouls_col], errors="coerce"),
})

team_long = pd.concat([home_long, away_long], ignore_index=True)
team_long = team_long.sort_values(
    ["team", "season", "match_datetime", "row_id"]
).reset_index(drop=True)


# =====================================================
# 9. recent_cards / recent_fouls
# =====================================================
# shift(1): 不使用当前比赛，只用之前比赛
# rolling(5): 过去 5 场平均
# groupby(["team", "season"]): 不跨赛季累计

team_long["recent_cards"] = (
    team_long
    .groupby(["team", "season"], group_keys=False)["cards"]
    .apply(lambda s: s.shift(1).rolling(RECENT_WINDOW, min_periods=1).mean())
)

team_long["recent_fouls"] = (
    team_long
    .groupby(["team", "season"], group_keys=False)["fouls"]
    .apply(lambda s: s.shift(1).rolling(RECENT_WINDOW, min_periods=1).mean())
)

team_long["recent_cards"] = team_long["recent_cards"].fillna(0)
team_long["recent_fouls"] = team_long["recent_fouls"].fillna(0)


# =====================================================
# 10. match_density
# =====================================================
# 定义：
# 当前比赛前 DENSITY_DAYS 天内，该队已经踢过多少场。
# 不包含当前比赛。
# 不跨赛季。

def compute_match_density_for_group(g: pd.DataFrame, days: int) -> pd.Series:
    g = g.sort_values(["match_datetime", "row_id"]).copy()
    dates = g["match_datetime"].to_numpy()

    density = []

    for current_date in dates:
        start_date = current_date - np.timedelta64(days, "D")

        count = ((dates < current_date) & (dates >= start_date)).sum()
        density.append(int(count))

    return pd.Series(density, index=g.index)


team_long["match_density"] = (
    team_long
    .groupby(["team", "season"], group_keys=False)
    .apply(lambda g: compute_match_density_for_group(g, DENSITY_DAYS))
)


# =====================================================
# 11. 正确 merge 回 match-level 表
# =====================================================
# 这是修复 away_recent_* 错位的关键部分。
# 不使用连续 merge(left_index=True, right_on="row_id")。
# 统一 set_index("row_id") 后 join，保证 home 和 away 都按原始 row_id 回填。

home_features = (
    team_long[team_long["is_home"] == 1]
    [["row_id", "recent_cards", "recent_fouls", "match_density"]]
    .set_index("row_id")
    .rename(columns={
        "recent_cards": "home_recent_cards",
        "recent_fouls": "home_recent_fouls",
        "match_density": "home_match_density",
    })
)

away_features = (
    team_long[team_long["is_home"] == 0]
    [["row_id", "recent_cards", "recent_fouls", "match_density"]]
    .set_index("row_id")
    .rename(columns={
        "recent_cards": "away_recent_cards",
        "recent_fouls": "away_recent_fouls",
        "match_density": "away_match_density",
    })
)

# 强校验：每场比赛必须正好有一条 home feature 和一条 away feature
assert len(home_features) == len(df), "home_features row count does not match df."
assert len(away_features) == len(df), "away_features row count does not match df."
assert home_features.index.is_unique, "home_features row_id is not unique."
assert away_features.index.is_unique, "away_features row_id is not unique."

df = df.set_index("row_id", drop=False)

df = df.join(home_features, how="left")
df = df.join(away_features, how="left")

df = df.reset_index(drop=True)


# =====================================================
# 12. 完整性检查
# =====================================================

recent_feature_cols = [
    "home_recent_cards",
    "away_recent_cards",
    "home_recent_fouls",
    "away_recent_fouls",
    "home_match_density",
    "away_match_density",
]

simple_feature_cols = [
    "match_id",
    "ppg_differential",
    "xg_power_index",
    "ht_result",
    "half_time_lead",
    "target_match_result",
    "target_home_team_goal_count",
    "target_away_team_goal_count",
    "target_total_goal_count",
]

print("\n==============================")
print("Feature engineering checks")
print("==============================")

print("Rows:", len(df))
print("Columns:", df.shape[1])

print("\nRecent feature missing values:")
print(df[recent_feature_cols].isna().sum())

assert df[recent_feature_cols].isna().sum().sum() == 0, \
    "ERROR: recent feature columns contain missing values."

print("\nSimple feature missing values:")
print(df[simple_feature_cols].isna().sum())

print("\nSeason counts:")
print(df[season_col].value_counts().sort_index())

print("\nTarget result distribution:")
print(df["target_match_result"].value_counts(dropna=False))

print("\nTime order check:")
print("match_datetime monotonic increasing:", df["match_datetime"].is_monotonic_increasing)

print("\nDuplicate match_id:")
print(df["match_id"].duplicated().sum())

assert df["match_id"].duplicated().sum() == 0, \
    "ERROR: duplicated match_id found."

goal_check = (
    pd.to_numeric(df["target_home_team_goal_count"], errors="coerce")
    + pd.to_numeric(df["target_away_team_goal_count"], errors="coerce")
    - pd.to_numeric(df["target_total_goal_count"], errors="coerce")
).abs()

print("\nGoal total mismatch rows:")
print((goal_check > 0).sum())

assert (goal_check.fillna(0) > 0).sum() == 0, \
    "ERROR: target_home + target_away != target_total."


# =====================================================
# 13. 可选：检查每赛季行数
# =====================================================

print("\n==============================")
print("Per-season row count")
print("==============================")
print(df[season_col].value_counts().sort_index())


# =====================================================
# 14. 整理字段顺序
# =====================================================

front_cols = [
    "match_id",
    season_col,
    "match_datetime",

    "home_recent_cards",
    "away_recent_cards",
    "home_recent_fouls",
    "away_recent_fouls",
    "home_match_density",
    "away_match_density",

    "ppg_differential",
    "xg_power_index",

    "ht_result",
    "half_time_lead",

    "target_match_result",
    "target_home_team_goal_count",
    "target_away_team_goal_count",
    "target_total_goal_count",
]

front_cols = [c for c in front_cols if c in df.columns]
other_cols = [c for c in df.columns if c not in front_cols]

df = df[front_cols + other_cols]


# =====================================================
# 15. 保存
# =====================================================

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\n==============================")
print("Saved feature engineered file")
print("==============================")
print(OUTPUT_FILE)