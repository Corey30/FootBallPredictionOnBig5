import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm



MATCHES_FILE = "statsbomb_big5_2015_16_matches.csv"
EVENTS_DIR = "sb_events"
OUTPUT_FILE = "event_state_table_big5_2015_16.parquet"

SHOT_WINDOW = 60
XG_WINDOW = 60
CORNER_WINDOW = 120
BOX_ENTRY_WINDOW = 60
HIGH_TURNOVER_WINDOW = 60


FILTER_ADMIN_EVENTS = True
ADMIN_EVENT_TYPES = {
    "Starting XI",
    "Half Start",
    "Half End",
    "Tactical Shift",
    "Referee Ball-Drop",
    "Injury Stoppage"
}



def parse_loc(val):
    
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return float(val[0]), float(val[1])

    if isinstance(val, str):
        s = val.strip()
        if s == "" or s.lower() == "none" or s.lower() == "nan":
            return np.nan, np.nan
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                return float(parsed[0]), float(parsed[1])
        except Exception:
            return np.nan, np.nan

    return np.nan, np.nan


def in_penalty_box(x, y):
    """
    StatsBomb 坐标通常是 120 x 80。：
      左侧禁区：x <= 18, 18 <= y <= 62
      右侧禁区：x >= 102, 18 <= y <= 62

    注意：
    不区分进攻方向，是“危险区域进入”近似。
    """
    if np.isnan(x) and np.isnan(y):
        return False
    left_box = (x <= 18) and (18 <= y <= 62)
    right_box = (x >= 102) and (18 <= y <= 62)
    return left_box or right_box


def compute_box_entry_flags(df):
    """
    判断事件是否构成 box entry
    规则：
    - 事件类型是 Pass 或 Carry
    - 起点不在禁区
    - 终点在禁区
    """
    flags = np.zeros(len(df), dtype=np.int8)

    for i, row in df.iterrows():
        event_type = row["type"]

        if event_type not in {"Pass", "Carry"}:
            continue

        sx, sy = parse_loc(row.get("location"))

        if event_type == "Pass":
            ex, ey = parse_loc(row.get("pass_end_location"))
        else:
            ex, ey = parse_loc(row.get("carry_end_location"))

        if np.isnan(sx) or np.isnan(ex):
            continue

        if (not in_penalty_box(sx, sy)) and in_penalty_box(ex, ey):
            flags[i] = 1

    return flags


def compute_high_turnover_flags(df):
    """
    high turnover 定义：
    - type 属于 Ball Recovery / Interception / Duel
    - 且事件位置 x >= 60
    """
    flags = np.zeros(len(df), dtype=np.int8)
    valid_types = {"Ball Recovery", "Interception", "Duel"}

    for i, row in df.iterrows():
        if row["type"] not in valid_types:
            continue
        x, y = parse_loc(row.get("location"))
        if np.isnan(x):
            continue
        if x >= 60:
            flags[i] = 1

    return flags


def rolling_feature_from_subset(current_times, subset_times, subset_values, window_sec):
    """
    向量化滚动窗口计算：
    对于 current_times 中每个 t，计算区间 [t-window_sec, t) 内 subset_values 的和

    参数：
    - current_times: 所有状态点时刻（按升序）
    - subset_times: 某类事件发生时刻（按升序）
    - subset_values: 某类事件值（与 subset_times 对齐）
    - window_sec: 窗口长度

    返回：
    - result: 与 current_times 同长度的数组
    """
    if len(subset_times) == 0:
        return np.zeros(len(current_times), dtype=float)

    subset_times = np.asarray(subset_times, dtype=float)
    subset_values = np.asarray(subset_values, dtype=float)

    cumsum = np.cumsum(subset_values)

    right_idx = np.searchsorted(subset_times, current_times, side="left")
    left_idx = np.searchsorted(subset_times, current_times - window_sec, side="left")

    right_sum = np.where(right_idx > 0, cumsum[right_idx - 1], 0.0)
    left_sum = np.where(left_idx > 0, cumsum[left_idx - 1], 0.0)

    return right_sum - left_sum


def build_next_goal_label(df, home_team, away_team):
    """
    为每一行状态构造 label_next_goal_class
    标签定义：
    - 从当前行之后往后找最近的一个进球事件
    - 如果下一个进球由主队打进 => home
    - 如果下一个进球由客队打进 => away
    - 如果之后没有进球 => none

    用倒序扫描实现，O(n)
    """
    labels = np.empty(len(df), dtype=object)
    next_goal_team = None

    for i in range(len(df) - 1, -1, -1):
        if next_goal_team is None:
            labels[i] = "none"
        elif next_goal_team == home_team:
            labels[i] = "home"
        elif next_goal_team == away_team:
            labels[i] = "away"
        else:
            labels[i] = "none"

        if df.iloc[i]["is_goal"] == 1:
            next_goal_team = df.iloc[i]["team"]

    return labels


def load_event_file(file_path):
   
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {file_path}")


# =========================================================
# 单场比赛：event -> event_state_table
# =========================================================

def build_event_state_table_for_match(ev, home_team, away_team):
    df = ev.copy()

    required_cols = ["match_id", "period", "minute", "second", "team", "type"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"缺少必要字段: {c}")

    sort_cols = [c for c in ["period", "minute", "second", "index"] if c in df.columns]
    df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    df["minute"] = pd.to_numeric(df["minute"], errors="coerce").fillna(0).astype(int)
    df["second"] = pd.to_numeric(df["second"], errors="coerce").fillna(0).astype(int)
    df["time_sec"] = df["minute"] * 60 + df["second"]

    if FILTER_ADMIN_EVENTS:
        df = df[~df["type"].isin(ADMIN_EVENT_TYPES)].reset_index(drop=True)

    # 事件标记
    df["is_shot"] = (df["type"] == "Shot").astype(np.int8)
    def safe_col(df, col_name, default_value):
        if col_name in df.columns:
            return df[col_name]
        return pd.Series(default_value, index=df.index)

    df["is_shot"] = (df["type"] == "Shot").astype(np.int8)

    shot_outcome = safe_col(df, "shot_outcome", "")
    pass_type = safe_col(df, "pass_type", "")
    foul_committed_card = safe_col(df, "foul_committed_card", "")
    shot_xg_raw = safe_col(df, "shot_statsbomb_xg", 0.0)

    df["is_goal"] = (
    (df["type"] == "Shot") &
    (shot_outcome.fillna("") == "Goal")
    ).astype(np.int8)

    df["is_corner"] = (pass_type.fillna("") == "Corner").astype(np.int8)

    red_values = {"Red Card", "Second Yellow"}
    df["is_red_card"] = foul_committed_card.fillna("").isin(red_values).astype(np.int8)

    df["shot_xg"] = pd.to_numeric(shot_xg_raw, errors="coerce").fillna(0.0)
    # 派生事件标记
    df["is_box_entry"] = compute_box_entry_flags(df)
    df["is_high_turnover"] = compute_high_turnover_flags(df)

    # 主客队事件
    df["home_event"] = (df["team"] == home_team).astype(np.int8)
    df["away_event"] = (df["team"] == away_team).astype(np.int8)

    # 比分累计
    df["home_goal_event"] = ((df["is_goal"] == 1) & (df["team"] == home_team)).astype(np.int8)
    df["away_goal_event"] = ((df["is_goal"] == 1) & (df["team"] == away_team)).astype(np.int8)

    df["home_goals_so_far"] = df["home_goal_event"].cumsum()
    df["away_goals_so_far"] = df["away_goal_event"].cumsum()
    df["score_diff"] = df["home_goals_so_far"] - df["away_goals_so_far"]

    # 红牌累计
    df["home_red_event"] = ((df["is_red_card"] == 1) & (df["team"] == home_team)).astype(np.int8)
    df["away_red_event"] = ((df["is_red_card"] == 1) & (df["team"] == away_team)).astype(np.int8)

    df["home_red_so_far"] = df["home_red_event"].cumsum()
    df["away_red_so_far"] = df["away_red_event"].cumsum()
    df["red_card_diff"] = df["home_red_so_far"] - df["away_red_so_far"]

    current_times = df["time_sec"].to_numpy()

    # 子集：shots
    shot_home_times = df.loc[(df["is_shot"] == 1) & (df["team"] == home_team), "time_sec"].to_numpy()
    shot_away_times = df.loc[(df["is_shot"] == 1) & (df["team"] == away_team), "time_sec"].to_numpy()

    recent_shots_home = rolling_feature_from_subset(
        current_times, shot_home_times, np.ones(len(shot_home_times)), SHOT_WINDOW
    )
    recent_shots_away = rolling_feature_from_subset(
        current_times, shot_away_times, np.ones(len(shot_away_times)), SHOT_WINDOW
    )

    # 子集：xG
    xg_home_df = df.loc[(df["is_shot"] == 1) & (df["team"] == home_team), ["time_sec", "shot_xg"]]
    xg_away_df = df.loc[(df["is_shot"] == 1) & (df["team"] == away_team), ["time_sec", "shot_xg"]]

    recent_xg_home = rolling_feature_from_subset(
        current_times,
        xg_home_df["time_sec"].to_numpy(),
        xg_home_df["shot_xg"].to_numpy(),
        XG_WINDOW
    )
    recent_xg_away = rolling_feature_from_subset(
        current_times,
        xg_away_df["time_sec"].to_numpy(),
        xg_away_df["shot_xg"].to_numpy(),
        XG_WINDOW
    )

    # 子集：corners
    corner_home_times = df.loc[(df["is_corner"] == 1) & (df["team"] == home_team), "time_sec"].to_numpy()
    corner_away_times = df.loc[(df["is_corner"] == 1) & (df["team"] == away_team), "time_sec"].to_numpy()

    recent_corners_home = rolling_feature_from_subset(
        current_times, corner_home_times, np.ones(len(corner_home_times)), CORNER_WINDOW
    )
    recent_corners_away = rolling_feature_from_subset(
        current_times, corner_away_times, np.ones(len(corner_away_times)), CORNER_WINDOW
    )

    # 子集：box entries
    box_home_times = df.loc[(df["is_box_entry"] == 1) & (df["team"] == home_team), "time_sec"].to_numpy()
    box_away_times = df.loc[(df["is_box_entry"] == 1) & (df["team"] == away_team), "time_sec"].to_numpy()

    recent_box_entries_home = rolling_feature_from_subset(
        current_times, box_home_times, np.ones(len(box_home_times)), BOX_ENTRY_WINDOW
    )
    recent_box_entries_away = rolling_feature_from_subset(
        current_times, box_away_times, np.ones(len(box_away_times)), BOX_ENTRY_WINDOW
    )

    # 子集：high turnovers
    ht_home_times = df.loc[(df["is_high_turnover"] == 1) & (df["team"] == home_team), "time_sec"].to_numpy()
    ht_away_times = df.loc[(df["is_high_turnover"] == 1) & (df["team"] == away_team), "time_sec"].to_numpy()

    recent_high_turnovers_home = rolling_feature_from_subset(
        current_times, ht_home_times, np.ones(len(ht_home_times)), HIGH_TURNOVER_WINDOW
    )
    recent_high_turnovers_away = rolling_feature_from_subset(
        current_times, ht_away_times, np.ones(len(ht_away_times)), HIGH_TURNOVER_WINDOW
    )

    # 标签
    label_next_goal_class = build_next_goal_label(df, home_team, away_team)

    # 输出状态表
    state_df = pd.DataFrame({
        "match_id": df["match_id"].values,
        "period": df["period"].values,
        "time_sec": df["time_sec"].values,

        "home_team": home_team,
        "away_team": away_team,

        "score_diff": df["score_diff"].values,
        "red_card_diff": df["red_card_diff"].values,

        "recent_shots_home": recent_shots_home,
        "recent_shots_away": recent_shots_away,

        "recent_xg_home": recent_xg_home,
        "recent_xg_away": recent_xg_away,

        "recent_corners_home": recent_corners_home,
        "recent_corners_away": recent_corners_away,

        "recent_box_entries_home": recent_box_entries_home,
        "recent_box_entries_away": recent_box_entries_away,

        "recent_high_turnovers_home": recent_high_turnovers_home,
        "recent_high_turnovers_away": recent_high_turnovers_away,

        "label_next_goal_class": label_next_goal_class,
    })

    # 差值特征
    state_df["recent_shots_diff"] = state_df["recent_shots_home"] - state_df["recent_shots_away"]
    state_df["recent_xg_diff"] = state_df["recent_xg_home"] - state_df["recent_xg_away"]
    state_df["recent_corners_diff"] = state_df["recent_corners_home"] - state_df["recent_corners_away"]
    state_df["recent_box_entries_diff"] = state_df["recent_box_entries_home"] - state_df["recent_box_entries_away"]
    state_df["recent_high_turnovers_diff"] = (
        state_df["recent_high_turnovers_home"] - state_df["recent_high_turnovers_away"]
    )

    return state_df


# =========================================================
# 批量处理全部比赛
# =========================================================

def find_event_file(events_dir, match_id):

    parquet_path = os.path.join(events_dir, f"{match_id}.parquet")
    csv_path = os.path.join(events_dir, f"{match_id}.csv")

    if os.path.exists(parquet_path):
        return parquet_path
    if os.path.exists(csv_path):
        return csv_path

    return None


def build_event_state_table_all(matches_file, events_dir, output_file):
    matches = pd.read_csv(matches_file)

    required_match_cols = ["match_id", "home_team", "away_team"]
    for c in required_match_cols:
        if c not in matches.columns:
            raise ValueError(f"matches 文件缺少字段: {c}")

    all_states = []

    for _, mrow in tqdm(matches.iterrows(), total=len(matches), desc="Building event_state_table"):
        match_id = mrow["match_id"]
        home_team = mrow["home_team"]
        away_team = mrow["away_team"]

        event_file = find_event_file(events_dir, match_id)
        if event_file is None:
            print(f"skip match_id={match_id}, 未找到事件文件")
            continue

        try:
            ev = load_event_file(event_file)
            state_df = build_event_state_table_for_match(ev, home_team, away_team)
            all_states.append(state_df)
        except Exception as e:
            print(f"failed on match_id={match_id}: {e}")

    if len(all_states) == 0:
        raise RuntimeError("没有成功构造任何比赛的 state table")

    final_df = pd.concat(all_states, ignore_index=True)

    if output_file.endswith(".parquet"):
        final_df.to_parquet(output_file, index=False)
    elif output_file.endswith(".csv"):
        final_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    else:
        raise ValueError("输出文件请使用 .parquet 或 .csv")

    return final_df


if __name__ == "__main__":
    final_df = build_event_state_table_all(
        matches_file=MATCHES_FILE,
        events_dir=EVENTS_DIR,
        output_file=OUTPUT_FILE
    )

    print("done.")
    print(final_df.shape)
    print(final_df.head())
    print(final_df["label_next_goal_class"].value_counts())