import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm


STATE_TABLE_PATH = "event_state_table_big5_2015_16.parquet"
MATCHES_FILE = "statsbomb_big5_2015_16_matches.csv"
EVENTS_DIR = "sb_events"
OUTPUT_FILE = "event_state_table_big5_2015_16_v2.parquet"

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


def load_table(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"不支持文件类型: {path}")


def load_event_file(events_dir, match_id):
    pq = os.path.join(events_dir, f"{match_id}.parquet")
    csv = os.path.join(events_dir, f"{match_id}.csv")

    if os.path.exists(pq):
        return pd.read_parquet(pq)
    elif os.path.exists(csv):
        return pd.read_csv(csv)
    else:
        raise FileNotFoundError(f"未找到事件文件: {match_id}")


def safe_col(df, col_name, default_value):
    if col_name in df.columns:
        return df[col_name]
    return pd.Series(default_value, index=df.index)


def parse_loc(val):
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return float(val[0]), float(val[1])

    if isinstance(val, str):
        s = val.strip()
        if s == "" or s.lower() in {"none", "nan"}:
            return np.nan, np.nan
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                return float(parsed[0]), float(parsed[1])
        except Exception:
            return np.nan, np.nan

    return np.nan, np.nan


def in_penalty_box(x, y):
    if np.isnan(x) or np.isnan(y):
        return False
    left_box = (x <= 18) and (18 <= y <= 62)
    right_box = (x >= 102) and (18 <= y <= 62)
    return left_box or right_box


def compute_box_entry_flags_v2(df):
    """
    更稳的 box entry 定义：
    - 仅对 Pass / Carry
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


def compute_high_turnover_flags_v2(df):
    """
    更合理的 high turnover 定义：
    优先视为高位赢回球权的情况：
    - offensive ball recovery
    - 或 counterpress 触发后的 ball recovery / interception
    - 或 interception 成功
    - 或 duel 赢下
    并且位置 x >= 60
    """
    flags = np.zeros(len(df), dtype=np.int8)

    ball_recovery_offensive = safe_col(df, "ball_recovery_offensive", False)
    counterpress = safe_col(df, "counterpress", False)
    interception_outcome = safe_col(df, "interception_outcome", "")
    duel_outcome = safe_col(df, "duel_outcome", "")

    for i, row in df.iterrows():
        event_type = row["type"]
        x, y = parse_loc(row.get("location"))
        if np.isnan(x) or x < 60:
            continue

        if event_type == "Ball Recovery":
            if bool(ball_recovery_offensive.iloc[i]) or bool(counterpress.iloc[i]):
                flags[i] = 1

        elif event_type == "Interception":
            outcome = str(interception_outcome.iloc[i]).strip().lower()
            if "won" in outcome or "success" in outcome:
                flags[i] = 1
            elif bool(counterpress.iloc[i]):
                flags[i] = 1

        elif event_type == "Duel":
            outcome = str(duel_outcome.iloc[i]).strip().lower()
            if "won" in outcome:
                flags[i] = 1

    return flags


def rolling_feature_from_subset(current_times, subset_times, subset_values, window_sec):
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


def prepare_raw_event(ev):
    df = ev.copy()

    sort_cols = [c for c in ["period", "minute", "second", "index"] if c in df.columns]
    df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    df["minute"] = pd.to_numeric(df["minute"], errors="coerce").fillna(0).astype(int)
    df["second"] = pd.to_numeric(df["second"], errors="coerce").fillna(0).astype(int)
    df["time_sec"] = df["minute"] * 60 + df["second"]

    if FILTER_ADMIN_EVENTS:
        df = df[~df["type"].isin(ADMIN_EVENT_TYPES)].reset_index(drop=True)

    return df


def build_patch_for_match(ev, home_team, away_team, match_id):
    df = prepare_raw_event(ev)

    df["is_box_entry_v2"] = compute_box_entry_flags_v2(df)
    df["is_high_turnover_v2"] = compute_high_turnover_flags_v2(df)

    df["state_row_id"] = np.arange(len(df))
    current_times = df["time_sec"].to_numpy()

    box_home_times = df.loc[(df["is_box_entry_v2"] == 1) & (df["team"] == home_team), "time_sec"].to_numpy()
    box_away_times = df.loc[(df["is_box_entry_v2"] == 1) & (df["team"] == away_team), "time_sec"].to_numpy()

    ht_home_times = df.loc[(df["is_high_turnover_v2"] == 1) & (df["team"] == home_team), "time_sec"].to_numpy()
    ht_away_times = df.loc[(df["is_high_turnover_v2"] == 1) & (df["team"] == away_team), "time_sec"].to_numpy()

    recent_box_entries_home_v2 = rolling_feature_from_subset(
        current_times, box_home_times, np.ones(len(box_home_times)), BOX_ENTRY_WINDOW
    )
    recent_box_entries_away_v2 = rolling_feature_from_subset(
        current_times, box_away_times, np.ones(len(box_away_times)), BOX_ENTRY_WINDOW
    )

    recent_high_turnovers_home_v2 = rolling_feature_from_subset(
        current_times, ht_home_times, np.ones(len(ht_home_times)), HIGH_TURNOVER_WINDOW
    )
    recent_high_turnovers_away_v2 = rolling_feature_from_subset(
        current_times, ht_away_times, np.ones(len(ht_away_times)), HIGH_TURNOVER_WINDOW
    )

    patch = pd.DataFrame({
        "match_id": match_id,
        "state_row_id": df["state_row_id"].values,

        "recent_box_entries_home_v2": recent_box_entries_home_v2,
        "recent_box_entries_away_v2": recent_box_entries_away_v2,
        "recent_box_entries_diff_v2": recent_box_entries_home_v2 - recent_box_entries_away_v2,

        "recent_high_turnovers_home_v2": recent_high_turnovers_home_v2,
        "recent_high_turnovers_away_v2": recent_high_turnovers_away_v2,
        "recent_high_turnovers_diff_v2": recent_high_turnovers_home_v2 - recent_high_turnovers_away_v2,
    })

    return patch, len(df)


def patch_state_table():
    state_df = load_table(STATE_TABLE_PATH)
    matches = pd.read_csv(MATCHES_FILE)

    state_df["state_row_id"] = state_df.groupby("match_id").cumcount()

    patches = []

    for _, mrow in tqdm(matches.iterrows(), total=len(matches), desc="Patching box/high-turnover"):
        match_id = mrow["match_id"]
        home_team = mrow["home_team"]
        away_team = mrow["away_team"]

        try:
            ev = load_event_file(EVENTS_DIR, match_id)
            patch_df, raw_state_len = build_patch_for_match(ev, home_team, away_team, match_id)

            old_len = (state_df["match_id"] == match_id).sum()
            if old_len != raw_state_len:
                print(f"[警告] match_id={match_id} 行数不一致 old={old_len}, new={raw_state_len}")
                continue

            patches.append(patch_df)

        except Exception as e:
            print(f"failed on match_id={match_id}: {e}")

    patch_all = pd.concat(patches, ignore_index=True)

    merged = state_df.merge(
        patch_all,
        on=["match_id", "state_row_id"],
        how="left",
        validate="one_to_one"
    )

    return merged


if __name__ == "__main__":
    merged = patch_state_table()

    print(merged.shape)
    print(merged[[
        "recent_box_entries_home_v2",
        "recent_box_entries_away_v2",
        "recent_high_turnovers_home_v2",
        "recent_high_turnovers_away_v2"
    ]].describe())

    if OUTPUT_FILE.endswith(".parquet"):
        merged.to_parquet(OUTPUT_FILE, index=False)
    else:
        merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("patched file saved:", OUTPUT_FILE)