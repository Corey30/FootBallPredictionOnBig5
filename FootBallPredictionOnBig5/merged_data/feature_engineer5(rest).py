import pandas as pd
import numpy as np
from pathlib import Path



BASE_DIR = Path(r"d:/program/SoccerPre/merged_data")

INPUT_DIR = BASE_DIR / "feature_engineer4"
OUTPUT_DIR = BASE_DIR / "feature_engineer5"
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_PATTERNS = [
    "*_selected_advanced_FINAL.csv"
]

WINDOW = 5
EWM_SPAN = 5
DENSITY_DAYS = 14

RESET_EACH_SEASON = False
RUN_CHECKS = True



NEW_FEATURES = [
    "home_rest_days",
    "away_rest_days",

    "home_match_density_mom_5",
    "away_match_density_mom_5",
    "home_match_density_ewm_5",
    "away_match_density_ewm_5",

    "home_travel_km_team",
    "away_travel_fatigue_score",

    "home_travel_km_mom_5",
    "away_travel_km_mom_5",
    "home_travel_km_ewm_5",
    "away_travel_km_ewm_5",

    "home_travel_fatigue_mom_5",
    "away_travel_fatigue_mom_5",
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


def prev_rolling_mean(s: pd.Series, window: int) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=1).mean()


def prev_ewm_mean(s: pd.Series, span: int) -> pd.Series:
    return s.shift(1).ewm(
        span=span,
        min_periods=1,
        adjust=False,
        ignore_na=True,
    ).mean()



def detect_required_columns(df: pd.DataFrame) -> dict:
    cols = {}

    cols["league"] = detect_col(df, ["league", "league_key"])
    cols["season"] = detect_col(df, ["season"])
    cols["timestamp"] = detect_col(df, ["timestamp"])
    cols["home_team"] = detect_col(df, ["home_team_name", "home_team", "home"])
    cols["away_team"] = detect_col(df, ["away_team_name", "away_team", "away"])

    cols["home_match_density"] = detect_col(df, ["home_match_density"])
    cols["away_match_density"] = detect_col(df, ["away_match_density"])

    cols["travel_distance_km"] = detect_col(df, [
        "travel_distance_km",
        "distance_km",
        "away_travel_km",
        "travel_km",
    ])

    required = [
        "season",
        "home_team",
        "away_team",
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
        bad_count = int(df["match_datetime"].isna().sum())
        raise ValueError(f"{input_path.name} 有 {bad_count} 行 match_datetime 无法解析。")

    df = df.drop(columns=[c for c in NEW_FEATURES if c in df.columns], errors="ignore")

    df["_orig_idx_fe5"] = np.arange(len(df))

    sort_cols = ["league_key", "match_datetime"]

    if cols["timestamp"] is not None:
        sort_cols.append(cols["timestamp"])

    sort_cols.append("_orig_idx_fe5")

    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["_row_id_fe5"] = np.arange(len(df))

    return df, cols



def compute_density_from_schedule(team_dates: pd.Series, days: int = 14) -> pd.Series:
    dates = pd.to_datetime(team_dates, errors="coerce").to_numpy()
    density = []

    for current_date in dates:
        start_date = current_date - np.timedelta64(days, "D")
        count = ((dates < current_date) & (dates >= start_date)).sum()
        density.append(int(count))

    return pd.Series(density, index=team_dates.index)


def build_team_long(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    home_col = cols["home_team"]
    away_col = cols["away_team"]

    if cols["travel_distance_km"] is None:
        print("[WARN] 没找到 travel_distance_km / distance_km，旅行距离相关特征会是 NaN。")
        travel_distance = pd.Series(np.nan, index=df.index)
    else:
        travel_distance = pd.to_numeric(df[cols["travel_distance_km"]], errors="coerce")

    if cols["home_match_density"] is not None and cols["away_match_density"] is not None:
        home_density = pd.to_numeric(df[cols["home_match_density"]], errors="coerce")
        away_density = pd.to_numeric(df[cols["away_match_density"]], errors="coerce")
    else:
        home_density = pd.Series(np.nan, index=df.index)
        away_density = pd.Series(np.nan, index=df.index)

    home_long = pd.DataFrame({
        "row_id": df["_row_id_fe5"],
        "league_key": df["league_key"],
        "season": df[cols["season"]].astype(str),
        "match_datetime": df["match_datetime"],
        "team": df[home_col].astype(str),
        "side": "home",

        # 当前比赛主队旅行距离默认 0
        "travel_km_current": 0.0,

        # 当前比赛主队已有赛程密度
        "match_density_current": home_density,
    })

    away_long = pd.DataFrame({
        "row_id": df["_row_id_fe5"],
        "league_key": df["league_key"],
        "season": df[cols["season"]].astype(str),
        "match_datetime": df["match_datetime"],
        "team": df[away_col].astype(str),
        "side": "away",

        # 当前比赛客队旅行距离 = travel_distance_km
        "travel_km_current": travel_distance,

        # 当前比赛客队已有赛程密度
        "match_density_current": away_density,
    })

    team_long = pd.concat([home_long, away_long], ignore_index=True)

    team_long = team_long.sort_values(
        ["league_key", "team", "match_datetime", "row_id"]
    ).reset_index(drop=True)

    group_cols = ["league_key", "team"]
    if RESET_EACH_SEASON:
        group_cols = ["league_key", "season", "team"]

    # rest days
    team_long["prev_match_datetime"] = (
        team_long.groupby(group_cols, sort=False)["match_datetime"]
        .shift(1)
    )

    team_long["rest_days"] = (
        team_long["match_datetime"] - team_long["prev_match_datetime"]
    ).dt.total_seconds() / 86400.0

    if team_long["match_density_current"].isna().all():
        team_long["match_density_current"] = (
            team_long
            .groupby(group_cols, group_keys=False)["match_datetime"]
            .apply(lambda s: compute_density_from_schedule(s, DENSITY_DAYS))
        )


    rest_safe = team_long["rest_days"].clip(lower=0).fillna(7.0)
    density_safe = pd.to_numeric(team_long["match_density_current"], errors="coerce").fillna(0)

    team_long["travel_fatigue_score_current"] = (
        np.log1p(pd.to_numeric(team_long["travel_km_current"], errors="coerce").fillna(0))
        * (1.0 + density_safe)
        / np.sqrt(rest_safe + 1.0)
    )

    # 历史 momentum / EWM
    g = team_long.groupby(group_cols, sort=False)

    team_long["match_density_mom_5"] = g["match_density_current"].transform(
        lambda s: prev_rolling_mean(s, WINDOW)
    )

    team_long["match_density_ewm_5"] = g["match_density_current"].transform(
        lambda s: prev_ewm_mean(s, EWM_SPAN)
    )

    team_long["travel_km_mom_5"] = g["travel_km_current"].transform(
        lambda s: prev_rolling_mean(s, WINDOW)
    )

    team_long["travel_km_ewm_5"] = g["travel_km_current"].transform(
        lambda s: prev_ewm_mean(s, EWM_SPAN)
    )

    team_long["travel_fatigue_mom_5"] = g["travel_fatigue_score_current"].transform(
        lambda s: prev_rolling_mean(s, WINDOW)
    )

    return team_long



def merge_back_to_matches(df: pd.DataFrame, team_long: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    feature_cols = [
        "row_id",
        "rest_days",
        "match_density_mom_5",
        "match_density_ewm_5",
        "travel_km_current",
        "travel_km_mom_5",
        "travel_km_ewm_5",
        "travel_fatigue_score_current",
        "travel_fatigue_mom_5",
    ]

    home_features = (
        team_long[team_long["side"] == "home"][feature_cols]
        .set_index("row_id")
        .rename(columns={
            "rest_days": "home_rest_days",
            "match_density_mom_5": "home_match_density_mom_5",
            "match_density_ewm_5": "home_match_density_ewm_5",
            "travel_km_current": "home_travel_km_team",
            "travel_km_mom_5": "home_travel_km_mom_5",
            "travel_km_ewm_5": "home_travel_km_ewm_5",
            "travel_fatigue_mom_5": "home_travel_fatigue_mom_5",
            "travel_fatigue_score_current": "_home_travel_fatigue_score_current_tmp",
        })
    )

    away_features = (
        team_long[team_long["side"] == "away"][feature_cols]
        .set_index("row_id")
        .rename(columns={
            "rest_days": "away_rest_days",
            "match_density_mom_5": "away_match_density_mom_5",
            "match_density_ewm_5": "away_match_density_ewm_5",
            "travel_km_current": "_away_travel_km_team_tmp",
            "travel_km_mom_5": "away_travel_km_mom_5",
            "travel_km_ewm_5": "away_travel_km_ewm_5",
            "travel_fatigue_score_current": "away_travel_fatigue_score",
            "travel_fatigue_mom_5": "away_travel_fatigue_mom_5",
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

    out = out.set_index("_row_id_fe5", drop=False)
    out = out.join(home_features, how="left")
    out = out.join(away_features, how="left")
    out = out.reset_index(drop=True)

    out = out.drop(
        columns=[
            "_home_travel_fatigue_score_current_tmp",
            "_away_travel_km_team_tmp",
        ],
        errors="ignore"
    )

    return out

def validate_features(df: pd.DataFrame, label: str):
    if not RUN_CHECKS:
        return

    print("\n" + "-" * 80)
    print(f"CHECK: {label}")
    print("-" * 80)

    print("Rows:", len(df))
    print("Columns:", df.shape[1])

    print("\nNew feature missing values:")
    print(df[NEW_FEATURES].isna().sum().to_string())

    if "match_id" in df.columns:
        dup = int(df["match_id"].duplicated().sum())
        print("\nDuplicate match_id:", dup)
        if dup > 0:
            raise RuntimeError(f"{label}: duplicated match_id {dup}")

    non_negative_cols = [
        "home_rest_days",
        "away_rest_days",
        "home_match_density_mom_5",
        "away_match_density_mom_5",
        "home_match_density_ewm_5",
        "away_match_density_ewm_5",
        "home_travel_km_team",
        "away_travel_fatigue_score",
        "home_travel_km_mom_5",
        "away_travel_km_mom_5",
        "home_travel_km_ewm_5",
        "away_travel_km_ewm_5",
        "home_travel_fatigue_mom_5",
        "away_travel_fatigue_mom_5",
    ]

    bad_negative = {}
    for c in non_negative_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        bad_negative[c] = int((s.dropna() < 0).sum())

    print("\nNegative value counts:")
    print(pd.Series(bad_negative).to_string())

    if any(v > 0 for v in bad_negative.values()):
        raise RuntimeError(f"{label}: 有新增特征出现负值。")

    # home_travel_km_team 应该全为 0
    home_travel_nonzero = int((pd.to_numeric(df["home_travel_km_team"], errors="coerce").fillna(0).abs() > 1e-9).sum())
    print("\nhome_travel_km_team non-zero rows:", home_travel_nonzero)

    if home_travel_nonzero > 0:
        raise RuntimeError(f"{label}: home_travel_km_team 应该为 0，但出现非 0。")



def build_features_for_file(input_path: Path, output_path: Path) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print(f"PROCESSING: {input_path.name}")
    print("=" * 80)

    df, cols = load_and_prepare_data(input_path)

    rows_before = len(df)
    cols_before = df.shape[1]

    team_long = build_team_long(df, cols)
    df = merge_back_to_matches(df, team_long)

    missing = [c for c in NEW_FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(f"以下特征未生成: {missing}")

    if len(df) != rows_before:
        raise RuntimeError(f"行数变化: before={rows_before}, after={len(df)}")

    validate_features(df, input_path.name)

    df = df.drop(columns=["_orig_idx_fe5", "_row_id_fe5"], errors="ignore")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns before: {cols_before}")
    print(f"Columns after:  {df.shape[1]}")
    print(f"New features added: {len(NEW_FEATURES)}")

    return df



def main():
    print("INPUT_DIR:", INPUT_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"输入目录不存在: {INPUT_DIR}")

    files = []

    for pattern in INPUT_PATTERNS:
        files.extend(INPUT_DIR.glob(pattern))

    files = sorted(set(files))

    files = [
        f for f in files
        if f.name.startswith(("England_", "France_", "Germany_", "Italy_", "Spain_"))
        and "summary" not in f.name.lower()
        and "failure" not in f.name.lower()
        and "failures" not in f.name.lower()
    ]

    if not files:
        print("\nAll CSV files in INPUT_DIR:")
        for f in INPUT_DIR.glob("*.csv"):
            print("-", f.name)

        raise FileNotFoundError(f"没有在 {INPUT_DIR} 找到五大联赛输入文件。")

    print("\nMatched input files:")
    for f in files:
        print("-", f.name)

    summary = []

    for input_path in files:
        output_path = OUTPUT_DIR / input_path.name.replace(
            ".csv",
            "_rest_travel_FINAL.csv"
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

    summary_path = OUTPUT_DIR / "rest_travel_feature_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("ALL DONE")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("\nSaved summary:")
    print(summary_path)


if __name__ == "__main__":
    main()