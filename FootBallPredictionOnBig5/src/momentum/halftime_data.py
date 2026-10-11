"""Shared 2015-16 halftime experiment data: P1 aggregates, merge table, splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.data.loader import clean_data
from src.features.halftime_features import process_halftime_features
from src.momentum.config import PARQUET_PATHS

ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = ROOT / "merged_data"
MOMENTUM_MERGED = MERGED_DIR / "feature_engineer5_momentum" / "all_leagues_2015-2016_momentum.merged.csv"
BRIDGE_PATH = MERGED_DIR / "feature_engineer5_momentum" / "momentum_match_bridge.csv"
AGG_CACHE = MERGED_DIR / "feature_engineer5_momentum" / "match_period1_aggregates.parquet"
SEASON = "2015-2016"

MOMENTUM_PREFIXES = ("eq_", "mod_raw_", "mod_ewm_", "mod_conf_w_", "mod_conf_", "n_rows_period1")
AGG_PREFIXES = ("agg_", "p1_", "proxy_")
SPARSE_PREFIXES = ("sparse_",)

NUMERIC_STATE_COLS = [
    "score_diff",
    "red_card_diff",
    "recent_shots_home",
    "recent_shots_away",
    "recent_xg_home",
    "recent_xg_away",
    "recent_corners_home",
    "recent_corners_away",
    "recent_box_entries_home",
    "recent_box_entries_away",
    "recent_high_turnovers_home",
    "recent_high_turnovers_away",
    "recent_shots_diff",
    "recent_xg_diff",
    "recent_corners_diff",
    "recent_box_entries_diff",
    "recent_high_turnovers_diff",
]


def resolve_parquet() -> Path:
    for p in PARQUET_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError("event_state_table parquet not found")


def _slope(y: np.ndarray, t: np.ndarray) -> float:
    if len(t) < 2:
        return 0.0
    t = t.astype(float)
    y = y.astype(float)
    if np.std(t) < 1e-9:
        return 0.0
    coef = np.polyfit(t, y, 1)
    return float(coef[0])


def _zero_crossings(y: np.ndarray) -> int:
    if len(y) < 2:
        return 0
    s = np.sign(y.astype(float))
    return int(np.sum(s[1:] != s[0:-1]))


def _area_trapz(y: np.ndarray, t: np.ndarray) -> float:
    if len(t) < 2:
        return 0.0
    return float(np.trapezoid(y.astype(float), t.astype(float)))


def build_period1_aggregates(parquet_path: Path) -> pd.DataFrame:
    """One row per StatsBomb match_id with first-half event aggregates."""
    pq = pd.read_parquet(parquet_path)
    p1 = pq[pq["period"] == 1].copy()
    rows = []

    for mid, g in p1.groupby("match_id", sort=False):
        g = g.sort_values("time_sec")
        t = g["time_sec"].to_numpy()
        row: dict = {"statsbomb_match_id": str(mid), "p1_n_events": len(g)}
        if len(t):
            row["p1_duration_sec"] = float(t.max())
            t_end = float(t.max())

            for col in NUMERIC_STATE_COLS:
                if col not in g.columns:
                    continue
                y = pd.to_numeric(g[col], errors="coerce").fillna(0).to_numpy()
                prefix = f"agg_{col}"
                row[f"{prefix}_mean"] = float(np.mean(y))
                row[f"{prefix}_max"] = float(np.max(y))
                row[f"{prefix}_min"] = float(np.min(y))
                row[f"{prefix}_last"] = float(y[-1])
                row[f"{prefix}_std"] = float(np.std(y))

                last_win = t >= (t_end - 600)
                if last_win.any():
                    row[f"{prefix}_mean_last600"] = float(np.mean(y[last_win]))
                else:
                    row[f"{prefix}_mean_last600"] = float(np.mean(y))

                third = t >= (t_end - t_end / 3) if t_end > 0 else np.ones_like(t, dtype=bool)
                row[f"{prefix}_slope_last_third"] = _slope(y[third], t[third])
                row[f"{prefix}_slope_all"] = _slope(y, t)

            if "recent_xg_diff" in g.columns:
                yd = pd.to_numeric(g["recent_xg_diff"], errors="coerce").fillna(0).to_numpy()
                row["proxy_area_signed"] = _area_trapz(yd, t)
                row["proxy_zero_crossings"] = _zero_crossings(yd)
            if "score_diff" in g.columns:
                row["proxy_score_diff_last"] = float(
                    pd.to_numeric(g["score_diff"], errors="coerce").fillna(0).iloc[-1]
                )

        rows.append(row)

    out = pd.DataFrame(rows)
    out["statsbomb_match_id"] = out["statsbomb_match_id"].astype(str)
    return out


def load_aggregates(rebuild: bool = False) -> pd.DataFrame:
    if not rebuild and AGG_CACHE.exists():
        print(f"[Aggregates] Loaded cache: {AGG_CACHE.name}")
        return pd.read_parquet(AGG_CACHE)
    print("[Aggregates] Building from parquet (first run may take ~1 min)...")
    agg = build_period1_aggregates(resolve_parquet())
    AGG_CACHE.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(AGG_CACHE, index=False)
    print(f"[Aggregates] Cached {len(agg):,} matches -> {AGG_CACHE.name}")
    return agg


def load_merged_table() -> pd.DataFrame:
    df = pd.read_csv(MOMENTUM_MERGED, encoding="utf-8-sig")
    df = df[df["season"] == SEASON].copy()
    if BRIDGE_PATH.exists():
        bridge = pd.read_csv(BRIDGE_PATH, encoding="utf-8-sig")
        bridge["statsbomb_match_id"] = bridge["statsbomb_match_id"].astype(str)
        if "statsbomb_match_id" in df.columns:
            df = df.drop(columns=["statsbomb_match_id"])
        df = df.merge(
            bridge[["match_id", "statsbomb_match_id"]],
            on="match_id",
            how="left",
        )
    df["statsbomb_match_id"] = df["statsbomb_match_id"].astype(str)
    return clean_data(df)


def chronological_split(df: pd.DataFrame, train_ratio=0.6, val_ratio=0.2):
    date_col = "match_datetime" if "match_datetime" in df.columns else "date_GMT"
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    return df.iloc[:t1].copy(), df.iloc[t1:t2].copy(), df.iloc[t2:].copy()


def agg_feature_columns(agg: pd.DataFrame) -> list[str]:
    return [c for c in agg.columns if c != "statsbomb_match_id"]


def process_halftime_with_extra(
    df: pd.DataFrame, extra: pd.DataFrame | None
) -> tuple[pd.DataFrame, np.ndarray]:
    X, y = process_halftime_features(df)
    drop_cols = [
        c
        for c in X.columns
        if c.startswith(MOMENTUM_PREFIXES)
        or c.startswith(AGG_PREFIXES)
        or c.startswith(SPARSE_PREFIXES)
        or c == "statsbomb_match_id"
    ]
    if drop_cols:
        X = X.drop(columns=drop_cols)
    if extra is not None and len(extra.columns):
        block = extra.reset_index(drop=True).copy()
        for c in block.columns:
            block[c] = pd.to_numeric(block[c], errors="coerce")
        X = pd.concat([X.reset_index(drop=True), block], axis=1)
    X = X.loc[:, ~X.columns.duplicated()].reset_index(drop=True)
    y = y.reset_index(drop=True)
    mask = y.notna().to_numpy()
    le = LabelEncoder()
    return X.loc[mask], le.fit_transform(y.astype(str)[mask])
