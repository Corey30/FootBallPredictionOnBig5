# -*- coding: utf-8 -*-
"""
Real-time momentum feature construction (first half only).


Two curves per match:
  - M_eq(t)   directional curve (who is dominating)
  - M_int(t)  intensity curve  (how active the match is)

Each curve is restricted to the first half (period=1), resampled onto a
uniform grid of 91 points, and summarised by 10 curve features, giving
20 momentum features per match.

input：event_state_table_big5_2015_16.csv
Output: momentum_features_big5_2015_16.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
EVENT_STATE_CSV = SCRIPT_DIR / "real_time_data" / "event_state_table_big5_2015_16.csv"
OUTPUT_CSV = SCRIPT_DIR / "real_time_data" / "momentum_features_big5_2015_16.csv"

# ---------------------------------------------------------------------------
# Expert weights for M_eq
# ---------------------------------------------------------------------------
W_INTERCEPT = 0.07
W_XG = 0.07
W_SHOTS = 0.01
W_BOX = 0.03
W_CORNERS = 0.02
W_TURNOVERS = 0.00
W_RED = -0.23
W_SCORE = 0.09
BETA_SCORE = 0.5  # tanh saturation

# Feature list for M_int uses |w_i| from the rolling-window block
W_INT = {
    "xg": abs(W_XG),
    "shots": abs(W_SHOTS),
    "box": abs(W_BOX),
    "corners": abs(W_CORNERS),
    "turnovers": abs(W_TURNOVERS),
}

# Rolling-window diff column names in the event state table
DIFF_COLS = {
    "xg": "recent_xg_diff",
    "shots": "recent_shots_diff",
    "box": "recent_box_entries_diff_v3",
    "corners": "recent_corners_diff",
    "turnovers": "recent_high_turnovers_diff_v3",
}

# Rolling-window home/away column names for M_int
HOME_COLS = {
    "xg": "recent_xg_home",
    "shots": "recent_shots_home",
    "box": "recent_box_entries_home_v3",
    "corners": "recent_corners_home",
    "turnovers": "recent_high_turnovers_home_v3",
}
AWAY_COLS = {
    "xg": "recent_xg_away",
    "shots": "recent_shots_away",
    "box": "recent_box_entries_away_v3",
    "corners": "recent_corners_away",
    "turnovers": "recent_high_turnovers_away_v3",
}

# Grid size for resampling (paper: 91 points spanning the half)
N_GRID = 91
# Minimum number of first-half state rows required (paper: 50)
MIN_ROWS = 50
# Time constant for end-loaded decay feature (paper: 600 s)
TAU_DECAY = 600.0


# ---------------------------------------------------------------------------
# Curve computation
# ---------------------------------------------------------------------------
def compute_meq(df_match: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute directional momentum M_eq(t) for a single match (first half)."""
    meq = np.full(len(df_match), W_INTERCEPT, dtype=float)
    # rolling-window differentials
    for key, col in DIFF_COLS.items():
        w = {"xg": W_XG, "shots": W_SHOTS, "box": W_BOX,
             "corners": W_CORNERS, "turnovers": W_TURNOVERS}[key]
        if w != 0.0:
            meq = meq + w * df_match[col].to_numpy(dtype=float)
    # cumulative state
    meq = meq + W_RED * df_match["red_card_diff"].to_numpy(dtype=float)
    meq = meq + W_SCORE * np.tanh(BETA_SCORE * df_match["score_diff"].to_numpy(dtype=float))
    t = df_match["time_sec"].to_numpy(dtype=float)
    return t, meq


def compute_mint(df_match: pd.DataFrame) -> np.ndarray:
    """Compute intensity momentum M_int(t) for a single match (first half)."""
    mint = np.zeros(len(df_match), dtype=float)
    for key, w in W_INT.items():
        if w == 0.0:
            continue
        mint = mint + w * (
            df_match[HOME_COLS[key]].to_numpy(dtype=float)
            + df_match[AWAY_COLS[key]].to_numpy(dtype=float)
        )
    return mint


# ---------------------------------------------------------------------------
# Resampling onto uniform grid
# ---------------------------------------------------------------------------
def resample_curve(t: np.ndarray, m: np.ndarray, n_grid: int = N_GRID) -> np.ndarray:
    """Resample (t, m) onto a uniform grid of n_grid points spanning [t_min, t_max]."""
    if len(t) < 2:
        return np.full(n_grid, np.nan)
    t_min, t_max = float(t.min()), float(t.max())
    if t_max <= t_min:
        return np.full(n_grid, float(m[0]))
    grid = np.linspace(t_min, t_max, n_grid)
    return np.interp(grid, t, m)


# ---------------------------------------------------------------------------
# Curve feature extraction (10 features per curve)
# ---------------------------------------------------------------------------
def extract_curve_features(m_grid: np.ndarray, t_grid: np.ndarray | None = None) -> dict:
    """Extract 10 summary features from a resampled momentum curve."""
    feats: dict = {}
    if m_grid is None or np.all(np.isnan(m_grid)):
        return {k: np.nan for k in [
            "mean", "max", "min", "area_signed", "zero_crossings",
            "slope_overall", "value_at_end", "mean_last_600s",
            "slope_last_third", "area_decay_tau600s",
        ]}

    m = np.asarray(m_grid, dtype=float)
    n = len(m)
    # Build a time axis in seconds for the grid (0 .. T)
    if t_grid is None:
        # default: assume grid spans 0..2700s (typical first half)
        t_grid = np.linspace(0.0, 2700.0, n)
    t = np.asarray(t_grid, dtype=float)
    T = float(t[-1])

    # --- distribution shape ---
    feats["mean"] = float(np.mean(m))
    feats["max"] = float(np.max(m))
    feats["min"] = float(np.min(m))
    feats["area_signed"] = float(np.trapezoid(m, t))
    # zero crossings: count sign changes (ignore zeros)
    signs = np.sign(m)
    # remove zeros to avoid spurious crossings
    nonzero = signs[signs != 0]
    if len(nonzero) > 1:
        feats["zero_crossings"] = int(np.sum(np.diff(nonzero) != 0))
    else:
        feats["zero_crossings"] = 0

    # --- trend ---
    if n > 1 and np.std(t) > 0:
        slope, _ = np.polyfit(t, m, 1)
        feats["slope_overall"] = float(slope)
    else:
        feats["slope_overall"] = 0.0

    # --- end-of-half emphasis ---
    feats["value_at_end"] = float(m[-1])
    # mean over last 600s
    mask_last = t >= (T - 600.0)
    if mask_last.any():
        feats["mean_last_600s"] = float(np.mean(m[mask_last]))
    else:
        feats["mean_last_600s"] = float(m[-1])
    # slope over last third
    third_idx = int(np.ceil(n * 2 / 3))
    if n - third_idx > 1 and np.std(t[third_idx:]) > 0:
        slope_last, _ = np.polyfit(t[third_idx:], m[third_idx:], 1)
        feats["slope_last_third"] = float(slope_last)
    else:
        feats["slope_last_third"] = 0.0
    # end-loaded decay integral: ∫ M(t) * exp((t-T)/tau) dt
    decay_w = np.exp((t - T) / TAU_DECAY)
    feats["area_decay_tau600s"] = float(np.trapezoid(m * decay_w, t))

    return feats


def build_momentum_features() -> pd.DataFrame:
    print(f"[1/4] Loading event state table from {EVENT_STATE_CSV} ...")
    usecols = (
        ["match_id", "period", "time_sec", "home_team", "away_team",
         "score_diff", "red_card_diff"]
        + list(DIFF_COLS.values())
        + list(HOME_COLS.values())
        + list(AWAY_COLS.values())
    )
    # de-duplicate (some columns overlap)
    usecols = sorted(set(usecols))
    df = pd.read_csv(EVENT_STATE_CSV, usecols=usecols)
    print(f"      Loaded {len(df):,} rows, {df['match_id'].nunique()} matches")

    # keep first half only
    df = df[df["period"] == 1].copy()
    print(f"[2/4] Filtered to first half: {len(df):,} rows, {df['match_id'].nunique()} matches")

    # sort
    df = df.sort_values(["match_id", "time_sec"]).reset_index(drop=True)

    print("[3/4] Computing momentum curves and features per match ...")
    rows = []
    match_ids = df["match_id"].unique()
    for i, mid in enumerate(match_ids):
        df_m = df[df["match_id"] == mid].reset_index(drop=True)
        home_team = df_m["home_team"].iloc[0]
        away_team = df_m["away_team"].iloc[0]

        if len(df_m) < MIN_ROWS:
            # flag and assign NaN
            row = {
                "match_id": int(mid),
                "home_team": home_team,
                "away_team": away_team,
                "n_first_half_rows": int(len(df_m)),
            }
            for prefix in ("eq", "int"):
                for fname in ["mean", "max", "min", "area_signed", "zero_crossings",
                              "slope_overall", "value_at_end", "mean_last_600s",
                              "slope_last_third", "area_decay_tau600s"]:
                    row[f"{prefix}_{fname}"] = np.nan
            rows.append(row)
            continue

        # compute curves
        t_arr, meq = compute_meq(df_m)
        mint = compute_mint(df_m)

        # resample onto 91-point uniform grid
        meq_grid = resample_curve(t_arr, meq)
        mint_grid = resample_curve(t_arr, mint)

        # build a time grid in seconds for feature extraction
        t_grid = np.linspace(float(t_arr.min()), float(t_arr.max()), N_GRID)

        # extract features
        eq_feats = extract_curve_features(meq_grid, t_grid)
        int_feats = extract_curve_features(mint_grid, t_grid)

        row = {
            "match_id": int(mid),
            "home_team": home_team,
            "away_team": away_team,
            "n_first_half_rows": int(len(df_m)),
        }
        for fname, val in eq_feats.items():
            row[f"eq_{fname}"] = val
        for fname, val in int_feats.items():
            row[f"int_{fname}"] = val
        rows.append(row)

        if (i + 1) % 200 == 0:
            print(f"      processed {i + 1}/{len(match_ids)} matches")

    print(f"[4/4] Saving momentum features to {OUTPUT_CSV} ...")
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"      Done. {len(out)} matches, {out.shape[1]} columns.")
    # quick summary
    nan_count = out[[c for c in out.columns if c.startswith(("eq_", "int_"))]].isna().any(axis=1).sum()
    print(f"      Matches with NaN momentum (rows < {MIN_ROWS}): {nan_count}")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    build_momentum_features()
