"""Pointwise M_eq(t) and M_intensity(t) from event state rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.momentum.config import (
    BETA_SCORE,
    EQ_RAW_COLS,
    EXPERT_WEIGHTS,
    INTENSITY_COLS,
    INTENSITY_WEIGHTS,
)


def compute_momentum_eq(df: pd.DataFrame) -> np.ndarray:
    """Directional momentum: home − away (+ score/red)."""
    m = np.full(len(df), EXPERT_WEIGHTS["intercept"], dtype=np.float64)
    for col in EQ_RAW_COLS:
        m += EXPERT_WEIGHTS[col] * pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
    sd = pd.to_numeric(df["score_diff"], errors="coerce").fillna(0.0).values
    m += EXPERT_WEIGHTS["tanh_score_diff"] * np.tanh(BETA_SCORE * sd)
    return m


def compute_momentum_intensity(df: pd.DataFrame) -> np.ndarray:
    """Match intensity: weighted sum of (home + away) event rates."""
    m = np.zeros(len(df), dtype=np.float64)
    for stem, (home_col, away_col) in INTENSITY_COLS.items():
        w = INTENSITY_WEIGHTS[stem]
        if w == 0.0:
            continue
        h = pd.to_numeric(df[home_col], errors="coerce").fillna(0.0).values
        a = pd.to_numeric(df[away_col], errors="coerce").fillna(0.0).values
        m += w * np.clip(h + a, 0.0, None)
    return m
