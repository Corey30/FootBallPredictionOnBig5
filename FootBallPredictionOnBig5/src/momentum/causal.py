"""Causal prefix curve features for real-time next-goal (ceq_*, cint_*)."""

from __future__ import annotations

import numpy as np

from src.momentum.config import CAUSAL_SUFFIXES, MIN_PREFIX_ROWS
from src.momentum.curve_features import _resample_uniform
from src.momentum.config import DECAY_TAUS_SEC, GRID_SIZE, LAST_WINDOWS_SEC


def extract_causal_curve_features(
    time_sec: np.ndarray,
    values: np.ndarray,
    prefix: str,
    *,
    min_rows: int = MIN_PREFIX_ROWS,
) -> dict[str, float]:
    names = [f"{prefix}_{s}" for s in CAUSAL_SUFFIXES]
    nan_out = {n: 0.0 for n in names}

    if len(time_sec) < min_rows:
        return nan_out

    finite = np.isfinite(values)
    if finite.sum() < min_rows:
        return nan_out

    t = time_sec[finite].astype(np.float64)
    m = values[finite].astype(np.float64)

    try:
        t_grid, m_grid = _resample_uniform(t, m, GRID_SIZE)
    except ValueError:
        return nan_out

    t_abs = t_grid - t_grid[0]
    t_end = float(t_abs[-1])
    if t_end <= 0:
        return nan_out

    feats: dict[str, float] = {}
    feats[f"{prefix}_mean"] = float(np.mean(m_grid))
    feats[f"{prefix}_max"] = float(np.max(m_grid))
    feats[f"{prefix}_min"] = float(np.min(m_grid))
    feats[f"{prefix}_value_at_end"] = float(m_grid[-1])

    for w in LAST_WINDOWS_SEC:
        mask = t_abs >= (t_end - w)
        feats[f"{prefix}_mean_last_{int(w)}s"] = (
            float(np.mean(m_grid[mask])) if mask.any() else float(m_grid[-1])
        )

    slope, _ = np.polyfit(t_abs, m_grid, 1)
    feats[f"{prefix}_slope_overall"] = float(slope)

    mask = t_abs >= (t_end - t_end / 3.0)
    if mask.sum() >= 2:
        s, _ = np.polyfit(t_abs[mask], m_grid[mask], 1)
        feats[f"{prefix}_slope_last_third"] = float(s)
    else:
        feats[f"{prefix}_slope_last_third"] = 0.0

    feats[f"{prefix}_area_signed"] = float(np.trapezoid(m_grid, t_abs))

    for tau in DECAY_TAUS_SEC:
        w_decay = np.exp((t_abs - t_end) / tau)
        feats[f"{prefix}_area_decay_tau{int(tau)}s"] = float(
            np.trapezoid(m_grid * w_decay, t_abs)
        )

    sgn = np.sign(m_grid)
    sgn[sgn == 0] = 1
    feats[f"{prefix}_zero_crossings"] = float(np.sum(np.diff(sgn) != 0))

    for n in names:
        feats.setdefault(n, 0.0)
    return feats
