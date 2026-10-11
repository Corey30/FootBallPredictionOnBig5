"""Match-level curve summaries (teammate build_features / period==1)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.momentum.config import DECAY_TAUS_SEC, GRID_SIZE, LAST_WINDOWS_SEC, MIN_PERIOD1_ROWS


def feature_names(prefix: str) -> list[str]:
    names = [f"{prefix}_mean", f"{prefix}_max", f"{prefix}_min", f"{prefix}_value_at_end"]
    for w in LAST_WINDOWS_SEC:
        names.append(f"{prefix}_mean_last_{int(w)}s")
    names += [f"{prefix}_slope_overall", f"{prefix}_slope_last_third", f"{prefix}_area_signed"]
    for tau in DECAY_TAUS_SEC:
        names.append(f"{prefix}_area_decay_tau{int(tau)}s")
    names.append(f"{prefix}_zero_crossings")
    return names


def _resample_uniform(t: np.ndarray, m: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    if np.any(np.diff(t) <= 0):
        order = np.argsort(t, kind="stable")
        t = t[order]
        m = m[order]
        unique_t, inv = np.unique(t, return_inverse=True)
        sums = np.bincount(inv, weights=m)
        counts = np.bincount(inv)
        m = sums / counts
        t = unique_t
    if len(t) < 2:
        raise ValueError("fewer than 2 unique time points")
    grid = np.linspace(t[0], t[-1], n)
    return grid, np.interp(grid, t, m)


def extract_curve_features(
    time_sec: np.ndarray,
    values: np.ndarray,
    prefix: str,
    *,
    min_rows: int = MIN_PERIOD1_ROWS,
) -> Dict[str, float]:
    """10 features for one curve segment (typically period==1)."""
    names = feature_names(prefix)
    nan_out = {n: float("nan") for n in names}

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

    feats: Dict[str, float] = {}
    feats[f"{prefix}_mean"] = float(np.mean(m_grid))
    feats[f"{prefix}_max"] = float(np.max(m_grid))
    feats[f"{prefix}_min"] = float(np.min(m_grid))
    feats[f"{prefix}_value_at_end"] = float(m_grid[-1])

    for w in LAST_WINDOWS_SEC:
        mask = t_abs >= (t_end - w)
        feats[f"{prefix}_mean_last_{int(w)}s"] = (
            float(np.mean(m_grid[mask])) if mask.any() else float("nan")
        )

    slope, _ = np.polyfit(t_abs, m_grid, 1)
    feats[f"{prefix}_slope_overall"] = float(slope)

    mask = t_abs >= (t_end - t_end / 3.0)
    if mask.sum() >= 2:
        s, _ = np.polyfit(t_abs[mask], m_grid[mask], 1)
        feats[f"{prefix}_slope_last_third"] = float(s)
    else:
        feats[f"{prefix}_slope_last_third"] = float("nan")

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
        feats.setdefault(n, float("nan"))
    return feats
