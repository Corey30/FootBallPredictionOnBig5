"""Plot formula-based momentum curves for the paper (top row only)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.momentum.config import PARQUET_PATHS, EQ_RAW_COLS, INTENSITY_COLS
from src.momentum.equation import compute_momentum_eq, compute_momentum_intensity


def _resolve_parquet() -> Path:
    for p in PARQUET_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError("event_state_table parquet not found")


def main():
    pq_path = _resolve_parquet()
    intensity_cols = [c for pair in INTENSITY_COLS.values() for c in pair]
    cols = (
        ["match_id", "period", "time_sec", "score_diff", "red_card_diff"]
        + EQ_RAW_COLS
        + intensity_cols
    )
    cols = list(dict.fromkeys(cols))
    df = pd.read_parquet(pq_path, columns=cols)
    df["match_id"] = df["match_id"].astype(str)

    p1 = df[df["period"] == 1].copy()
    matches = sorted(p1["match_id"].unique())
    print(f"Total period-1 matches: {len(matches)}")

    p1["momentum_eq"] = compute_momentum_eq(p1)
    p1["momentum_intensity"] = compute_momentum_intensity(p1)

    rng = np.random.default_rng(42)
    shuffled = rng.permutation(matches)
    mid = None
    for m in shuffled:
        g = p1[p1["match_id"] == m]
        meq = g["momentum_eq"].values
        if len(meq) > 20 and np.min(meq) < -0.1 and np.max(meq) > 0.1:
            mid = m
            break
    if mid is None:
        mid = matches[len(matches) // 2]
    print(f"Selected match: {mid}")

    match_df = p1[p1["match_id"] == mid].sort_values("time_sec").reset_index(drop=True)
    t = match_df["time_sec"].values / 60.0
    m_eq = match_df["momentum_eq"].values
    m_int = match_df["momentum_intensity"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharex=True)

    ax = axes[0]
    ax.plot(t, m_eq, color="#2166ac", linewidth=1.2)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.fill_between(t, 0, m_eq, where=(m_eq > 0), alpha=0.15, color="#2166ac")
    ax.fill_between(t, 0, m_eq, where=(m_eq < 0), alpha=0.15, color="#b2182b")
    ax.set_ylabel("$M_{eq}(t)$")
    ax.set_title("Directional momentum curve")
    ax.annotate("Home\ndominant", xy=(0.02, 0.92), xycoords="axes fraction",
                fontsize=8, color="#2166ac", va="top")
    ax.annotate("Away\ndominant", xy=(0.02, 0.08), xycoords="axes fraction",
                fontsize=8, color="#b2182b", va="bottom")
    ax.set_xlabel("First-half time (min)")

    ax = axes[1]
    ax.plot(t, m_int, color="#4dac26", linewidth=1.2)
    ax.fill_between(t, 0, m_int, alpha=0.15, color="#4dac26")
    ax.set_ylabel("$M_{int}(t)$")
    ax.set_title("Intensity momentum curve")
    ax.set_xlabel("First-half time (min)")

    plt.tight_layout()

    out_pdf = ROOT / "paper" / "momentum_curves_formula.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_pdf}")

    out_png = ROOT / "paper" / "momentum_curves_formula.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")

    plt.close()


if __name__ == "__main__":
    main()
