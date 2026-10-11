"""CLI: compute per-event M_eq / M_intensity and save parquet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.momentum.config import (
    EQ_RAW_COLS,
    INTENSITY_COLS,
    PARQUET_PATHS,
    EXPERT_WEIGHTS,
    INTENSITY_WEIGHTS,
    BETA_SCORE,
)
from src.momentum.equation import compute_momentum_eq, compute_momentum_intensity


def _resolve_parquet() -> Path:
    for p in PARQUET_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError("event_state_table parquet not found")


def run(output_dir: Path | None = None) -> Path:
    out_dir = output_dir or ROOT / "outputs" / "momentum"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = _resolve_parquet()

    intensity_cols = [c for pair in INTENSITY_COLS.values() for c in pair]
    keep = (
        ["match_id", "state_row_id", "period", "time_sec", "score_diff"]
        + EQ_RAW_COLS
        + intensity_cols
    )
    df = pd.read_parquet(data_path, columns=list(dict.fromkeys(keep)))

    print(f"[momentum] rows={len(df):,} matches={df['match_id'].nunique()}")
    out = df[["match_id", "state_row_id", "period", "time_sec"]].copy()
    out["momentum_eq"] = compute_momentum_eq(df)
    out["momentum_intensity"] = compute_momentum_intensity(df)

    pq_path = out_dir / "equation_momentum.parquet"
    out.to_parquet(pq_path, index=False)

    weights = {
        "M_eq_weights": EXPERT_WEIGHTS,
        "M_intensity_weights": INTENSITY_WEIGHTS,
        "beta_score": BETA_SCORE,
    }
    with open(out_dir / "equation_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)

    print(f"[momentum] saved -> {pq_path}")
    return pq_path


if __name__ == "__main__":
    run()
