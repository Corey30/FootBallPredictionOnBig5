"""Configuration for equation-based momentum (aligned with teammate src.zip)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PARQUET_PATHS = [
    ROOT / "merged_data" / "event_state_table_big5_2015_16_v3(1).parquet",
    ROOT / "merged_data" / "event_state_table_big5_2015_16_v3.parquet",
    ROOT / "event_state_table_big5_2015_16_v3(1).parquet",
    ROOT / "event_state_table_big5_2015_16_v3.parquet",
]

EXPERT_WEIGHTS = {
    "intercept": 0.07,
    "recent_xg_diff": 0.07,
    "recent_shots_diff": 0.01,
    "recent_box_entries_diff_v3": 0.03,
    "recent_corners_diff": 0.02,
    "recent_high_turnovers_diff_v3": 0.00,
    "red_card_diff": -0.23,
    "tanh_score_diff": 0.09,
}
BETA_SCORE = 0.5

INTENSITY_WEIGHTS = {
    "recent_xg": 0.07,
    "recent_shots": 0.01,
    "recent_box_entries_v3": 0.03,
    "recent_corners": 0.02,
    "recent_high_turnovers_v3": 0.00,
}

EQ_RAW_COLS = [
    "recent_xg_diff",
    "recent_shots_diff",
    "recent_box_entries_diff_v3",
    "recent_corners_diff",
    "recent_high_turnovers_diff_v3",
    "red_card_diff",
]

INTENSITY_COLS = {
    "recent_xg": ("recent_xg_home", "recent_xg_away"),
    "recent_shots": ("recent_shots_home", "recent_shots_away"),
    "recent_box_entries_v3": ("recent_box_entries_home_v3", "recent_box_entries_away_v3"),
    "recent_corners": ("recent_corners_home", "recent_corners_away"),
    "recent_high_turnovers_v3": ("recent_high_turnovers_home_v3", "recent_high_turnovers_away_v3"),
}

GRID_SIZE = 91
LAST_WINDOWS_SEC = [600]
DECAY_TAUS_SEC = [600]
MIN_PERIOD1_ROWS = 50
MIN_PREFIX_ROWS = 10

CAUSAL_SUFFIXES = [
    "mean",
    "max",
    "min",
    "value_at_end",
    "mean_last_600s",
    "slope_overall",
    "slope_last_third",
    "area_signed",
    "area_decay_tau600s",
    "zero_crossings",
]

EQ_CORE_COLS = [
    "eq_mean",
    "eq_max",
    "eq_min",
    "eq_value_at_end",
    "eq_mean_last_600s",
    "eq_slope_overall",
    "eq_slope_last_third",
    "eq_area_signed",
    "eq_area_decay_tau600s",
    "eq_zero_crossings",
]

INT_CORE_COLS = [c.replace("eq_", "int_") for c in EQ_CORE_COLS]

# ceq_* suffix -> eq_* column for CSV comparison
CEQ_TO_EQ = {f"ceq_{s}": f"eq_{s}" for s in CAUSAL_SUFFIXES}
CINT_TO_INT = {f"cint_{s}": f"int_{s}" for s in CAUSAL_SUFFIXES}
