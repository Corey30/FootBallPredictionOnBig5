"""Equation-based momentum curves (M_eq, M_intensity) and curve feature extraction."""

from src.momentum.config import (
    BETA_SCORE,
    CAUSAL_SUFFIXES,
    EQ_CORE_COLS,
    EXPERT_WEIGHTS,
    INTENSITY_COLS,
    INTENSITY_WEIGHTS,
)
from src.momentum.curve_features import extract_curve_features, feature_names
from src.momentum.causal import extract_causal_curve_features
from src.momentum.equation import compute_momentum_eq, compute_momentum_intensity

__all__ = [
    "BETA_SCORE",
    "CAUSAL_SUFFIXES",
    "EQ_CORE_COLS",
    "EXPERT_WEIGHTS",
    "INTENSITY_COLS",
    "INTENSITY_WEIGHTS",
    "compute_momentum_eq",
    "compute_momentum_intensity",
    "extract_curve_features",
    "extract_causal_curve_features",
    "feature_names",
]
