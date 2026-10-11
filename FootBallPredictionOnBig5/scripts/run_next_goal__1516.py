
"""
Combined Formula-based real-time next-goal experiments (2015-2016).

Includes:
  - Realtime experiment (default)
  - Time-series cross-validation (--cv)
  - Regularization/overfitting comparison (--regularized)

Outputs:
  - outputs/reports/next_goal_1516_formula_realtime_3class.csv
  - outputs/reports/next_goal_1516_formula_timeseries_cv_3class.csv (if --cv)
  - outputs/reports/next_goal_1516_formula_regularized_3class.csv (if --regularized)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.config import MODEL_PARAMS
from src.momentum.config import (
    CAUSAL_SUFFIXES,
    EQ_RAW_COLS,
    INTENSITY_COLS,
    PARQUET_PATHS,
)
from src.momentum.causal import extract_causal_curve_features
from src.momentum.equation import compute_momentum_eq, compute_momentum_intensity

MERGED_CANDIDATES = [
    ROOT / "merged_data/feature_engineer5_momentum/all_leagues_2015-2016_momentum.merged.csv",
    ROOT / "merged_data/feature_engineer5_momentum_formula_curve/all_leagues_2015-2016_momentum.merged.csv",
    ROOT / "merged_data/feature_engineer_real_momentum/all_leagues_2015-2016_momentum.merged.csv",
]
BRIDGE_CANDIDATES = [
    ROOT / "merged_data/feature_engineer5_momentum/momentum_match_bridge.csv",
    ROOT / "merged_data/feature_engineer5_momentum_formula_curve/momentum_match_bridge.csv",
]
SEASON = "2015-2016"

STATE_COLS = [
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

CAUSAL_EQ_CORE = [f"ceq_{s}" for s in CAUSAL_SUFFIXES]
CAUSAL_INT_CORE = [f"cint_{s}" for s in CAUSAL_SUFFIXES]
CAUSAL_BOTH_CORE = CAUSAL_EQ_CORE + CAUSAL_INT_CORE

VARIANTS = ["state_only", "state_plus_causal_both_core"]


def _resolve_parquet() -> Path:
    for p in PARQUET_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError("event_state_table parquet not found")


def _find_merged() -> Path:
    for p in MERGED_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"No merged CSV found in candidates: {MERGED_CANDIDATES}")


def _find_bridge() -> Path:
    for p in BRIDGE_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"No bridge CSV found in candidates: {BRIDGE_CANDIDATES}")


def match_chronological_order() -> list[str]:
    merged_path = _find_merged()
    bridge_path = _find_bridge()
    merged = pd.read_csv(merged_path, encoding="utf-8-sig", usecols=["match_id", "match_datetime", "season"])
    merged = merged[merged["season"] == SEASON]
    bridge = pd.read_csv(bridge_path, encoding="utf-8-sig")
    bridge["statsbomb_match_id"] = bridge["statsbomb_match_id"].astype(str)
    df = merged.merge(bridge[["match_id", "statsbomb_match_id"]], on="match_id", how="inner")
    df["match_datetime"] = pd.to_datetime(df["match_datetime"], errors="coerce")
    df = df.sort_values("match_datetime")
    print(f"[Data] merged={merged_path.name}, bridge={bridge_path.name}, matches={len(df)}")
    return df["statsbomb_match_id"].astype(str).tolist()


def split_match_ids(match_ids: list[str], train_ratio=0.6, val_ratio=0.2):
    n = len(match_ids)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    return match_ids[:t1], match_ids[t1:t2], match_ids[t2:]


def _prefix_features_at_indices(
    t_all: np.ndarray,
    m_eq: np.ndarray,
    m_int: np.ndarray,
    sample_step_sec: int,
) -> list[dict[str, float]]:
    n = len(t_all)
    sample_idx = [0]
    for i in range(1, n):
        if t_all[i] - t_all[sample_idx[-1]] >= sample_step_sec:
            sample_idx.append(i)
    if sample_idx[-1] != n - 1:
        sample_idx.append(n - 1)

    si = 0
    cur_eq = extract_causal_curve_features(t_all[:1], m_eq[:1], "ceq")
    cur_int = extract_causal_curve_features(t_all[:1], m_int[:1], "cint")
    rows: list[dict[str, float]] = []
    for i in range(n):
        while si + 1 < len(sample_idx) and i >= sample_idx[si + 1]:
            si += 1
            j = sample_idx[si]
            cur_eq = extract_causal_curve_features(t_all[:j + 1], m_eq[:j + 1], "ceq")
            cur_int = extract_causal_curve_features(t_all[:j + 1], m_int[:j + 1], "cint")
        rows.append({**cur_eq, **cur_int})
    return rows


def augment_events(events: pd.DataFrame, sample_step_sec: int) -> pd.DataFrame:
    ev = events.copy()
    ev["match_id"] = ev["match_id"].astype(str)
    ev["momentum_eq"] = compute_momentum_eq(ev)
    ev["momentum_intensity"] = compute_momentum_intensity(ev)

    parts: list[pd.DataFrame] = []
    for _, g in ev.groupby("match_id", sort=False):
        g = g.sort_values(["time_sec", "state_row_id"]).reset_index(drop=True)
        t_all = g["time_sec"].to_numpy(dtype=float)
        m_eq = g["momentum_eq"].to_numpy(dtype=float)
        m_int = g["momentum_intensity"].to_numpy(dtype=float)
        feat_rows = _prefix_features_at_indices(t_all, m_eq, m_int, sample_step_sec)
        g = pd.concat([g, pd.DataFrame(feat_rows)], axis=1)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def load_events(match_ids: set[str], parquet_path: Path) -> pd.DataFrame:
    intensity_cols = [c for pair in INTENSITY_COLS.values() for c in pair]
    cols = (
        ["match_id", "state_row_id", "period", "time_sec", "label_next_goal_class"]
        + STATE_COLS
        + EQ_RAW_COLS
        + intensity_cols
    )
    cols = list(dict.fromkeys(cols))
    pq = pd.read_parquet(parquet_path, columns=cols)
    pq["match_id"] = pq["match_id"].astype(str)
    return pq[pq["match_id"].isin(match_ids)].copy()


def variant_feature_cols(variant: str) -> list[str]:
    if variant == "state_only":
        return STATE_COLS
    if variant == "state_plus_causal_both_core":
        return STATE_COLS + CAUSAL_BOTH_CORE
    raise ValueError(variant)


def build_xy(events: pd.DataFrame, variant: str) -> tuple[np.ndarray, np.ndarray]:
    ev = events.copy()
    y = ev["label_next_goal_class"].astype(str)
    mask = y.notna() & (y != "")
    ev = ev.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    feat_cols = variant_feature_cols(variant)
    X = ev[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X.to_numpy(dtype=np.float32), y.to_numpy()


def subsample_fit(X: np.ndarray, y_enc: np.ndarray, max_rows: int, seed: int):
    if len(y_enc) <= max_rows:
        return X, y_enc
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_enc), size=max_rows, replace=False)
    return X[idx], y_enc[idx]


def eval_split(clf, le: LabelEncoder, X, y_enc) -> dict:
    pred = le.inverse_transform(clf.predict(X))
    y_str = le.inverse_transform(y_enc)
    return {
        "accuracy": accuracy_score(y_str, pred),
        "f1_macro": f1_score(y_str, pred, average="macro", zero_division=0),
        "n_events": len(y_enc),
    }


def build_clf(name: str = "hist_gb_shallow"):
    if name == "hist_gb_default":
        return HistGradientBoostingClassifier(
            max_depth=10, learning_rate=0.08, max_iter=200, random_state=42
        )
    if name == "hist_gb_shallow":
        return HistGradientBoostingClassifier(
            max_depth=5,
            learning_rate=0.05,
            max_iter=150,
            min_samples_leaf=80,
            random_state=42,
        )
    if name == "hist_gb_l2":
        return HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=150,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=42,
        )
    if name == "random_forest":
        p = {k: v for k, v in MODEL_PARAMS["random_forest"].items() if k != "n_jobs"}
        p["n_jobs"] = -1
        p["max_depth"] = 12
        p["min_samples_leaf"] = 20
        return RandomForestClassifier(**p)
    raise ValueError(name)


def run_realtime_experiment(args, match_ids, parquet_path):
    """Run realtime feature comparison experiment"""
    print("\n" + "=" * 60)
    print("Running Realtime Feature Comparison Experiment")
    print("=" * 60)

    train_ids, val_ids, test_ids = split_match_ids(match_ids)
    print(f"[Matches] train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    print("[Load + augment] M_eq + M_intensity + causal ceq_* / cint_* per match...")
    train_ev = augment_events(load_events(set(train_ids), parquet_path), args.sample_step_sec)
    val_ev = augment_events(load_events(set(val_ids), parquet_path), args.sample_step_sec)
    test_ev = augment_events(load_events(set(test_ids), parquet_path), args.sample_step_sec)

    label_mode = "home_away_none"
    all_rows: list[dict] = []

    for variant in VARIANTS:
        print(f"\n[Next-goal] variant={variant}")
        X_train, y_train = build_xy(train_ev, variant)
        X_val, y_val = build_xy(val_ev, variant)
        X_test, y_test = build_xy(test_ev, variant)

        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_train)
        y_val_enc = le.transform(y_val)
        y_test_enc = le.transform(y_test)

        X_fit, y_fit = subsample_fit(X_train, y_tr_enc, args.max_train_events, seed=42)
        clf = build_clf("hist_gb_shallow")
        clf.fit(X_fit, y_fit)

        for split_name, X, y_enc in [
            ("train", X_fit, y_fit),
            ("val", X_val, y_val_enc),
            ("test", X_test, y_test_enc),
        ]:
            metrics = eval_split(clf, le, X, y_enc)
            all_rows.append(
                {
                    "task": "next_goal_formula_realtime",
                    "label_mode": label_mode,
                    "variant": variant,
                    "formula": "M_eq_and_M_intensity",
                    "clf_model": "hist_gb_shallow",
                    "split": split_name,
                    **metrics,
                }
            )
        test_acc = [r for r in all_rows if r["variant"] == variant and r["split"] == "test"][-1]
        print(f"  test acc={test_acc['accuracy']:.3f} f1={test_acc['f1_macro']:.3f}")

    # Majority baseline
    _, y_test = build_xy(test_ev, "state_only")
    le0 = LabelEncoder()
    y_test_enc = le0.fit_transform(y_test)
    maj = le0.classes_[np.bincount(le0.transform(y_test)).argmax()]
    all_rows.append(
        {
            "task": "next_goal_formula_realtime",
            "label_mode": label_mode,
            "variant": "majority_baseline",
            "formula": "—",
            "clf_model": "—",
            "split": "test",
            "n_events": len(y_test),
            "accuracy": accuracy_score(y_test, np.full(len(y_test), maj)),
            "f1_macro": f1_score(y_test, np.full(len(y_test), maj), average="macro", zero_division=0),
        }
    )

    # Save results
    out = pd.DataFrame(all_rows)
    output_dir = ROOT / "outputs" / "reports" / "Next-Goal Prediction outcome"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "next_goal_1516_formula_realtime_3class.csv"
    out.to_csv(path, index=False)

    print("\n[Summary — test split]")
    for _, r in out[out["split"] == "test"].sort_values("accuracy", ascending=False).iterrows():
        print(f"  {r['variant']:32} acc={r['accuracy']:.3f}  f1={r['f1_macro']:.3f}")
    print(f"\nSaved: {path}")

    return train_ev, val_ev, test_ev


def expanding_folds(
    match_ids: list[str],
    n_folds: int,
    min_train_ratio: float,
    test_ratio_per_fold: float,
) -> list[tuple[int, slice, slice]]:
    n = len(match_ids)
    test_chunk = max(1, int(n * test_ratio_per_fold))
    min_train = max(test_chunk + 1, int(n * min_train_ratio))

    folds = []
    test_start = min_train
    for k in range(n_folds):
        test_end = min(test_start + test_chunk, n)
        if test_start >= n or test_end <= test_start:
            break
        folds.append((k, slice(0, test_start), slice(test_start, test_end)))
        test_start = test_end
        if test_start >= n:
            break
    return folds


def run_cv_experiment(args, match_ids, parquet_path):
    """Run time-series cross-validation experiment"""
    print("\n" + "=" * 60)
    print("Running Time-Series Cross-Validation Experiment")
    print("=" * 60)

    folds = expanding_folds(
        match_ids, args.n_folds, args.min_train_ratio, args.test_ratio_per_fold
    )
    print(f"[Matches] total={len(match_ids)}  folds={len(folds)}")
    for fid, tr_sl, te_sl in folds:
        print(
            f"  fold {fid}: train matches {tr_sl.start}-{tr_sl.stop} "
            f"({tr_sl.stop - tr_sl.start}), test {te_sl.start}-{te_sl.stop} "
            f"({te_sl.stop - te_sl.start})"
        )

    # Load and augment all matches once
    all_ids = set(match_ids)
    print("\n[Augment] all matches once (may take several minutes)...")
    events = augment_events(load_events(all_ids, parquet_path), args.sample_step_sec)

    label_mode = "home_away_none"
    rows: list[dict] = []

    for fold_id, tr_sl, te_sl in folds:
        train_m = set(match_ids[tr_sl])
        test_m = set(match_ids[te_sl])
        train_ev = events[events["match_id"].isin(train_m)]
        test_ev = events[events["match_id"].isin(test_m)]
        print(
            f"\n[Fold {fold_id}] train_events={len(train_ev):,} "
            f"test_events={len(test_ev):,}"
        )

        for variant in VARIANTS:
            X_train, y_train = build_xy(train_ev, variant)
            X_test, y_test = build_xy(test_ev, variant)

            le = LabelEncoder()
            y_tr_enc = le.fit_transform(y_train)
            y_te_enc = le.transform(y_test)
            X_fit, y_fit = subsample_fit(
                X_train, y_tr_enc, args.max_train_events, seed=42 + fold_id
            )

            clf = build_clf("hist_gb_shallow")
            clf.fit(X_fit, y_fit)

            train_mets = eval_split(clf, le, X_fit, y_fit)
            test_mets = eval_split(clf, le, X_test, y_te_enc)

            rows.append(
                {
                    "task": "next_goal_formula_timeseries_cv",
                    "fold": fold_id,
                    "label_mode": label_mode,
                    "variant": variant,
                    "clf_model": "hist_gb_shallow",
                    "split": "train",
                    "n_train_matches": tr_sl.stop - tr_sl.start,
                    "n_test_matches": te_sl.stop - te_sl.start,
                    **train_mets,
                }
            )
            rows.append(
                {
                    "task": "next_goal_formula_timeseries_cv",
                    "fold": fold_id,
                    "label_mode": label_mode,
                    "variant": variant,
                    "clf_model": "hist_gb_shallow",
                    "split": "test",
                    "n_train_matches": tr_sl.stop - tr_sl.start,
                    "n_test_matches": te_sl.stop - te_sl.start,
                    **test_mets,
                }
            )
            print(
                f"  {variant:28} train={train_mets['accuracy']:.3f} "
                f"test={test_mets['accuracy']:.3f} f1={test_mets['f1_macro']:.3f}"
            )

    out = pd.DataFrame(rows)
    test_df = out[out["split"] == "test"]

    summary_rows = []
    for variant in VARIANTS:
        sub = test_df[test_df["variant"] == variant]
        summary_rows.append(
            {
                "task": "next_goal_formula_timeseries_cv",
                "fold": "mean_std",
                "label_mode": label_mode,
                "variant": variant,
                "clf_model": "hist_gb_shallow",
                "split": "test",
                "n_folds": len(sub),
                "accuracy_mean": sub["accuracy"].mean(),
                "accuracy_std": sub["accuracy"].std(),
                "f1_macro_mean": sub["f1_macro"].mean(),
                "f1_macro_std": sub["f1_macro"].std(),
                "n_events": int(sub["n_events"].sum()),
            }
        )

    out = pd.concat([out, pd.DataFrame(summary_rows)], ignore_index=True)
    output_dir = ROOT / "outputs" / "reports" / "Next-Goal Prediction outcome"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "next_goal_1516_formula_timeseries_cv_3class.csv"
    out.to_csv(path, index=False)

    print("\n[Summary — test across folds]")
    for s in summary_rows:
        print(
            f"  {s['variant']:28} acc={s['accuracy_mean']:.3f}±{s['accuracy_std']:.3f}  "
            f"f1={s['f1_macro_mean']:.3f}±{s['f1_macro_std']:.3f}  "
            f"({s['n_folds']} folds)"
        )
    print(f"\nSaved: {path}")


def run_regularized_experiment(args, match_ids, parquet_path):
    """Run regularization/overfitting comparison experiment"""
    print("\n" + "=" * 60)
    print("Running Regularization/Overfitting Comparison Experiment")
    print("=" * 60)

    train_ids, val_ids, test_ids = split_match_ids(match_ids)

    print("[Augment] loading train/val/test...")
    train_ev = augment_events(load_events(set(train_ids), parquet_path), 120)
    val_ev = augment_events(load_events(set(val_ids), parquet_path), 120)
    test_ev = augment_events(load_events(set(test_ids), parquet_path), 120)

    label_mode = "home_away_none"
    models = ["hist_gb_default", "hist_gb_shallow", "hist_gb_l2", "random_forest"]
    variant = "state_plus_causal_both_core"
    rows = []

    X_train, y_train = build_xy(train_ev, variant)
    X_val, y_val = build_xy(val_ev, variant)
    X_test, y_test = build_xy(test_ev, variant)

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    X_fit, y_fit = subsample_fit(X_train, y_tr_enc, args.max_train_events, 42)

    for model_name in models:
        print(f"\n[Model] {model_name}")
        clf = build_clf(model_name)
        clf.fit(X_fit, y_fit)

        for split, X, y_enc in [
            ("train", X_fit, y_fit),
            ("val", X_val, y_val_enc),
            ("test", X_test, y_test_enc),
        ]:
            m = eval_split(clf, le, X, y_enc)
            rows.append(
                {
                    "task": "next_goal_formula_regularized",
                    "label_mode": label_mode,
                    "variant": variant,
                    "clf_model": model_name,
                    "split": split,
                    "train_test_gap": None,
                    **m,
                }
            )

        tr_acc = rows[-3]["accuracy"]
        te_acc = rows[-1]["accuracy"]
        gap = tr_acc - te_acc
        rows[-1]["train_test_gap"] = gap
        rows[-2]["train_test_gap"] = None
        rows[-3]["train_test_gap"] = None
        print(f"  train={tr_acc:.3f} val={rows[-2]['accuracy']:.3f} test={te_acc:.3f} gap={gap:.3f}")

    out = pd.DataFrame(rows)
    output_dir = ROOT / "outputs" / "reports" / "Next-Goal Prediction outcome"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "next_goal_1516_formula_regularized_3class.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Combined Formula-based real-time next-goal experiments (2015-2016)"
    )
    parser.add_argument("--max-train-events", type=int, default=800_000, help="Max training events")
    parser.add_argument("--sample-step-sec", type=int, default=120, help="Feature sample step (seconds)")
    parser.add_argument("--cv", action="store_true", help="Run time-series cross-validation")
    parser.add_argument("--regularized", action="store_true", help="Run regularization comparison")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--min-train-ratio", type=float, default=0.5, help="Min training ratio for CV")
    parser.add_argument("--test-ratio-per-fold", type=float, default=0.1, help="Test ratio per CV fold")
    args = parser.parse_args()

    print("=" * 60)
    print("Formula-based Real-time Next-Goal Experiments (2015-2016)")
    print("=" * 60)
    print(f"Experiments to run:")
    print(f"  - Realtime feature comparison: YES (default)")
    print(f"  - Time-series cross-validation: {'YES' if args.cv else 'NO'}")
    print(f"  - Regularization comparison: {'YES' if args.regularized else 'NO'}")

    parquet_path = _resolve_parquet()
    match_ids = match_chronological_order()

    # Run realtime (always)
    train_ev, val_ev, test_ev = run_realtime_experiment(args, match_ids, parquet_path)

    # Run CV if requested
    if args.cv:
        run_cv_experiment(args, match_ids, parquet_path)

    # Run regularized if requested
    if args.regularized:
        run_regularized_experiment(args, match_ids, parquet_path)

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

