"""
Halftime: compare three merged CSVs with Random Forest.

Three data sources (all from period-1 baseline event data):
  1) Baseline:  all_leagues_2015-2016.csv (no momentum features)
  2) Formula:   feature_engineer5_momentum_formula_curve/all_leagues_2015-2016_momentum.merged.csv
                -> eq_*, int_* from M_eq / M_int -> extract_curve_features
  3) Learned:   feature_engineer5_momentum_learned_curve/all_leagues_2015-2016_momentum.merged.csv
                -> leq_*, lint_* from next-goal OOF prediction curves (CV)

Learned curve generation uses Out-of-Fold (OOF) predictions:
  - Expanding window time-series CV (same as run_next_goal__1516.py --cv)
  - Each fold: train on matches[:test_start], predict on matches[test_start:test_end]
  - OOF predictions cover all matches without data leakage
  - Best config: state_plus_causal_both_core + hist_gb_shallow

Output:
  outputs/reports/Real-time momentum_merged_result/halftime_1516_merged_compare.csv
  merged_data/match_level_formula_curve_features.csv
  merged_data/match_level_learned_curve_features.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.config import MODEL_PARAMS
from src.data.loader import clean_data
from src.features.halftime_features import process_halftime_features
from src.momentum.config import (
    CAUSAL_SUFFIXES,
    EQ_RAW_COLS,
    INTENSITY_COLS,
    MIN_PERIOD1_ROWS,
    PARQUET_PATHS,
    EQ_CORE_COLS,
    INT_CORE_COLS,
)
from src.momentum.causal import extract_causal_curve_features
from src.momentum.curve_features import extract_curve_features, feature_names
from src.momentum.equation import compute_momentum_eq, compute_momentum_intensity

SEASON = "2015-2016"
LEAGUES = ["all", "England", "Spain", "Germany", "Italy", "France"]

BASELINE_CSV = ROOT / "merged_data/all_leagues_2015-2016.csv"
FORMULA_MERGED_CSV = ROOT / "merged_data/feature_engineer5_momentum_formula_curve/all_leagues_2015-2016_momentum.merged.csv"
LEARNED_MERGED_CSV = ROOT / "merged_data/feature_engineer5_momentum_learned_curve/all_leagues_2015-2016_momentum.merged.csv"

FORMULA_CURVE_COLS = EQ_CORE_COLS + INT_CORE_COLS
LEARNED_CURVE_COLS = feature_names("leq") + feature_names("lint")

OUTPUT_DIR = ROOT / "outputs/reports/Real-time momentum_merged_result"

LEAK_PREFIXES = ("fml_", "leq_", "lint_", "agg_", "p1_", "proxy_", "grid_", "sparse_")

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

BEST_VARIANT = "state_plus_causal_both_core"
BEST_MODEL = "hist_gb_shallow"

FORMULA_CSV = ROOT / "merged_data/match_level_formula_curve_features.csv"
LEARNED_CSV = ROOT / "merged_data/match_level_learned_curve_features.csv"

MERGED_CANDIDATES = [
    ROOT / "merged_data/feature_engineer5_momentum/all_leagues_2015-2016_momentum.merged.csv",
    ROOT / "merged_data/feature_engineer5_momentum_formula_curve/all_leagues_2015-2016_momentum.merged.csv",
    ROOT / "merged_data/feature_engineer_real_momentum/all_leagues_2015-2016_momentum.merged.csv",
]
BRIDGE_CANDIDATES = [
    ROOT / "merged_data/feature_engineer5_momentum/momentum_match_bridge.csv",
    ROOT / "merged_data/feature_engineer5_momentum_formula_curve/momentum_match_bridge.csv",
]

MOMENTUM_PREFIXES = ("eq_", "mod_raw_", "mod_ewm_", "mod_conf_w_", "mod_conf_", "n_rows_period1")
AGG_PREFIXES = ("agg_", "p1_", "proxy_")
SPARSE_PREFIXES = ("sparse_",)


def _resolve_parquet() -> Path:
    for p in PARQUET_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError("event_state_table parquet not found")


def _find_merged() -> Path:
    for p in MERGED_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"No merged CSV found in candidates")


def _find_bridge() -> Path:
    for p in BRIDGE_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"No bridge CSV found in candidates")


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


def expanding_folds(
    match_ids: list[str],
    n_folds: int = 5,
    min_train_ratio: float = 0.5,
    test_ratio_per_fold: float = 0.1,
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


def _build_clf(name: str = BEST_MODEL):
    if name == "hist_gb_default":
        return HistGradientBoostingClassifier(
            max_depth=10, learning_rate=0.08, max_iter=200, random_state=42
        )
    if name == "hist_gb_shallow":
        return HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.05, max_iter=150,
            min_samples_leaf=80, random_state=42,
        )
    if name == "hist_gb_l2":
        return HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, max_iter=150,
            min_samples_leaf=50, l2_regularization=1.0, random_state=42,
        )
    if name == "random_forest":
        p = {k: v for k, v in MODEL_PARAMS["random_forest"].items() if k != "n_jobs"}
        p["n_jobs"] = -1
        p["max_depth"] = 12
        p["min_samples_leaf"] = 20
        return RandomForestClassifier(**p)
    raise ValueError(name)


def _variant_feature_cols(variant: str) -> list[str]:
    if variant == "state_only":
        return STATE_COLS
    if variant == "state_plus_causal_both_core":
        return STATE_COLS + CAUSAL_BOTH_CORE
    raise ValueError(variant)


def _load_events(match_ids: set[str], parquet_path: Path) -> pd.DataFrame:
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


def _augment_events(events: pd.DataFrame, sample_step_sec: int) -> pd.DataFrame:
    ev = events.copy()
    ev["match_id"] = ev["match_id"].astype(str)
    ev["momentum_eq"] = compute_momentum_eq(ev)
    ev["momentum_intensity"] = compute_momentum_intensity(ev)

    parts: list[pd.DataFrame] = []
    for mid, g in ev.groupby("match_id", sort=False):
        g = g.sort_values(["time_sec", "state_row_id"]).reset_index(drop=True)
        t_all = g["time_sec"].to_numpy(dtype=float)
        m_eq = g["momentum_eq"].to_numpy(dtype=float)
        m_int = g["momentum_intensity"].to_numpy(dtype=float)
        feat_rows = _prefix_features_at_indices(t_all, m_eq, m_int, sample_step_sec)
        g = pd.concat([g, pd.DataFrame(feat_rows)], axis=1)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _augment_fold_events(
    parquet_path: Path,
    match_ids_set: set[str],
    sample_step_sec: int,
    label: str = "",
) -> pd.DataFrame:
    raw = _load_events(match_ids_set, parquet_path)
    print(f"    [{label}] Loaded {len(raw):,} events for {len(match_ids_set)} matches", flush=True)
    aug = _augment_events(raw, sample_step_sec)
    print(f"    [{label}] Augmented {len(aug):,} events", flush=True)
    del raw
    return aug


def _build_xy(events: pd.DataFrame, variant: str) -> tuple[np.ndarray, np.ndarray]:
    ev = events.copy()
    y = ev["label_next_goal_class"].astype(str)
    mask = y.notna() & (y != "")
    ev = ev.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    feat_cols = _variant_feature_cols(variant)
    X = ev[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X.to_numpy(dtype=np.float32), y.to_numpy()


def _subsample_fit(X: np.ndarray, y_enc: np.ndarray, max_rows: int, seed: int):
    if len(y_enc) <= max_rows:
        return X, y_enc
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_enc), size=max_rows, replace=False)
    return X[idx], y_enc[idx]


def build_oof_prediction_curves(
    variant: str = BEST_VARIANT,
    model_name: str = BEST_MODEL,
    sample_step_sec: int = 120,
    max_train_events: int = 800_000,
    n_folds: int = 5,
    min_train_ratio: float = 0.5,
    test_ratio_per_fold: float = 0.1,
) -> pd.DataFrame:
    parquet_path = _resolve_parquet()
    match_ids = match_chronological_order()
    folds = expanding_folds(match_ids, n_folds, min_train_ratio, test_ratio_per_fold)

    print(f"  [OOF] {len(folds)} folds, {len(match_ids)} matches total", flush=True)
    for fid, tr_sl, te_sl in folds:
        n_tr = tr_sl.stop - tr_sl.start
        n_te = te_sl.stop - te_sl.start
        print(f"    fold {fid}: train={n_tr} matches test={n_te} matches", flush=True)

    feat_cols = _variant_feature_cols(variant)
    oof_proba_parts: list[pd.DataFrame] = []
    last_fold_clf = None
    last_fold_le = None

    for fold_id, tr_sl, te_sl in folds:
        train_m = set(match_ids[tr_sl])
        test_m = set(match_ids[te_sl])

        print(f"  [OOF] Fold {fold_id}: augmenting train events...", flush=True)
        train_ev = _augment_fold_events(parquet_path, train_m, sample_step_sec, f"fold{fold_id}-train")

        print(f"  [OOF] Fold {fold_id}: augmenting test events...", flush=True)
        test_ev = _augment_fold_events(parquet_path, test_m, sample_step_sec, f"fold{fold_id}-test")

        X_train, y_train = _build_xy(train_ev, variant)
        X_test, y_test = _build_xy(test_ev, variant)

        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_train)
        y_te_enc = le.transform(y_test)

        X_fit, y_fit = _subsample_fit(X_train, y_tr_enc, max_train_events, seed=42 + fold_id)

        print(f"  [OOF] Fold {fold_id}: training {model_name} on {len(X_fit):,} events...", flush=True)
        clf = _build_clf(model_name)
        clf.fit(X_fit, y_fit)

        train_acc = accuracy_score(y_fit, clf.predict(X_fit))
        test_acc = accuracy_score(y_te_enc, clf.predict(X_test))
        print(f"  [OOF] Fold {fold_id}: train_acc={train_acc:.3f} test_acc={test_acc:.3f}", flush=True)

        p1_test = test_ev[test_ev["period"] == 1].copy()
        if len(p1_test) > 0:
            X_p1 = p1_test[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            proba = clf.predict_proba(X_p1)
            classes = list(le.classes_)
            p_home = proba[:, classes.index("home")] if "home" in classes else np.zeros(len(proba))
            p_away = proba[:, classes.index("away")] if "away" in classes else np.zeros(len(proba))

            p1_test = p1_test[["match_id", "time_sec"]].copy()
            p1_test["_p_home"] = p_home
            p1_test["_p_away"] = p_away
            p1_test["_directional"] = p_home - p_away
            p1_test["_intensity"] = p_home + p_away
            oof_proba_parts.append(p1_test)

        last_fold_clf = clf
        last_fold_le = le
        del train_ev, test_ev

    oof_match_ids = set()
    for part in oof_proba_parts:
        oof_match_ids.update(part["match_id"].unique())
    all_match_set = set(match_ids)
    missing_m = all_match_set - oof_match_ids
    print(f"  [OOF] OOF covers {len(oof_match_ids)} matches, {len(missing_m)} missing (train-only)", flush=True)

    if missing_m and last_fold_clf is not None:
        print(f"  [OOF] Predicting {len(missing_m)} train-only matches with last fold model...", flush=True)
        missing_ev = _augment_fold_events(parquet_path, missing_m, sample_step_sec, "missing-train")
        p1_missing = missing_ev[missing_ev["period"] == 1].copy()
        if len(p1_missing) > 0:
            X_p1 = p1_missing[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            proba = last_fold_clf.predict_proba(X_p1)
            classes = list(last_fold_le.classes_)
            p_home = proba[:, classes.index("home")] if "home" in classes else np.zeros(len(proba))
            p_away = proba[:, classes.index("away")] if "away" in classes else np.zeros(len(proba))

            p1_missing = p1_missing[["match_id", "time_sec"]].copy()
            p1_missing["_p_home"] = p_home
            p1_missing["_p_away"] = p_away
            p1_missing["_directional"] = p_home - p_away
            p1_missing["_intensity"] = p_home + p_away
            oof_proba_parts.append(p1_missing)
        del missing_ev, last_fold_clf

    print("  [OOF] Combining OOF predictions across folds...", flush=True)
    oof_all = pd.concat(oof_proba_parts, ignore_index=True)

    print("  [OOF] Extracting curve features per match...", flush=True)
    feat_rows: list[dict] = []
    for mid, g in oof_all.groupby("match_id", sort=False):
        g = g.sort_values("time_sec")
        t = g["time_sec"].to_numpy(dtype=float)
        if len(t) < MIN_PERIOD1_ROWS:
            continue
        directional = g["_directional"].to_numpy(dtype=float)
        intensity = g["_intensity"].to_numpy(dtype=float)
        leq_f = extract_curve_features(t, directional, "leq")
        lint_f = extract_curve_features(t, intensity, "lint")
        feat_rows.append({"statsbomb_match_id": str(mid), **leq_f, **lint_f})

    out = pd.DataFrame(feat_rows)
    print(f"  [OOF] Prediction curves: {len(out)} matches (full coverage)", flush=True)
    return out


def load_baseline() -> pd.DataFrame:
    df = pd.read_csv(BASELINE_CSV, encoding="utf-8-sig")
    df = df[df["season"] == SEASON].copy()
    df["match_id"] = df["match_id"].astype(str)
    df = clean_data(df)
    print(f"[Baseline] {len(df):,} rows, {df.shape[1]:,} columns")
    return df


def load_merged_csv(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["season"] == SEASON].copy()
    df["match_id"] = df["match_id"].astype(str)
    if "statsbomb_match_id" in df.columns:
        df["statsbomb_match_id"] = df["statsbomb_match_id"].astype(str)
    df = clean_data(df)
    print(f"[{label}] {len(df):,} rows, {df.shape[1]:,} columns")
    return df


def chronological_split(df: pd.DataFrame, train_ratio=0.6, val_ratio=0.2):
    date_col = "match_datetime" if "match_datetime" in df.columns else "date_GMT"
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    return df.iloc[:t1].copy(), df.iloc[t1:t2].copy(), df.iloc[t2:].copy()


def prepare_features(
    df: pd.DataFrame,
    keep_curve_prefixes: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    X, y = process_halftime_features(df)

    drop_cols = []
    for c in X.columns:
        if c.startswith(MOMENTUM_PREFIXES) and (keep_curve_prefixes is None or not c.startswith(keep_curve_prefixes)):
            drop_cols.append(c)
        elif c.startswith(AGG_PREFIXES):
            drop_cols.append(c)
        elif c.startswith(SPARSE_PREFIXES):
            drop_cols.append(c)
        elif c.startswith(LEAK_PREFIXES):
            drop_cols.append(c)
        elif c == "statsbomb_match_id":
            drop_cols.append(c)

    if keep_curve_prefixes is not None:
        for c in list(drop_cols):
            if any(c.startswith(p) for p in keep_curve_prefixes):
                drop_cols.remove(c)

    if drop_cols:
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")

    X = X.loc[:, ~X.columns.duplicated()].reset_index(drop=True)
    y = y.reset_index(drop=True)
    mask = y.notna().to_numpy()
    le = LabelEncoder()
    return X.loc[mask], le.fit_transform(y.astype(str)[mask])


def run_rf(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    keep_curve_prefixes: tuple[str, ...] | None = None,
) -> dict:
    X_train, y_train = prepare_features(train_df, keep_curve_prefixes)
    X_val, y_val = prepare_features(val_df, keep_curve_prefixes)
    X_test, y_test = prepare_features(test_df, keep_curve_prefixes)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    p = {k: v for k, v in MODEL_PARAMS["random_forest"].items() if k != "n_jobs"}
    p["n_jobs"] = -1
    clf = RandomForestClassifier(**p)
    clf.fit(X_train, y_train)

    return {
        "train_accuracy": accuracy_score(y_train, clf.predict(X_train)),
        "val_accuracy": accuracy_score(y_val, clf.predict(X_val)),
        "test_accuracy": accuracy_score(y_test, clf.predict(X_test)),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "n_features": X_train.shape[1],
    }


def main():
    parser = argparse.ArgumentParser(description="Halftime 3-way merged CSV comparison")
    parser.add_argument("--skip-oof", action="store_true",
                        help="Skip OOF curve generation (use cached LEARNED_CSV)")
    args = parser.parse_args()

    print("=" * 60)
    print("Halftime: 3-way merged CSV comparison (Random Forest)")
    print("=" * 60)

    print("\n[1/5] Generating OOF learned curve features...")
    if args.skip_oof and LEARNED_CSV.exists():
        learned_all = pd.read_csv(LEARNED_CSV, encoding="utf-8-sig")
        learned_all["statsbomb_match_id"] = learned_all["statsbomb_match_id"].astype(str)
        print(f"  Loaded cached: {LEARNED_CSV} ({len(learned_all)} matches)")
    else:
        learned_all = build_oof_prediction_curves()
        learned_out = learned_all.copy()
        learned_out["match_id"] = learned_out["statsbomb_match_id"]
        learned_out.to_csv(LEARNED_CSV, index=False, encoding="utf-8-sig")
        print(f"  Saved: {LEARNED_CSV} ({len(learned_out)} matches)")

    print("\n[2/5] Loading three data sources...")
    baseline = load_baseline()
    formula = load_merged_csv(FORMULA_MERGED_CSV, "Formula")
    learned = load_merged_csv(LEARNED_MERGED_CSV, "Learned")

    print("\n[3/5] Chronological splits...")
    bl_tr, bl_va, bl_te = chronological_split(baseline)
    fm_tr, fm_va, fm_te = chronological_split(formula)
    lr_tr, lr_va, lr_te = chronological_split(learned)

    data_sources = [
        ("baseline", bl_tr, bl_va, bl_te, None),
        ("formula_curve", fm_tr, fm_va, fm_te, ("eq_", "int_")),
        ("learned_curve", lr_tr, lr_va, lr_te, ("leq_", "lint_")),
    ]

    print("\n[4/5] Running Random Forest for each data source & league...")
    rows = []

    for source_name, tr, va, te, curve_prefixes in data_sources:
        for league in LEAGUES:
            if league == "all":
                ltr, lva, lte = tr, va, te
            else:
                ltr = tr[tr["league"].str.lower() == league.lower()]
                lva = va[va["league"].str.lower() == league.lower()]
                lte = te[te["league"].str.lower() == league.lower()]
            if len(lte) == 0:
                continue
            try:
                m = run_rf(ltr, lva, lte, curve_prefixes)
            except Exception as e:
                print(f"  skip {source_name}/{league}: {e}")
                continue
            rows.append({
                "season": SEASON,
                "league": league,
                "data_source": source_name,
                "model": "random_forest",
                **m,
            })
            if league == "all":
                print(f"  {source_name:18} test={m['test_accuracy']:.3f}  val={m['val_accuracy']:.3f}  n_feat={m['n_features']}")

    print("\n[5/5] Saving results...")
    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "halftime_1516_merged_compare.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 60)
    print("Summary (all leagues):")
    print("=" * 60)
    all_rows = out[out["league"] == "all"]
    for _, r in all_rows.iterrows():
        print(
            f"  {r['data_source']:18} | "
            f"test_acc={r['test_accuracy']:.3f} | "
            f"val_acc={r['val_accuracy']:.3f} | "
            f"train_acc={r['train_accuracy']:.3f} | "
            f"n_feat={r['n_features']}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
