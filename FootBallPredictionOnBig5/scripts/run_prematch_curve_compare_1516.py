"""
Pre-match prediction: compare three data sources WITHOUT halftime information.

Research question: Does adding real-time momentum trend features help
predict full-time match result when halftime result is NOT known?

Three data sources:
  1) Baseline:   all_leagues_2015-2016.csv
                 -> remove halftime columns -> pure pre-match features
  2) Formula:    feature_engineer5_momentum_formula_curve/...merged.csv
                 -> remove halftime columns + keep eq_*, int_* momentum features
  3) Learned:    feature_engineer5_momentum_learned_curve/...merged.csv
                 -> remove halftime columns + keep leq_*, lint_* momentum features

Output:
  outputs/reports/Pre-match momentum_merged_result/prematch_1516_merged_compare.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.config import MODEL_PARAMS, LABEL_ENCODING
from src.data.loader import clean_data
from src.features.prematch_features import handle_missing_values

SEASON = "2015-2016"
LEAGUES = ["all", "England", "Spain", "Germany", "Italy", "France"]

BASELINE_CSV = ROOT / "merged_data/all_leagues_2015-2016.csv"
FORMULA_MERGED_CSV = ROOT / "merged_data/feature_engineer5_momentum_formula_curve/all_leagues_2015-2016_momentum.merged.csv"
LEARNED_MERGED_CSV = ROOT / "merged_data/feature_engineer5_momentum_learned_curve/all_leagues_2015-2016_momentum.merged.csv"

OUTPUT_DIR = ROOT / "outputs/reports/Pre-match momentum_merged_result"

HALFTIME_COLS_TO_DROP = [
    "ht_result",
    "half_time_lead",
    "total_goals_at_half_time",
    "home_team_goal_count_half_time",
    "away_team_goal_count_half_time",
    "home_team_first_half_cards",
    "away_team_first_half_cards",
]

SECOND_HALF_COLS_TO_DROP = [
    "home_team_second_half_cards",
    "away_team_second_half_cards",
]

IDENTIFIER_COLS = [
    "match_id", "timestamp", "date_GMT", "match_datetime", "match_date",
    "season", "status", "stadium_name", "referee", "home_team_name",
    "away_team_name", "Game Week", "league", "row_id", "_main_row_id",
    "league_key", "home_team_key", "away_team_key", "date_offset_used",
    "odds_open_source", "odds_close_source", "attendance",
    "statsbomb_match_id",
]

TARGET_COL = "target_match_result"

EXTRA_TARGET_COLS = [
    "target_home_team_goal_count",
    "target_away_team_goal_count",
    "target_total_goal_count",
]

MOMENTUM_PREFIXES = ("eq_", "mod_raw_", "mod_ewm_", "mod_conf_w_", "mod_conf_", "n_rows_period1")
AGG_PREFIXES = ("agg_", "p1_", "proxy_")
SPARSE_PREFIXES = ("sparse_",)
LEAK_PREFIXES = ("fml_", "leq_", "lint_")

FORMULA_KEEP_PREFIXES = ("eq_", "int_")
LEARNED_KEEP_PREFIXES = ("leq_", "lint_")


def load_and_filter(path: Path, label: str, keep_momentum_prefixes: tuple[str, ...] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["season"] == SEASON].copy()
    df["match_id"] = df["match_id"].astype(str)
    if "statsbomb_match_id" in df.columns:
        df["statsbomb_match_id"] = df["statsbomb_match_id"].astype(str)

    date_col = None
    for c in ["match_datetime", "date_GMT"]:
        if c in df.columns:
            date_col = c
            break
    if date_col:
        df["_sort_date"] = pd.to_datetime(df[date_col], errors="coerce")
    if "league" in df.columns:
        df["_league"] = df["league"].astype(str).str.lower()
    df = clean_data(df)

    drop_cols = set()
    for c in df.columns:
        if c in HALFTIME_COLS_TO_DROP or c in SECOND_HALF_COLS_TO_DROP:
            drop_cols.add(c)
        elif c in IDENTIFIER_COLS:
            drop_cols.add(c)
        elif c in EXTRA_TARGET_COLS:
            drop_cols.add(c)
        elif c == TARGET_COL:
            continue
        elif c == "_sort_date":
            continue
        elif c == "_league":
            continue
        elif keep_momentum_prefixes is not None and any(c.startswith(p) for p in keep_momentum_prefixes):
            continue
        elif c.startswith(AGG_PREFIXES):
            drop_cols.add(c)
        elif c.startswith(SPARSE_PREFIXES):
            drop_cols.add(c)
        elif c.startswith(LEAK_PREFIXES):
            drop_cols.add(c)
        elif c.startswith(MOMENTUM_PREFIXES):
            drop_cols.add(c)

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)
    print(f"[{label}] {len(df):,} rows, {df.shape[1]:,} columns (after removing halftime)")
    return df


def chronological_split(df: pd.DataFrame, train_ratio=0.6, val_ratio=0.2):
    date_col = None
    for c in ["_sort_date", "date_GMT", "match_datetime", "timestamp"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError("No date column found for chronological split")
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    return df.iloc[:t1].copy(), df.iloc[t1:t2].copy(), df.iloc[t2:].copy()


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")

    y = df[TARGET_COL].copy()
    X = df.drop(columns=[TARGET_COL], errors="ignore")

    X = handle_missing_values(X)
    X = X.select_dtypes(include=[np.number]).reset_index(drop=True)
    y = y.reset_index(drop=True)

    mask = y.notna().to_numpy()
    le = LabelEncoder()
    return X.loc[mask], le.fit_transform(y.astype(str)[mask])


def run_rf(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    p = {k: v for k, v in MODEL_PARAMS["random_forest"].items() if k != "n_jobs"}
    p["n_jobs"] = -1
    clf = RandomForestClassifier(**p)
    clf.fit(X_train, y_train)

    y_pred_test = clf.predict(X_test)
    y_pred_val = clf.predict(X_val)
    y_pred_train = clf.predict(X_train)

    return {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "val_accuracy": accuracy_score(y_val, y_pred_val),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "train_f1_macro": f1_score(y_train, y_pred_train, average="macro", zero_division=0),
        "val_f1_macro": f1_score(y_val, y_pred_val, average="macro", zero_division=0),
        "test_f1_macro": f1_score(y_test, y_pred_test, average="macro", zero_division=0),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "n_features": X_train.shape[1],
    }


def main():
    print("=" * 60)
    print("Pre-match: 3-way comparison WITHOUT halftime info")
    print("Research: Does momentum trend help when HT result is unknown?")
    print("=" * 60)

    print("\n[1/4] Loading three data sources (removing halftime columns)...")
    baseline = load_and_filter(BASELINE_CSV, "Baseline", keep_momentum_prefixes=None)
    formula = load_and_filter(FORMULA_MERGED_CSV, "Formula", keep_momentum_prefixes=FORMULA_KEEP_PREFIXES)
    learned = load_and_filter(LEARNED_MERGED_CSV, "Learned", keep_momentum_prefixes=LEARNED_KEEP_PREFIXES)

    print("\n[2/4] Chronological splits...")
    bl_tr, bl_va, bl_te = chronological_split(baseline)
    fm_tr, fm_va, fm_te = chronological_split(formula)
    lr_tr, lr_va, lr_te = chronological_split(learned)

    data_sources = [
        ("baseline", bl_tr, bl_va, bl_te),
        ("formula_curve", fm_tr, fm_va, fm_te),
        ("learned_curve", lr_tr, lr_va, lr_te),
    ]

    print("\n[3/4] Running Random Forest for each data source & league...")
    rows = []

    for source_name, tr, va, te in data_sources:
        for league in LEAGUES:
            if league == "all":
                ltr, lva, lte = tr, va, te
            else:
                league_col = "_league" if "_league" in tr.columns else "league"
                ltr = tr[tr[league_col] == league.lower()] if league_col in tr.columns else tr
                lva = va[va[league_col] == league.lower()] if league_col in va.columns else va
                lte = te[te[league_col] == league.lower()] if league_col in te.columns else te
            if len(lte) == 0:
                continue
            try:
                m = run_rf(ltr, lva, lte)
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
                print(
                    f"  {source_name:18} test={m['test_accuracy']:.3f} "
                    f"val={m['val_accuracy']:.3f} "
                    f"f1={m['test_f1_macro']:.3f} "
                    f"n_feat={m['n_features']}"
                )

    print("\n[4/4] Saving results...")
    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "prematch_1516_merged_compare.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 60)
    print("Summary (all leagues, NO halftime info):")
    print("=" * 60)
    all_rows = out[out["league"] == "all"]
    for _, r in all_rows.iterrows():
        print(
            f"  {r['data_source']:18} | "
            f"test_acc={r['test_accuracy']:.3f} | "
            f"val_acc={r['val_accuracy']:.3f} | "
            f"train_acc={r['train_accuracy']:.3f} | "
            f"test_f1={r['test_f1_macro']:.3f} | "
            f"n_feat={r['n_features']}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
