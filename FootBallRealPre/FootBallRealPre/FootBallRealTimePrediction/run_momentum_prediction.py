
"""
League-specific CatBoost prediction with/without real-time momentum features.

4 experimental conditions per league:
  1. pre_baseline:    baseline features (no HT, no momentum)
  2. pre_momentum:    baseline features + 20 momentum features
  3. ht_baseline:     baseline features + 8 HT columns (no momentum)
  4. ht_momentum:     baseline features + 8 HT columns + 20 momentum features

This allows isolating the contribution of real-time momentum in both tasks.

CatBoost tuned by Optuna (accuracy as sole criterion), 60/20/20 time split.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SCRIPT_DIR = Path(__file__).parent
OFFICIAL_DIR = SCRIPT_DIR / "15-16_official_data"
MOMENTUM_CSV = SCRIPT_DIR / "real_time_data" / "momentum_features_big5_2015_16.csv"
OUTPUT_DIR = SCRIPT_DIR / "momentum_prediction_results"
OUTPUT_DIR.mkdir(exist_ok=True)

LEAGUE_FILES = {
    "England": "England.csv",
    "France": "France.csv",
    "Germany": "Germany.csv",
    "Italy": "Italy.csv",
    "Spain": "Spain.csv",
}

# Half-time columns (8 columns)
HT_COLUMNS = [
    "home_team_goal_count_half_time",
    "away_team_goal_count_half_time",
    "total_goals_at_half_time",
    "ht_goal_difference",
    "home_team_first_half_cards",
    "away_team_first_half_cards",
    "half_time_lead",
    "ht_result_encoded",
]

# Target
TARGET = "target_match_result"

# Momentum feature columns (20)
MOMENTUM_COLS = [
    "eq_mean", "eq_max", "eq_min", "eq_area_signed", "eq_zero_crossings",
    "eq_slope_overall", "eq_value_at_end", "eq_mean_last_600s",
    "eq_slope_last_third", "eq_area_decay_tau600s",
    "int_mean", "int_max", "int_min", "int_area_signed", "int_zero_crossings",
    "int_slope_overall", "int_value_at_end", "int_mean_last_600s",
    "int_slope_last_third", "int_area_decay_tau600s",
]

# Columns to always exclude (identifiers, targets, leakage)
EXCLUDE_ALWAYS = {
    # identifiers / meta
    "timestamp", "date_GMT", "status", "attendance", "referee", "stadium_name",
    "home_team_name", "away_team_name", "home_team_key", "away_team_key",
    "match_date", "match_datetime", "league", "season", "league_key",
    "match_id", "year", "month", "day_of_week", "is_weekend",
    # targets
    TARGET, "target_home_team_goal_count", "target_away_team_goal_count",
    "home_team_goal_count", "away_team_goal_count", "total_goal_count",
    # post-match / full-time leakage
    "home_team_corner_count", "away_team_corner_count",
    "home_team_yellow_cards", "home_team_red_cards",
    "away_team_yellow_cards", "away_team_red_cards",
    "home_team_second_half_cards", "away_team_second_half_cards",
    "home_team_shots", "away_team_shots",
    "home_team_shots_on_target", "away_team_shots_on_target",
    "home_team_shots_off_target", "away_team_shots_off_target",
    "home_team_fouls", "away_team_fouls",
    "home_team_possession", "away_team_possession",
    "team_a_xg", "team_b_xg",
    "home_team_goal_timings", "away_team_goal_timings",
}

# 4 experimental conditions
CONDITIONS = {
    "pre_baseline":  {"use_ht": False, "use_momentum": False},
    "pre_momentum":  {"use_ht": False, "use_momentum": True},
    "ht_baseline":   {"use_ht": True,  "use_momentum": False},
    "ht_momentum":   {"use_ht": True,  "use_momentum": True},
}


# ---------------------------------------------------------------------------
# Team name standardization
# ---------------------------------------------------------------------------
import unicodedata


def strip_accents(s: str) -> str:
    if pd.isna(s):
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(s))
        if not unicodedata.combining(c)
    )


TEAM_ALIASES = {
    # --- Spain ---
    "fc barcelona": "barcelona", "barcelona": "barcelona",
    "athletic club": "athletic bilbao", "athletic club bilbao": "athletic bilbao",
    "celta vigo": "celta", "celta de vigo": "celta",
    "deportivo la coruna": "deportivo", "rc deportivo la coruna": "deportivo",
    "sporting gijon": "sporting gijon",
    "getafe cf": "getafe", "getafe": "getafe",
    "granada cf": "granada", "granada": "granada",
    "malaga cf": "malaga", "malaga": "malaga",
    "sevilla fc": "sevilla", "sevilla": "sevilla",
    "valencia cf": "valencia", "valencia": "valencia",
    "rcd espanyol": "espanyol", "espanyol": "espanyol",
    "sd eibar": "eibar", "eibar": "eibar",
    "ud las palmas": "las palmas", "las palmas": "las palmas",
    # --- Germany ---
    "bayern munich": "bayern", "bayern munchen": "bayern",
    "borussia monchengladbach": "gladbach", "borussia m gladbach": "gladbach",
    "hertha bsc": "hertha", "hertha berlin": "hertha",
    "koln": "koln", "fc koln": "koln",
    "mainz 05": "mainz", "fsv mainz 05": "mainz",
    "stuttgart": "stuttgart", "vfb stuttgart": "stuttgart",
    # --- France ---
    "angers": "angers", "angers sco": "angers",
    "as monaco": "monaco", "monaco": "monaco",
    "nice": "nice", "ogc nice": "nice",
    "olympique lyonnais": "lyon", "lyon": "lyon",
    "olympique marseille": "marseille", "marseille": "marseille",
    "paris saint germain": "psg", "paris saint-germain": "psg",
    "reims": "reims", "stade de reims": "reims",
    # --- Italy ---
    "as roma": "roma", "roma": "roma",
    "inter milan": "inter", "inter": "inter",
}


def normalize_team(name: str) -> str:
    if pd.isna(name):
        return ""
    s = str(name).strip().lower()
    s = strip_accents(s)
    s = s.replace("'", " ").replace("`", " ").replace("´", " ")
    s = " ".join(s.split())
    if s in TEAM_ALIASES:
        return TEAM_ALIASES[s]
    return s


def build_match_key(home: str, away: str) -> str:
    return f"{normalize_team(home)}__{normalize_team(away)}"


# ---------------------------------------------------------------------------
# Data loading & merging
# ---------------------------------------------------------------------------
def load_official_data() -> pd.DataFrame:
    frames = []
    for league, fname in LEAGUE_FILES.items():
        fp = OFFICIAL_DIR / fname
        df = pd.read_csv(fp)
        df["league"] = league
        frames.append(df)
        print(f"  {league}: {len(df)} matches")
    return pd.concat(frames, ignore_index=True)


def load_momentum() -> pd.DataFrame:
    return pd.read_csv(MOMENTUM_CSV)


def merge_momentum(official: pd.DataFrame, momentum: pd.DataFrame) -> pd.DataFrame:
    official["_key"] = official.apply(
        lambda r: build_match_key(r["home_team_name"], r["away_team_name"]), axis=1
    )
    momentum["_key"] = momentum.apply(
        lambda r: build_match_key(r["home_team"], r["away_team"]), axis=1
    )
    dup = momentum["_key"].value_counts()
    dup = dup[dup > 1]
    if len(dup):
        print(f"  [warn] {len(dup)} duplicate keys in momentum; keeping first.")
        momentum = momentum.drop_duplicates("_key", keep="first")

    merged = official.merge(
        momentum[["_key"] + MOMENTUM_COLS],
        on="_key", how="left", indicator=True
    )
    n_unmatched = (merged["_merge"] == "left_only").sum()
    print(f"  Matched: {(merged['_merge'] == 'both').sum()}, "
          f"Unmatched (deleted): {n_unmatched}")
    if n_unmatched > 0:
        unmatched = merged.loc[merged["_merge"] == "left_only",
                               ["league", "home_team_name", "away_team_name", "_key"]]
        print("  Unmatched matches:")
        for _, r in unmatched.iterrows():
            print(f"    [{r['league']}] {r['home_team_name']} vs {r['away_team_name']}")

    merged = merged[merged["_merge"] == "both"].drop(columns=["_merge", "_key"])
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame, condition: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for a given experimental condition."""
    cfg = CONDITIONS[condition]
    y = df[TARGET].map({"H": 0, "D": 1, "A": 2}).astype(int)

    # build exclusion set
    exclude = set(EXCLUDE_ALWAYS)
    if not cfg["use_ht"]:
        exclude = exclude | set(HT_COLUMNS)
    if not cfg["use_momentum"]:
        exclude = exclude | set(MOMENTUM_COLS)

    candidate_cols = [c for c in df.columns if c not in exclude]
    X = df[candidate_cols].copy()
    X = X.select_dtypes(include=[np.number])

    # drop constant columns
    nunique = X.nunique(dropna=True)
    drop_cols = nunique[nunique <= 1].index.tolist()
    if drop_cols:
        X = X.drop(columns=drop_cols)

    return X, y


def chronological_split(df: pd.DataFrame, ratios=(0.6, 0.2, 0.2)):
    sort_col = "match_datetime" if "match_datetime" in df.columns else "date_GMT"
    df_sorted = df.sort_values(sort_col).reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = df_sorted.iloc[:n_train].reset_index(drop=True)
    val = df_sorted.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test = df_sorted.iloc[n_train + n_val:].reset_index(drop=True)
    return train, val, test


# ---------------------------------------------------------------------------
# CatBoost + Optuna
# ---------------------------------------------------------------------------
def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "auto_class_weights": trial.suggest_categorical(
            "auto_class_weights", ["None", "Balanced", "SqrtBalanced"]
        ),
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)


def tune_and_eval(df_league, condition, league, n_trials=40):
    """Tune CatBoost with Optuna and evaluate on the test set."""
    train, val, test = chronological_split(df_league)
    X_train, y_train = prepare_features(train, condition)
    X_val, y_val = prepare_features(val, condition)
    X_test, y_test = prepare_features(test, condition)

    # align columns
    common_cols = sorted(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_val = X_val[common_cols]
    X_test = X_test[common_cols]

    # impute NaN with training median
    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_val = X_val.fillna(med)
    X_test = X_test.fillna(med)

    print(f"  [{league}/{condition}] train={len(X_train)} val={len(X_val)} "
          f"test={len(X_test)} features={len(common_cols)}")

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda tr: objective(tr, X_train, y_train, X_val, y_val),
                   n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({"random_seed": 42, "verbose": False, "allow_writing_files": False})
    print(f"  [{league}/{condition}] best val acc = {study.best_value:.4f}")

    # retrain on train+val
    X_trval = pd.concat([X_train, X_val], ignore_index=True)
    y_trval = pd.concat([y_train, y_val], ignore_index=True)
    model = CatBoostClassifier(**best_params)
    model.fit(X_trval, y_trval)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")
    f1_weighted = f1_score(y_test, preds, average="weighted")
    ll = log_loss(y_test, probs, labels=[0, 1, 2])
    try:
        auc_macro = roc_auc_score(y_test, probs, multi_class="ovr", average="macro", labels=[0, 1, 2])
    except Exception:
        auc_macro = np.nan
    f1_per = f1_score(y_test, preds, average=None, labels=[0, 1, 2])
    draw_f1 = f1_per[1]

    return {
        "league": league,
        "condition": condition,
        "n_features": len(common_cols),
        "test_accuracy": acc,
        "test_f1_macro": f1_macro,
        "test_f1_weighted": f1_weighted,
        "test_log_loss": ll,
        "test_auc_macro": auc_macro,
        "test_draw_f1": draw_f1,
        "val_accuracy_best": study.best_value,
    }



def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 70)
    print("Loading official data ...")
    official = load_official_data()
    print(f"  Total: {len(official)} matches\n")

    print("Loading momentum features ...")
    momentum = load_momentum()
    print(f"  Total: {len(momentum)} matches\n")

    print("Merging momentum -> official ...")
    merged = merge_momentum(official, momentum)
    print(f"  Merged: {len(merged)} matches\n")

    # encode HT features
    merged["ht_result_encoded"] = merged["ht_goal_difference"].apply(
        lambda x: 0 if x > 0 else (2 if x < 0 else 1)
    )
    merged["half_time_lead"] = merged["ht_goal_difference"]

    all_results = []
    for league in LEAGUE_FILES:
        df_l = merged[merged["league"] == league].copy()
        if len(df_l) == 0:
            continue
        print("=" * 70)
        print(f"League: {league}  ({len(df_l)} matches)")
        for condition in ["pre_baseline", "pre_momentum", "ht_baseline", "ht_momentum"]:
            res = tune_and_eval(df_l, condition, league, n_trials=40)
            all_results.append(res)
            print(f"  -> {league}/{condition}: acc={res['test_accuracy']:.4f}  "
                  f"f1m={res['test_f1_macro']:.4f}  "
                  f"logloss={res['test_log_loss']:.4f}  "
                  f"auc={res['test_auc_macro']:.4f}  "
                  f"draw_f1={res['test_draw_f1']:.4f}\n")

    # save results
    results_df = pd.DataFrame(all_results)
    results_csv = OUTPUT_DIR / "momentum_catboost_results.csv"
    results_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
    print(f"\nResults saved to {results_csv}")


    print("\n" + "=" * 70)
    print("MOMENTUM CONTRIBUTION ANALYSIS")
    print("=" * 70)

    for league in LEAGUE_FILES:
        print(f"\n--- {league} ---")
        lr = results_df[results_df["league"] == league]
        for task_prefix in ["pre", "ht"]:
            base = lr[lr["condition"] == f"{task_prefix}_baseline"].iloc[0]
            mom = lr[lr["condition"] == f"{task_prefix}_momentum"].iloc[0]
            delta_acc = mom["test_accuracy"] - base["test_accuracy"]
            delta_f1 = mom["test_f1_macro"] - base["test_f1_macro"]
            delta_auc = mom["test_auc_macro"] - base["test_auc_macro"]
            delta_draw = mom["test_draw_f1"] - base["test_draw_f1"]
            delta_ll = mom["test_log_loss"] - base["test_log_loss"]
            print(f"  {task_prefix.upper()}: "
                  f"baseline acc={base['test_accuracy']:.4f} -> "
                  f"+momentum acc={mom['test_accuracy']:.4f}  "
                  f"Δacc={delta_acc:+.4f}  Δf1={delta_f1:+.4f}  "
                  f"Δauc={delta_auc:+.4f}  Δdraw_f1={delta_draw:+.4f}  "
                  f"Δlogloss={delta_ll:+.4f}")

    # ---- Print full table ----
    print("\n" + "=" * 70)
    print("FULL RESULTS TABLE")
    print("=" * 70)
    print(results_df[["league", "condition", "n_features", "test_accuracy",
                       "test_f1_macro", "test_log_loss", "test_auc_macro",
                       "test_draw_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
