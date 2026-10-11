# -*- coding: utf-8 -*-
"""Focused tuning for England pre-match momentum to beat baseline."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
import optuna
import unicodedata
from pathlib import Path

optuna.logging.set_verbosity(optuna.logging.WARNING)

SCRIPT_DIR = Path(__file__).parent
OFFICIAL_DIR = SCRIPT_DIR / "15-16_official_data"
MOMENTUM_CSV = SCRIPT_DIR / "real_time_data" / "momentum_features_big5_2015_16.csv"

MOMENTUM_COLS = [
    "eq_mean", "eq_max", "eq_min", "eq_area_signed", "eq_zero_crossings",
    "eq_slope_overall", "eq_value_at_end", "eq_mean_last_600s",
    "eq_slope_last_third", "eq_area_decay_tau600s",
    "int_mean", "int_max", "int_min", "int_area_signed", "int_zero_crossings",
    "int_slope_overall", "int_value_at_end", "int_mean_last_600s",
    "int_slope_last_third", "int_area_decay_tau600s",
]
HT_COLUMNS = [
    "home_team_goal_count_half_time", "away_team_goal_count_half_time",
    "total_goals_at_half_time", "ht_goal_difference",
    "home_team_first_half_cards", "away_team_first_half_cards",
    "half_time_lead", "ht_result_encoded",
]
EXCLUDE_ALWAYS = {
    "timestamp", "date_GMT", "status", "attendance", "referee", "stadium_name",
    "home_team_name", "away_team_name", "home_team_key", "away_team_key",
    "match_date", "match_datetime", "league", "season", "league_key",
    "match_id", "year", "month", "day_of_week", "is_weekend",
    "target_match_result", "target_home_team_goal_count", "target_away_team_goal_count",
    "home_team_goal_count", "away_team_goal_count", "total_goal_count",
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
TARGET = "target_match_result"

TEAM_ALIASES = {
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
    "bayern munich": "bayern", "bayern munchen": "bayern",
    "borussia monchengladbach": "gladbach", "borussia m gladbach": "gladbach",
    "hertha bsc": "hertha", "hertha berlin": "hertha",
    "koln": "koln", "fc koln": "koln",
    "mainz 05": "mainz", "fsv mainz 05": "mainz",
    "stuttgart": "stuttgart", "vfb stuttgart": "stuttgart",
    "angers": "angers", "angers sco": "angers",
    "as monaco": "monaco", "monaco": "monaco",
    "nice": "nice", "ogc nice": "nice",
    "olympique lyonnais": "lyon", "lyon": "lyon",
    "olympique marseille": "marseille", "marseille": "marseille",
    "paris saint germain": "psg", "paris saint-germain": "psg",
    "reims": "reims", "stade de reims": "reims",
    "as roma": "roma", "roma": "roma",
    "inter milan": "inter", "inter": "inter",
}


def strip_accents(s):
    if pd.isna(s):
        return ""
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def normalize_team(name):
    if pd.isna(name):
        return ""
    s = str(name).strip().lower()
    s = strip_accents(s)
    s = s.replace("'", " ").replace("`", " ").replace("\u00b4", " ")
    s = " ".join(s.split())
    return TEAM_ALIASES.get(s, s)


def build_match_key(home, away):
    return f"{normalize_team(home)}__{normalize_team(away)}"


# Load & merge
official = pd.read_csv(OFFICIAL_DIR / "England.csv")
official["league"] = "England"
momentum = pd.read_csv(MOMENTUM_CSV)

official["_key"] = official.apply(
    lambda r: build_match_key(r["home_team_name"], r["away_team_name"]), axis=1
)
momentum["_key"] = momentum.apply(
    lambda r: build_match_key(r["home_team"], r["away_team"]), axis=1
)
momentum = momentum.drop_duplicates("_key", keep="first")
merged = official.merge(momentum[["_key"] + MOMENTUM_COLS], on="_key", how="inner").drop(columns=["_key"])
merged["ht_result_encoded"] = merged["ht_goal_difference"].apply(lambda x: 0 if x > 0 else (2 if x < 0 else 1))
merged["half_time_lead"] = merged["ht_goal_difference"]

# Prepare features for pre_momentum
exclude = EXCLUDE_ALWAYS | set(HT_COLUMNS)
y = merged[TARGET].map({"H": 0, "D": 1, "A": 2}).astype(int)
candidate_cols = [c for c in merged.columns if c not in exclude]
X = merged[candidate_cols].select_dtypes(include=[np.number])
nunique = X.nunique(dropna=True)
X = X.drop(columns=nunique[nunique <= 1].index.tolist())

# Time split
df_sorted = merged.sort_values("match_datetime").reset_index(drop=True)
n = len(df_sorted)
n_train = int(n * 0.6)
n_val = int(n * 0.2)
train_idx = df_sorted.index[:n_train]
val_idx = df_sorted.index[n_train:n_train + n_val]
test_idx = df_sorted.index[n_train + n_val:]

X_train, y_train = X.loc[train_idx], y.loc[train_idx]
X_val, y_val = X.loc[val_idx], y.loc[val_idx]
X_test, y_test = X.loc[test_idx], y.loc[test_idx]

med = X_train.median(numeric_only=True)
X_train = X_train.fillna(med)
X_val = X_val.fillna(med)
X_test = X_test.fillna(med)

print(f"England pre_momentum: train={len(X_train)} val={len(X_val)} test={len(X_test)} features={X_train.shape[1]}", flush=True)


def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 20.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "auto_class_weights": trial.suggest_categorical(
            "auto_class_weights", ["None", "Balanced", "SqrtBalanced"]
        ),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val),
              use_best_model=True, early_stopping_rounds=50)
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=False)

print(f"Best val accuracy: {study.best_value:.4f}", flush=True)
best_params = study.best_params
best_params.update({"random_seed": 42, "verbose": False, "allow_writing_files": False})

# Retrain on train+val
X_trval = pd.concat([X_train, X_val], ignore_index=True)
y_trval = pd.concat([y_train, y_val], ignore_index=True)
model = CatBoostClassifier(**best_params)
model.fit(X_trval, y_trval)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)
acc = accuracy_score(y_test, preds)
f1_macro = f1_score(y_test, preds, average="macro")
ll = log_loss(y_test, probs, labels=[0, 1, 2])
auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro", labels=[0, 1, 2])
f1_per = f1_score(y_test, preds, average=None, labels=[0, 1, 2])

print(f"\nEngland pre_momentum (100 trials, wider search):")
print(f"  test_accuracy  = {acc:.4f}")
print(f"  test_f1_macro  = {f1_macro:.4f}")
print(f"  test_log_loss  = {ll:.4f}")
print(f"  test_auc_macro = {auc:.4f}")
print(f"  test_draw_f1   = {f1_per[1]:.4f}")
print(f"  baseline acc   = 0.5921")
print(f"  delta          = {acc - 0.5921:+.4f}")
print(f"  best_params    = {study.best_params}")
