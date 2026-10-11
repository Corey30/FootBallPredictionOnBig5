#!/usr/bin/env python3
"""
Fetch and normalize open/close match odds for Top 5 European leagues
from football-data.co.uk.

Default range:
2015-2016 to 2024-2025

Supported leagues:
- England Premier League: E0
- Spain La Liga: SP1
- Germany Bundesliga: D1
- Italy Serie A: I1
- France Ligue 1: F1

Main output odds features:
- league
- league_code
- league_display_name
- season
- match_date
- home_team_name
- away_team_name
- home_team_key
- away_team_key
- odds_open_home / draw / away
- odds_close_home / draw / away
- odds_open_source
- odds_close_source
- odds_move_home / draw / away
- odds_move_abs_home / draw / away
- implied_prob_open_home / draw / away
- implied_prob_close_home / draw / away
- implied_prob_shift_home / draw / away
- overround_open
- overround_close

Optional:
- Use --main-csv to check merge coverage against your main feature table.
- Outputs unmatched reports when --main-csv is provided.
"""

from __future__ import annotations

import argparse
import io
import time
import urllib.error
import urllib.request
from pathlib import Path
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd


# =========================================================
# 1. League config
# =========================================================

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"

LEAGUE_CONFIG = {
    "england": {
        "league": "England",
        "league_code": "E0",
        "display_name": "Premier League",
    },
    "spain": {
        "league": "Spain",
        "league_code": "SP1",
        "display_name": "La Liga",
    },
    "germany": {
        "league": "Germany",
        "league_code": "D1",
        "display_name": "Bundesliga",
    },
    "italy": {
        "league": "Italy",
        "league_code": "I1",
        "display_name": "Serie A",
    },
    "france": {
        "league": "France",
        "league_code": "F1",
        "display_name": "Ligue 1",
    },
}


# =========================================================
# 2. Odds source priority
# =========================================================

OPEN_PRIORITY = [
    ("B365", ("B365H", "B365D", "B365A")),
    ("PS", ("PSH", "PSD", "PSA")),
    ("Avg", ("AvgH", "AvgD", "AvgA")),
    ("Max", ("MaxH", "MaxD", "MaxA")),
]

CLOSE_PRIORITY = [
    ("B365", ("B365CH", "B365CD", "B365CA")),
    ("PS", ("PSCH", "PSCD", "PSCA")),
    ("Avg", ("AvgCH", "AvgCD", "AvgCA")),
    ("Max", ("MaxCH", "MaxCD", "MaxCA")),
]


# =========================================================
# 3. Built-in team aliases
# =========================================================

BUILTIN_TEAM_ALIAS_GROUPS = {
    "england": [
        ("Arsenal", "Arsenal"),
        ("Aston Villa", "Aston Villa"),
        ("Bournemouth", "Bournemouth", "AFC Bournemouth"),
        ("Brentford", "Brentford"),
        ("Brighton & Hove Albion", "Brighton", "Brighton and Hove Albion"),
        ("Burnley", "Burnley"),
        ("Cardiff City", "Cardiff"),
        ("Chelsea", "Chelsea"),
        ("Crystal Palace", "Crystal Palace"),
        ("Everton", "Everton"),
        ("Fulham", "Fulham"),
        ("Huddersfield Town", "Huddersfield"),
        ("Hull City", "Hull"),
        ("Leeds United", "Leeds"),
        ("Leicester City", "Leicester"),
        ("Liverpool", "Liverpool"),
        ("Luton Town", "Luton"),
        ("Manchester City", "Man City", "Manchester City"),
        ("Manchester United", "Man United", "Manchester United"),
        ("Middlesbrough", "Middlesbrough"),
        ("Newcastle United", "Newcastle"),
        ("Norwich City", "Norwich"),
        ("Nottingham Forest", "Nott'm Forest", "Nottm Forest", "Nottingham Forest"),
        ("Queens Park Rangers", "QPR"),
        ("Sheffield United", "Sheffield United", "Sheffield Utd"),
        ("Southampton", "Southampton"),
        ("Stoke City", "Stoke"),
        ("Sunderland", "Sunderland"),
        ("Swansea City", "Swansea"),
        ("Tottenham Hotspur", "Tottenham", "Spurs"),
        ("Watford", "Watford"),
        ("West Bromwich Albion", "West Brom", "West Bromwich Albion"),
        ("West Ham United", "West Ham"),
        ("Wolverhampton Wanderers", "Wolves", "Wolverhampton Wanderers"),
    ],
    "spain": [
        ("Athletic Club Bilbao", "Ath Bilbao", "Athletic Bilbao"),
        ("Atlético Madrid", "Ath Madrid", "Atletico Madrid", "Atl Madrid"),
        ("Deportivo Alavés", "Alaves", "Deportivo Alaves"),
        ("Deportivo La Coruña", "La Coruna", "Deportivo La Coruna"),
        ("RCD Espanyol", "Espanol", "Espanyol"),
        ("UD Las Palmas", "Las Palmas"),
        ("Real Valladolid", "Valladolid"),
        ("Cádiz", "Cadiz"),
        ("Málaga CF", "Malaga", "Malaga CF"),
        ("Real Betis", "Betis"),
        ("Real Sociedad", "Sociedad"),
        ("FC Barcelona", "Barcelona"),
        ("Valencia CF", "Valencia"),
        ("Villarreal", "Villarreal CF"),
        ("Rayo Vallecano", "Vallecano", "Rayo"),
        ("Celta Vigo", "Celta", "Celta de Vigo"),
        ("Eibar", "Eibar"),
        ("Elche", "Elche"),
        ("Getafe", "Getafe"),
        ("Girona", "Girona"),
        ("Granada", "Granada"),
        ("Huesca", "Huesca"),
        ("Leganés", "Leganes"),
        ("Levante", "Levante"),
        ("Mallorca", "Mallorca"),
        ("Osasuna", "Osasuna"),
        ("Real Madrid", "Real Madrid"),
        ("Sevilla", "Sevilla"),
        ("Sporting Gijón", "Sp Gijon", "Sporting Gijon"),
    ],
    "germany": [
        ("Bayern Munich", "Bayern Munich", "Bayern Munchen", "Bayern München"),
        ("Borussia Dortmund", "Dortmund", "Borussia Dortmund"),
        ("Bayer Leverkusen", "Leverkusen", "Bayer Leverkusen"),
        ("Borussia Mönchengladbach", "M'gladbach", "Mgladbach", "Borussia Monchengladbach"),
        ("RB Leipzig", "RB Leipzig", "Leipzig"),
        ("Eintracht Frankfurt", "Ein Frankfurt", "Frankfurt"),
        ("SC Freiburg", "Freiburg"),
        ("VfL Wolfsburg", "Wolfsburg"),
        ("Werder Bremen", "Werder Bremen", "Bremen"),
        ("VfB Stuttgart", "Stuttgart"),
        ("Mainz 05", "Mainz"),
        ("FC Augsburg", "Augsburg"),
        ("Hoffenheim", "Hoffenheim"),
        ("Hertha Berlin", "Hertha"),
        ("Schalke 04", "Schalke 04", "Schalke"),
        ("FC Köln", "FC Koln", "Köln", "Koln"),
        ("Union Berlin", "Union Berlin"),
        ("Bochum", "Bochum"),
        ("Darmstadt", "Darmstadt"),
        ("Hamburger SV", "Hamburg"),
        ("Hannover 96", "Hannover"),
        ("Ingolstadt", "Ingolstadt"),
        ("Paderborn", "Paderborn"),
        ("Fortuna Düsseldorf", "Fortuna Dusseldorf", "Dusseldorf"),
        ("Arminia Bielefeld", "Bielefeld"),
        ("Greuther Fürth", "Greuther Furth", "Greuther Fürth"),
        ("Heidenheim", "Heidenheim"),
        ("Holstein Kiel", "Holstein Kiel"),
    ],
    "italy": [
        ("Atalanta", "Atalanta"),
        ("Bologna", "Bologna"),
        ("Cagliari", "Cagliari"),
        ("Carpi", "Carpi"),
        ("Chievo", "Chievo"),
        ("Como", "Como"),
        ("Cremonese", "Cremonese"),
        ("Crotone", "Crotone"),
        ("Empoli", "Empoli"),
        ("Fiorentina", "Fiorentina"),
        ("Frosinone", "Frosinone"),
        ("Genoa", "Genoa"),
        ("Hellas Verona", "Verona", "Hellas Verona"),
        ("Inter Milan", "Inter", "Internazionale"),
        ("Juventus", "Juventus"),
        ("Lazio", "Lazio"),
        ("Lecce", "Lecce"),
        ("AC Milan", "Milan", "AC Milan"),
        ("Monza", "Monza"),
        ("Napoli", "Napoli"),
        ("Palermo", "Palermo"),
        ("Parma", "Parma"),
        ("Pescara", "Pescara"),
        ("Roma", "Roma"),
        ("Salernitana", "Salernitana"),
        ("Sampdoria", "Sampdoria"),
        ("Sassuolo", "Sassuolo"),
        ("Spezia", "Spezia"),
        ("SPAL", "Spal", "SPAL"),
        ("Torino", "Torino"),
        ("Udinese", "Udinese"),
        ("Venezia", "Venezia"),
    ],
    "france": [
        ("Ajaccio", "Ajaccio", "AC Ajaccio"),
        ("Amiens", "Amiens"),
        ("Angers", "Angers"),
        ("Auxerre", "Auxerre"),
        ("Bastia", "Bastia"),
        ("Bordeaux", "Bordeaux"),
        ("Brest", "Brest"),
        ("Caen", "Caen"),
        ("Clermont", "Clermont"),
        ("Dijon", "Dijon"),
        ("Guingamp", "Guingamp"),
        ("Lens", "Lens"),
        ("Lille", "Lille"),
        ("Lorient", "Lorient"),
        ("Lyon", "Lyon"),
        ("Marseille", "Marseille"),
        ("Metz", "Metz"),
        ("Monaco", "Monaco"),
        ("Montpellier", "Montpellier"),
        ("Nancy", "Nancy"),
        ("Nantes", "Nantes"),
        ("Nice", "Nice"),
        ("Nîmes", "Nimes"),
        ("Paris Saint-Germain", "Paris SG", "PSG", "Paris Saint Germain"),
        ("Reims", "Reims"),
        ("Rennes", "Rennes"),
        ("Saint-Étienne", "St Etienne", "Saint Etienne"),
        ("Strasbourg", "Strasbourg"),
        ("Toulouse", "Toulouse"),
        ("Troyes", "Troyes"),
    ],
}


# =========================================================
# 4. Normalization utilities
# =========================================================

def normalize_team_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = text.replace("&", "and")
    return "".join(ch for ch in text if ch.isalnum())


def normalize_league_key(value: str) -> str:
    x = str(value).strip().lower()

    if x in LEAGUE_CONFIG:
        return x
    if "england" in x or "premier" in x or x == "epl":
        return "england"
    if "spain" in x or "laliga" in x or "la liga" in x or "span" in x:
        return "spain"
    if "germany" in x or "german" in x or "bundesliga" in x:
        return "germany"
    if "italy" in x or "serie" in x:
        return "italy"
    if "france" in x or "ligue" in x:
        return "france"

    return x


def build_alias_map(
    league_key: str,
    alias_csv: Optional[Path] = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Returns:
    - alias_map: normalized variant key -> normalized canonical key
    - display_map: normalized canonical key -> display canonical name

    Optional alias CSV accepted formats:
    1. league, master_name, football_data_name
    2. league, canonical_name, variant_name
    3. master_name, football_data_name
    4. canonical_name, variant_name
    5. first two columns treated as canonical, variant
    """

    alias_map: dict[str, str] = {}
    display_map: dict[str, str] = {}

    groups = BUILTIN_TEAM_ALIAS_GROUPS.get(league_key, [])

    for group in groups:
        canonical_display = group[0]
        canonical_key = normalize_team_key(canonical_display)
        display_map[canonical_key] = canonical_display

        for alias in group:
            alias_map[normalize_team_key(alias)] = canonical_key

    if alias_csv is None or not alias_csv.exists():
        return alias_map, display_map

    alias_df = pd.read_csv(alias_csv)

    if alias_df.empty or alias_df.shape[1] < 2:
        return alias_map, display_map

    cols_lower = {str(c).lower(): c for c in alias_df.columns}

    if "league" in cols_lower:
        league_col = cols_lower["league"]
        tmp = alias_df[
            alias_df[league_col].astype(str).map(normalize_league_key).eq(league_key)
        ].copy()
        if not tmp.empty:
            alias_df = tmp

    cols_lower = {str(c).lower(): c for c in alias_df.columns}

    if "master_name" in cols_lower and "football_data_name" in cols_lower:
        canonical_col = cols_lower["master_name"]
        variant_col = cols_lower["football_data_name"]
    elif "canonical_name" in cols_lower and "variant_name" in cols_lower:
        canonical_col = cols_lower["canonical_name"]
        variant_col = cols_lower["variant_name"]
    else:
        canonical_col, variant_col = alias_df.columns[:2]

    for _, row in alias_df[[canonical_col, variant_col]].dropna().iterrows():
        canonical_display = str(row[canonical_col]).strip()
        variant_display = str(row[variant_col]).strip()

        canonical_key = normalize_team_key(canonical_display)
        variant_key = normalize_team_key(variant_display)

        if not canonical_key or not variant_key:
            continue

        alias_map[variant_key] = canonical_key
        alias_map.setdefault(canonical_key, canonical_key)
        display_map[canonical_key] = canonical_display

    return alias_map, display_map


def canonical_team_name(
    name: str,
    alias_map: dict[str, str],
    display_map: dict[str, str],
) -> str:
    key = normalize_team_key(name)
    canonical_key = alias_map.get(key, key)
    return display_map.get(canonical_key, str(name).strip())


def canonical_team_key(
    name: str,
    alias_map: dict[str, str],
) -> str:
    key = normalize_team_key(name)
    return alias_map.get(key, key)


def season_code(season: str) -> str:
    start, end = season.split("-")
    return f"{int(start) % 100:02d}{int(end) % 100:02d}"


def iter_seasons(start_season: str, end_season: str) -> list[str]:
    a0, b0 = [int(x) for x in start_season.split("-")]
    a1, b1 = [int(x) for x in end_season.split("-")]

    if b0 != a0 + 1 or b1 != a1 + 1:
        raise ValueError("Season format should be like 2015-2016 or 2024-2025.")

    seasons = []
    cur_a, cur_b = a0, b0

    while (cur_a, cur_b) <= (a1, b1):
        seasons.append(f"{cur_a}-{cur_b}")
        cur_a += 1
        cur_b += 1

    return seasons


# =========================================================
# 5. Fetch and odds parsing
# =========================================================

def fetch_csv(url: str, timeout: int, retries: int, sleep_seconds: float) -> pd.DataFrame:
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "soccer-odds-fetcher/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()

            return pd.read_csv(io.BytesIO(raw))

        except Exception as e:
            last_err = e
            if attempt < retries:
                wait = sleep_seconds * attempt
                print(f"[WARN] fetch failed attempt={attempt}/{retries}: {e}; retry in {wait:.1f}s")
                time.sleep(wait)

    raise last_err


def valid_triplet(
    row: pd.Series,
    cols: tuple[str, str, str],
) -> Optional[tuple[float, float, float]]:
    vals = []

    for c in cols:
        if c not in row.index:
            return None

        v = pd.to_numeric(row[c], errors="coerce")

        if pd.isna(v) or float(v) <= 1.0:
            return None

        vals.append(float(v))

    return vals[0], vals[1], vals[2]


def pick_open_close(row: pd.Series) -> tuple[dict[str, float], dict[str, str]]:
    open_vals = {"home": np.nan, "draw": np.nan, "away": np.nan}
    close_vals = {"home": np.nan, "draw": np.nan, "away": np.nan}

    open_source = ""
    close_source = ""

    for source, cols in OPEN_PRIORITY:
        triplet = valid_triplet(row, cols)
        if triplet is not None:
            open_vals = {
                "home": triplet[0],
                "draw": triplet[1],
                "away": triplet[2],
            }
            open_source = source
            break

    for source, cols in CLOSE_PRIORITY:
        triplet = valid_triplet(row, cols)
        if triplet is not None:
            close_vals = {
                "home": triplet[0],
                "draw": triplet[1],
                "away": triplet[2],
            }
            close_source = source
            break

    if np.isnan(close_vals["home"]) and not np.isnan(open_vals["home"]):
        close_vals = dict(open_vals)
        close_source = f"{open_source}_fallback_open"

    return (
        {
            "odds_open_home": open_vals["home"],
            "odds_open_draw": open_vals["draw"],
            "odds_open_away": open_vals["away"],
            "odds_close_home": close_vals["home"],
            "odds_close_draw": close_vals["draw"],
            "odds_close_away": close_vals["away"],
        },
        {
            "odds_open_source": open_source,
            "odds_close_source": close_source,
        },
    )


def safe_prob(odds: pd.Series) -> pd.Series:
    odds = pd.to_numeric(odds, errors="coerce")
    odds = odds.where(odds > 1.0)
    return 1.0 / odds


def add_odds_features(out_df: pd.DataFrame) -> pd.DataFrame:
    out_df = out_df.copy()

    for side in ("home", "draw", "away"):
        open_col = f"odds_open_{side}"
        close_col = f"odds_close_{side}"

        out_df[open_col] = pd.to_numeric(out_df[open_col], errors="coerce")
        out_df[close_col] = pd.to_numeric(out_df[close_col], errors="coerce")

        out_df[f"odds_move_{side}"] = out_df[close_col] - out_df[open_col]
        out_df[f"odds_move_abs_{side}"] = out_df[f"odds_move_{side}"].abs()

        out_df[f"implied_prob_open_{side}"] = safe_prob(out_df[open_col])
        out_df[f"implied_prob_close_{side}"] = safe_prob(out_df[close_col])
        out_df[f"implied_prob_shift_{side}"] = (
            out_df[f"implied_prob_close_{side}"]
            - out_df[f"implied_prob_open_{side}"]
        )

    out_df["overround_open"] = (
        out_df["implied_prob_open_home"]
        + out_df["implied_prob_open_draw"]
        + out_df["implied_prob_open_away"]
    )

    out_df["overround_close"] = (
        out_df["implied_prob_close_home"]
        + out_df["implied_prob_close_draw"]
        + out_df["implied_prob_close_away"]
    )

    return out_df


def parse_match_date(raw: pd.DataFrame) -> pd.Series:
    if "Date" not in raw.columns:
        return pd.Series(pd.NaT, index=raw.index)

    return pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce").dt.date


def fetch_one_league_season(
    league_key: str,
    season: str,
    timeout: int,
    retries: int,
    sleep_seconds: float,
    alias_csv: Optional[Path],
) -> Optional[pd.DataFrame]:
    cfg = LEAGUE_CONFIG[league_key]
    scode = season_code(season)

    url = FOOTBALL_DATA_URL.format(
        season_code=scode,
        league_code=cfg["league_code"],
    )

    try:
        raw = fetch_csv(
            url=url,
            timeout=timeout,
            retries=retries,
            sleep_seconds=sleep_seconds,
        )
    except urllib.error.HTTPError as e:
        print(f"[WARN] {cfg['league']} {season}: HTTP {e.code} -> {url}")
        return None
    except Exception as e:
        print(f"[WARN] {cfg['league']} {season}: fetch failed -> {e}")
        return None

    if raw.empty:
        print(f"[WARN] {cfg['league']} {season}: empty csv")
        return None

    required = ["Date", "HomeTeam", "AwayTeam"]
    missing = [c for c in required if c not in raw.columns]

    if missing:
        print(f"[WARN] {cfg['league']} {season}: missing key columns {missing}")
        return None

    alias_map, display_map = build_alias_map(league_key, alias_csv)

    raw = raw.copy()
    raw["match_date"] = parse_match_date(raw)
    raw = raw[raw["match_date"].notna()].copy()

    rows = []

    for _, row in raw.iterrows():
        odds_dict, source_dict = pick_open_close(row)

        home_raw = str(row["HomeTeam"])
        away_raw = str(row["AwayTeam"])

        home_name = canonical_team_name(home_raw, alias_map, display_map)
        away_name = canonical_team_name(away_raw, alias_map, display_map)

        home_key = canonical_team_key(home_raw, alias_map)
        away_key = canonical_team_key(away_raw, alias_map)

        rows.append({
            "league": cfg["league"],
            "league_key": league_key,
            "league_code": cfg["league_code"],
            "league_display_name": cfg["display_name"],
            "season": season,
            "match_date": row["match_date"],

            "home_team_name": home_name,
            "away_team_name": away_name,
            "home_team_key": home_key,
            "away_team_key": away_key,

            "home_team_name_raw_football_data": home_raw,
            "away_team_name_raw_football_data": away_raw,

            "odds_open_home": odds_dict["odds_open_home"],
            "odds_open_draw": odds_dict["odds_open_draw"],
            "odds_open_away": odds_dict["odds_open_away"],
            "odds_close_home": odds_dict["odds_close_home"],
            "odds_close_draw": odds_dict["odds_close_draw"],
            "odds_close_away": odds_dict["odds_close_away"],

            "odds_open_source": source_dict["odds_open_source"],
            "odds_close_source": source_dict["odds_close_source"],
        })

    out = pd.DataFrame(rows)

    print(
        f"[INFO] {cfg['league']} {season}: "
        f"raw_rows={len(raw)}, output_rows={len(out)}"
    )

    return out


# =========================================================
# 6. Quality checks
# =========================================================

def run_odds_quality_checks(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("ODDS QUALITY CHECK")
    print("=" * 70)

    print("\nRows by league and season:")
    print(
        df.groupby(["league", "season"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    key_cols = ["league_key", "season", "match_date", "home_team_key", "away_team_key"]
    dup_count = df.duplicated(subset=key_cols).sum()
    print(f"\nDuplicate odds match keys: {dup_count}")

    odds_cols = [
        "odds_open_home", "odds_open_draw", "odds_open_away",
        "odds_close_home", "odds_close_draw", "odds_close_away",
    ]

    print("\nMissing odds counts:")
    print(df[odds_cols].isna().sum().to_string())

    bad_odds = {}
    for c in odds_cols:
        bad_odds[c] = int((pd.to_numeric(df[c], errors="coerce") <= 1.0).sum())

    print("\nOdds <= 1 counts:")
    print(pd.Series(bad_odds).to_string())

    formula_checks = {}

    for side in ("home", "draw", "away"):
        move_expected = df[f"odds_close_{side}"] - df[f"odds_open_{side}"]
        move_diff = (df[f"odds_move_{side}"] - move_expected).abs()
        formula_checks[f"odds_move_{side}_mismatch"] = int((move_diff > 1e-9).sum())

        abs_expected = df[f"odds_move_{side}"].abs()
        abs_diff = (df[f"odds_move_abs_{side}"] - abs_expected).abs()
        formula_checks[f"odds_move_abs_{side}_mismatch"] = int((abs_diff > 1e-9).sum())

        open_prob_expected = 1.0 / df[f"odds_open_{side}"]
        close_prob_expected = 1.0 / df[f"odds_close_{side}"]

        open_prob_diff = (df[f"implied_prob_open_{side}"] - open_prob_expected).abs()
        close_prob_diff = (df[f"implied_prob_close_{side}"] - close_prob_expected).abs()

        formula_checks[f"implied_prob_open_{side}_mismatch"] = int((open_prob_diff > 1e-9).sum())
        formula_checks[f"implied_prob_close_{side}_mismatch"] = int((close_prob_diff > 1e-9).sum())

        shift_expected = df[f"implied_prob_close_{side}"] - df[f"implied_prob_open_{side}"]
        shift_diff = (df[f"implied_prob_shift_{side}"] - shift_expected).abs()
        formula_checks[f"implied_prob_shift_{side}_mismatch"] = int((shift_diff > 1e-9).sum())

    overround_open_expected = (
        df["implied_prob_open_home"]
        + df["implied_prob_open_draw"]
        + df["implied_prob_open_away"]
    )
    overround_close_expected = (
        df["implied_prob_close_home"]
        + df["implied_prob_close_draw"]
        + df["implied_prob_close_away"]
    )

    formula_checks["overround_open_mismatch"] = int(
        ((df["overround_open"] - overround_open_expected).abs() > 1e-9).sum()
    )
    formula_checks["overround_close_mismatch"] = int(
        ((df["overround_close"] - overround_close_expected).abs() > 1e-9).sum()
    )

    print("\nFormula mismatch counts:")
    print(pd.Series(formula_checks).to_string())

    if dup_count == 0 and all(v == 0 for v in formula_checks.values()):
        print("\nPASS: odds table keys and formulas look consistent.")
    else:
        print("\nCHECK: duplicates or formula mismatches found.")


# =========================================================
# 7. Optional main table matching
# =========================================================

def detect_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c

    lower_map = {str(c).lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    return None


def prepare_main_for_matching(
    main: pd.DataFrame,
    alias_csv: Optional[Path],
) -> pd.DataFrame:
    main = main.copy()

    league_col = detect_col(main, ["league", "league_key"])
    season_col = detect_col(main, ["season"])
    home_col = detect_col(main, ["home_team_name", "home_team", "home"])
    away_col = detect_col(main, ["away_team_name", "away_team", "away"])

    if league_col is None:
        raise ValueError("Main CSV missing league column.")
    if season_col is None:
        raise ValueError("Main CSV missing season column.")
    if home_col is None or away_col is None:
        raise ValueError("Main CSV missing home/away team columns.")

    if "match_datetime" in main.columns:
        main["match_date"] = pd.to_datetime(main["match_datetime"], errors="coerce").dt.date
    elif "date_GMT" in main.columns:
        main["match_date"] = pd.to_datetime(main["date_GMT"], errors="coerce").dt.date
    elif "date_gmt" in main.columns:
        main["match_date"] = pd.to_datetime(main["date_gmt"], errors="coerce").dt.date
    elif "timestamp" in main.columns:
        main["match_date"] = pd.to_datetime(
            pd.to_numeric(main["timestamp"], errors="coerce"),
            unit="s",
            errors="coerce"
        ).dt.date
    else:
        raise ValueError("Main CSV missing match_datetime / date_GMT / date_gmt / timestamp.")

    main["league_key"] = main[league_col].map(normalize_league_key)
    main["season"] = main[season_col].astype(str)

    home_keys = []
    away_keys = []
    home_names = []
    away_names = []

    for _, row in main.iterrows():
        lk = row["league_key"]

        alias_map, display_map = build_alias_map(
            league_key=lk,
            alias_csv=alias_csv,
        )

        home_raw = str(row[home_col])
        away_raw = str(row[away_col])

        home_keys.append(canonical_team_key(home_raw, alias_map))
        away_keys.append(canonical_team_key(away_raw, alias_map))
        home_names.append(canonical_team_name(home_raw, alias_map, display_map))
        away_names.append(canonical_team_name(away_raw, alias_map, display_map))

    main["home_team_key"] = home_keys
    main["away_team_key"] = away_keys
    main["home_team_name_canonical"] = home_names
    main["away_team_name_canonical"] = away_names

    return main


def add_date_shift_keys(df: pd.DataFrame, days: int) -> pd.DataFrame:
    out = df.copy()
    base = pd.to_datetime(out["match_date"], errors="coerce")

    if days == 0:
        out["match_date_match"] = base.dt.date
    else:
        out["match_date_match"] = (base + pd.to_timedelta(days, unit="D")).dt.date

    return out


def match_with_date_tolerance(
    main: pd.DataFrame,
    odds: pd.DataFrame,
    tolerance_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Exact match first.
    Then unmatched main rows try date offsets from -tolerance_days to +tolerance_days.

    Returns:
    - matched rows
    - unmatched_main
    - unmatched_odds
    """

    main = main.copy()
    odds = odds.copy()

    main["_main_row_id"] = np.arange(len(main))
    odds["_odds_row_id"] = np.arange(len(odds))

    key_base = ["league_key", "season", "home_team_key", "away_team_key"]

    matched_parts = []
    used_main = set()
    used_odds = set()

    offsets = [0]
    for d in range(1, tolerance_days + 1):
        offsets.extend([-d, d])

    for offset in offsets:
        main_left = main[~main["_main_row_id"].isin(used_main)].copy()
        odds_left = odds[~odds["_odds_row_id"].isin(used_odds)].copy()

        if main_left.empty or odds_left.empty:
            break

        main_tmp = add_date_shift_keys(main_left, 0)
        odds_tmp = add_date_shift_keys(odds_left, offset)

        merge_keys = key_base + ["match_date_match"]

        m = main_tmp.merge(
            odds_tmp,
            on=merge_keys,
            how="inner",
            suffixes=("_main", "_odds"),
        )

        if m.empty:
            continue

        # Keep only unique one-to-one matches for this offset.
        m["_main_count"] = m.groupby("_main_row_id")["_main_row_id"].transform("size")
        m["_odds_count"] = m.groupby("_odds_row_id")["_odds_row_id"].transform("size")
        m = m[(m["_main_count"] == 1) & (m["_odds_count"] == 1)].copy()

        if m.empty:
            continue

        m["date_offset_used"] = offset

        matched_parts.append(m)

        used_main.update(m["_main_row_id"].tolist())
        used_odds.update(m["_odds_row_id"].tolist())

    matched = pd.concat(matched_parts, ignore_index=True, sort=False) if matched_parts else pd.DataFrame()

    unmatched_main = main[~main["_main_row_id"].isin(used_main)].copy()
    unmatched_odds = odds[~odds["_odds_row_id"].isin(used_odds)].copy()

    return matched, unmatched_main, unmatched_odds


def run_main_match_check(
    main_csv: Path,
    odds_df: pd.DataFrame,
    alias_csv: Optional[Path],
    out_dir: Path,
    tolerance_days: int,
    merged_out_csv: Optional[Path],
) -> None:
    print("\n" + "=" * 70)
    print("MAIN TABLE MATCH CHECK")
    print("=" * 70)

    main = pd.read_csv(main_csv, low_memory=False)
    main_prepared = prepare_main_for_matching(main, alias_csv=alias_csv)

    odds = odds_df.copy()
    odds["match_date"] = pd.to_datetime(odds["match_date"], errors="coerce").dt.date

    matched, unmatched_main, unmatched_odds = match_with_date_tolerance(
        main=main_prepared,
        odds=odds,
        tolerance_days=tolerance_days,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    unmatched_main_path = out_dir / "unmatched_main_to_odds.csv"
    unmatched_odds_path = out_dir / "unmatched_odds_to_main.csv"
    matched_path = out_dir / "matched_main_to_odds.csv"

    unmatched_main.to_csv(unmatched_main_path, index=False, encoding="utf-8-sig")
    unmatched_odds.to_csv(unmatched_odds_path, index=False, encoding="utf-8-sig")
    matched.to_csv(matched_path, index=False, encoding="utf-8-sig")

    print(f"Main rows: {len(main_prepared)}")
    print(f"Odds rows: {len(odds)}")
    print(f"Matched rows: {len(matched)}")
    print(f"Unmatched main rows: {len(unmatched_main)}")
    print(f"Unmatched odds rows: {len(unmatched_odds)}")

    if not unmatched_main.empty:
        print("\nUnmatched main by league-season:")
        print(
            unmatched_main.groupby(["league_key", "season"])
            .size()
            .sort_index()
            .to_string()
        )

    if not unmatched_odds.empty:
        print("\nUnmatched odds by league-season:")
        print(
            unmatched_odds.groupby(["league_key", "season"])
            .size()
            .sort_index()
            .to_string()
        )

    print("\nSaved reports:")
    print(matched_path)
    print(unmatched_main_path)
    print(unmatched_odds_path)

    if merged_out_csv is not None:
        odds_feature_cols = [
            "league_key", "season", "match_date", "home_team_key", "away_team_key",
            "odds_open_home", "odds_open_draw", "odds_open_away",
            "odds_close_home", "odds_close_draw", "odds_close_away",
            "odds_open_source", "odds_close_source",
            "odds_move_home", "odds_move_draw", "odds_move_away",
            "odds_move_abs_home", "odds_move_abs_draw", "odds_move_abs_away",
            "implied_prob_open_home", "implied_prob_open_draw", "implied_prob_open_away",
            "implied_prob_close_home", "implied_prob_close_draw", "implied_prob_close_away",
            "implied_prob_shift_home", "implied_prob_shift_draw", "implied_prob_shift_away",
            "overround_open", "overround_close",
        ]

        # Exact-date merge for the saved merged table.
        # If unmatched caused by ±1 day, use the matched report to inspect before production merge.
        main_exact = main_prepared.merge(
            odds[odds_feature_cols],
            on=["league_key", "season", "match_date", "home_team_key", "away_team_key"],
            how="left",
            suffixes=("", "_odds"),
        )

        merged_out_csv.parent.mkdir(parents=True, exist_ok=True)
        main_exact.to_csv(merged_out_csv, index=False, encoding="utf-8-sig")
        print(f"\nSaved exact-date merged main table to: {merged_out_csv}")


# =========================================================
# 8. CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Top 5 European league open/close odds from football-data.co.uk"
    )

    parser.add_argument(
        "--leagues",
        nargs="+",
        default=["all"],
        help="Leagues to fetch: all, england, spain, germany, italy, france.",
    )

    parser.add_argument("--start-season", type=str, default="2015-2016")
    parser.add_argument("--end-season", type=str, default="2024-2025")

    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("top5_odds_2015_2025.csv"),
    )

    parser.add_argument(
        "--alias-csv",
        type=Path,
        default=None,
        help="Optional alias CSV. Recommended columns: league, master_name, football_data_name.",
    )

    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.8)

    parser.add_argument("--no-checks", action="store_true")

    parser.add_argument(
        "--main-csv",
        type=Path,
        default=None,
        help="Optional main feature table to check odds merge coverage.",
    )

    parser.add_argument(
        "--match-report-dir",
        type=Path,
        default=Path("odds_match_reports"),
        help="Where to save matched/unmatched reports if --main-csv is provided.",
    )

    parser.add_argument(
        "--date-tolerance-days",
        type=int,
        default=1,
        help="Date tolerance for matching reports. Exact match first, then ±N days.",
    )

    parser.add_argument(
        "--merged-out-csv",
        type=Path,
        default=None,
        help="Optional exact-date merged output when --main-csv is provided.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    requested = [x.lower() for x in args.leagues]

    if "all" in requested:
        league_keys = list(LEAGUE_CONFIG.keys())
    else:
        unknown = [x for x in requested if x not in LEAGUE_CONFIG]
        if unknown:
            raise ValueError(
                f"Unknown leagues: {unknown}. "
                f"Valid options: {list(LEAGUE_CONFIG.keys()) + ['all']}"
            )
        league_keys = requested

    seasons = iter_seasons(args.start_season, args.end_season)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_parts = []

    for league_key in league_keys:
        for season in seasons:
            part = fetch_one_league_season(
                league_key=league_key,
                season=season,
                timeout=args.timeout,
                retries=args.retries,
                sleep_seconds=args.sleep_seconds,
                alias_csv=args.alias_csv,
            )

            if part is not None and not part.empty:
                all_parts.append(part)

            time.sleep(max(0.0, args.sleep_seconds))

    if not all_parts:
        raise RuntimeError("No odds rows fetched.")

    out_df = pd.concat(all_parts, ignore_index=True, sort=False)
    out_df = add_odds_features(out_df)

    out_df = out_df.sort_values(
        ["league", "season", "match_date", "home_team_name", "away_team_name"]
    ).reset_index(drop=True)

    out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("FETCH FINISHED")
    print("=" * 70)
    print(f"Rows: {len(out_df)}")
    print(f"Saved odds table to: {args.out_csv}")

    if not args.no_checks:
        run_odds_quality_checks(out_df)

    if args.main_csv is not None:
        run_main_match_check(
            main_csv=args.main_csv,
            odds_df=out_df,
            alias_csv=args.alias_csv,
            out_dir=args.match_report_dir,
            tolerance_days=args.date_tolerance_days,
            merged_out_csv=args.merged_out_csv,
        )


if __name__ == "__main__":
    main()