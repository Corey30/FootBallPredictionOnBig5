"""
momentum_merge.py
将动量特征 CSV 与 15-16 赛季主表合并。

输入：
  all_leagues_2015-2016.csv（筛 season == 2015-2016）
  match_level_momentum_features.csv（或 --momentum-csv 指定的 CSV）
  event_state_table parquet（提供 StatsBomb 主客队名）

输出：
  feature_engineer5_momentum/all_leagues_2015-2016_momentum.merged.csv
  feature_engineer5_momentum/momentum_match_bridge.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import unicodedata
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

FEATURES_PATH = BASE_DIR / "all_leagues_2015-2016.csv"
MOMENTUM_PATH = BASE_DIR / "match_level_momentum_features.csv"
PARQUET_PATHS = [
    ROOT_DIR / "merged_data" / "event_state_table_big5_2015_16.parquet",
    ROOT_DIR / "merged_data" / "event_state_table_big5_2015_16.parquet",
    ROOT_DIR / "event_state_table_big5_2015_16.parquet",
    ROOT_DIR / "event_state_table_big5_2015_16.parquet",
]
REFERENCE_MERGED = (
    BASE_DIR / "feature_engineer_real_momentum" / "all_leagues_2015-2016_momentum.merged.csv"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "feature_engineer5_momentum"
DEFAULT_BRIDGE_PATH = DEFAULT_OUTPUT_DIR / "momentum_match_bridge.csv"

TARGET_SEASON = "2015-2016"
LEAGUES = ["England", "Spain", "Germany", "Italy", "France"]

MOMENTUM_COLS_PREFIX = ("eq_", "mod_raw_", "mod_ewm_", "mod_conf_w_", "mod_conf_", "n_rows_period1")
CURVE_COLS_PREFIX = ("eq_", "int_", "leq_", "lint_")


def _load_alias_module():
    path = BASE_DIR / "feature_enginner2(fetch_odd).py"
    spec = importlib.util.spec_from_file_location("fetch_odd", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _feat_slug(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _resolve_parquet_path() -> Path:
    for p in PARQUET_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(f"No event_state parquet found under {ROOT_DIR}")


def load_features_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, encoding="utf-8-sig")
    df = df[df["season"] == TARGET_SEASON].copy()
    df["match_id"] = df["match_id"].astype(str)
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"]).dt.strftime("%Y-%m-%d")
    print(f"[Features] {TARGET_SEASON}: {len(df):,} rows, {df.shape[1]:,} columns")
    return df


def load_momentum_data(momentum_path: Path | None = None) -> pd.DataFrame:
    path = momentum_path or MOMENTUM_PATH
    df = pd.read_csv(path, encoding="utf-8-sig")

    id_col = "match_id"
    if "statsbomb_match_id" in df.columns and "match_id" not in df.columns:
        df = df.rename(columns={"statsbomb_match_id": "match_id"})
        id_col = "match_id"

    df[id_col] = df[id_col].astype(str).str.replace(r"\.0$", "", regex=True)

    if "home_team" not in df.columns or "away_team" not in df.columns:
        pq = pd.read_parquet(
            _resolve_parquet_path(),
            columns=["match_id", "home_team", "away_team"],
        )
        pq = pq.drop_duplicates("match_id")
        pq["match_id"] = pq["match_id"].astype(str)
        df = df.merge(pq, on="match_id", how="left", suffixes=("", "_pq"))
        missing_teams = df["home_team"].isna().sum()
        if missing_teams:
            print(f"  [Warning] {missing_teams} momentum rows missing teams from parquet")
        else:
            print(f"  [Momentum] Teams joined from parquet for {path.name}")

    missing_teams = df["home_team"].isna().sum() if "home_team" in df.columns else len(df)
    if missing_teams:
        raise ValueError(f"{missing_teams} momentum rows missing teams")
    print(f"[Momentum] {len(df):,} rows with StatsBomb team names")
    return df


def momentum_feature_columns(df: pd.DataFrame, custom_prefixes: tuple[str, ...] | None = None) -> list[str]:
    prefixes = custom_prefixes or MOMENTUM_COLS_PREFIX
    return [c for c in df.columns if c.startswith(prefixes)]


def _merge_by_canonical_names(
    features: pd.DataFrame, momentum: pd.DataFrame, alias_mod,
    mom_cols: list[str] | None = None,
) -> pd.DataFrame:
    if mom_cols is None:
        mom_cols = momentum_feature_columns(momentum)
    parts: list[pd.DataFrame] = []

    for league in LEAGUES:
        lk = alias_mod.normalize_league_key(league)
        alias_map, _ = alias_mod.build_alias_map(lk, None)

        fl = features[features["league"] == league].copy()
        mm = momentum.copy()
        fl["join_home"] = fl["home_team_name"].map(
            lambda x, a=alias_map: alias_mod.canonical_team_key(x, a)
        )
        fl["join_away"] = fl["away_team_name"].map(
            lambda x, a=alias_map: alias_mod.canonical_team_key(x, a)
        )
        mm["join_home"] = mm["home_team"].map(
            lambda x, a=alias_map: alias_mod.canonical_team_key(x, a)
        )
        mm["join_away"] = mm["away_team"].map(
            lambda x, a=alias_map: alias_mod.canonical_team_key(x, a)
        )

        hit = fl.merge(
            mm[["match_id", "join_home", "join_away"] + mom_cols],
            on=["join_home", "join_away"],
            how="inner",
            suffixes=("_feature", "_statsbomb"),
        )
        hit = hit.rename(columns={"match_id_feature": "match_id", "match_id_statsbomb": "statsbomb_match_id"})
        hit["merge_method"] = "canonical_name"
        parts.append(hit[["match_id", "statsbomb_match_id", "merge_method"]])

    if not parts:
        return pd.DataFrame(columns=["match_id", "statsbomb_match_id", "merge_method"])

    bridge = pd.concat(parts, ignore_index=True)
    bridge = bridge.drop_duplicates("match_id", keep="first")
    return bridge


def _merge_by_slug_keys(
    features: pd.DataFrame, momentum: pd.DataFrame, already_matched: set[str],
    mom_cols: list[str] | None = None,
) -> pd.DataFrame:
    if mom_cols is None:
        mom_cols = momentum_feature_columns(momentum)
    mm = momentum.copy()
    mm["join_home"] = mm["home_team"].map(_feat_slug)
    mm["join_away"] = mm["away_team"].map(_feat_slug)

    fl = features[~features["match_id"].isin(already_matched)].copy()
    if fl.empty:
        return pd.DataFrame(columns=["match_id", "statsbomb_match_id", "merge_method"])

    hit = fl.merge(
        mm[["match_id", "join_home", "join_away"] + mom_cols],
        left_on=["home_team_key", "away_team_key"],
        right_on=["join_home", "join_away"],
        how="inner",
        suffixes=("_feature", "_statsbomb"),
    )
    hit = hit.rename(columns={"match_id_feature": "match_id", "match_id_statsbomb": "statsbomb_match_id"})
    hit["merge_method"] = "team_key_slug"
    return hit[["match_id", "statsbomb_match_id", "merge_method"]]


def build_bridge_from_reference(
    features: pd.DataFrame, momentum: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    ref = reference[reference["season"] == TARGET_SEASON][["match_id", "eq_mean"]].copy()
    ref = ref.drop_duplicates(subset=["eq_mean"])
    sb = momentum[["match_id", "eq_mean"]].drop_duplicates(subset=["eq_mean"])
    bridge = ref.merge(sb, on="eq_mean", suffixes=("_feature", "_statsbomb"))
    bridge = bridge.rename(
        columns={"match_id_feature": "match_id", "match_id_statsbomb": "statsbomb_match_id"}
    )
    bridge["merge_method"] = "reference_eq_mean"
    return bridge[["match_id", "statsbomb_match_id", "merge_method"]]


def build_match_bridge(
    features: pd.DataFrame,
    momentum: pd.DataFrame,
    *,
    use_reference: bool = True,
    mom_cols: list[str] | None = None,
) -> pd.DataFrame:
    alias_mod = _load_alias_module()

    if use_reference and mom_cols is None and REFERENCE_MERGED.exists():
        ref = pd.read_csv(REFERENCE_MERGED, encoding="utf-8-sig")
        bridge = build_bridge_from_reference(features, momentum, ref)
        print(f"  [Bridge] Built from reference file: {len(bridge):,} pairs")
        return bridge

    bridge1 = _merge_by_canonical_names(features, momentum, alias_mod, mom_cols)
    matched_ids = set(bridge1["match_id"])
    bridge2 = _merge_by_slug_keys(features, momentum, matched_ids, mom_cols)

    bridges = [bridge1]
    if len(bridge2):
        bridge2 = bridge2[~bridge2["match_id"].isin(matched_ids)]
        bridges.append(bridge2)

    bridge = pd.concat(bridges, ignore_index=True)
    bridge = bridge.drop_duplicates("match_id", keep="first")
    print(f"  [Bridge] Heuristic: canonical={len(bridge1):,}, +slug={len(bridge2):,}, total={len(bridge):,}")
    return bridge


def merge_datasets(
    features: pd.DataFrame,
    momentum: pd.DataFrame,
    bridge: pd.DataFrame,
    mom_cols: list[str] | None = None,
) -> pd.DataFrame:
    if mom_cols is None:
        mom_cols = momentum_feature_columns(momentum)
    mom_sub = momentum[["match_id"] + mom_cols].rename(columns={"match_id": "statsbomb_match_id"})

    out = features.merge(
        bridge[["match_id", "statsbomb_match_id", "merge_method"]],
        on="match_id",
        how="left",
        validate="one_to_one",
    )
    out = out.merge(mom_sub, on="statsbomb_match_id", how="left", validate="many_to_one")
    return out


def validate_merge(
    merged: pd.DataFrame,
    features: pd.DataFrame,
    momentum: pd.DataFrame,
    bridge: pd.DataFrame,
    reference: pd.DataFrame | None,
    mom_cols: list[str] | None = None,
) -> None:
    if mom_cols is None:
        mom_cols = momentum_feature_columns(momentum)
    first_col = mom_cols[0] if mom_cols else None
    print("\n[Validation]")
    print(f"  Feature rows ({TARGET_SEASON}): {len(features):,}")
    print(f"  Momentum rows:               {len(momentum):,}")
    print(f"  Bridge pairs:                {len(bridge):,}")

    if first_col and first_col in merged.columns:
        matched = merged[first_col].notna().sum()
        print(f"  Rows with momentum:          {matched:,} ({matched / len(features) * 100:.1f}%)")
        print(f"  Rows missing momentum:       {len(features) - matched:,}")

    if "merge_method" in bridge.columns:
        print("\n  Merge method counts:")
        for method, cnt in bridge["merge_method"].value_counts().items():
            print(f"    {method}: {cnt:,}")

    if "league" in merged.columns and first_col and first_col in merged.columns:
        print("\n  League distribution (with momentum):")
        has = merged[merged[first_col].notna()]
        for league, cnt in has["league"].value_counts().sort_index().items():
            total = (merged["league"] == league).sum()
            print(f"    {league}: {cnt:,} / {total:,}")

    dup = merged["match_id"].duplicated().sum()
    print(f"\n  Duplicate match_id: {dup} ({'OK' if dup == 0 else 'WARN'})")

    if reference is not None and "eq_mean" in reference.columns and "eq_mean" in merged.columns:
        ref = reference[reference["season"] == TARGET_SEASON]
        common = merged[merged["match_id"].isin(ref["match_id"])]
        if len(common) and "eq_mean" in common.columns:
            cmp = common[["match_id", "eq_mean"]].merge(
                ref[["match_id", "eq_mean"]],
                on="match_id",
                suffixes=("_new", "_ref"),
            )
            diff = (cmp["eq_mean_new"] - cmp["eq_mean_ref"]).abs()
            print(f"\n  vs reference merged file ({len(cmp):,} common matches):")
            print(f"    max |eq_mean diff|: {diff.max():.6f}")
            print(f"    rows with diff > 1e-6: {(diff > 1e-6).sum()}")

    if first_col and first_col in merged.columns:
        print("\n  Sample matched rows:")
        sample_cols = ["match_id", "home_team_name", "away_team_name", "league", first_col, "statsbomb_match_id"]
        avail = [c for c in sample_cols if c in merged.columns]
        sample = merged[merged[first_col].notna()][avail].head(3)
        for _, row in sample.iterrows():
            print(
                f"    {row.get('league', 'NA')}: {row.get('home_team_name', 'NA')} vs {row.get('away_team_name', 'NA')} | "
                f"sb={row.get('statsbomb_match_id', 'NA')} | {first_col}={row[first_col]:.4f}"
            )

        unmatched = merged[merged[first_col].isna()][["match_id", "home_team_name", "away_team_name", "league"]]
        if len(unmatched):
            print(f"\n  Unmatched feature rows (first 5 of {len(unmatched)}):")
            for _, row in unmatched.head(5).iterrows():
                print(f"    {row['league']}: {row['home_team_name']} vs {row['away_team_name']}")


def save_outputs(
    merged: pd.DataFrame,
    bridge: pd.DataFrame,
    output_dir: Path | None = None,
    bridge_path: Path | None = None,
) -> None:
    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    br_path = bridge_path or DEFAULT_BRIDGE_PATH
    out_dir.mkdir(parents=True, exist_ok=True)
    season_tag = TARGET_SEASON.replace("/", "-")
    out_path = out_dir / f"all_leagues_{season_tag}_momentum.merged.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    br_path.parent.mkdir(parents=True, exist_ok=True)
    bridge[["match_id", "statsbomb_match_id", "merge_method"]].to_csv(
        br_path, index=False, encoding="utf-8-sig"
    )
    print(f"\n[Saved] {out_path.name} ({len(merged):,} rows)")
    print(f"[Saved] {br_path.name} ({len(bridge):,} pairs)")


def run_merge(
    momentum_csv: Path | None = None,
    output_dir: Path | None = None,
    output_tag: str | None = None,
) -> None:
    if momentum_csv is not None:
        print(f"Momentum merge with custom CSV: {momentum_csv.name}")
    else:
        print("Momentum merge (default)")

    features = load_features_data()
    momentum = load_momentum_data(momentum_csv)

    is_custom = momentum_csv is not None
    if is_custom:
        mom_cols = momentum_feature_columns(momentum, custom_prefixes=CURVE_COLS_PREFIX)
        if not mom_cols:
            mom_cols = [c for c in momentum.columns if c not in ("match_id", "home_team", "away_team")]
        print(f"  [Custom] Feature columns: {len(mom_cols)}")
    else:
        mom_cols = None

    reference = None
    if REFERENCE_MERGED.exists():
        reference = pd.read_csv(REFERENCE_MERGED, encoding="utf-8-sig")
        print(f"[Reference] {REFERENCE_MERGED.name} for validation")

    if is_custom:
        print("\n[Merge] Loading match bridge from default merge output...")
        existing_bridge = DEFAULT_BRIDGE_PATH
        if existing_bridge.exists():
            bridge = pd.read_csv(existing_bridge, encoding="utf-8-sig")
            bridge["match_id"] = bridge["match_id"].astype(str)
            bridge["statsbomb_match_id"] = bridge["statsbomb_match_id"].astype(str)
            print(f"  [Bridge] Loaded from {existing_bridge.name}: {len(bridge):,} pairs")
        elif REFERENCE_MERGED.exists():
            print("  [Bridge] Default bridge not found, building from reference...")
            ref = pd.read_csv(REFERENCE_MERGED, encoding="utf-8-sig")
            ref_sb = ref[ref["season"] == TARGET_SEASON][["match_id", "statsbomb_match_id"]].copy()
            ref_sb["match_id"] = ref_sb["match_id"].astype(str)
            ref_sb["statsbomb_match_id"] = ref_sb["statsbomb_match_id"].astype(str)
            ref_sb["merge_method"] = "reference"
            bridge = ref_sb
            print(f"  [Bridge] Built from reference: {len(bridge):,} pairs")
        else:
            print("  [Bridge] No existing bridge or reference, using heuristic...")
            bridge = build_match_bridge(features, momentum, use_reference=False, mom_cols=mom_cols)
    else:
        print("\n[Merge] Building match bridge...")
        bridge = build_match_bridge(features, momentum, use_reference=True)

    print("\n[Merge] Joining momentum features onto feature table...")
    merged = merge_datasets(features, momentum, bridge, mom_cols)

    validate_merge(merged, features, momentum, bridge, reference, mom_cols)

    if output_dir is not None:
        odir = output_dir
    elif is_custom:
        tag = output_tag or momentum_csv.stem.replace("match_level_", "")
        odir = BASE_DIR / f"feature_engineer5_momentum_{tag}"
    else:
        odir = DEFAULT_OUTPUT_DIR

    br_path = odir / "momentum_match_bridge.csv"
    save_outputs(merged, bridge, output_dir=odir, bridge_path=br_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Momentum merge")
    parser.add_argument(
        "--momentum-csv",
        type=str,
        default=None,
        help="Path to custom momentum features CSV (e.g., match_level_formula_curve_features.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for merged CSV",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Tag for auto-generated output dir name (used when --output-dir not set)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Momentum merge (date/team keys)")
    print("=" * 60)

    momentum_csv = Path(args.momentum_csv) if args.momentum_csv else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    run_merge(
        momentum_csv=momentum_csv,
        output_dir=output_dir,
        output_tag=args.output_tag,
    )

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
