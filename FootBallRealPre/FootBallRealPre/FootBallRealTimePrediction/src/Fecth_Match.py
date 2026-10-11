
from statsbombpy import sb
import pandas as pd
from pathlib import Path

BIG5_1516 = [
    {"competition_id": 2, "season_id": 27, "league": "Premier League"},
    {"competition_id": 11, "season_id": 27, "league": "La Liga"},
    {"competition_id": 12, "season_id": 27, "league": "Serie A"},
    {"competition_id": 9, "season_id": 27, "league": "1. Bundesliga"},
    {"competition_id": 7, "season_id": 27, "league": "Ligue 1"},
]

def fetch_matches():
    frames = []
    for item in BIG5_1516:
        df = sb.matches(item["competition_id"], item["season_id"]).copy()
        df["league"] = item["league"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    out = Path("statsbomb_big5_2015_16_matches.csv")
    matches = fetch_matches()
    keep = [c for c in [
        "match_id", "match_date", "home_team", "away_team", "competition", "season", "league"
    ] if c in matches.columns]
    matches[keep].to_csv(out, index=False)
    print(f"saved {len(matches)} matches to {out}")
