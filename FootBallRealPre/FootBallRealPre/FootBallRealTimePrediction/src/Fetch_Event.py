from statsbombpy import sb
import pandas as pd
from tqdm import tqdm
import os

matches = pd.read_csv("statsbomb_big5_2015_16_matches.csv")
os.makedirs("sb_events", exist_ok=True)

for mid in tqdm(matches["match_id"].unique()):
    try:
        ev = sb.events(match_id=mid).copy()
        ev["match_id"] = mid
        ev.to_parquet(f"sb_events/{mid}.parquet", index=False)
    except Exception as e:
        print(f"failed on {mid}: {e}")

print("done")