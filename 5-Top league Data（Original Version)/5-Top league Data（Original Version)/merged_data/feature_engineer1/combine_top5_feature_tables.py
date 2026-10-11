import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"E:\soccerf\5-Top league Data（Original Version)\5-Top league Data（Original Version)\merged_data\feature_engineer1")

files = [
    DATA_DIR / "England_merged_2015_2025_feature_engineered.csv",
    DATA_DIR / "France_merged_2015_2025_feature_engineered.csv",
    DATA_DIR / "Germany_merged_2015_2025_feature_engineered.csv",
    DATA_DIR / "Italy_merged_2015_2025_feature_engineered.csv",
    DATA_DIR / "Spain_merged_2015_2025_feature_engineered.csv",
]

dfs = []

for f in files:
    df = pd.read_csv(f, low_memory=False)

    # 如果某些表 league 列缺失，就从文件名补
    if "league" not in df.columns:
        league = f.name.split("_")[0]
        df["league"] = league

    dfs.append(df)
    print(f"OK: {f.name} | rows={len(df)} | cols={df.shape[1]}")

top5 = pd.concat(dfs, ignore_index=True, sort=False)

# 推荐建模前过滤 canceled / incomplete
if "status" in top5.columns:
    print("\nStatus counts before filter:")
    print(top5["status"].value_counts(dropna=False))

    # 如果你想保留所有比赛，就注释掉下面这一行
    top5 = top5[top5["status"] == "complete"].copy()

top5 = top5.sort_values(["league", "season", "match_datetime"]).reset_index(drop=True)

out = DATA_DIR / "top5_feature_engineered.csv"
top5.to_csv(out, index=False, encoding="utf-8-sig")

print("\nSaved:")
print(out)
print("Rows:", len(top5))
print("Columns:", top5.shape[1])
print("\nRows by league:")
print(top5["league"].value_counts())