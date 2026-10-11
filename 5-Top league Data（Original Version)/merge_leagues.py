import os
import pandas as pd
import re
from pathlib import Path

def extract_season_year(filename):
    match = re.search(r'(\d{4})-to-(\d{4})', filename)
    if match:
        return int(match.group(1))
    return 0

def merge_league_data(base_path, output_path=None):
    leagues = {
        'England': 'England-match2015-2026(original version)',
        'France': 'France-match2015-2026(original version)',
        'Germany': 'German-match2015-2026(original version)',
        'Italy': 'Italy-match2015-2026(original version)',
        'Spain': 'Span-match2015-2026(original version)'
    }
    
    merged_results = {}
    
    for league_name, league_folder in leagues.items():
        league_path = Path(base_path) / league_folder
        
        if not league_path.exists():
            print(f"警告: 找不到 {league_name} 的目录: {league_path}")
            continue
        
        csv_files = []
        for root, dirs, files in os.walk(league_path):
            for file in files:
                if file.endswith('.csv') and 'matches' in file.lower():
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            print(f"警告: {league_name} 没有找到CSV文件")
            continue
        
        csv_files_sorted = sorted(csv_files, key=lambda x: extract_season_year(os.path.basename(x)))
        
        print(f"\n{league_name} 联赛找到 {len(csv_files_sorted)} 个文件:")
        for f in csv_files_sorted:
            print(f"  - {os.path.basename(f)}")
        
        dfs = []
        for csv_file in csv_files_sorted:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                season_year = extract_season_year(os.path.basename(csv_file))
                df['season'] = f"{season_year}-{season_year + 1}"
                dfs.append(df)
                print(f"  读取 {os.path.basename(csv_file)}: {len(df)} 行")
            except Exception as e:
                print(f"  错误读取 {csv_file}: {e}")
        
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            
            if 'timestamp' in merged_df.columns:
                merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
            
            merged_results[league_name] = merged_df
            
            if output_path:
                output_file = os.path.join(output_path, f"{league_name}_merged_2015_2026.csv")
            else:
                output_file = os.path.join(league_path, f"{league_name}_merged_2015_2026.csv")
            
            merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n{league_name} 合并完成: {len(merged_df)} 行, 保存到: {output_file}")
    
    return merged_results

if __name__ == "__main__":
    base_path = r"e:\soccerf\5-Top league Data（Original Version)\5-Top league Data（Original Version)"
    
    output_path = r"e:\soccerf\5-Top league Data（Original Version)\5-Top league Data（Original Version)\merged_data"
    os.makedirs(output_path, exist_ok=True)
    
    merged_data = merge_league_data(base_path, output_path)
    
    print("\n" + "="*60)
    print("合并完成! 各联赛数据统计:")
    print("="*60)
    for league, df in merged_data.items():
        print(f"\n{league}:")
        print(f"  总行数: {len(df)}")
        print(f"  列数: {len(df.columns)}")
        if 'season' in df.columns:
            print(f"  赛季范围: {df['season'].min()} 到 {df['season'].max()}")
