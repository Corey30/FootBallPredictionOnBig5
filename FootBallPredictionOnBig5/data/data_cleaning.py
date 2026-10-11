import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from src.config.config import (
    IDENTIFIER_COLUMNS, TARGET_COLUMNS, TARGET_COLUMN, 
    LABEL_ENCODING, EXCLUDE_SEASONS
)


LEAKAGE_COLUMNS = [
    "home_ppg",
    "away_ppg",
    "home_team_goal_count",
    "away_team_goal_count",
    "total_goal_count",
    "full_time_result",
    "ft_result",
    "result",
    "home_team_goal_timings",
    "away_team_goal_timings",
    "home_team_possession",
    "away_team_possession",
    "home_team_shots",
    "away_team_shots",
    "home_team_shots_on_target",
    "away_team_shots_on_target",
    "home_team_shots_off_target",
    "away_team_shots_off_target",
    "home_team_corners",
    "away_team_corners",
    "home_team_corner_count",
    "away_team_corner_count",
    "home_team_fouls",
    "away_team_fouls",
    "home_team_yellow_cards",
    "away_team_yellow_cards",
    "home_team_red_cards",
    "away_team_red_cards",
    "home_team_cards",
    "away_team_cards",
    "home_team_goal_count_second_half",
    "away_team_goal_count_second_half",
    "total_goals_second_half",
    "home_team_second_half_cards",
    "away_team_second_half_cards",
    "home_team_yellow_cards_second_half",
    "away_team_yellow_cards_second_half",
    "home_team_red_cards_second_half",
    "away_team_red_cards_second_half",
    "home_team_shots_second_half",
    "away_team_shots_second_half",
    "home_team_shots_on_target_second_half",
    "away_team_shots_on_target_second_half",
    "home_team_corners_second_half",
    "away_team_corners_second_half",
    "home_team_fouls_second_half",
    "away_team_fouls_second_half",
    "team_a_xg",
    "team_b_xg"
]


class DataCleaner:
    """统一的数据清洗器"""

    def __init__(self,
                 keep_only_complete_status: bool = True,
                 drop_columns_missing_rate_gt: float = 0.6,
                 valid_target_labels: List[str] = None):
        """
        初始化数据清洗器
        Args:
            keep_only_complete_status: 只保留已完成的比赛
            drop_columns_missing_rate_gt: 删除缺失率超过此阈值的列
            valid_target_labels: 合法的目标标签列表
        """
        self.keep_only_complete_status = keep_only_complete_status
        self.drop_columns_missing_rate_gt = drop_columns_missing_rate_gt
        self.valid_target_labels = valid_target_labels or ['H', 'D', 'A']
        self.cleaning_report = {}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行完整的数据清洗流程
        """
        print("\n" + "=" * 80)
        print("数据清洗报告")
        print("=" * 80)
        
        original_shape = df.shape
        out = df.copy()
        
        # 1. 只保留已完成的比赛
        out = self._filter_complete_matches(out)
        
        # 2. 删除数据泄露列
        out = self._drop_leakage_columns(out)
        
        # 3. 处理比赛时间
        out = self._process_match_time(out)
        
        # 4. 检查目标标签是否合法
        out = self._validate_target_labels(out)
        
        # 5. 删除缺失率太高的列
        out = self._drop_high_missing_columns(out)
        
        # 6. 按match_id去重
        out = self._remove_duplicates(out)
        
        # 7. 排除特定赛季
        out = self._exclude_seasons(out)
        
        # 8. 排除标识符列和目标列（特征工程时使用）
        # 这个步骤在特征工程中进行，这里只记录
        
        # 生成清洗报告
        self._generate_cleaning_report(original_shape, out.shape)
        
        return out
    
    def _filter_complete_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """只保留已完成的比赛"""
        if not self.keep_only_complete_status:
            return df
        
        if "status" not in df.columns:
            print("\n[1] 跳过状态过滤 - 未找到status列")
            self.cleaning_report['status_filter'] = 'skipped (no status column)'
            return df
        
        before = len(df)
        s = df["status"].astype(str).str.strip().str.lower()
        df = df[s.eq("complete")].copy()
        after = len(df)
        
        removed = before - after
        print(f"\n[1] 状态过滤: 保留已完成的比赛")
        print(f"    原始记录数: {before}")
        print(f"    删除记录数: {removed}")
        print(f"    保留记录数: {after}")
        
        self.cleaning_report['status_filter'] = {
            'before': before,
            'removed': removed,
            'after': after
        }
        
        return df
    
    def _drop_leakage_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除数据泄露列（赛后统计数据）"""
        print(f"\n[2] 删除数据泄露列")

        cols_to_drop = [col for col in LEAKAGE_COLUMNS if col in df.columns]

        if cols_to_drop:
            print(f"    删除 {len(cols_to_drop)} 个数据泄露列:")
            for col in cols_to_drop[:10]:
                print(f"      - {col}")
            if len(cols_to_drop) > 10:
                print(f"      ... 还有 {len(cols_to_drop) - 10} 个列")

            df = df.drop(columns=cols_to_drop)
        else:
            print("    未发现数据泄露列")

        self.cleaning_report['leakage_columns'] = {
            'dropped_count': len(cols_to_drop),
            'dropped_columns': cols_to_drop
        }

        return df
    
    def _process_match_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理比赛时间
        按优先级查找时间字段：match_datetime > date_GMT > timestamp
        """
        print(f"\n[3] 处理比赛时间")
        
        # 定义时间字段的优先级
        time_columns_priority = ['match_datetime', 'date_GMT', 'timestamp']
        
        # 尝试每个时间字段，直到找到可用的
        time_col = None
        for col in time_columns_priority:
            if col not in df.columns:
                continue
            
            print(f"    尝试时间字段: {col}")
            
            # 检查是否已经是datetime类型
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                time_col = col
                print(f"    使用时间字段: {time_col} (已是datetime类型)")
                break
            
            # 尝试多种格式转换
            formats_to_try = [
                '%Y-%m-%d %H:%M:%S%z',  # "2020-01-01 15:00:00+00:00" (带时区)
                '%b %d %Y - %I:%M%p',   # "Jan 01 2020 - 03:00PM"
                '%Y-%m-%d %H:%M:%S',    # "2020-01-01 15:00:00"
                '%Y-%m-%d',             # "2020-01-01"
                '%d/%m/%Y %H:%M',       # "01/01/2020 15:00"
                '%d/%m/%Y',             # "01/01/2020"
            ]
            
            converted = False
            for fmt in formats_to_try:
                try:
                    temp_col = pd.to_datetime(df[col], format=fmt, errors='coerce')
                    # 检查转换成功率
                    success_rate = temp_col.notna().sum() / len(df)
                    if success_rate > 0.8:  # 如果成功率超过80%，认为转换成功
                        df[col] = temp_col
                        print(f"    转换{col}为datetime类型 (成功率: {success_rate:.2%})")
                        time_col = col
                        converted = True
                        break
                except:
                    continue
            
            if not converted:
                # 最后尝试自动推断（可以处理带时区的格式）
                temp_col = pd.to_datetime(df[col], errors='coerce')
                success_rate = temp_col.notna().sum() / len(df)
                if success_rate > 0.8:  # 如果成功率超过80%，认为转换成功
                    df[col] = temp_col
                    print(f"    自动推断转换{col}为datetime类型 (成功率: {success_rate:.2%})")
                    time_col = col
                    converted = True
            
            if converted:
                break
            else:
                print(f"    时间字段 {col} 转换失败，尝试下一个字段")
        
        if time_col is None:
            print("    错误: 所有时间字段转换失败 (match_datetime, date_GMT, timestamp)")
            raise ValueError("All time columns conversion failed (match_datetime, date_GMT, timestamp)")
        
        # 检查时间缺失情况
        before = len(df)
        missing_time = df[time_col].isna().sum()
        
        if missing_time > 0:
            print(f"    警告: 发现 {missing_time} 条记录时间缺失")
            
            # 客观判断：删除时间缺失的记录
            # 原因：
            # 1. 时间序列划分需要时间字段
            # 2. 时间缺失的数据可能不完整
            # 3. 设为NaT会导致排序问题
            print(f"    删除时间缺失的记录")
            df = df.dropna(subset=[time_col])
            after = len(df)
            removed = before - after
            print(f"    删除记录数: {removed}")
        else:
            print("    所有记录时间完整")
        
        # 检查未来比赛
        now = pd.Timestamp.now()
        # 处理带时区的时间戳
        if df[time_col].dt.tz is not None:
            now = pd.Timestamp.now(tz=df[time_col].dt.tz)
        
        future_matches = df[df[time_col] > now]
        
        if len(future_matches) > 0:
            print(f"    警告: 发现 {len(future_matches)} 场未来比赛")
            self.cleaning_report['future_matches'] = len(future_matches)
        else:
            print("    未发现未来比赛")
        
        # 按时间排序
        df = df.sort_values(time_col).reset_index(drop=True)
        print(f"    按时间排序完成")
        
        # 统一时间列名为date_GMT（用于后续处理）
        if time_col != 'date_GMT':
            df['date_GMT'] = df[time_col]
            print(f"    统一时间列名为date_GMT")
        
        self.cleaning_report['time_processing'] = {
            'time_column_used': time_col,
            'missing_time_records': missing_time,
            'records_removed': missing_time,
            'future_matches': len(future_matches)
        }
        
        return df
    
    def _validate_target_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """检查目标标签是否合法"""
        print(f"\n[4] 验证目标标签")
        
        if TARGET_COLUMN not in df.columns:
            print(f"    错误: 未找到目标列 '{TARGET_COLUMN}'")
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found")
        
        before = len(df)
        
        # 删除缺失值
        df = df.dropna(subset=[TARGET_COLUMN])
        
        # 检查标签是否合法
        invalid_labels = df[~df[TARGET_COLUMN].isin(self.valid_target_labels)]
        
        if len(invalid_labels) > 0:
            print(f"    警告: 发现 {len(invalid_labels)} 个非法标签")
            print(f"    非法标签: {invalid_labels[TARGET_COLUMN].unique()}")
            # 删除非法标签
            df = df[df[TARGET_COLUMN].isin(self.valid_target_labels)]
        
        after = len(df)
        removed = before - after
        
        print(f"    原始记录数: {before}")
        print(f"    删除记录数: {removed}")
        print(f"    保留记录数: {after}")
        print(f"    标签分布: {df[TARGET_COLUMN].value_counts().to_dict()}")
        
        self.cleaning_report['target_validation'] = {
            'before': before,
            'removed': removed,
            'after': after,
            'valid_labels': self.valid_target_labels,
            'label_distribution': df[TARGET_COLUMN].value_counts().to_dict()
        }
        
        return df
    
    def _drop_high_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除缺失率太高的列"""
        print(f"\n[5] 删除高缺失率列 (阈值: {self.drop_columns_missing_rate_gt})")
        
        # 计算缺失率
        missing_ratio = df.isnull().sum() / len(df)
        high_missing_cols = missing_ratio[missing_ratio > self.drop_columns_missing_rate_gt].index.tolist()
        
        # 排除标识符列和目标列（这些列即使缺失率高也要保留）
        protected_cols = set(IDENTIFIER_COLUMNS + TARGET_COLUMNS)
        cols_to_drop = [col for col in high_missing_cols if col not in protected_cols]
        
        if cols_to_drop:
            print(f"    删除 {len(cols_to_drop)} 个高缺失率列:")
            for col in cols_to_drop[:10]:  # 只显示前10个
                print(f"      - {col}: {missing_ratio[col]:.2%}")
            if len(cols_to_drop) > 10:
                print(f"      ... 还有 {len(cols_to_drop) - 10} 个列")
            
            df = df.drop(columns=cols_to_drop)
        else:
            print("    未发现高缺失率列")
        
        self.cleaning_report['high_missing_columns'] = {
            'threshold': self.drop_columns_missing_rate_gt,
            'dropped_count': len(cols_to_drop),
            'dropped_columns': cols_to_drop
        }
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """按match_id去重"""
        print(f"\n[6] 去重处理")
        
        if 'match_id' not in df.columns:
            print("    跳过 - 未找到match_id列")
            self.cleaning_report['deduplication'] = 'skipped (no match_id column)'
            return df
        
        before = len(df)
        df = df.drop_duplicates(subset=["match_id"], keep="first")
        after = len(df)
        removed = before - after
        
        print(f"    原始记录数: {before}")
        print(f"    删除重复数: {removed}")
        print(f"    保留记录数: {after}")
        
        self.cleaning_report['deduplication'] = {
            'before': before,
            'removed': removed,
            'after': after
        }
        
        return df
    
    def _exclude_seasons(self, df: pd.DataFrame) -> pd.DataFrame:
        """排除特定赛季"""
        print(f"\n[7] 排除特定赛季")
        
        if 'season' not in df.columns:
            print("    跳过 - 未找到season列")
            self.cleaning_report['season_exclusion'] = 'skipped (no season column)'
            return df
        
        before = len(df)
        df = df[~df['season'].isin(EXCLUDE_SEASONS)]
        after = len(df)
        removed = before - after
        
        print(f"    排除赛季: {EXCLUDE_SEASONS}")
        print(f"    删除记录数: {removed}")
        print(f"    保留记录数: {after}")
        
        self.cleaning_report['season_exclusion'] = {
            'excluded_seasons': EXCLUDE_SEASONS,
            'removed': removed,
            'after': after
        }
        
        return df
    
    def _generate_cleaning_report(self, original_shape: Tuple[int, int], 
                                  final_shape: Tuple[int, int]):
        """生成清洗报告"""
        print("\n" + "=" * 80)
        print("清洗汇总")
        print("=" * 80)
        
        print(f"\n原始数据: {original_shape[0]} 行, {original_shape[1]} 列")
        print(f"清洗后数据: {final_shape[0]} 行, {final_shape[1]} 列")
        print(f"删除行数: {original_shape[0] - final_shape[0]}")
        print(f"删除列数: {original_shape[1] - final_shape[1]}")
        
        self.cleaning_report['summary'] = {
            'original_shape': original_shape,
            'final_shape': final_shape,
            'rows_removed': original_shape[0] - final_shape[0],
            'columns_removed': original_shape[1] - final_shape[1]
        }
    
    def get_cleaning_report(self) -> Dict:
        """获取清洗报告"""
        return self.cleaning_report
    
    def save_cleaning_report(self, filepath: Path):
        """保存清洗报告"""
        import json
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.cleaning_report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n清洗报告已保存至: {filepath}")


def clean_dataset(input_path: str, output_path: str = None, 
                  keep_only_complete_status: bool = True,
                  drop_columns_missing_rate_gt: float = 0.6) -> pd.DataFrame:
    """
    清洗数据集的便捷函数
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径（可选）
        keep_only_complete_status: 是否只保留已完成的比赛
        drop_columns_missing_rate_gt: 删除缺失率超过此阈值的列
        
    Returns:
        清洗后的数据框
    """
    # 读取数据
    print(f"读取数据: {input_path}")
    df = pd.read_csv(input_path)
    print(f"原始数据形状: {df.shape}")
    
    # 创建清洗器
    cleaner = DataCleaner(
        keep_only_complete_status=keep_only_complete_status,
        drop_columns_missing_rate_gt=drop_columns_missing_rate_gt
    )
    
    # 执行清洗
    df_cleaned = cleaner.clean_data(df)
    
    # 保存结果
    if output_path:
        df_cleaned.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n清洗后数据已保存至: {output_path}")
        
        # 保存清洗报告
        report_path = Path(output_path).parent / 'cleaning_report.json'
        cleaner.save_cleaning_report(report_path)
    
    return df_cleaned


if __name__ == "__main__":
    # 示例用法
    import sys
    from pathlib import Path
    
    # 设置路径
    data_dir = Path(__file__).resolve().parent
    input_file = data_dir / 'Italy.csv'
    output_file = data_dir / 'Italy_cleaned.csv'
    
    # 执行清洗
    df_cleaned = clean_dataset(
        input_path=str(input_file),
        output_path=str(output_file),
        keep_only_complete_status=True,
        drop_columns_missing_rate_gt=0.6
    )
    
    print(f"\n清洗完成！最终数据形状: {df_cleaned.shape}")
