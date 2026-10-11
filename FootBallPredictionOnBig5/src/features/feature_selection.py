from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureSelectionConfig:
    """
    特征筛选配置
    
    默认值设置得比较宽容，避免过早删除有用特征
    
    注意：缺失率筛选已在数据清洗阶段完成
    """
    

    variance_drop_lte: float = 0.0
    

    corr_abs_drop_gte: float = 0.97
    corr_method: Literal["pearson", "spearman"] = "pearson"
    

    drop_zero_importance: bool = False
    

    max_features_for_corr: int = 300
    

    random_state: int = 42


def _variance(series: pd.Series) -> float:
    """计算方差"""
    s = pd.to_numeric(series, errors="coerce")
    return float(np.nanvar(s.to_numpy(dtype=float), ddof=0))


def _compute_abs_corr(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """计算绝对值相关性矩阵"""
    c = df.corr(method=method).abs()

    arr = c.values.copy()
    np.fill_diagonal(arr, 0.0)
    return pd.DataFrame(arr, index=c.index, columns=c.columns)


def _pick_drop_from_correlated_pair(a: str, b: str) -> str:
    """
    决策删除哪一个：
    - 按名字排序稳定删除
    """
    return sorted([a, b])[1]


def select_features(
    X_train: pd.DataFrame,
    y_train: Optional[pd.Series] = None,
    X_val: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    cfg: Optional[FeatureSelectionConfig] = None,
    *,
    feature_names_hint: Optional[list[str]] = None,
    feature_importance: Optional[dict[str, float]] = None,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], dict]:
    """
    对特征做可复现筛选，返回筛选后的 X_train/X_val/X_test 以及详细日志
    
    注意：
    - 只对数值列操作
    - 相关性去冗余默认较宽容（0.97），避免过早删掉信息量
    - 缺失率筛选已在数据清洗阶段完成
    
    Args:
        X_train: 训练集特征
        y_train: 训练集标签（可选，用于未来扩展）
        X_val: 验证集特征（可选）
        X_test: 测试集特征（可选）
        cfg: 特征选择配置
        feature_names_hint: 特征名称提示（可选）
        feature_importance: 特征重要性字典（可选），键为特征名，值为重要性分数
    
    Returns:
        (X_train_selected, X_val_selected, X_test_selected, log)
    """
    cfg = cfg or FeatureSelectionConfig()
    

    Xtr0 = X_train.select_dtypes(include=[np.number]).copy()
    Xva0 = X_val.select_dtypes(include=[np.number]).copy() if X_val is not None else None
    Xte0 = X_test.select_dtypes(include=[np.number]).copy() if X_test is not None else None
    

    all_nan = [c for c in Xtr0.columns if Xtr0[c].isna().all()]
    if all_nan:
        Xtr0 = Xtr0.drop(columns=all_nan, errors="ignore")
        if Xva0 is not None:
            Xva0 = Xva0.drop(columns=all_nan, errors="ignore")
        if Xte0 is not None:
            Xte0 = Xte0.drop(columns=all_nan, errors="ignore")
    
    if feature_names_hint is not None:

        cols = [c for c in feature_names_hint if c in Xtr0.columns]
        Xtr0 = Xtr0[cols].copy()
        if Xva0 is not None:
            Xva0 = Xva0.reindex(columns=cols)
        if Xte0 is not None:
            Xte0 = Xte0.reindex(columns=cols)
    
    original_cols = list(Xtr0.columns)
    removed: list[dict] = []
    

    if cfg.drop_zero_importance and feature_importance is not None:
        print("\n[特征筛选] 步骤0: 特征重要性筛选")

        zero_importance = [
            c for c in Xtr0.columns 
            if c in feature_importance and feature_importance[c] == 0.0
        ]
        
        if zero_importance:
            print(f"  删除 {len(zero_importance)} 个重要性为0的特征")
            for c in zero_importance:
                removed.append({"feature": c, "reason": "zero_importance", "importance": 0.0})
            Xtr0 = Xtr0.drop(columns=zero_importance, errors="ignore")
            if Xva0 is not None:
                Xva0 = Xva0.drop(columns=zero_importance, errors="ignore")
            if Xte0 is not None:
                Xte0 = Xte0.drop(columns=zero_importance, errors="ignore")
        else:
            print("  未发现重要性为0的特征")
    elif cfg.drop_zero_importance and feature_importance is None:
        print("\n[特征筛选] 步骤0: 特征重要性筛选 - 跳过（未提供feature_importance）")
    

    print("\n[特征筛选] 步骤1: 方差筛选")
    variances = {c: _variance(Xtr0[c]) for c in Xtr0.columns}
    drop_var = [c for c, v in variances.items() if v <= cfg.variance_drop_lte]
    
    if drop_var:
        print(f"  删除 {len(drop_var)} 个低方差列 (阈值: {cfg.variance_drop_lte})")
        for c in drop_var:
            removed.append({"feature": c, "reason": "low_variance", "variance": variances[c]})
        Xtr0 = Xtr0.drop(columns=drop_var, errors="ignore")
        if Xva0 is not None:
            Xva0 = Xva0.drop(columns=drop_var, errors="ignore")
        if Xte0 is not None:
            Xte0 = Xte0.drop(columns=drop_var, errors="ignore")
    else:
        print("  未发现低方差列")
    

    print("\n[特征筛选] 步骤2: 相关性去冗余")
    cols_for_corr = list(Xtr0.columns)
    corr_used_cols = cols_for_corr
    corr_skipped = False
    
    if len(cols_for_corr) > cfg.max_features_for_corr:
        corr_used_cols = cols_for_corr[: cfg.max_features_for_corr]
        corr_skipped = True
        print(f"  特征数过多，只处理前 {cfg.max_features_for_corr} 个特征")
    
    drop_corr: list[str] = []
    corr_pairs: list[dict] = []
    
    if len(corr_used_cols) >= 2:
        corr_df = _compute_abs_corr(Xtr0[corr_used_cols], method=cfg.corr_method)

        used = set(corr_used_cols)
        for i, a in enumerate(corr_used_cols):
            if a not in used:
                continue
            row = corr_df.loc[a]

            strong = row[row >= cfg.corr_abs_drop_gte].sort_values(ascending=False)
            for b, corr_val in strong.items():
                if b not in used or b == a:
                    continue
                to_drop = _pick_drop_from_correlated_pair(a, b)
                if to_drop in used:
                    used.remove(to_drop)
                    drop_corr.append(to_drop)
                    corr_pairs.append({"a": a, "b": b, "abs_corr": float(corr_val), "dropped": to_drop})
        
        if drop_corr:
            print(f"  删除 {len(set(drop_corr))} 个高相关列 (阈值: {cfg.corr_abs_drop_gte})")
            for c in drop_corr:
                removed.append({"feature": c, "reason": "high_correlation"})
            Xtr0 = Xtr0.drop(columns=sorted(set(drop_corr)), errors="ignore")
            if Xva0 is not None:
                Xva0 = Xva0.drop(columns=sorted(set(drop_corr)), errors="ignore")
            if Xte0 is not None:
                Xte0 = Xte0.drop(columns=sorted(set(drop_corr)), errors="ignore")
        else:
            print("  未发现高相关列")
    else:
        print("  特征数不足，跳过相关性筛选")
    

    log = {
        "config": asdict(cfg),
        "original_feature_count": int(len(original_cols)),
        "selected_feature_count": int(Xtr0.shape[1]),
        "removed_count": int(len(removed)),
        "removed": removed,
        "correlation": {
            "method": cfg.corr_method,
            "threshold": cfg.corr_abs_drop_gte,
            "max_features_for_corr": cfg.max_features_for_corr,
            "skipped_due_to_feature_count": bool(corr_skipped),

            "dropped_due_to_corr": sorted(set(drop_corr)),
        },
        "importance": {
            "dropped_zero_importance": [r["feature"] for r in removed if r["reason"] == "zero_importance"],
            "feature_importance_provided": feature_importance is not None,
        },
    }
    
    final_cols = list(Xtr0.columns)
    if Xva0 is not None:
        Xva0 = Xva0.reindex(columns=final_cols)
    if Xte0 is not None:
        Xte0 = Xte0.reindex(columns=final_cols)
    
    for col in final_cols:
        fill_val = Xtr0[col].median()
        if np.isnan(fill_val):
            fill_val = 0.0
        if Xva0 is not None and Xva0[col].isna().any():
            Xva0[col] = Xva0[col].fillna(fill_val)
        if Xte0 is not None and Xte0[col].isna().any():
            Xte0[col] = Xte0[col].fillna(fill_val)
    
    print(f"\n[特征筛选] 完成: {len(original_cols)} -> {Xtr0.shape[1]} 个特征")
    
    return Xtr0, Xva0, Xte0, log


def get_feature_selection_summary(log: dict) -> str:
    """生成特征筛选摘要报告"""
    summary = []
    summary.append("=" * 80)
    summary.append("特征筛选报告")
    summary.append("=" * 80)
    
    summary.append(f"\n原始特征数: {log['original_feature_count']}")
    summary.append(f"筛选后特征数: {log['selected_feature_count']}")
    summary.append(f"删除特征数: {log['removed_count']}")
    

    reasons = {}
    for item in log['removed']:
        reason = item['reason']
        if reason not in reasons:
            reasons[reason] = []
        reasons[reason].append(item['feature'])
    
    summary.append("\n删除原因统计:")
    for reason, features in reasons.items():
        summary.append(f"  {reason}: {len(features)} 个")
    

    if log['importance']['feature_importance_provided']:
        summary.append("\n特征重要性筛选:")
        summary.append(f"  删除重要性为0的特征数: {len(log['importance']['dropped_zero_importance'])}")
    

    summary.append("\n相关性筛选:")
    summary.append(f"  方法: {log['correlation']['method']}")
    summary.append(f"  阈值: {log['correlation']['threshold']}")
    summary.append(f"  删除特征数: {len(log['correlation']['dropped_due_to_corr'])}")
    
    return "\n".join(summary)