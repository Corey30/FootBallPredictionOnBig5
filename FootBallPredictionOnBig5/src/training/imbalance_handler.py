"""
类别不平衡处理模块

提供多种策略处理足球比赛结果预测中的类别不平衡问题
特别是平局(D)样本通常较少的情况

包括:
  * 损失/决策侧:
      - compute_sample_weights / get_class_weight_dict (cost-sensitive)
      - adjust_decision_threshold / find_optimal_thresholds (阈值调优)
  * 训练分布侧 (data augmentation, 参考 Atta Mills et al. 2024):
      - apply_resampling(X, y, method)
        method ∈ {svm_smote, smote_nn, near_miss, random_oversample,
                  smote_tomek, none}
"""

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter


def analyze_class_imbalance(y):
    """
    分析类别不平衡情况

    Args:
        y: 标签数据

    Returns:
        dict: 包含类别分布和不平衡比率的字典
    """
    counter = Counter(y)
    total = len(y)

    distribution = {label: count / total for label, count in counter.items()}
    max_count = max(counter.values())
    min_count = min(counter.values())
    imbalance_ratio = max_count / min_count

    print(f"\n{'='*60}")
    print("Class Imbalance Analysis")
    print('='*60)
    print(f"Total samples: {total}")
    print(f"\nClass distribution:")
    for label in ['H', 'D', 'A']:
        if label in counter:
            count = counter[label]
            pct = distribution[label]
            print(f"  {label}: {count:5d} ({pct:6.2%})")

    print(f"\nImbalance ratio (max/min): {imbalance_ratio:.2f}")

    if imbalance_ratio > 2.0:
        print("  Status: Significant imbalance detected")
    elif imbalance_ratio > 1.5:
        print("  Status: Moderate imbalance")
    else:
        print("  Status: Relatively balanced")

    return {
        'distribution': distribution,
        'counts': dict(counter),
        'imbalance_ratio': imbalance_ratio
    }


def compute_sample_weights(y, strategy='balanced'):
    """
    计算样本权重

    Args:
        y: 标签数据
        strategy: 'balanced' (sklearn默认), 'sqrt' (平方根), 'log' (对数)

    Returns:
        样本权重数组
    """
    classes = np.unique(y)

    if strategy == 'balanced':

        class_weights = compute_class_weight('balanced', classes=classes, y=y)
    elif strategy == 'sqrt':

        counter = Counter(y)
        total = len(y)
        class_weights = np.array([
            np.sqrt(total / (len(classes) * counter[c])) for c in classes
        ])
    elif strategy == 'log':

        counter = Counter(y)
        total = len(y)
        class_weights = np.array([
            np.log(total / counter[c]) + 1 for c in classes
        ])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


    class_weight_dict = dict(zip(classes, class_weights))


    sample_weights = np.array([class_weight_dict[label] for label in y])

    print(f"\n{'='*60}")
    print(f"Sample Weights (strategy: {strategy})")
    print('='*60)
    for cls, weight in class_weight_dict.items():
        print(f"  Class {cls}: weight = {weight:.4f}")

    return sample_weights


def get_class_weight_dict(y, strategy='balanced'):
    """
    获取类别权重字典（用于sklearn模型的class_weight参数）

    Args:
        y: 标签数据
        strategy: 权重策略

    Returns:
        类别权重字典
    """
    from sklearn.preprocessing import LabelEncoder


    if isinstance(y.iloc[0] if isinstance(y, pd.Series) else y[0], str):
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        classes = np.arange(len(le.classes_))
    else:
        y_encoded = y
        classes = np.unique(y)

    if strategy == 'balanced':
        class_weights = compute_class_weight('balanced', classes=classes, y=y_encoded)
    elif strategy == 'sqrt':
        counter = Counter(y_encoded)
        total = len(y_encoded)
        class_weights = np.array([
            np.sqrt(total / (len(classes) * counter[c])) for c in classes
        ])
    elif strategy == 'log':
        counter = Counter(y_encoded)
        total = len(y_encoded)
        class_weights = np.array([
            np.log(total / counter[c]) + 1 for c in classes
        ])
    else:
        return None

    return dict(zip(classes, class_weights))


def adjust_decision_threshold(y_proba, thresholds={'H': 0.33, 'D': 0.25, 'A': 0.33}):
    """
    调整决策阈值以提高少数类的召回率

    Args:
        y_proba: 预测概率 (n_samples, 3)
        thresholds: 各类别的阈值字典

    Returns:
        调整后的预测标签
    """
    labels = ['H', 'D', 'A']
    y_pred_adjusted = []

    for proba in y_proba:

        above_threshold = []
        for i, label in enumerate(labels):
            if proba[i] >= thresholds[label]:
                above_threshold.append((i, proba[i]))

        if above_threshold:

            best_idx = max(above_threshold, key=lambda x: x[1])[0]
            y_pred_adjusted.append(labels[best_idx])
        else:

            y_pred_adjusted.append(labels[np.argmax(proba)])

    return np.array(y_pred_adjusted)


def find_optimal_thresholds(y_true, y_proba, metric='f1_macro', labels=['H', 'D', 'A']):
    """
    通过网格搜索找到最优决策阈值

    Args:
        y_true: 真实标签
        y_proba: 预测概率
        metric: 优化指标 ('f1_macro', 'f1_weighted', 'accuracy')
        labels: 类别标签

    Returns:
        最优阈值字典
    """
    from sklearn.metrics import f1_score, accuracy_score

    print(f"\n{'='*60}")
    print(f"Searching for optimal thresholds (metric: {metric})")
    print('='*60)


    threshold_ranges = {
        'H': np.arange(0.20, 0.50, 0.05),
        'D': np.arange(0.15, 0.40, 0.05),
        'A': np.arange(0.20, 0.50, 0.05)
    }

    best_score = 0
    best_thresholds = {'H': 0.33, 'D': 0.33, 'A': 0.33}


    for th_h in threshold_ranges['H']:
        for th_d in threshold_ranges['D']:
            for th_a in threshold_ranges['A']:
                thresholds = {'H': th_h, 'D': th_d, 'A': th_a}


                y_pred = adjust_decision_threshold(y_proba, thresholds)


                if metric == 'f1_macro':
                    score = f1_score(y_true, y_pred, average='macro', labels=labels)
                elif metric == 'f1_weighted':
                    score = f1_score(y_true, y_pred, average='weighted', labels=labels)
                elif metric == 'accuracy':
                    score = accuracy_score(y_true, y_pred)
                else:
                    score = f1_score(y_true, y_pred, average='macro', labels=labels)

                if score > best_score:
                    best_score = score
                    best_thresholds = thresholds

    print(f"\nBest thresholds found:")
    for label, threshold in best_thresholds.items():
        print(f"  {label}: {threshold:.2f}")
    print(f"Best {metric}: {best_score:.4f}")

    return best_thresholds


def recommend_imbalance_strategy(y_train):
    """
    根据数据分析推荐最合适的不平衡处理策略

    Args:
        y_train: 训练集标签

    Returns:
        推荐策略字典
    """
    analysis = analyze_class_imbalance(y_train)
    imbalance_ratio = analysis['imbalance_ratio']

    print(f"\n{'='*60}")
    print("Recommended Imbalance Handling Strategy")
    print('='*60)

    recommendations = {}

    if imbalance_ratio < 1.5:
        print("\n✓ 类别相对平衡，建议策略：")
        print("  1. 使用默认设置（无需特殊处理）")
        print("  2. 可选：class_weight='balanced' 作为保险")
        recommendations['class_weight'] = None
        recommendations['sample_weight'] = False
        recommendations['threshold_adjustment'] = False

    elif imbalance_ratio < 2.0:
        print("\n[!] 中度不平衡，建议策略：")
        print("  1. 使用 class_weight='balanced'")
        print("  2. 关注平局(D)类别的F1分数")
        recommendations['class_weight'] = 'balanced'
        recommendations['sample_weight'] = False
        recommendations['threshold_adjustment'] = False

    else:
        print("\n[!!] 显著不平衡，建议策略：")
        print("  1. 使用 class_weight='balanced' 或 'sqrt'")
        print("  2. 考虑调整决策阈值（降低少数类阈值）")
        print("  3. 在验证集上优化阈值")
        print("  4. 重点监控少数类的召回率")
        recommendations['class_weight'] = 'balanced'
        recommendations['sample_weight'] = True
        recommendations['threshold_adjustment'] = True

    print(f"\n推荐配置:")
    print(f"  class_weight: {recommendations['class_weight']}")
    print(f"  使用sample_weight: {recommendations['sample_weight']}")
    print(f"  调整决策阈值: {recommendations['threshold_adjustment']}")

    return recommendations


_RESAMPLING_METHODS = {
    'svm_smote', 'smote_nn', 'near_miss', 'random_oversample',
    'smote_tomek', 'none',
}


def apply_resampling(X, y, method='none', random_state=42, **kwargs):
    """
    在训练集上做数据增强 / 重采样以缓解类别不平衡。

    五种方法对应 Atta Mills et al. (2024, J. Big Data) 中描述的
    四种 augmentation 技术，加一个 SMOTE-Tomek 作为更稳健的
    清洗式扩充。仅在训练集上应用，验证 / 测试集不动。

    Args:
        X: pandas.DataFrame 或 ndarray，特征
        y: pandas.Series 或 ndarray，标签
        method: 'svm_smote'         - SVM-SMOTE (沿决策边界生成少数类合成样本)
                'smote_nn'          - SMOTE 的 SMOTE-N 变体 (近邻插值)
                'near_miss'         - NearMiss-3 (欠采样多数类，靠近少数)
                'random_oversample' - 随机过采样 (复制少数类样本)
                'smote_tomek'       - SMOTE + Tomek-link 清洗 (推荐)
                'none'              - 不重采样 (透传)
        random_state: 随机种子
        **kwargs: 传给 imblearn 采样器的额外参数 (如 k_neighbors)

    Returns:
        (X_resampled, y_resampled) 与输入同类型；method='none' 时直接返回输入。

    Notes:
        - 仅当 X 全为数值列时安全。建议先做 fillna 与编码再调用。
        - 该函数惰性 import imbalanced-learn；未安装时仅在被调用时报错。
        - SMOTE-N (smote_nn) 期望全为类别特征；若特征为连续值，回退到
          标准 SMOTE 使其更鲁棒，并打印一行说明。
    """
    if method == 'none' or method is None:
        return X, y

    if method not in _RESAMPLING_METHODS:
        raise ValueError(
            f"Unknown resampling method: {method}. "
            f"Supported: {sorted(_RESAMPLING_METHODS)}"
        )

    try:
        from imblearn.over_sampling import (
            SMOTE, SMOTEN, SVMSMOTE, RandomOverSampler,
        )
        from imblearn.under_sampling import NearMiss
        from imblearn.combine import SMOTETomek
    except ImportError as exc:
        raise ImportError(
            "imbalanced-learn is required for apply_resampling(). "
            "Install with: pip install imbalanced-learn"
        ) from exc

    counts_before = Counter(y)

    if method == 'svm_smote':
        sampler = SVMSMOTE(random_state=random_state, **kwargs)
    elif method == 'smote_nn':
        all_int_or_obj = True
        if hasattr(X, 'dtypes'):
            all_int_or_obj = all(
                np.issubdtype(d, np.integer) or d == object for d in X.dtypes
            )
        if all_int_or_obj:
            sampler = SMOTEN(random_state=random_state, **kwargs)
        else:
            print("[apply_resampling] smote_nn: detected continuous "
                  "features; falling back to standard SMOTE.")
            sampler = SMOTE(random_state=random_state, **kwargs)
    elif method == 'near_miss':
        sampler = NearMiss(version=kwargs.pop('version', 3), **kwargs)
    elif method == 'random_oversample':
        sampler = RandomOverSampler(random_state=random_state, **kwargs)
    elif method == 'smote_tomek':
        sampler = SMOTETomek(random_state=random_state, **kwargs)
    else:
        return X, y

    X_res, y_res = sampler.fit_resample(X, y)
    counts_after = Counter(y_res)

    print(f"\n[apply_resampling] method={method}")
    print(f"  before: {dict(counts_before)} (n={len(y)})")
    print(f"  after : {dict(counts_after)} (n={len(y_res)})")

    return X_res, y_res
