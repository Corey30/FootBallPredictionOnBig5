import sys
import argparse
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_data, preprocess_data, clean_data
from src.data.splitter import time_series_split
from src.features.prematch_features import process_prematch_features
from src.features.feature_selection import select_features
from src.training.trainer import Trainer
from src.training.ensemble import (
    VotingEnsemble, StackingEnsemble,
    VanillaRNNClassifier,
)
from src.evaluation.metrics import compute_class_distribution, evaluate_classification
from src.evaluation.visualization import generate_model_comparison_report
from src.config.config import MODEL_DISPLAY_NAMES, MODEL_PARAMS


BASE_MODELS = ['random_forest', 'logistic_regression', 'xgboost', 'svm', 'fnn', 'vanilla_rnn']
ENSEMBLE_MODELS = ['voting_ensemble', 'stacking_ensemble']


def parse_args():
    parser = argparse.ArgumentParser(
        description='赛前预测模型训练',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--feature-selection', '-fs',
        action='store_true',
        default=True,
        help='是否启用特征选择'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='输出目录路径'
    )
    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default=None,
        help='数据文件路径'
    )
    return parser.parse_args()


def _build_base_model(model_type):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.neural_network import MLPClassifier

    params = MODEL_PARAMS.get(model_type, {})
    if model_type == 'random_forest':
        return RandomForestClassifier(**params)
    elif model_type == 'logistic_regression':
        return LogisticRegression(**params)
    elif model_type == 'xgboost':
        import xgboost as xgb
        xgb_params = {k: v for k, v in params.items() if k != 'early_stopping_rounds'}
        return xgb.XGBClassifier(**xgb_params)
    elif model_type == 'svm':
        base_svm = LinearSVC(**params)
        return CalibratedClassifierCV(base_svm, cv=3)
    elif model_type == 'fnn':
        return MLPClassifier(**params)
    else:
        raise ValueError(f"Unknown base model type: {model_type}")


def train_base_model(model_type, X_train, y_train, X_val, y_val, X_test, y_test):
    display_name = MODEL_DISPLAY_NAMES.get(model_type, model_type)
    print(f"\n{'='*60}")
    print(f"训练模型: {display_name} ({model_type})")
    print('='*60)

    if model_type == 'vanilla_rnn':
        return _train_rnn(X_train, y_train, X_val, y_val, X_test, y_test)

    use_scaling = model_type in ['svm', 'fnn', 'logistic_regression', 'naive_bayes']
    trainer = Trainer(model_type=model_type, task_name='prematch', use_internal_scaling=use_scaling)
    train_metrics, val_metrics = trainer.train(X_train, y_train, X_val, y_val)

    print("\n训练集指标:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\n验证集指标:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    test_metrics, y_pred, y_proba = trainer.evaluate(X_test, y_test)
    print("\n测试集指标:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    model_path = trainer.save_model()
    print(f"\n模型保存至: {model_path}")

    importance = trainer.get_feature_importance()
    if importance:
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 10 重要特征:")
        for feat, imp in sorted_importance[:10]:
            print(f"  {feat}: {imp:.4f}")

    return {
        'trainer': trainer,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'feature_importance': importance
    }


def _train_rnn(X_train, y_train, X_val, y_val, X_test, y_test):
    cfg = MODEL_PARAMS['vanilla_rnn']
    model = VanillaRNNClassifier(
        rnn_units=cfg['rnn_units'],
        dense_units=cfg['dense_units'],
        dropout=cfg['dropout'],
        learning_rate=cfg['learning_rate'],
        epochs=cfg['epochs'],
        batch_size=cfg['batch_size'],
    )
    model.train(X_train, y_train, X_val=X_val, y_val=y_val, verbose=1)

    train_pred = model.predict(X_train)
    train_proba = model.predict_proba(X_train)
    train_metrics = evaluate_classification(y_train, train_pred, train_proba)

    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)
    val_metrics = evaluate_classification(y_val, val_pred, val_proba)

    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)
    test_metrics = evaluate_classification(y_test, test_pred, test_proba)

    print("\n训练集指标:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\n验证集指标:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\n测试集指标:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    return {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'y_pred': test_pred,
        'y_proba': test_proba,
        'feature_importance': None,
    }


def train_ensemble_model(model_type, X_train, y_train, X_val, y_val, X_test, y_test):
    display_name = MODEL_DISPLAY_NAMES.get(model_type, model_type)
    print(f"\n{'='*60}")
    print(f"训练模型: {display_name} ({model_type})")
    print('='*60)

    if model_type == 'voting_ensemble':
        result = _train_voting(X_train, y_train, X_val, y_val, X_test, y_test)
    elif model_type == 'stacking_ensemble':
        result = _train_stacking(X_train, y_train, X_val, y_val, X_test, y_test)
    else:
        raise ValueError(f"Unknown ensemble type: {model_type}")

    return result


def _train_voting(X_train, y_train, X_val, y_val, X_test, y_test):
    cfg = MODEL_PARAMS['voting_ensemble']
    models_dict = {}
    for name in cfg['base_models']:
        models_dict[name] = _build_base_model(name)

    ensemble = VotingEnsemble(
        models_dict=models_dict,
        voting=cfg['voting'],
        weights=cfg['weights'],
    )
    ensemble.train(X_train, y_train)

    train_pred = ensemble.predict(X_train)
    train_proba = ensemble.predict_proba(X_train)
    train_metrics = evaluate_classification(y_train, train_pred, train_proba)

    val_pred = ensemble.predict(X_val)
    val_proba = ensemble.predict_proba(X_val)
    val_metrics = evaluate_classification(y_val, val_pred, val_proba)

    test_pred = ensemble.predict(X_test)
    test_proba = ensemble.predict_proba(X_test)
    test_metrics = evaluate_classification(y_test, test_pred, test_proba)

    print("\n测试集指标:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    return {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'y_pred': test_pred,
        'y_proba': test_proba,
        'feature_importance': None,
    }


def _train_stacking(X_train, y_train, X_val, y_val, X_test, y_test):
    cfg = MODEL_PARAMS['stacking_ensemble']
    base_models_dict = {}
    for name in cfg['base_models']:
        base_models_dict[name] = _build_base_model(name)

    meta_model = _build_base_model(cfg['meta_model'])

    ensemble = StackingEnsemble(
        base_models_dict=base_models_dict,
        meta_model=meta_model,
    )
    ensemble.train(X_train, y_train, cv=cfg['cv'])

    train_pred = ensemble.predict(X_train)
    train_proba = ensemble.predict_proba(X_train)
    train_metrics = evaluate_classification(y_train, train_pred, train_proba)

    val_pred = ensemble.predict(X_val)
    val_proba = ensemble.predict_proba(X_val)
    val_metrics = evaluate_classification(y_val, val_pred, val_proba)

    test_pred = ensemble.predict(X_test)
    test_proba = ensemble.predict_proba(X_test)
    test_metrics = evaluate_classification(y_test, test_pred, test_proba)

    print("\n测试集指标:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    return {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'y_pred': test_pred,
        'y_proba': test_proba,
        'feature_importance': None,
    }


def print_comparison_summary(models_results):
    model_names = list(models_results.keys())

    print("\n" + "=" * 80)
    print("模型对比汇总")
    print("=" * 80)

    metrics_to_compare = ['accuracy', 'f1_macro', 'f1_weighted', 'log_loss', 'precision_macro', 'recall_macro', 'auc_macro', 'auc_weighted']

    header = f"{'指标':<18}"
    for name in model_names:
        display_name = MODEL_DISPLAY_NAMES.get(name, name)
        header += f"{display_name[:12]:>14}"
    print(f"\n{header}")
    print("-" * (18 + 14 * len(model_names)))

    for metric in metrics_to_compare:
        line = f"{metric:<18}"
        for name in model_names:
            val = models_results[name]['test_metrics'].get(metric, 0)
            line += f"{val:>14.4f}"
        print(line)

    if not model_names:
        print("\n所有模型训练失败，无对比结果")
        return

    best_model = max(model_names, key=lambda x: models_results[x]['test_metrics'].get('f1_macro', 0))
    best_f1 = models_results[best_model]['test_metrics'].get('f1_macro', 0)
    best_display = MODEL_DISPLAY_NAMES.get(best_model, best_model)
    print(f"\n最佳模型 (F1-macro): {best_display} = {best_f1:.4f}")


def main():
    args = parse_args()

    print("=" * 60)
    print("赛前预测模型训练 - 多模型对比")
    print("=" * 60)

    print("\n[配置参数]")
    print(f"  特征选择: {'启用' if args.feature_selection else '禁用'}")

    print("\n[1] 加载数据...")
    if args.data_path:
        df = load_data(args.data_path)
    else:
        df = load_data()
    print(f"原始数据形状: {df.shape}")

    print("\n[2] 数据预处理...")
    df = preprocess_data(df)
    df = clean_data(df)
    print(f"处理后数据形状: {df.shape}")

    print("\n[3] 时间序列划分...")
    train_df, val_df, test_df = time_series_split(df)
    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")
    print(f"测试集: {len(test_df)} 条")

    print("\n[4] 构建赛前特征...")
    X_train, y_train = process_prematch_features(train_df)
    X_val, y_val = process_prematch_features(val_df)
    X_test, y_test = process_prematch_features(test_df)

    print(f"特征数量: {X_train.shape[1]}")

    if args.feature_selection:
        print("\n[5] 特征选择...")
        X_train, X_val, X_test, fs_log = select_features(
            X_train, y_train, X_val, X_test
        )
        print(f"筛选后特征数量: {X_train.shape[1]}")

    print("\n[6] 类别分布...")
    print("训练集:", compute_class_distribution(y_train))
    print("验证集:", compute_class_distribution(y_val))
    print("测试集:", compute_class_distribution(y_test))

    models_results = {}
    all_model_types = BASE_MODELS + ENSEMBLE_MODELS

    for i, model_type in enumerate(all_model_types, 1):
        display_name = MODEL_DISPLAY_NAMES.get(model_type, model_type)
        print(f"\n[{i}/{len(all_model_types)}] 训练 {display_name}...")
        try:
            if model_type in BASE_MODELS:
                models_results[model_type] = train_base_model(
                    model_type, X_train, y_train, X_val, y_val, X_test, y_test
                )
            else:
                models_results[model_type] = train_ensemble_model(
                    model_type, X_train, y_train, X_val, y_val, X_test, y_test
                )
        except ImportError as e:
            print(f"跳过 {model_type}: {e}")
        except Exception as e:
            print(f"训练 {model_type} 时出错: {e}")

    print_comparison_summary(models_results)

    print(f"\n[{len(models_results)+1}] 生成模型对比可视化报告...")
    generate_model_comparison_report(
        task_name='prematch',
        models_results=models_results,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test
    )

    print("\n" + "=" * 60)
    print("赛前预测模型训练完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
