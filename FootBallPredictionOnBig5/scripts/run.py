import sys
import argparse
from pathlib import Path
from typing import Optional


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_data, preprocess_data, clean_data
from src.data.splitter import time_series_split
from src.features.prematch_features import process_prematch_features
from src.features.halftime_features import process_halftime_features
from src.features.feature_selection import select_features
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_class_distribution
from src.evaluation.visualization import generate_model_comparison_report
from src.config.config import MODEL_DISPLAY_NAMES, MERGED_DATA_DIR


def _resolve_input_csv(args) -> Optional[str]:
    if args.input_csv:
        return args.input_csv
    if args.data_path:
        return args.data_path
    if args.league:
        candidate = Path(MERGED_DATA_DIR) / "feature_engineer5" / f"{args.league}.csv"
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(
            f"未找到 league 对应 CSV：{candidate}。请用 --input-csv 指定完整路径，"
            f"或确认 merged_data/feature_engineer5 下存在该文件。"
        )
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="训练入口（支持赛前/中场），可用命令行参数运行",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--task",
        choices=["prematch", "halftime"],
        default="prematch",
        help="训练任务：prematch=赛前，halftime=上半场",
    )


    parser.add_argument(
        "--stage",
        type=str,
        default="E",
        choices=["A", "B", "C", "D", "E"],
        help="使用哪个阶段的特征集合（累积阶段）。当前用于标识输出，不改变特征生成逻辑。",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="baseline",
        choices=["baseline", "selected"],
        help="baseline=不做 Step4 筛选；selected=使用 Step4（特征选择）",
    )

    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="指定输入 CSV 路径（优先级最高）。例如 data/processed/prematch_clean/Germany.csv",
    )
    parser.add_argument(
        "--league",
        type=str,
        default=None,
        help="联赛文件名（不含 .csv），在 merged_data/feature_engineer5 目录下查找。例如 ENG-premier",
    )


    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default=None,
        help="数据文件路径（同 --input-csv，优先级低于 --input-csv）",
    )

    parser.add_argument(
        "--feature-selection",
        "-fs",
        action="store_true",
        default=False,
        help="是否启用特征选择（方差筛选 + 相关性去冗余）。若 --version=selected 会自动启用。",
    )

    parser.add_argument(
        "--models",
        type=str,
        default="random_forest,logistic_regression,xgboost,svm,fnn",
        help="要训练的模型列表，英文逗号分隔",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="输出目录（默认 outputs/）。模型保存到 output-dir/models，报告保存到 output-dir/reports",
    )

    return parser.parse_args()


def _train_and_eval_one(model_type, task_name, X_train, y_train, X_val, y_val, X_test, y_test, model_dir):
    display_name = MODEL_DISPLAY_NAMES.get(model_type, model_type)
    print(f"\n{'='*60}")
    print(f"训练模型: {display_name} ({model_type})")
    print("=" * 60)

    use_scaling = model_type in ["svm", "fnn", "logistic_regression", "naive_bayes"]
    trainer = Trainer(
        model_type=model_type,
        task_name=task_name,
        use_internal_scaling=use_scaling,
        model_dir=model_dir,
    )
    train_metrics, val_metrics = trainer.train(X_train, y_train, X_val, y_val)

    print("\n训练集指标:")
    for k, v in (train_metrics or {}).items():
        print(f"  {k}: {v:.4f}")

    print("\n验证集指标:")
    for k, v in (val_metrics or {}).items():
        print(f"  {k}: {v:.4f}")

    test_metrics, y_pred, y_proba = trainer.evaluate(X_test, y_test)

    print("\n测试集指标:")
    for k, v in (test_metrics or {}).items():
        print(f"  {k}: {v:.4f}")

    model_path = trainer.save_model(filename=f"{task_name}_{model_type}.joblib")
    print(f"\n模型保存至: {model_path}")

    importance = trainer.get_feature_importance()
    if importance:
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 10 重要特征:")
        for feat, imp in sorted_importance[:10]:
            print(f"  {feat}: {imp:.4f}")

    return {
        "trainer": trainer,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "feature_importance": importance,
    }


def main():
    args = parse_args()


    feature_selection = args.feature_selection or (args.version == "selected")

    input_csv = _resolve_input_csv(args)

    output_dir = Path(args.output_dir) if args.output_dir else None
    model_dir = (output_dir / "models") if output_dir else None
    report_dir = (output_dir / "reports") if output_dir else None

    task_label = f"{args.task}_stage{args.stage}_{args.version}"
    print("=" * 60)
    print(f"训练任务: {args.task} | stage={args.stage} | version={args.version}")
    print("=" * 60)
    print("\n[配置参数]")
    print(f"  输入CSV: {input_csv or '(默认配置 RAW_DATA_PATH)'}")
    print(f"  特征选择: {'启用' if feature_selection else '禁用'}")
    if output_dir:
        print(f"  输出目录: {output_dir}")

    print("\n[1] 加载数据...")
    df = load_data(input_csv) if input_csv else load_data()
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

    print("\n[4] 构建特征...")
    if args.task == "prematch":
        X_train, y_train = process_prematch_features(train_df)
        X_val, y_val = process_prematch_features(val_df)
        X_test, y_test = process_prematch_features(test_df)
    else:
        X_train, y_train = process_halftime_features(train_df)
        X_val, y_val = process_halftime_features(val_df)
        X_test, y_test = process_halftime_features(test_df)

    print(f"特征数量: {X_train.shape[1]}")

    if feature_selection:
        print("\n[5] 特征选择...")
        X_train, X_val, X_test, _ = select_features(X_train, y_train, X_val, X_test)
        print(f"筛选后特征数量: {X_train.shape[1]}")

    print("\n[6] 类别分布...")
    print("训练集:", compute_class_distribution(y_train))
    print("验证集:", compute_class_distribution(y_val))
    print("测试集:", compute_class_distribution(y_test))

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    models_results = {}

    print("\n[7] 多模型训练与评估...")
    for i, model_type in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] 训练 {MODEL_DISPLAY_NAMES.get(model_type, model_type)}...")
        try:
            models_results[model_type] = _train_and_eval_one(
                model_type=model_type,
                task_name=task_label,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                model_dir=model_dir,
            )
        except ImportError as e:
            print(f"跳过 {model_type}: {e}")
        except Exception as e:
            print(f"训练 {model_type} 时出错: {e}")

    print(f"\n[{len(models_results) + 1}] 生成模型对比可视化报告...")
    generate_model_comparison_report(
        task_name=task_label,
        models_results=models_results,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        save_dir=report_dir,
    )

    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

