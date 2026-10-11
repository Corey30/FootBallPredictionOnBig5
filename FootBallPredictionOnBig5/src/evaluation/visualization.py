import os
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from src.config.config import REPORT_DIR

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_metrics_comparison(train_metrics, val_metrics, test_metrics, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics_names = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    x = np.arange(len(metrics_names))
    width = 0.25
    
    train_values = [train_metrics.get(m, 0) for m in metrics_names]
    val_values = [val_metrics.get(m, 0) for m in metrics_names]
    test_values = [test_metrics.get(m, 0) for m in metrics_names]
    
    ax1 = axes[0]
    bars1 = ax1.bar(x - width, train_values, width, label='Train', color='#2ecc71')
    bars2 = ax1.bar(x, val_values, width, label='Validation', color='#3498db')
    bars3 = ax1.bar(x + width, test_values, width, label='Test', color='#e74c3c')
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Metrics Comparison Across Datasets')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, rotation=15)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    ax2 = axes[1]
    datasets = ['Train', 'Validation', 'Test']
    accuracies = [train_metrics.get('accuracy', 0), val_metrics.get('accuracy', 0), test_metrics.get('accuracy', 0)]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = ax2.bar(datasets, accuracies, color=colors)
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Comparison')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        ax2.annotate(f'{acc:.2%}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics comparison saved to: {save_path}")
    
    plt.close()
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    from sklearn.metrics import confusion_matrix
    
    if labels is None:
        labels = ['H', 'D', 'A']
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix')
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close()
    return fig


def plot_feature_importance(importance_dict, top_n=15, save_path=None):
    if not importance_dict:
        print("No feature importance to plot")
        return None
    
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [item[0] for item in sorted_items]
    importances = [item[1] for item in sorted_items]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importances, align='center', color='#3498db')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Feature Importance')
    ax.grid(axis='x', alpha=0.3)
    
    for bar, imp in zip(bars, importances):
        ax.annotate(f'{imp:.4f}',
                   xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                   xytext=(3, 0), textcoords="offset points",
                   ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance saved to: {save_path}")
    
    plt.close()
    return fig


def plot_class_distribution(y_train, y_val, y_test, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    datasets = [
        ('Train', y_train),
        ('Validation', y_val),
        ('Test', y_test)
    ]
    
    colors = {'H': '#2ecc71', 'D': '#f39c12', 'A': '#e74c3c'}
    
    for ax, (name, y) in zip(axes, datasets):
        if y is not None:
            counts = pd.Series(y).value_counts()
            labels = counts.index.tolist()
            values = counts.values
            
            wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                               colors=[colors.get(l, '#95a5a6') for l in labels],
                                               startangle=90)
            ax.set_title(f'{name} Set\n(n={len(y)})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Class distribution saved to: {save_path}")
    
    plt.close()
    return fig


def plot_per_class_metrics(y_true, y_pred, labels=None, save_path=None):
    from sklearn.metrics import precision_score, recall_score, f1_score

    if labels is None:
        labels = ['H', 'D', 'A']

    precision = precision_score(y_true, y_pred, labels=labels, average=None)
    recall = recall_score(y_true, y_pred, labels=labels, average=None)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None)

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}\n({"Home Win" if l=="H" else "Draw" if l=="D" else "Away Win"})' for l in labels])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-class metrics saved to: {save_path}")

    plt.close()
    return fig


def plot_roc_curves(y_true, y_proba, labels=None, save_path=None):
    """
    绘制多分类ROC曲线（one-vs-rest）

    Args:
        y_true: 真实标签
        y_proba: 预测概率
        labels: 类别标签列表
        save_path: 保存路径
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.preprocessing import label_binarize

    if labels is None:
        labels = ['H', 'D', 'A']


    y_true_bin = label_binarize(y_true, classes=labels)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    label_names = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}


    for i, (label, color) in enumerate(zip(labels, colors)):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])

            ax.plot(fpr, tpr, color=color, lw=2,
                   label=f'{label_names[label]} (AUC = {auc:.3f})')
        except Exception as e:
            print(f"Warning: Could not plot ROC for class {label}: {e}")


    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curves saved to: {save_path}")

    plt.close()
    return fig


def plot_per_class_auc(y_true, y_proba, labels=None, save_path=None):
    """
    绘制每个类别的AUC柱状图

    Args:
        y_true: 真实标签
        y_proba: 预测概率
        labels: 类别标签列表
        save_path: 保存路径
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    if labels is None:
        labels = ['H', 'D', 'A']


    y_true_bin = label_binarize(y_true, classes=labels)


    aucs = []
    label_names = []
    for i, label in enumerate(labels):
        try:
            auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
            aucs.append(auc)
            label_name = 'Home Win' if label == 'H' else 'Draw' if label == 'D' else 'Away Win'
            label_names.append(f'{label}\n({label_name})')
        except Exception as e:
            print(f"Warning: Could not compute AUC for class {label}: {e}")
            aucs.append(0)
            label_names.append(f'{label}\n(Error)')

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(range(len(labels)), aucs, color=colors)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('AUC Score', fontsize=12)
    ax.set_title('AUC Score by Class', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(label_names)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Random Classifier')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()


    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.annotate(f'{auc:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-class AUC saved to: {save_path}")

    plt.close()
    return fig


def generate_training_report(task_name, model_type, train_metrics, val_metrics, test_metrics,
                             y_train, y_val, y_test, y_pred_test, y_proba_test, feature_importance,
                             save_dir=None):
    if save_dir is None:
        save_dir = Path(REPORT_DIR)
    else:
        save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = save_dir / f'{task_name}_{timestamp}'
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating Visualization Report: {report_dir}")
    print('='*60)

    plot_metrics_comparison(
        train_metrics, val_metrics, test_metrics,
        save_path=report_dir / 'metrics_comparison.png'
    )

    plot_confusion_matrix(
        y_test, y_pred_test,
        save_path=report_dir / 'confusion_matrix.png'
    )

    if feature_importance:
        plot_feature_importance(
            feature_importance,
            save_path=report_dir / 'feature_importance.png'
        )

    plot_class_distribution(
        y_train, y_val, y_test,
        save_path=report_dir / 'class_distribution.png'
    )

    plot_per_class_metrics(
        y_test, y_pred_test,
        save_path=report_dir / 'per_class_metrics.png'
    )


    if y_proba_test is not None:
        plot_roc_curves(
            y_test, y_proba_test,
            save_path=report_dir / 'roc_curves.png'
        )

        plot_per_class_auc(
            y_test, y_proba_test,
            save_path=report_dir / 'per_class_auc.png'
        )

    report_data = {
        'task_name': task_name,
        'model_type': model_type,
        'timestamp': timestamp,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'train_samples': len(y_train) if y_train is not None else 0,
        'val_samples': len(y_val) if y_val is not None else 0,
        'test_samples': len(y_test) if y_test is not None else 0
    }

    with open(report_dir / 'report.json', 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    summary_path = report_dir / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"Training Report: {task_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Dataset Sizes:\n")
        f.write(f"  Train: {report_data['train_samples']}\n")
        f.write(f"  Validation: {report_data['val_samples']}\n")
        f.write(f"  Test: {report_data['test_samples']}\n\n")
        f.write(f"Metrics:\n")
        f.write(f"  Train Accuracy: {train_metrics.get('accuracy', 0):.4f}\n")
        f.write(f"  Val Accuracy: {val_metrics.get('accuracy', 0):.4f}\n")
        f.write(f"  Test Accuracy: {test_metrics.get('accuracy', 0):.4f}\n")
        f.write(f"  Test F1-Macro: {test_metrics.get('f1_macro', 0):.4f}\n")
        f.write(f"  Test Log Loss: {test_metrics.get('log_loss', 0):.4f}\n")
        f.write(f"  Test AUC-Macro: {test_metrics.get('auc_macro', 0):.4f}\n\n")


        f.write(f"Per-Class Metrics (Test Set):\n")
        for label in ['H', 'D', 'A']:
            label_name = 'Home Win' if label == 'H' else 'Draw' if label == 'D' else 'Away Win'
            f.write(f"  {label} ({label_name}):\n")
            f.write(f"    Precision: {test_metrics.get(f'precision_{label}', 0):.4f}\n")
            f.write(f"    Recall: {test_metrics.get(f'recall_{label}', 0):.4f}\n")
            f.write(f"    F1-Score: {test_metrics.get(f'f1_{label}', 0):.4f}\n")
            f.write(f"    AUC: {test_metrics.get(f'auc_{label}', 0):.4f}\n")

        f.write(f"\nOverfitting Analysis:\n")
        gap = train_metrics.get('accuracy', 0) - test_metrics.get('accuracy', 0)
        f.write(f"  Train-Test Gap: {gap:.4f}\n")
        if gap > 0.15:
            f.write(f"  Status: Possible overfitting (gap > 0.15)\n")
        elif gap < 0:
            f.write(f"  Status: Good generalization\n")
        else:
            f.write(f"  Status: Acceptable\n")

    print(f"\nReport generated successfully!")
    print(f"  - Metrics comparison: {report_dir / 'metrics_comparison.png'}")
    print(f"  - Confusion matrix: {report_dir / 'confusion_matrix.png'}")
    print(f"  - Feature importance: {report_dir / 'feature_importance.png'}")
    print(f"  - Class distribution: {report_dir / 'class_distribution.png'}")
    print(f"  - Per-class metrics: {report_dir / 'per_class_metrics.png'}")
    print(f"  - ROC curves: {report_dir / 'roc_curves.png'}")
    print(f"  - Per-class AUC: {report_dir / 'per_class_auc.png'}")
    print(f"  - Summary: {summary_path}")

    return report_dir


def plot_model_comparison_bar(models_results, metrics_names, save_path=None):
    model_names = list(models_results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics_names)
    
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']
    
    for i, model_name in enumerate(model_names):
        test_metrics = models_results[model_name]['test_metrics']
        values = [test_metrics.get(m, 0) for m in metrics_names]
        bars = ax.bar(x + i * width, values, width, label=model_name, color=colors[i % len(colors)])
        
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison (Test Set)')
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metrics_names, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Model comparison bar chart saved to: {save_path}")
    
    plt.close()
    return fig


def plot_model_comparison_radar(models_results, save_path=None):
    metrics_names = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    model_names = list(models_results.keys())
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']
    
    for i, model_name in enumerate(model_names):
        test_metrics = models_results[model_name]['test_metrics']
        values = [test_metrics.get(m, 0) for m in metrics_names]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Model Performance Radar Chart', y=1.08)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Model comparison radar chart saved to: {save_path}")
    
    plt.close()
    return fig


def plot_confusion_matrix_comparison(models_results, y_test, save_path=None):
    from sklearn.metrics import confusion_matrix
    
    model_names = list(models_results.keys())
    n_models = len(model_names)
    labels = ['H', 'D', 'A']
    
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for ax, model_name in zip(axes, model_names):
        y_pred = models_results[model_name]['y_pred']
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=labels, yticklabels=labels,
               xlabel='Predicted', ylabel='Actual',
               title=f'{model_name}')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=12, fontweight='bold')
    
    fig.suptitle('Confusion Matrix Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix comparison saved to: {save_path}")
    
    plt.close()
    return fig


def plot_per_class_comparison(models_results, y_test, save_path=None):
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    labels = ['H', 'D', 'A']
    model_names = list(models_results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_funcs = [
        ('Precision', precision_score),
        ('Recall', recall_score),
        ('F1-Score', f1_score)
    ]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']
    
    for ax, (metric_name, metric_func) in zip(axes, metrics_funcs):
        x = np.arange(len(labels))
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            y_pred = models_results[model_name]['y_pred']
            values = metric_func(y_test, y_pred, labels=labels, average=None)
            bars = ax.bar(x + i * width, values, width, label=model_name, color=colors[i % len(colors)])
            
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Class')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} by Class')
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(['H (Home)', 'D (Draw)', 'A (Away)'])
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-class comparison saved to: {save_path}")
    
    plt.close()
    return fig


def plot_roc_curves_comparison(models_results, y_test, labels=None, save_path=None):
    """
    多模型 ROC 曲线对比（one-vs-rest）

    每个子图对应一个类别（H/D/A），展示所有模型在该类别上的 ROC 曲线。
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.preprocessing import label_binarize
    from src.config.config import MODEL_DISPLAY_NAMES

    if labels is None:
        labels = ['H', 'D', 'A']

    sorted_labels = sorted(labels)
    y_true_bin = label_binarize(y_test, classes=sorted_labels)

    label_names = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
              '#1abc9c', '#34495e', '#e67e22', '#16a085']

    model_names = list(models_results.keys())

    fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 6))
    if len(labels) == 1:
        axes = [axes]

    for idx, (label, ax) in enumerate(zip(labels, axes)):
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

        proba_col = sorted_labels.index(label)

        for i, model_name in enumerate(model_names):
            y_proba = models_results[model_name].get('y_proba')
            if y_proba is None:
                continue
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, proba_col], y_proba[:, proba_col])
                auc = roc_auc_score(y_true_bin[:, proba_col], y_proba[:, proba_col])
                display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
                ax.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                        label=f'{display_name} (AUC={auc:.3f})')
            except Exception:
                continue

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'ROC: {label_names[label]} vs Rest', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle('ROC Curves Comparison (One-vs-Rest)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curves comparison saved to: {save_path}")

    plt.close()
    return fig


def plot_auc_comparison_bar(models_results, labels=None, save_path=None):
    """
    多模型 AUC 对比柱状图

    展示每个模型在每个类别上的 AUC，以及 macro / weighted 平均 AUC。
    """
    from src.config.config import MODEL_DISPLAY_NAMES

    if labels is None:
        labels = ['H', 'D', 'A']

    label_names = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
    model_names = list(models_results.keys())
    display_names = [MODEL_DISPLAY_NAMES.get(n, n) for n in model_names]

    categories = [f'{l} ({label_names[l]})' for l in labels] + ['Macro', 'Weighted']
    n_cats = len(categories)
    n_models = len(model_names)

    x = np.arange(n_cats)
    width = 0.8 / max(n_models, 1)

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
              '#1abc9c', '#34495e', '#e67e22', '#16a085']

    fig, ax = plt.subplots(figsize=(4 + 2 * n_cats, 6))

    for i, model_name in enumerate(model_names):
        test_metrics = models_results[model_name]['test_metrics']
        values = []
        for label in labels:
            values.append(test_metrics.get(f'auc_{label}', 0))
        values.append(test_metrics.get('auc_macro', 0))
        values.append(test_metrics.get('auc_weighted', 0))

        bars = ax.bar(x + i * width, values, width,
                      label=display_names[i], color=colors[i % len(colors)])
        for bar, val in zip(bars, values):
            if val > 0:
                ax.annotate(f'{val:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.set_xlabel('Class / Average', fontsize=11)
    ax.set_ylabel('AUC Score', fontsize=11)
    ax.set_title('ROC-AUC Comparison Across Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(categories, fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"AUC comparison bar chart saved to: {save_path}")

    plt.close()
    return fig


def generate_model_comparison_report(task_name, models_results, y_train, y_val, y_test, save_dir=None):
    if save_dir is None:
        save_dir = Path(REPORT_DIR)
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = save_dir / f'{task_name}_comparison_{timestamp}'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating Model Comparison Report: {report_dir}")
    print('='*60)
    
    metrics_names = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    plot_model_comparison_bar(models_results, metrics_names, save_path=report_dir / 'model_comparison_bar.png')
    
    plot_model_comparison_radar(models_results, save_path=report_dir / 'model_comparison_radar.png')
    
    plot_confusion_matrix_comparison(models_results, y_test, save_path=report_dir / 'confusion_matrix_comparison.png')
    
    plot_per_class_comparison(models_results, y_test, save_path=report_dir / 'per_class_comparison.png')

    has_proba = any(models_results[n].get('y_proba') is not None for n in models_results)
    if has_proba:
        plot_roc_curves_comparison(models_results, y_test, save_path=report_dir / 'roc_curves_comparison.png')
        plot_auc_comparison_bar(models_results, save_path=report_dir / 'auc_comparison.png')

    comparison_data = {
        'task_name': task_name,
        'timestamp': timestamp,
        'models': {}
    }
    
    for model_name, results in models_results.items():
        comparison_data['models'][model_name] = {
            'train_metrics': results['train_metrics'],
            'val_metrics': results['val_metrics'],
            'test_metrics': results['test_metrics']
        }
    
    with open(report_dir / 'comparison_report.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    summary_path = report_dir / 'comparison_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"Model Comparison Report: {task_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write(f"Test Set Metrics Comparison:\n")
        f.write(f"{'-'*60}\n")
        
        metrics_to_show = ['accuracy', 'f1_macro', 'f1_weighted', 'log_loss', 'precision_macro', 'recall_macro', 'auc_macro', 'auc_weighted']
        header = f"{'Metric':<20}"
        model_names = list(models_results.keys())
        for name in model_names:
            header += f"{name:>15}"
        f.write(header + "\n")
        f.write("-" * (20 + 15 * len(model_names)) + "\n")
        
        for metric in metrics_to_show:
            line = f"{metric:<20}"
            for model_name in model_names:
                val = models_results[model_name]['test_metrics'].get(metric, 0)
                line += f"{val:>15.4f}"
            f.write(line + "\n")
        
        f.write(f"\n{'='*60}\n")
        f.write("Per-Class F1-Score:\n")
        f.write(f"{'-'*60}\n")
        
        from sklearn.metrics import f1_score
        labels = ['H', 'D', 'A']
        for label in labels:
            line = f"{label:<20}"
            for model_name in model_names:
                y_pred = models_results[model_name]['y_pred']
                f1 = f1_score(y_test, y_pred, labels=labels, average=None)
                idx = labels.index(label)
                line += f"{f1[idx]:>15.4f}"
            f.write(line + "\n")
        
        f.write(f"\n{'='*60}\n")
        f.write("Per-Class AUC (One-vs-Rest):\n")
        f.write(f"{'-'*60}\n")

        for label in labels:
            line = f"auc_{label:<16}"
            for model_name in model_names:
                val = models_results[model_name]['test_metrics'].get(f'auc_{label}', 0)
                line += f"{val:>15.4f}"
            f.write(line + "\n")

        f.write(f"\n{'='*60}\n")
        f.write("Best Model per Metric:\n")
        f.write(f"{'-'*60}\n")
        
        for metric in ['accuracy', 'f1_macro', 'auc_macro']:
            best_model = max(model_names, key=lambda x: models_results[x]['test_metrics'].get(metric, 0))
            best_val = models_results[best_model]['test_metrics'].get(metric, 0)
            f.write(f"  {metric}: {best_model} ({best_val:.4f})\n")
    
    print(f"\nComparison report generated successfully!")
    print(f"  - Model comparison bar: {report_dir / 'model_comparison_bar.png'}")
    print(f"  - Model comparison radar: {report_dir / 'model_comparison_radar.png'}")
    print(f"  - Confusion matrix comparison: {report_dir / 'confusion_matrix_comparison.png'}")
    print(f"  - Per-class comparison: {report_dir / 'per_class_comparison.png'}")
    if has_proba:
        print(f"  - ROC curves comparison: {report_dir / 'roc_curves_comparison.png'}")
        print(f"  - AUC comparison: {report_dir / 'auc_comparison.png'}")
    print(f"  - Summary: {summary_path}")
    
    return report_dir
