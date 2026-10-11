import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


def evaluate_classification(y_true, y_pred, y_proba):
    """
    全面的分类评估指标

    包括:
    - 整体指标: accuracy, log_loss, f1_macro, f1_weighted
    - 分类别指标: 每个类别的precision, recall, f1
    - ROC-AUC: one-vs-rest多分类AUC
    """
    labels = sorted(['H', 'D', 'A'])


    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'log_loss': log_loss(y_true, y_proba, labels=labels),
        'f1_macro': f1_score(y_true, y_pred, average='macro', labels=labels),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', labels=labels),
        'precision_macro': precision_score(y_true, y_pred, average='macro', labels=labels),
        'recall_macro': recall_score(y_true, y_pred, average='macro', labels=labels)
    }


    precision_per_class = precision_score(y_true, y_pred, average=None, labels=labels)
    recall_per_class = recall_score(y_true, y_pred, average=None, labels=labels)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels)

    # Compute AUC for each class
    for i, label in enumerate(labels):
        metrics[f'precision_{label}'] = precision_per_class[i]
        metrics[f'recall_{label}'] = recall_per_class[i]
        metrics[f'f1_{label}'] = f1_per_class[i]

    # ROC-AUC (one-vs-rest)
    try:

        from sklearn.preprocessing import label_binarize
        # Convert labels to one-hot encoding
        y_true_bin = label_binarize(y_true, classes=labels)


        auc_per_class = []
        for i in range(len(labels)):
            try:
                auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                auc_per_class.append(auc)
                metrics[f'auc_{labels[i]}'] = auc
            except:
                auc_per_class.append(np.nan)
                metrics[f'auc_{labels[i]}'] = np.nan


        valid_aucs = [auc for auc in auc_per_class if not np.isnan(auc)]
        if valid_aucs:
            # Macro-average AUC
            metrics['auc_macro'] = np.mean(valid_aucs)
        else:
            metrics['auc_macro'] = np.nan


        try:
            metrics['auc_weighted'] = roc_auc_score(y_true_bin, y_proba,
                                                    average='weighted', multi_class='ovr')
        except:
            metrics['auc_weighted'] = np.nan

    except Exception as e:
        print(f"Warning: Could not compute AUC metrics: {e}")
        metrics['auc_macro'] = np.nan
        metrics['auc_weighted'] = np.nan
        for label in labels:
            metrics[f'auc_{label}'] = np.nan

    return metrics


def print_evaluation_report(y_true, y_pred, target_names=None):
    if target_names is None:
        target_names = ['H', 'D', 'A']

    print("\n" + "=" * 50)
    print("Classification Report")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=target_names))

    print("\nConfusion Matrix")
    print("-" * 30)
    cm = confusion_matrix(y_true, y_pred)
    print(f"              Predicted")
    print(f"              H    D    A")
    for i, row_name in enumerate(target_names):
        print(f"Actual {row_name}  {cm[i][0]:4d} {cm[i][1]:4d} {cm[i][2]:4d}")


def compute_class_distribution(y):
    if isinstance(y, pd.Series):
        return y.value_counts(normalize=True).to_dict()
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts / len(y)))


def get_roc_curves(y_true, y_proba, labels=None):
    if labels is None:
        labels = sorted(['H', 'D', 'A'])
    """
    计算每个类别的ROC曲线数据

    Returns:
        dict: {label: {'fpr': fpr, 'tpr': tpr, 'auc': auc}}
    """
    from sklearn.preprocessing import label_binarize

    y_true_bin = label_binarize(y_true, classes=labels)

    roc_data = {}
    for i, label in enumerate(labels):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
            roc_data[label] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': auc
            }
        except Exception as e:
            print(f"Warning: Could not compute ROC for class {label}: {e}")
            roc_data[label] = None

    return roc_data
