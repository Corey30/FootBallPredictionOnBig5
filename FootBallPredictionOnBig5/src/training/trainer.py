import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.evaluation.metrics import evaluate_classification, print_evaluation_report
from src.config.config import MODEL_DIR, MODEL_PARAMS, LABEL_ENCODING, LABEL_DECODING


class Trainer:
    """
    统一的模型训练器
    
    整合了模型初始化、训练、评估、保存/加载功能
    支持多种模型类型:
    - random_forest: 随机森林
    - logistic_regression: 逻辑回归
    - xgboost: 极端梯度提升
    - naive_bayes: 朴素贝叶斯
    - svm: 支持向量机 (LinearSVC with calibration)
    - fnn: 前馈神经网络 (MLP)
    """
    
    SKLEARN_MODELS = {
        'random_forest': RandomForestClassifier,
        'logistic_regression': LogisticRegression,
        'naive_bayes': GaussianNB,
        'fnn': MLPClassifier,
    }
    
    def __init__(self, model_type='random_forest', params=None, task_name='match_result', use_internal_scaling=False, class_weight=None):
        self.model_type = model_type
        self.params = params if params is not None else MODEL_PARAMS.get(model_type, {})
        self.task_name = task_name
        self.use_internal_scaling = use_internal_scaling
        self.class_weight = class_weight
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.scaler = None
        self.history = {}
    
    def _init_model(self):
        if self.model_type == 'svm':
            svm_params = self.params.copy()
            if 'probability' in svm_params:
                del svm_params['probability']
            if self.class_weight is not None:
                svm_params['class_weight'] = self.class_weight
            base_svm = LinearSVC(**svm_params)
            self.model = CalibratedClassifierCV(base_svm, cv=3)
        elif self.model_type in self.SKLEARN_MODELS:
            model_class = self.SKLEARN_MODELS[self.model_type]
            model_params = self.params.copy()
            if self.class_weight is not None and self.model_type in ['random_forest', 'logistic_regression']:
                model_params['class_weight'] = self.class_weight
            self.model = model_class(**model_params)
        elif self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier(**self.params)
            except ImportError:
                raise ImportError("XGBoost not installed. Use: pip install xgboost")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. "
                           f"Supported: {list(self.SKLEARN_MODELS.keys()) + ['xgboost']}")
    
    def _encode_labels(self, y):
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        return y_encoded
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self._init_model()
        
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train_values = X_train.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            X_train_values = X_train
        
        if self.use_internal_scaling and self.model_type in ['svm', 'fnn', 'logistic_regression', 'naive_bayes']:
            self.scaler = StandardScaler()
            X_train_values = self.scaler.fit_transform(X_train_values)
        
        y_train_encoded = self._encode_labels(y_train)

        sample_weight = None
        if self.model_type == 'xgboost':
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weight = compute_sample_weight('balanced', y_train_encoded)

        if self.model_type == 'xgboost' and X_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val_values = X_val.values
            else:
                X_val_values = X_val
            if self.scaler is not None:
                X_val_values = self.scaler.transform(X_val_values)
            y_val_encoded = self.label_encoder.transform(y_val)
            self.model.fit(
                X_train_values, y_train_encoded,
                sample_weight=sample_weight,
                eval_set=[(X_val_values, y_val_encoded)],
                verbose=False
            )
        else:
            self.model.fit(X_train_values, y_train_encoded, sample_weight=sample_weight)
        
        train_metrics = self._evaluate_internal(X_train, y_train)
        val_metrics = self._evaluate_internal(X_val, y_val) if X_val is not None else None
        
        self.history = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        return train_metrics, val_metrics
    
    def _evaluate_internal(self, X, y):
        if X is None or y is None:
            return None
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        return evaluate_classification(y, y_pred, y_proba)
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = evaluate_classification(y_test, y_pred, y_proba)
        print_evaluation_report(y_test, y_pred, target_names=['H', 'D', 'A'])
        
        return metrics, y_pred, y_proba
    
    def save_model(self, filename=None):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if filename is None:
            filename = f"{self.task_name}_{self.model_type}.joblib"
        
        save_path = Path(MODEL_DIR) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'params': self.params,
            'scaler': self.scaler
        }
        joblib.dump(model_data, save_path)
        
        return save_path
    
    def load_model(self, filename):
        load_path = Path(MODEL_DIR) / filename
        model_data = joblib.load(load_path)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.params = model_data['params']
        self.scaler = model_data.get('scaler', None)
        
        return self.model
    
    def get_feature_importance(self):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_).mean(axis=0)
            return dict(zip(self.feature_names, importance))
        return None
    
    def get_model_info(self):
        return {
            'model_type': self.model_type,
            'params': self.params,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'uses_scaler': self.scaler is not None,
            'use_internal_scaling': self.use_internal_scaling
        }
