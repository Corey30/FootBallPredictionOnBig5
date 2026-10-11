"""
模型集成模块

支持Voting、Stacking集成模型 和 Vanilla RNN基础模型
"""

import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.config.config import RANDOM_SEED, MODEL_PARAMS


class VotingEnsemble:
    """投票集成模型"""

    def __init__(self, models_dict, voting='soft', weights=None):
        self.models_dict = models_dict
        self.voting = voting
        self.weights = weights
        self.label_encoder = None
        self.ensemble = None

    def train(self, X_train, y_train):
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        estimators = [(name, model) for name, model in self.models_dict.items()]

        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting=self.voting,
            weights=self.weights,
            n_jobs=-1
        )
        self.ensemble.fit(X_train, y_train_encoded)
        return self

    def predict(self, X):
        y_pred_encoded = self.ensemble.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X):
        return self.ensemble.predict_proba(X)


class StackingEnsemble:
    """堆叠集成模型"""

    def __init__(self, base_models_dict, meta_model=None):
        self.base_models_dict = base_models_dict
        self.meta_model = meta_model or LogisticRegression(
            C=1.0, random_state=RANDOM_SEED, max_iter=1000
        )
        self.label_encoder = None
        self.ensemble = None

    def train(self, X_train, y_train, cv=5):
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        estimators = [(name, model) for name, model in self.base_models_dict.items()]

        self.ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=self.meta_model,
            cv=cv,
            n_jobs=-1,
            passthrough=False
        )
        self.ensemble.fit(X_train, y_train_encoded)
        return self

    def predict(self, X):
        y_pred_encoded = self.ensemble.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X):
        return self.ensemble.predict_proba(X)


class VanillaRNNClassifier:
    """Vanilla RNN分类器"""

    def __init__(self, rnn_units=64, dense_units=32, dropout=0.3,
                 learning_rate=0.001, epochs=50, batch_size=32,
                 class_weight=None):
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.n_features = None

    def _build_model(self, n_features, n_classes):
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            raise ImportError("TensorFlow is required for RNN. Install: pip install tensorflow")

        model = keras.Sequential([
            layers.Input(shape=(1, n_features)),
            layers.SimpleRNN(self.rnn_units, return_sequences=False, activation='tanh'),
            layers.Dropout(self.dropout),
            layers.Dense(self.dense_units, activation='relu'),
            layers.Dropout(self.dropout),
            layers.Dense(n_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=0):
        try:
            from tensorflow import keras
        except ImportError:
            raise ImportError("TensorFlow is required for RNN. Install: pip install tensorflow")

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        self.n_features = X_train_scaled.shape[1]
        X_train_reshaped = X_train_scaled.reshape(-1, 1, self.n_features)

        n_classes = len(self.label_encoder.classes_)
        self.model = self._build_model(self.n_features, n_classes)

        cw = self.class_weight
        if cw is None:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train_encoded)
            weights = compute_class_weight('balanced', classes=classes, y=y_train_encoded)
            cw = dict(enumerate(weights))

        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_reshaped = X_val_scaled.reshape(-1, 1, self.n_features)
            y_val_encoded = self.label_encoder.transform(y_val)
            validation_data = (X_val_reshaped, y_val_encoded)

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=15,
            restore_best_weights=True
        )

        self.model.fit(
            X_train_reshaped,
            y_train_encoded,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            class_weight=cw,
            callbacks=[early_stopping],
            verbose=verbose
        )
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_reshaped = X_scaled.reshape(-1, 1, self.n_features)
        y_pred_proba = self.model.predict(X_reshaped, verbose=0)
        y_pred_encoded = np.argmax(y_pred_proba, axis=1)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        X_reshaped = X_scaled.reshape(-1, 1, self.n_features)
        return self.model.predict(X_reshaped, verbose=0)


def create_voting_ensemble(model_types=None, voting='soft', weights=None):
    if model_types is None:
        model_types = MODEL_PARAMS['voting_ensemble']['base_models']

    models_dict = {}
    for model_type in model_types:
        params = MODEL_PARAMS.get(model_type, {})
        if model_type == 'random_forest':
            model = RandomForestClassifier(**params)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(**params)
        elif model_type == 'xgboost':
            import xgboost as xgb
            xgb_params = {k: v for k, v in params.items() if k != 'early_stopping_rounds'}
            model = xgb.XGBClassifier(**xgb_params)
        elif model_type == 'svm':
            base_svm = LinearSVC(**params)
            model = CalibratedClassifierCV(base_svm, cv=3)
        elif model_type == 'fnn':
            model = MLPClassifier(**params)
        else:
            continue
        models_dict[model_type] = model

    return VotingEnsemble(models_dict, voting=voting, weights=weights)


def create_stacking_ensemble(base_model_types=None, meta_model_type='logistic_regression'):
    if base_model_types is None:
        base_model_types = MODEL_PARAMS['stacking_ensemble']['base_models']

    base_models_dict = {}
    for model_type in base_model_types:
        params = MODEL_PARAMS.get(model_type, {})
        if model_type == 'random_forest':
            model = RandomForestClassifier(**params)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(**params)
        elif model_type == 'xgboost':
            import xgboost as xgb
            xgb_params = {k: v for k, v in params.items() if k != 'early_stopping_rounds'}
            model = xgb.XGBClassifier(**xgb_params)
        elif model_type == 'svm':
            base_svm = LinearSVC(**params)
            model = CalibratedClassifierCV(base_svm, cv=3)
        elif model_type == 'fnn':
            model = MLPClassifier(**params)
        else:
            continue
        base_models_dict[model_type] = model

    meta_params = MODEL_PARAMS.get(meta_model_type, {})
    if meta_model_type == 'logistic_regression':
        meta_model = LogisticRegression(**meta_params)
    elif meta_model_type == 'random_forest':
        meta_model = RandomForestClassifier(**meta_params)
    else:
        meta_model = LogisticRegression(C=1.0, random_state=RANDOM_SEED, max_iter=1000)

    return StackingEnsemble(base_models_dict, meta_model=meta_model)
