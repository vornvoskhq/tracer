"""
Machine Learning models for stock signal prediction
Implements the logistic regression approach from convo.txt
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from .config import model_config

# Import model interpretability (SHAP alternative for Python 3.12)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    from .model_interpretability import ModelInterpreter, SHAPCompatibilityWrapper

# Import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)

class EqualWeightSignalClassifier(BaseEstimator, ClassifierMixin):
    """
    A simple no-training classifier that computes probabilities based on the
    fraction of features that are positively activated.
    - mode='buy': probability is high when most features are activated
    - mode='sell': probability is high when few features are activated
    Optionally uses the sign of correlation with y to orient features so
    that "positive" means supportive of the positive class.
    If fixed_signs is provided, those are used instead of data-derived signs.
    """
    def __init__(self, mode: str = 'buy', correlation_signs: bool = True,
                 activation_threshold: float = 0.0, temperature: float = 1.0,
                 calibrate_prior: bool = True, fixed_signs: Optional[np.ndarray] = None,
                 aggregation: str = 'fraction'):
        self.mode = mode
        self.correlation_signs = correlation_signs
        self.activation_threshold = activation_threshold
        self.temperature = max(1e-6, float(temperature))
        self.calibrate_prior = calibrate_prior
        # Fitted attributes
        self.n_features_in_ = None
        self.signs_ = None
        self.coef_ = None  # shape (1, n_features)
        self.intercept_ = 0.0
        self.fixed_signs = fixed_signs
        self.aggregation = aggregation
    
    def _compute_corr_signs(self, X: np.ndarray, y: Optional[np.ndarray]) -> np.ndarray:
        if (y is None) or (len(np.unique(y)) < 2) or (not self.correlation_signs):
            return np.ones(X.shape[1], dtype=float)
        signs = np.ones(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            try:
                c = np.corrcoef(X[:, j], y)[0, 1]
                if np.isnan(c):
                    c = 0.0
            except Exception:
                c = 0.0
            signs[j] = 1.0 if c >= 0 else -1.0
        return signs
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        if self.fixed_signs is not None and len(self.fixed_signs) == self.n_features_in_:
            self.signs_ = self.fixed_signs.astype(float)
        else:
            self.signs_ = self._compute_corr_signs(X, y)
        # Equal magnitude weights with sign direction for interpretability
        self.coef_ = (self.signs_ / self.n_features_in_).reshape(1, -1)
        # Prior calibration (maps the fraction-based score to be centered)
        if self.calibrate_prior and y is not None and len(y) > 0:
            p = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))
            # store as logit prior shift around 0.5 baseline
            self.intercept_ = np.log(p / (1.0 - p))
        else:
            self.intercept_ = 0.0
        return self
    
    def _aggregate_activation(self, X: np.ndarray) -> np.ndarray:
        X_signed = X * self.signs_
        # Binary activation based on threshold
        activated = (X_signed > self.activation_threshold).astype(float)
        if self.aggregation == 'mean':
            # Use signed mean (allows negative contributions if signs_ = -1)
            return X_signed.mean(axis=1)
        if self.aggregation in ('all', 'all_activated', 'conjunction'):
            # 1 only if all features are activated
            return (activated.sum(axis=1) == activated.shape[1]).astype(float)
        if self.aggregation == 'any_deactivated':
            # 1 if any feature is deactivated
            return (activated.sum(axis=1) < activated.shape[1]).astype(float)
        # default: fraction of features above threshold
        return activated.mean(axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        agg = self._aggregate_activation(X)
        # For 'mean' aggregation, center around 0, else around 0.5
        if self.aggregation == 'mean':
            raw = agg
            if self.mode == 'sell':
                raw = -raw
            z = raw / self.temperature + self.intercept_
        elif self.aggregation == 'any_deactivated':
            # agg is 1 if any feature is deactivated, else 0
            # BUY should favor all-activated (agg==0) -> low p; SELL should favor any-deactivated (agg==1) -> high p
            raw = agg if self.mode == 'sell' else (1.0 - agg)
            z = (raw - 0.5) / self.temperature + self.intercept_
        else:
            raw = agg if self.mode == 'buy' else (1.0 - agg)
            z = (raw - 0.5) / self.temperature + self.intercept_
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - p, p])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)

class EnsembleModel:
    """Simple averaging ensemble of configured base classifiers.
    Each base model must implement predict_proba and be trained on the same scaler/features.
    """
    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = None
        if weights and len(weights) == len(models):
            s = float(sum(weights)) or 1.0
            self.weights = [w / s for w in weights]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise ValueError("No base models in ensemble")
        # Sum probabilities
        probs = None
        for i, m in enumerate(self.models):
            p = m.predict_proba(X)
            if probs is None:
                probs = p * (self.weights[i] if self.weights else 1.0)
            else:
                probs += p * (self.weights[i] if self.weights else 1.0)
        if self.weights is None:
            probs = probs / float(len(self.models))
        return probs


class StockSignalPredictor:
    """
    ML model for predicting stock buy/sell signals
    Implements dual model approach from convo.txt
    """
    
    def __init__(self, config=None):
        self.config = config or model_config
        self.buy_model = None
        self.sell_model = None
        self.scaler = None
        self.feature_names = None
        self.buy_explainer = None
        self.sell_explainer = None
    
    def _create_model(self, model_type: str, signal_type: str = 'buy'):
        """Create a model based on type and configuration"""
        if model_type.lower() == 'equal_weight':
            # DEBUG: Print config object details
            logger.info(f"🔍 DEBUG: Config object type: {type(self.config)}")
            logger.info(f"🔍 DEBUG: Config attributes: {dir(self.config)}")
            if hasattr(self.config, '__dict__'):
                config_dict = self.config.__dict__
                logger.info(f"🔍 DEBUG: Config dict keys: {list(config_dict.keys())}")
                ew_keys = [k for k in config_dict.keys() if k.startswith('ew_')]
                logger.info(f"🔍 DEBUG: EW keys in config: {ew_keys}")
                for k in ew_keys:
                    logger.info(f"🔍 DEBUG: {k} = {getattr(self.config, k, 'NOT_FOUND')}")
            
            # Use ExperimentConfig values instead of global model_config
            corr = getattr(self.config, 'ew_correlation_signs', False)
            cal = getattr(self.config, 'ew_calibrate_prior', False)
            thr = float(getattr(self.config, 'ew_activation_threshold', 0.0))
            temp = float(getattr(self.config, 'ew_temperature', 1.0))
            buy_aggr = getattr(self.config, 'ew_buy_aggregation', 'fraction')
            sell_aggr = getattr(self.config, 'ew_sell_aggregation', 'fraction')
            
            logger.info(f"🔍 DEBUG: Retrieved values - corr={corr}, cal={cal}, thr={thr}, temp={temp}, buy_aggr={buy_aggr}, sell_aggr={sell_aggr}")
            
            agg = sell_aggr if signal_type == 'sell' else buy_aggr
            logger.info(f"Initializing EqualWeightSignalClassifier(mode={signal_type}, agg={agg}, thr={thr}, temp={temp}, corr_signs={corr}, calibrate={cal})")
            return EqualWeightSignalClassifier(mode=signal_type, correlation_signs=corr,
                                              activation_threshold=thr, temperature=temp,
                                              calibrate_prior=cal, aggregation=agg)
        if model_type.lower() == 'xgboost':
            if not XGBOOST_AVAILABLE:
                logger.warning("XGBoost not available, falling back to LogisticRegression")
                model_type = 'logistic'
            else:
                return xgb.XGBClassifier(
                    n_estimators=getattr(self.config, 'n_estimators', 100),
                    max_depth=getattr(self.config, 'max_depth', 6),
                    learning_rate=getattr(self.config, 'learning_rate', 0.1),
                    subsample=getattr(self.config, 'subsample', 0.8),
                    colsample_bytree=getattr(self.config, 'colsample_bytree', 0.8),
                    random_state=getattr(self.config, 'random_state', 42),
                    eval_metric='logloss',
                    verbosity=0
                )
        
        # Default to LogisticRegression
        return LogisticRegression(
            penalty=getattr(self.config, 'penalty', 'l1'),
            solver=getattr(self.config, 'solver', 'liblinear'),
            class_weight=getattr(self.config, 'class_weight', 'balanced'),
            max_iter=getattr(self.config, 'max_iter', 1000),
            random_state=getattr(self.config, 'random_state', 42)
        )
        
    def prepare_data(self, df: pd.DataFrame, feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training/prediction
        
        Args:
            df: DataFrame with features and targets
            feature_columns: List of feature column names
            
        Returns:
            X: Feature matrix
            y_buy: Buy target vector
            y_sell: Sell target vector
        """
        # Remove rows with missing values
        df_clean = df[feature_columns + ['target_buy', 'target_sell']].dropna()
        
        X = df_clean[feature_columns].values
        y_buy = df_clean['target_buy'].values
        y_sell = df_clean['target_sell'].values
        
        self.feature_names = feature_columns
        
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Buy signals: {y_buy.sum()} ({y_buy.mean():.3f})")
        logger.info(f"Sell signals: {y_sell.sum()} ({y_sell.mean():.3f})")
        
        return X, y_buy, y_sell
    
    def train_models(self, X: np.ndarray, y_buy: np.ndarray, y_sell: np.ndarray) -> Dict[str, Any]:
        """
        Train both buy and sell models (supports ensemble if base_models provided)
        
        Args:
            X: Feature matrix
            y_buy: Buy target vector
            y_sell: Sell target vector
            
        Returns:
            Dictionary with training results
        """
        # Split data chronologically (no shuffling for time series)
        split_idx = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_buy_train, y_buy_test = y_buy[:split_idx], y_buy[split_idx:]
        y_sell_train, y_sell_test = y_sell[:split_idx], y_sell[split_idx:]
        
        # Scale features (skip scaling for equal_weight to preserve binary feature semantics)
        model_type = getattr(self.config, 'model_type', 'logistic')
        if str(model_type).lower() == 'equal_weight':
            self.scaler = None
            X_train_scaled = X_train
            X_test_scaled = X_test
        else:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        
        # Optional simple tuning for logistic (Phase 4)
        from .config import model_config as mc
        if getattr(mc, 'tuning', 'none') == 'simple' and getattr(self.config, 'model_type', 'logistic').lower() == 'logistic':
            try:
                Cs = [0.1, 0.5, 1.0, 2.0]
                best_auc = -1
                best_C = getattr(mc, 'C', 1.0)
                for C in Cs:
                    tmp = LogisticRegression(
                        penalty=getattr(self.config, 'penalty', 'l1'),
                        solver=getattr(self.config, 'solver', 'liblinear'),
                        class_weight=getattr(self.config, 'class_weight', 'balanced'),
                        max_iter=getattr(self.config, 'max_iter', 1000),
                        random_state=getattr(self.config, 'random_state', 42),
                        C=C
                    )
                    if len(np.unique(y_buy_train)) > 1:
                        tmp.fit(X_train_scaled, y_buy_train)
                        auc = roc_auc_score(y_buy_test, tmp.predict_proba(X_test)[:,1]) if len(np.unique(y_buy_test)) > 1 else 0.5
                        if auc > best_auc:
                            best_auc, best_C = auc, C
                if best_auc >= 0:
                    mc.C = best_C
            except Exception:
                pass
        
        # Ensemble or single-model path
        base_models = getattr(self.config, 'base_models', None)
        weights = getattr(self.config, 'weights', None)
        model_type = getattr(self.config, 'model_type', 'logistic')
        
        if base_models and isinstance(base_models, (list, tuple)):
            logger.info(f"Training ensemble with base models: {base_models}")
            buy_models = []
            sell_models = []
            from sklearn.dummy import DummyClassifier
            for mname in base_models:
                mtype = str(mname)
                # Train buy
                if len(np.unique(y_buy_train)) < 2:
                    bmodel = DummyClassifier(strategy='constant', constant=0)
                else:
                    bmodel = self._create_model(mtype, 'buy')
                bmodel.fit(X_train_scaled, y_buy_train)
                buy_models.append(bmodel)
                # Train sell
                if len(np.unique(y_sell_train)) < 2:
                    smodel = DummyClassifier(strategy='constant', constant=0)
                else:
                    smodel = self._create_model(mtype, 'sell')
                smodel.fit(X_train_scaled, y_sell_train)
                sell_models.append(smodel)
            # Wrap in ensemble
            self.buy_model = EnsembleModel(buy_models, weights)
            self.sell_model = EnsembleModel(sell_models, weights)
        else:
            # Single model path
            logger.info(f"Training {model_type} buy signal model...")
            # Train buy model
            if len(np.unique(y_buy_train)) < 2:
                logger.warning("Insufficient buy signal diversity - using dummy classifier")
                from sklearn.dummy import DummyClassifier
                self.buy_model = DummyClassifier(strategy='constant', constant=0)
            else:
                # For logistic, apply tuned C if present
                if model_type.lower() == 'logistic':
                    try:
                        from .config import model_config as mc
                        self.buy_model = LogisticRegression(
                            penalty=getattr(self.config, 'penalty', 'l1'),
                            solver=getattr(self.config, 'solver', 'liblinear'),
                            class_weight=getattr(self.config, 'class_weight', 'balanced'),
                            max_iter=getattr(self.config, 'max_iter', 1000),
                            random_state=getattr(self.config, 'random_state', 42),
                            C=getattr(mc, 'C', 1.0)
                        )
                    except Exception:
                        self.buy_model = self._create_model(model_type, 'buy')
                else:
                    self.buy_model = self._create_model(model_type, 'buy')
            self.buy_model.fit(X_train_scaled, y_buy_train)
            
            # Train sell model
            logger.info(f"Training {model_type} sell signal model...")
            if len(np.unique(y_sell_train)) < 2:
                logger.warning("Insufficient sell signal diversity - using dummy classifier")
                from sklearn.dummy import DummyClassifier
                self.sell_model = DummyClassifier(strategy='constant', constant=0)
            else:
                if model_type.lower() == 'logistic':
                    try:
                        from .config import model_config as mc
                        self.sell_model = LogisticRegression(
                            penalty=getattr(self.config, 'penalty', 'l1'),
                            solver=getattr(self.config, 'solver', 'liblinear'),
                            class_weight=getattr(self.config, 'class_weight', 'balanced'),
                            max_iter=getattr(self.config, 'max_iter', 1000),
                            random_state=getattr(self.config, 'random_state', 42),
                            C=getattr(mc, 'C', 1.0)
                        )
                    except Exception:
                        self.sell_model = self._create_model(model_type, 'sell')
                else:
                    self.sell_model = self._create_model(model_type, 'sell')
            self.sell_model.fit(X_train_scaled, y_sell_train)
        
        # Evaluate models
        results = self._evaluate_models(X_train_scaled, X_test_scaled, 
                                      y_buy_train, y_buy_test, 
                                      y_sell_train, y_sell_test)
        
        # Initialize SHAP explainers
        self._initialize_shap_explainers(X_train_scaled)
        
        return results
    
    def _evaluate_models(self, X_train: np.ndarray, X_test: np.ndarray,
                        y_buy_train: np.ndarray, y_buy_test: np.ndarray,
                        y_sell_train: np.ndarray, y_sell_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance"""
        results = {}

        def pos_proba(model, proba_matrix):
            try:
                classes = getattr(model, 'classes_', None)
                if classes is not None:
                    classes = list(classes)
                    if proba_matrix.ndim == 2 and proba_matrix.shape[1] >= 2:
                        if 1 in classes:
                            idx = classes.index(1)
                            return proba_matrix[:, idx]
                        else:
                            return np.zeros(proba_matrix.shape[0])
                    elif proba_matrix.ndim == 2 and proba_matrix.shape[1] == 1:
                        # single column => only one class present
                        return np.ones(proba_matrix.shape[0]) if 1 in classes else np.zeros(proba_matrix.shape[0])
                    else:
                        return np.zeros(len(proba_matrix))
                else:
                    # Fallback assumption: second column is positive if exists
                    if proba_matrix.ndim == 2 and proba_matrix.shape[1] >= 2:
                        return proba_matrix[:, 1]
                    elif proba_matrix.ndim == 2 and proba_matrix.shape[1] == 1:
                        return np.zeros(proba_matrix.shape[0])
                    else:
                        return np.zeros(len(proba_matrix))
            except Exception:
                return proba_matrix[:, -1] if proba_matrix.ndim == 2 and proba_matrix.shape[1] >= 1 else np.zeros(len(proba_matrix))
        
        # Buy model evaluation
        try:
            buy_train_proba = self.buy_model.predict_proba(X_train)
            buy_test_proba = self.buy_model.predict_proba(X_test)
            buy_train_pred = pos_proba(self.buy_model, buy_train_proba)
            buy_test_pred = pos_proba(self.buy_model, buy_test_proba)
            
            # Calculate AUC only if we have both classes
            if len(np.unique(y_buy_test)) > 1:
                train_auc = roc_auc_score(y_buy_train, buy_train_pred)
                test_auc = roc_auc_score(y_buy_test, buy_test_pred)
            else:
                train_auc = 0.5  # Random performance
                test_auc = 0.5
            
            # Get feature importance (only for models with coef_)
            if not self.feature_names:
                self.feature_names = [f"f{i}" for i in range(X_train.shape[1])]
            if hasattr(self.buy_model, 'coef_'):
                feature_importance = dict(zip(self.feature_names, self.buy_model.coef_[0]))
            else:
                feature_importance = {name: 0.0 for name in self.feature_names}
            
            results['buy_model'] = {
                'train_auc': train_auc,
                'test_auc': test_auc,
                'feature_importance': feature_importance,
                'classification_report': classification_report(y_buy_test, 
                                                             (buy_test_pred > 0.5).astype(int),
                                                             output_dict=True)
            }
        except Exception as e:
            logger.error(f"Buy model evaluation failed: {e}")
            if not self.feature_names:
                self.feature_names = [f"f{i}" for i in range(X_train.shape[1])]
            results['buy_model'] = {
                'train_auc': 0.5,
                'test_auc': 0.5,
                'feature_importance': {name: 0.0 for name in self.feature_names},
                'error': str(e)
            }
        
        # Sell model evaluation
        try:
            sell_train_proba = self.sell_model.predict_proba(X_train)
            sell_test_proba = self.sell_model.predict_proba(X_test)
            sell_train_pred = pos_proba(self.sell_model, sell_train_proba)
            sell_test_pred = pos_proba(self.sell_model, sell_test_proba)
            
            # Calculate AUC only if we have both classes
            if len(np.unique(y_sell_test)) > 1:
                train_auc = roc_auc_score(y_sell_train, sell_train_pred)
                test_auc = roc_auc_score(y_sell_test, sell_test_pred)
            else:
                train_auc = 0.5  # Random performance
                test_auc = 0.5
            
            # Get feature importance (only for LogisticRegression)
            if hasattr(self.sell_model, 'coef_'):
                feature_importance = dict(zip(self.feature_names, self.sell_model.coef_[0]))
            else:
                feature_importance = {name: 0.0 for name in self.feature_names}
            
            results['sell_model'] = {
                'train_auc': train_auc,
                'test_auc': test_auc,
                'feature_importance': feature_importance,
                'classification_report': classification_report(y_sell_test,
                                                             (sell_test_pred > 0.5).astype(int),
                                                             output_dict=True)
            }
        except Exception as e:
            logger.error(f"Sell model evaluation failed: {e}")
            results['sell_model'] = {
                'train_auc': 0.5,
                'test_auc': 0.5,
                'feature_importance': {name: 0.0 for name in self.feature_names},
                'error': str(e)
            }
        
        logger.info(f"Buy model - Train AUC: {results['buy_model']['train_auc']:.3f}, "
                   f"Test AUC: {results['buy_model']['test_auc']:.3f}")
        logger.info(f"Sell model - Train AUC: {results['sell_model']['train_auc']:.3f}, "
                   f"Test AUC: {results['sell_model']['test_auc']:.3f}")
        
        return results
    
    def _initialize_shap_explainers(self, X_train: np.ndarray):
        """Initialize SHAP explainers for model interpretability (with Python 3.12 compatibility)"""
        try:
            # Use a sample of training data for background
            background_size = min(100, len(X_train))
            background_indices = np.random.choice(len(X_train), background_size, replace=False)
            background_data = X_train[background_indices]
            
            # Try SHAP first (if available), else fallback to LIME wrapper
            if SHAP_AVAILABLE:
                try:
                    self.buy_explainer = shap.LinearExplainer(self.buy_model, background_data)
                    self.sell_explainer = shap.LinearExplainer(self.sell_model, background_data)
                    logger.info("SHAP explainers initialized successfully")
                except Exception as shap_err:
                    logger.warning(f"SHAP explainer init failed, falling back to LIME wrapper: {shap_err}")
                    from .model_interpretability import SHAPCompatibilityWrapper
                    self.buy_explainer = SHAPCompatibilityWrapper(
                        self.buy_model, background_data, self.feature_names
                    )
                    self.sell_explainer = SHAPCompatibilityWrapper(
                        self.sell_model, background_data, self.feature_names
                    )
                    logger.info("LIME-based explainers initialized successfully (fallback)")
            else:
                # Use LIME-based alternative for Python 3.12
                from .model_interpretability import SHAPCompatibilityWrapper
                self.buy_explainer = SHAPCompatibilityWrapper(
                    self.buy_model, background_data, self.feature_names
                )
                self.sell_explainer = SHAPCompatibilityWrapper(
                    self.sell_model, background_data, self.feature_names
                )
                logger.info("LIME-based explainers initialized successfully (SHAP alternative)")
                
        except Exception as e:
            logger.warning(f"Failed to initialize explainers: {e}")
            self.buy_explainer = None
            self.sell_explainer = None
    
    def predict_signals(self, X: np.ndarray, confidence_threshold: float = None) -> Dict[str, np.ndarray]:
        """
        Generate buy/sell signals for new data
        
        Args:
            X: Feature matrix
            confidence_threshold: Minimum probability for signal generation
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if self.buy_model is None or self.sell_model is None:
            raise ValueError("Models not trained. Call train_models() first.")
        
        if confidence_threshold is None:
            confidence_threshold = 0.5
        
        # Scale features (skip for equal_weight)
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        def pos_proba(model, proba_matrix):
            try:
                classes = getattr(model, 'classes_', None)
                if classes is not None:
                    classes = list(classes)
                    if proba_matrix.ndim == 2 and proba_matrix.shape[1] >= 2:
                        if 1 in classes:
                            idx = classes.index(1)
                            return proba_matrix[:, idx]
                        else:
                            return np.zeros(proba_matrix.shape[0])
                    elif proba_matrix.ndim == 2 and proba_matrix.shape[1] == 1:
                        return np.ones(proba_matrix.shape[0]) if 1 in classes else np.zeros(proba_matrix.shape[0])
                    else:
                        return np.zeros(len(proba_matrix))
                else:
                    # Fallback assumption: second column is positive if exists
                    if proba_matrix.ndim == 2 and proba_matrix.shape[1] >= 2:
                        return proba_matrix[:, 1]
                    elif proba_matrix.ndim == 2 and proba_matrix.shape[1] == 1:
                        return np.zeros(proba_matrix.shape[0])
                    else:
                        return np.zeros(len(proba_matrix))
            except Exception:
                return proba_matrix[:, -1] if proba_matrix.ndim == 2 and proba_matrix.shape[1] >= 1 else np.zeros(len(proba_matrix))
        
        # Get probabilities robustly (handle single-class models/dummy classifiers)
        buy_proba = pos_proba(self.buy_model, self.buy_model.predict_proba(X_scaled))
        sell_proba = pos_proba(self.sell_model, self.sell_model.predict_proba(X_scaled))
        
        # Generate signals based on confidence threshold and relative strength
        buy_signals = (buy_proba > confidence_threshold) & (buy_proba > sell_proba)
        sell_signals = (sell_proba > confidence_threshold) & (sell_proba > buy_proba)
        
        return {
            'buy_probability': buy_proba,
            'sell_probability': sell_proba,
            'buy_signals': buy_signals.astype(int),
            'sell_signals': sell_signals.astype(int),
            'signal_strength': np.maximum(buy_proba, sell_proba),
            'signal_direction': np.where(buy_proba > sell_proba, 1, -1)
        }
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from both models.
        For EqualWeightSignalClassifier, return uniform magnitudes (with signs) to reflect design.
        """
        if self.buy_model is None or self.sell_model is None:
            raise ValueError("Models not trained.")
        
        def importance_for(model):
            # Equal-weight model has coef_ too, but ensure uniformity if desired
            if isinstance(model, EqualWeightSignalClassifier) and hasattr(model, 'coef_'):
                return dict(zip(self.feature_names, model.coef_[0]))
            if hasattr(model, 'coef_'):
                return dict(zip(self.feature_names, model.coef_[0]))
            if hasattr(model, 'feature_importances_'):
                return dict(zip(self.feature_names, getattr(model, 'feature_importances_')))
            return {name: 0.0 for name in self.feature_names}
        
        return {
            'buy_model': importance_for(self.buy_model),
            'sell_model': importance_for(self.sell_model)
        }
    
    def get_shap_values(self, X: np.ndarray, model_type: str = 'buy') -> Optional[np.ndarray]:
        """
        Get SHAP values for model interpretability (with Python 3.12 compatibility)
        
        Args:
            X: Feature matrix
            model_type: 'buy' or 'sell'
            
        Returns:
            SHAP values array or None if explainer not available
        """
        try:
            X_scaled = self.scaler.transform(X)
            
            if model_type == 'buy' and self.buy_explainer is not None:
                if SHAP_AVAILABLE and hasattr(self.buy_explainer, 'shap_values'):
                    return self.buy_explainer.shap_values(X_scaled)
                elif hasattr(self.buy_explainer, 'shap_values'):
                    # LIME-based alternative
                    return self.buy_explainer.shap_values(X_scaled)
                else:
                    logger.warning("Explainer does not support shap_values method")
                    return None
                    
            elif model_type == 'sell' and self.sell_explainer is not None:
                if SHAP_AVAILABLE and hasattr(self.sell_explainer, 'shap_values'):
                    return self.sell_explainer.shap_values(X_scaled)
                elif hasattr(self.sell_explainer, 'shap_values'):
                    # LIME-based alternative
                    return self.sell_explainer.shap_values(X_scaled)
                else:
                    logger.warning("Explainer does not support shap_values method")
                    return None
            else:
                logger.warning(f"Explainer for {model_type} model not available")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get SHAP values for {model_type} model: {e}")
            return None
    
    def get_comprehensive_feature_importance(self, X: np.ndarray, y_buy: np.ndarray, y_sell: np.ndarray) -> Dict[str, Any]:
        """
        Get comprehensive feature importance analysis using multiple methods
        
        Args:
            X: Feature matrix
            y_buy: Buy target vector
            y_sell: Sell target vector
            
        Returns:
            Dictionary with comprehensive importance analysis
        """
        if not SHAP_AVAILABLE:
            # Use LIME-based comprehensive analysis
            interpreter = ModelInterpreter()
            
            results = {
                'buy_model': interpreter.get_feature_importance_summary(
                    self.buy_model, X, y_buy, self.feature_names
                ),
                'sell_model': interpreter.get_feature_importance_summary(
                    self.sell_model, X, y_sell, self.feature_names
                )
            }
            
            logger.info("Comprehensive feature importance analysis completed using LIME")
            return results
        else:
            # Use traditional SHAP analysis
            logger.info("Using SHAP for feature importance analysis")
            return {
                'buy_model': {'linear_coefficients': dict(zip(self.feature_names, self.buy_model.coef_[0]))},
                'sell_model': {'linear_coefficients': dict(zip(self.feature_names, self.sell_model.coef_[0]))}
            }
    
    def save_models(self, filepath: str):
        """Save trained models and scaler"""
        model_data = {
            'buy_model': self.buy_model,
            'sell_model': self.sell_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models and scaler"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.buy_model = model_data['buy_model']
        self.sell_model = model_data['sell_model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.config = model_data.get('config', self.config)
        
        logger.info(f"Models loaded from {filepath}")

class AlternativeModels:
    """Alternative ML models for comparison"""
    
    @staticmethod
    def train_lasso_regression(X: np.ndarray, y: np.ndarray, alpha: float = 0.01) -> Tuple[Lasso, StandardScaler]:
        """Train Lasso regression for continuous target prediction"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = Lasso(alpha=alpha, random_state=model_config.random_state)
        model.fit(X_scaled, y)
        
        return model, scaler
    
    @staticmethod
    def train_random_forest(X: np.ndarray, y: np.ndarray, **kwargs) -> RandomForestClassifier:
        """Train Random Forest for comparison"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': model_config.random_state,
            'class_weight': 'balanced'
        }
        default_params.update(kwargs)
        
        model = RandomForestClassifier(**default_params)
        model.fit(X, y)
        
        return model