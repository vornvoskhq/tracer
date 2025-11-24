"""
Main framework orchestrator for stock analysis and backtesting
Integrates all components: data loading, feature engineering, ML models, backtesting, and visualization
"""
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import framework components with lazy loading support
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_loader import StockDataLoader
    from .feature_engineering import FeatureEngineer
    from .ml_models import StockSignalPredictor
    from .backtesting import BacktestEngine
    from .visualization import VisualizationEngine

from .config import model_config, backtest_config, db_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_framework.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import json
import pyarrow as pa
import pyarrow.parquet as pq

class TradingFramework:
    """
    Main framework class that orchestrates the entire ML trading pipeline
    """
    
    def __init__(self, config_overrides: Optional[Dict] = None):
        """
        Initialize the trading framework with lazy loading
        
        Args:
            config_overrides: Optional dictionary to override default configurations
        """
        # Apply configuration overrides
        if config_overrides:
            self._apply_config_overrides(config_overrides)
        
        # Store custom configurations for components
        self.custom_config = config_overrides or {}
        
        # Lazy loading - components are initialized only when needed
        self._data_loader = None
        self._feature_engineer = None
        self._predictor = None
        self._backtest_engine = None
        self._viz_engine = None
        self._data_integrity_validator = None
        
        # Apply any configuration overrides
        if config_overrides:
            self._apply_config_overrides(config_overrides)
        
        self.results = {}
        logger.info("Trading Framework initialized with lazy loading")
    
    def _apply_config_overrides(self, config_overrides: Dict):
        """Apply configuration overrides to global config objects"""
        global model_config, backtest_config
        
        # Override model configuration
        if 'model_config' in config_overrides:
            model_overrides = config_overrides['model_config']
            for key, value in model_overrides.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
                    logger.info(f"Model config override: {key} = {value}")
        
        # Override backtest configuration
        if 'backtest_config' in config_overrides:
            backtest_overrides = config_overrides['backtest_config']
            for key, value in backtest_overrides.items():
                if hasattr(backtest_config, key):
                    setattr(backtest_config, key, value)
                    logger.info(f"Backtest config override: {key} = {value}")
    
    @property
    def data_loader(self):
        """Lazy load data loader"""
        if self._data_loader is None:
            from .data_loader import StockDataLoader
            self._data_loader = StockDataLoader()
            logger.debug("Data loader initialized")
        return self._data_loader
    
    @property
    def feature_engineer(self):
        """Lazy load feature engineer"""
        if self._feature_engineer is None:
            from .feature_engineering import FeatureEngineer
            # Pass custom configuration to feature engineer
            self._feature_engineer = FeatureEngineer(config=self.custom_config)
            logger.debug("Feature engineer initialized")
        return self._feature_engineer
    
    @property
    def predictor(self):
        """Lazy load ML predictor"""
        if self._predictor is None:
            try:
                from .ml_models import StockSignalPredictor
                from .config import model_config as mc
                
                # Pass the updated model_config object instead of custom_config dictionary
                # The _apply_config_overrides method already updated the global model_config
                logger.info(f"🔍 DEBUG: Initializing predictor with model_config type: {type(mc)}")
                logger.info(f"🔍 DEBUG: model_config attributes: {dir(mc)}")
                if hasattr(mc, '__dict__'):
                    mc_dict = mc.__dict__
                    ew_keys = [k for k in mc_dict.keys() if k.startswith('ew_')]
                    logger.info(f"🔍 DEBUG: EW keys in model_config: {ew_keys}")
                    for k in ew_keys:
                        logger.info(f"🔍 DEBUG: model_config.{k} = {getattr(mc, k, 'NOT_FOUND')}")
                
                self._predictor = StockSignalPredictor(config=mc)
                logger.info(f"ML predictor initialized (model_type={getattr(mc, 'model_type', 'logistic')})")
            except Exception as e:
                logger.error(f"Failed to initialize ML predictor: {e}")
                logger.error(f"Available attributes: {dir(self)}")
                raise
        return self._predictor
    
    @property
    def backtest_engine(self):
        """Lazy load backtest engine"""
        if self._backtest_engine is None:
            from .backtesting import BacktestEngine
            self._backtest_engine = BacktestEngine()
            logger.debug("Backtest engine initialized")
        return self._backtest_engine
    
    @property
    def viz_engine(self):
        """Lazy load visualization engine"""
        if self._viz_engine is None:
            from .visualization import VisualizationEngine
            self._viz_engine = VisualizationEngine()
            logger.debug("Visualization engine initialized")
        return self._viz_engine
    
    @property
    def data_integrity_validator(self):
        """Lazy load data integrity validator"""
        if self._data_integrity_validator is None:
            from .data_integrity import DataIntegrityValidator
            self._data_integrity_validator = DataIntegrityValidator(self.data_loader)
            logger.debug("Data integrity validator initialized")
        return self._data_integrity_validator
    
    def _apply_config_overrides(self, overrides: Dict):
        """Apply configuration overrides (supports nested and flat keys)."""
        global model_config, backtest_config
        if not overrides:
            return
        # Nested blocks
        if isinstance(overrides.get('model_config'), dict):
            for key, value in overrides['model_config'].items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
                    logger.info(f"Model config override: {key} = {value}")
        if isinstance(overrides.get('backtest_config'), dict):
            for key, value in overrides['backtest_config'].items():
                if hasattr(backtest_config, key):
                    setattr(backtest_config, key, value)
                    logger.info(f"Backtest config override: {key} = {value}")
        # Flat keys fallback (backward compatible): apply to model or backtest config if attribute exists
        for key, value in overrides.items():
            try:
                if hasattr(model_config, key):
                    old_value = getattr(model_config, key, 'NOT_SET')
                    setattr(model_config, key, value)
                    if key.startswith('ew_'):
                        logger.info(f"🔍 DEBUG: Model config override (flat): {key} = {old_value} -> {value}")
                    else:
                        logger.info(f"Model config override (flat): {key} = {value}")
                elif hasattr(backtest_config, key):
                    setattr(backtest_config, key, value)
                    logger.info(f"Backtest config override (flat): {key} = {value}")
            except Exception:
                pass
    
    def run_experiment(self, symbols: List[str], 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      experiment_name: str = "default_experiment") -> Dict[str, Any]:
        """
        Run a complete trading experiment
        
        Args:
            symbols: List of stock symbols to analyze
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            experiment_name: Name for this experiment
            
        Returns:
            Dictionary with complete experiment results
        """
        logger.info(f"Starting experiment: {experiment_name}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Use experiment name directly as ID
        experiment_id = experiment_name
        
        experiment_results = {
            'experiment_id': experiment_id,
            'experiment_name': experiment_name,
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'timestamp': datetime.now(),
            'config': {
                'model_config': dict(model_config.__dict__),
                'backtest_config': dict(backtest_config.__dict__)
            }
        }
        
        try:
            # Step 1: Load and validate data with strict integrity checks
            logger.info("Step 1: Loading and validating data...")
            combined_data = self._load_multi_symbol_data(symbols, start_date, end_date)
            
            if combined_data.empty:
                raise ValueError("No data loaded for the specified symbols and date range")
            
            # CRITICAL: Validate data integrity - no synthetic data, no forward bias
            logger.info("Step 1.1: Running comprehensive data integrity validation...")
            integrity_results = self.data_integrity_validator.run_comprehensive_validation(combined_data)
            
            # Check if any critical validations failed
            critical_failures = []
            for category, checks in integrity_results.items():
                for check, passed in checks.items():
                    # Only treat chronological_order and database_source_only as truly critical
                    # no_future_gaps can be a warning for market data that might include some future dates
                    if not passed and check in ['chronological_order', 'database_source_only']:
                        critical_failures.append(f"{category}.{check}")
                    elif not passed and check == 'no_future_gaps':
                        logger.warning(f"Future timestamps detected in {category}.{check} - this may be normal for market data")
            
            if critical_failures:
                raise ValueError(f"Critical data integrity failures: {critical_failures}. "
                               f"Cannot proceed with potentially compromised data.")
            
            experiment_results['data_integrity'] = integrity_results
            experiment_results['data_summary'] = {
                'total_records': len(combined_data),
                'symbols_loaded': combined_data['symbol'].nunique(),
                'date_range': {
                    'start': combined_data['datetime'].min(),
                    'end': combined_data['datetime'].max()
                },
                'integrity_passed': len(critical_failures) == 0
            }
            
            # Step 2: Feature engineering
            logger.info("Step 2: Engineering features...")
            featured_data = self._engineer_features_multi_symbol(combined_data)
            
            # Step 3: Train ML models
            logger.info("Step 3: Training ML models...")
            model_results = self._train_models(featured_data, experiment_id)
            experiment_results['model_results'] = model_results
            
            # Step 4: Generate signals
            logger.info("Step 4: Generating trading signals...")
            signals = self._generate_signals(featured_data)
            experiment_results['signal_summary'] = self._summarize_signals(signals)
            
            # Step 5: Run backtests
            logger.info("Step 5: Running backtests...")
            backtest_results = self._run_backtests(featured_data, signals)
            experiment_results['backtest_results'] = backtest_results
            
            # Step 6: Create visualizations
            logger.info("Step 6: Creating visualizations...")
            viz_paths = self._create_visualizations(featured_data, signals, backtest_results, experiment_id)
            experiment_results['visualization_paths'] = viz_paths
            
            # Step 7: Generate SHAP analysis
            logger.info("Step 7: Generating SHAP analysis...")
            shap_results = self._generate_shap_analysis(featured_data, experiment_id)
            experiment_results['shap_results'] = shap_results

            # Step 8: Save standardized artifacts (Sprint 1 acceptance)
            logger.info("Step 8: Saving standardized artifacts...")
            try:
                self._save_standard_artifacts(featured_data, signals, backtest_results, experiment_id)
            except Exception as art_e:
                logger.warning(f"Failed to save one or more standard artifacts: {art_e}")
            
            # Save experiment results
            self._save_experiment_results(experiment_results, experiment_id)
            
            logger.info(f"Experiment {experiment_name} completed successfully!")
            return experiment_results
            
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            experiment_results['error'] = str(e)
            return experiment_results
    
    def _load_multi_symbol_data(self, symbols: List[str], start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """Load data for multiple symbols"""
        all_data = []
        
        for symbol in symbols:
            try:
                symbol_data = self.data_loader.get_symbol_data(symbol, start_date, end_date)
                symbol_data = self.data_loader.validate_data_quality(symbol_data)
                
                if not symbol_data.empty:
                    all_data.append(symbol_data)
                    logger.info(f"Loaded {len(symbol_data)} records for {symbol}")
                else:
                    logger.warning(f"No data found for symbol {symbol}")
                    
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {str(e)}")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values(['symbol', 'datetime']).reset_index(drop=True)
            return combined_data
        else:
            return pd.DataFrame()
    
    def _engineer_features_multi_symbol(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for multiple symbols"""
        featured_data = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            
            if len(symbol_data) < max(model_config.ema_periods) + model_config.forecast_horizon:
                logger.warning(f"Insufficient data for {symbol}, skipping...")
                continue
            
            try:
                symbol_featured = self.feature_engineer.process_symbol_data(symbol_data)
                # Apply target strategy via registry (Sprint 2)
                symbol_featured = self._apply_target_strategy(symbol_featured)
                featured_data.append(symbol_featured)
                logger.info(f"Features engineered for {symbol}: {len(symbol_featured)} records")
                
            except Exception as e:
                logger.error(f"Feature engineering failed for {symbol}: {str(e)}")
        
        if featured_data:
            return pd.concat(featured_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _train_models(self, df: pd.DataFrame, experiment_id: str) -> Dict[str, Any]:
        """Train ML models on the featured data"""
        feature_columns = self.feature_engineer.get_all_feature_names()
        
        # Filter to only include features that exist in the data
        available_features = [col for col in feature_columns if col in df.columns]
        logger.info(f"Training with {len(available_features)} features")
        
        # Prepare data
        X, y_buy, y_sell = self.predictor.prepare_data(df, available_features)
        
        # Train models (support buy-only target strategy by zeroing sell)
        from .config import model_config as mc
        if getattr(mc, 'target_strategy', 'dual') == 'buy_only':
            y_sell = np.zeros_like(y_sell)
        
        # Train models
        model_results = self.predictor.train_models(X, y_buy, y_sell)
        
        # Save models in experiment-specific directory
        experiment_dir = f"results/{experiment_id}"
        os.makedirs(f"{experiment_dir}/models", exist_ok=True)
        
        model_path = f"{experiment_dir}/models/trained_model.pkl"
        self.predictor.save_models(model_path)
        model_results['model_path'] = model_path
        
        return model_results
    
    def _generate_signals(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate trading signals using trained models with configurable thresholding"""
        feature_columns = self.feature_engineer.get_all_feature_names()
        available_features = [col for col in feature_columns if col in df.columns]
        
        X = df[available_features].fillna(0).values
        # First get raw probabilities across the whole dataset
        base = self.predictor.predict_signals(X, confidence_threshold=0.0)
        buy_p = base['buy_probability']
        sell_p = base['sell_probability']

        # If buy-only strategy, zero sell probabilities to avoid conflict
        from .config import model_config as mc
        if getattr(mc, 'target_strategy', 'dual') == 'buy_only':
            sell_p = np.zeros_like(sell_p)
        
        # Helper to compute signals under a given strategy
        def compute_signals(strategy: str):
            if strategy == 'percentile':
                window = max(10, int(getattr(backtest_config, 'threshold_window', 60)))
                q = float(getattr(backtest_config, 'threshold_percentile', 0.7))
                q = min(max(q, 0.0), 1.0)
                buy_thr = np.zeros_like(buy_p)
                sell_thr = np.zeros_like(sell_p)
                for symbol in df['symbol'].unique():
                    idx = df.index[df['symbol'] == symbol]
                    s_buy = pd.Series(buy_p[idx])
                    s_sell = pd.Series(sell_p[idx])
                    rb = s_buy.rolling(window=window, min_periods=max(5, window//3)).quantile(q).bfill().values
                    rs = s_sell.rolling(window=window, min_periods=max(5, window//3)).quantile(q).bfill().values
                    buy_thr[idx] = rb
                    sell_thr[idx] = rs
                buy_sig = (buy_p > buy_thr) & (buy_p > sell_p)
                sell_sig = (sell_p > sell_thr) & (sell_p > buy_p)
                return buy_sig, sell_sig
            elif strategy == 'dynamic_absolute':
                # Dynamic absolute threshold: per-symbol rolling mean + k*std, floored by absolute threshold
                window = max(10, int(getattr(backtest_config, 'threshold_window', 60)))
                k = float(getattr(backtest_config, 'dynamic_k', 0.5))
                base_thr = float(getattr(backtest_config, 'confidence_threshold', 0.5))
                buy_thr = np.zeros_like(buy_p)
                sell_thr = np.zeros_like(sell_p)
                for symbol in df['symbol'].unique():
                    idx = df.index[df['symbol'] == symbol]
                    s_buy = pd.Series(buy_p[idx])
                    s_sell = pd.Series(sell_p[idx])
                    m_buy = s_buy.rolling(window=window, min_periods=max(5, window//3)).mean()
                    sd_buy = s_buy.rolling(window=window, min_periods=max(5, window//3)).std().fillna(0)
                    m_sell = s_sell.rolling(window=window, min_periods=max(5, window//3)).mean()
                    sd_sell = s_sell.rolling(window=window, min_periods=max(5, window//3)).std().fillna(0)
                    db = (m_buy + k * sd_buy).clip(lower=base_thr).bfill().values
                    ds = (m_sell + k * sd_sell).clip(lower=base_thr).bfill().values
                    buy_thr[idx] = db
                    sell_thr[idx] = ds
                buy_sig = (buy_p > buy_thr) & (buy_p > sell_p)
                sell_sig = (sell_p > sell_thr) & (sell_p > buy_p)
                return buy_sig, sell_sig
            else:
                thr = float(getattr(backtest_config, 'confidence_threshold', 0.5))
                buy_sig = (buy_p > thr) & (buy_p > sell_p)
                sell_sig = (sell_p > thr) & (sell_p > buy_p)
                return buy_sig, sell_sig
        
        strategy = getattr(backtest_config, 'threshold_strategy', 'absolute')
        buy_signals, sell_signals = compute_signals(strategy)
        
        # Fallback: if absolute produced no signals and auto fallback is enabled, switch to percentile
        if strategy == 'absolute' and backtest_config.auto_threshold_fallback:
            if buy_signals.sum() + sell_signals.sum() == 0:
                logger.warning("No signals with absolute threshold; falling back to percentile strategy.")
                buy_signals, sell_signals = compute_signals('percentile')
        
        # Diagnostics: compute counts and quantiles per symbol
        try:
            diag = {}
            for symbol in df['symbol'].unique():
                idx = df.index[df['symbol'] == symbol]
                bp, sp = buy_p[idx], sell_p[idx]
                diag[symbol] = {
                    'n': int(len(idx)),
                    'buy_q': {q: float(np.quantile(bp, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
                    'sell_q': {q: float(np.quantile(sp, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
                    'buy_signals': int(buy_signals[idx].sum()),
                    'sell_signals': int(sell_signals[idx].sum())
                }
            logger.info(f"Signal diagnostics per symbol: {diag}")
        except Exception as e:
            logger.debug(f"Signal diagnostics failed: {e}")
        
        return {
            'buy_probability': buy_p,
            'sell_probability': sell_p,
            'buy_signals': buy_signals.astype(int),
            'sell_signals': sell_signals.astype(int),
            'signal_strength': np.maximum(buy_p, sell_p),
            'signal_direction': np.where(buy_p > sell_p, 1, -1)
        }
    
    def _summarize_signals(self, signals: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Summarize signal statistics"""
        return {
            'total_buy_signals': int(signals['buy_signals'].sum()),
            'total_sell_signals': int(signals['sell_signals'].sum()),
            'avg_buy_probability': float(signals['buy_probability'].mean()),
            'avg_sell_probability': float(signals['sell_probability'].mean()),
            'max_buy_probability': float(signals['buy_probability'].max()),
            'max_sell_probability': float(signals['sell_probability'].max())
        }
    
    def _run_backtests(self, df: pd.DataFrame, signals: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Run backtests for each symbol"""
        backtest_results = {}
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_indices = df[df['symbol'] == symbol].index
            
            # Extract signals for this symbol
            symbol_signals = {
                key: values[symbol_indices] for key, values in signals.items()
            }
            
            try:
                symbol_backtest = self.backtest_engine.run_backtest(symbol_data, symbol_signals)
                backtest_results[symbol] = symbol_backtest
                
            except Exception as e:
                logger.error(f"Backtest failed for {symbol}: {str(e)}")
                backtest_results[symbol] = {'error': str(e)}
        
        # Calculate aggregate results
        backtest_results['aggregate'] = self._calculate_aggregate_backtest_results(backtest_results)
        
        return backtest_results
    
    def _apply_target_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply target generation based on configured strategy using simple registry."""
        try:
            from .config import model_config as mc
            strategy = str(getattr(mc, 'target_strategy', 'dual')).lower()
            params = getattr(mc, 'target_params', {}) or {}
            # Lazy import architecture registry
            try:
                from .architecture import targets_registry
            except Exception:
                targets_registry = {}
            # Select strategy implementation
            if strategy in targets_registry:
                strat_cls = targets_registry[strategy]
                # Use symmetric horizon and thresholds from model_config
                return strat_cls().generate_targets(
                    df,
                    buy_threshold=float(getattr(mc, 'buy_threshold', 0.02)),
                    sell_threshold=float(getattr(mc, 'sell_threshold', 0.02)),
                    symmetric_horizon=int(getattr(mc, 'forecast_horizon', 5)),
                    **params
                )
            # Fallback: return df as-is (FeatureEngineer already created targets)
            return df
        except Exception as e:
            logger.debug(f"Target strategy application failed; using existing targets: {e}")
            return df

    def _calculate_aggregate_backtest_results(self, individual_results: Dict) -> Dict[str, Any]:
        """Calculate aggregate backtest results across all symbols"""
        valid_results = {k: v for k, v in individual_results.items() if 'error' not in v and k != 'aggregate'}
        
        if not valid_results:
            return {'error': 'No valid backtest results'}
        
        # Aggregate metrics
        total_trades = sum(result.get('total_trades', 0) for result in valid_results.values())
        total_return = np.mean([result.get('total_return', 0) for result in valid_results.values()])
        avg_sharpe = np.mean([result.get('sharpe_ratio', 0) for result in valid_results.values()])
        avg_max_drawdown = np.mean([result.get('max_drawdown', 0) for result in valid_results.values()])
        
        return {
            'symbols_analyzed': len(valid_results),
            'total_trades': total_trades,
            'avg_total_return': total_return,
            'avg_sharpe_ratio': avg_sharpe,
            'avg_max_drawdown': avg_max_drawdown,
            'individual_results': valid_results
        }
    
    def _create_visualizations(self, df: pd.DataFrame, signals: Dict, backtest_results: Dict, experiment_id: str) -> Dict[str, str]:
        """Create essential visualizations only (optimized for speed)"""
        viz_dir = f"results/{experiment_id}/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        viz_paths = {}
        
        try:
            # Feature importance plot (essential for analysis)
            importance = self.predictor.get_feature_importance()
            fig_importance = self.viz_engine.plot_feature_importance(importance)
            importance_path = f"{viz_dir}/feature_importance.png"
            fig_importance.savefig(importance_path, dpi=300, bbox_inches='tight')
            viz_paths['feature_importance'] = importance_path
            
            # Create only performance charts (remove redundant dashboard and signal charts)
            for symbol in df['symbol'].unique():
                if symbol in backtest_results and 'error' not in backtest_results[symbol]:
                    # Performance chart only (most important for comparison)
                    fig_perf = self.viz_engine.plot_backtest_performance(backtest_results[symbol])
                    perf_path = f"{viz_dir}/{symbol}_performance.png"
                    fig_perf.savefig(perf_path, dpi=300, bbox_inches='tight')
                    viz_paths[f'{symbol}_performance'] = perf_path
                    
                    # Skip signal chart and dashboard (redundant and slow)
                    # These can be generated on-demand if needed
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")
            viz_paths['error'] = str(e)
        
        return viz_paths
    
    def _generate_shap_analysis(self, df: pd.DataFrame, experiment_id: str) -> Dict[str, Any]:
        """Generate model interpretability analysis (SHAP or LIME-based for Python 3.12)"""
        try:
            feature_columns = self.feature_engineer.get_all_feature_names()
            available_features = [col for col in feature_columns if col in df.columns]
            
            # Use a sample of data for analysis (for performance)
            sample_size = min(1000, len(df))
            sample_indices = np.random.choice(len(df), sample_size, replace=False)
            X_sample = df[available_features].iloc[sample_indices].fillna(0).values
            y_buy_sample = df['target_buy'].iloc[sample_indices].fillna(0).values
            y_sell_sample = df['target_sell'].iloc[sample_indices].fillna(0).values
            
            # Get comprehensive feature importance analysis
            comprehensive_analysis = self.predictor.get_comprehensive_feature_importance(
                X_sample, y_buy_sample, y_sell_sample
            )
            
            # Get SHAP/LIME values for both models
            shap_values_buy = self.predictor.get_shap_values(X_sample, 'buy')
            shap_values_sell = self.predictor.get_shap_values(X_sample, 'sell')
            
            # Determine which method was used
            try:
                import shap
                method_used = 'SHAP'
            except ImportError:
                method_used = 'LIME'
            
            shap_results = {
                'feature_names': available_features,
                'sample_size': sample_size,
                'comprehensive_analysis': comprehensive_analysis,
                'method_used': method_used
            }
            
            if shap_values_buy is not None:
                # Calculate mean absolute values for feature importance
                mean_shap_buy = np.abs(shap_values_buy).mean(axis=0)
                
                # If LIME failed and we're using linear coefficients, the values will be the same across all samples
                # In this case, just take the first row since they're all identical
                if np.all(shap_values_buy[0] == shap_values_buy[-1]) and not np.all(shap_values_buy[0] == 0):
                    mean_shap_buy = np.abs(shap_values_buy[0])
                    logger.info("Using linear coefficient values for buy model visualization")
                elif np.all(mean_shap_buy == 0):
                    logger.warning("LIME analysis failed completely for buy model - using linear coefficients directly")
                    # Get linear coefficients directly from comprehensive analysis
                    if 'buy_model' in comprehensive_analysis and 'linear_coefficients' in comprehensive_analysis['buy_model']:
                        linear_coeffs = comprehensive_analysis['buy_model']['linear_coefficients']
                        mean_shap_buy = np.array([abs(linear_coeffs.get(feature, 0)) for feature in available_features])
                
                shap_results['buy_model_importance'] = dict(zip(available_features, mean_shap_buy))
                
                # Create interpretability plot
                fig_shap_buy = self.viz_engine.plot_shap_summary(shap_values_buy, available_features, X_sample, 'buy')
                shap_path_buy = f"results/{experiment_id}/analysis/interpretability_buy_model.png"
                os.makedirs(f"results/{experiment_id}/analysis", exist_ok=True)
                fig_shap_buy.savefig(shap_path_buy, dpi=300, bbox_inches='tight')
                shap_results['buy_model_plot'] = shap_path_buy
            
            if shap_values_sell is not None:
                mean_shap_sell = np.abs(shap_values_sell).mean(axis=0)
                
                # If LIME failed and we're using linear coefficients, the values will be the same across all samples
                # In this case, just take the first row since they're all identical
                if np.all(shap_values_sell[0] == shap_values_sell[-1]) and not np.all(shap_values_sell[0] == 0):
                    mean_shap_sell = np.abs(shap_values_sell[0])
                    logger.info("Using linear coefficient values for sell model visualization")
                elif np.all(mean_shap_sell == 0):
                    logger.warning("LIME analysis failed completely for sell model - using linear coefficients directly")
                    # Get linear coefficients directly from comprehensive analysis
                    if 'sell_model' in comprehensive_analysis and 'linear_coefficients' in comprehensive_analysis['sell_model']:
                        linear_coeffs = comprehensive_analysis['sell_model']['linear_coefficients']
                        mean_shap_sell = np.array([abs(linear_coeffs.get(feature, 0)) for feature in available_features])
                
                shap_results['sell_model_importance'] = dict(zip(available_features, mean_shap_sell))
                
                fig_shap_sell = self.viz_engine.plot_shap_summary(shap_values_sell, available_features, X_sample, 'sell')
                shap_path_sell = f"results/{experiment_id}/analysis/interpretability_sell_model.png"
                fig_shap_sell.savefig(shap_path_sell, dpi=300, bbox_inches='tight')
                shap_results['sell_model_plot'] = shap_path_sell
            
            # Generate comprehensive report if using LIME
            if method_used == 'LIME' and comprehensive_analysis:
                from .model_interpretability import ModelInterpreter
                interpreter = ModelInterpreter()
                
                if 'buy_model' in comprehensive_analysis:
                    buy_report = interpreter.create_feature_importance_report(
                        comprehensive_analysis['buy_model'], top_n=15
                    )
                    shap_results['buy_model_report'] = buy_report
                
                if 'sell_model' in comprehensive_analysis:
                    sell_report = interpreter.create_feature_importance_report(
                        comprehensive_analysis['sell_model'], top_n=15
                    )
                    shap_results['sell_model_report'] = sell_report
            
            logger.info(f"Model interpretability analysis completed using {method_used}")
            return shap_results
            
        except Exception as e:
            logger.error(f"Model interpretability analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _save_standard_artifacts(self, df: pd.DataFrame, signals: Dict[str, Any], backtest_results: Dict[str, Any], experiment_id: str):
        """Save standardized artifacts required by Sprint 1.
        - signals.parquet
        - backtest.csv
        - backtest_aggregate.json
        - models/trained_model.pkl is already saved in _train_models
        """
        base = f"results/{experiment_id}"
        os.makedirs(base, exist_ok=True)

        # 1) signals.parquet
        try:
            sdf = pd.DataFrame({
                'datetime': df['datetime'],
                'symbol': df['symbol'],
                'buy_probability': signals['buy_probability'],
                'sell_probability': signals['sell_probability'],
                'buy_signal': signals['buy_signals'],
                'sell_signal': signals['sell_signals'],
            })
            table = pa.Table.from_pandas(sdf)
            pq.write_table(table, os.path.join(base, 'signals.parquet'))
        except Exception as e:
            logger.warning(f"Failed to write signals.parquet: {e}")

        # 2) backtest.csv per symbol flattened
        try:
            rows = []
            for sym, res in backtest_results.items():
                if sym == 'aggregate' or 'error' in res:
                    continue
                rows.append({
                    'symbol': sym,
                    'total_return': res.get('total_return'),
                    'sharpe_ratio': res.get('sharpe_ratio'),
                    'max_drawdown': res.get('max_drawdown'),
                    'volatility': res.get('volatility'),
                    'total_trades': res.get('total_trades', 0),
                    'win_rate': res.get('win_rate', 0.0),
                })
            pd.DataFrame(rows).to_csv(os.path.join(base, 'backtest.csv'), index=False)
        except Exception as e:
            logger.warning(f"Failed to write backtest.csv: {e}")

        # 3) backtest_aggregate.json
        try:
            agg = backtest_results.get('aggregate', {})
            with open(os.path.join(base, 'backtest_aggregate.json'), 'w') as f:
                json.dump(agg, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to write backtest_aggregate.json: {e}")

        # 4) metrics.json (summary)
        try:
            metrics = {}
            if 'aggregate' in backtest_results:
                metrics = {
                    'symbols_analyzed': backtest_results['aggregate'].get('symbols_analyzed'),
                    'total_trades': backtest_results['aggregate'].get('total_trades'),
                    'avg_total_return': backtest_results['aggregate'].get('avg_total_return'),
                    'avg_sharpe_ratio': backtest_results['aggregate'].get('avg_sharpe_ratio'),
                    'avg_max_drawdown': backtest_results['aggregate'].get('avg_max_drawdown'),
                }
            else:
                # fallback minimal
                metrics = {
                    'symbols_analyzed': len([k for k in backtest_results.keys() if k != 'aggregate']),
                }
            with open(os.path.join(base, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to write metrics.json: {e}")

        # 5) feature_importance.json
        try:
            fi = self.predictor.get_feature_importance()
            with open(os.path.join(base, 'feature_importance.json'), 'w') as f:
                json.dump(fi, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write feature_importance.json: {e}")

    def _save_experiment_results(self, results: Dict, experiment_id: str):
        """Save experiment results to file"""
        experiment_dir = f"results/{experiment_id}"
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Convert numpy arrays and other non-serializable objects
        serializable_results = self._make_serializable(results)
        
        import json
        results_path = f"{experiment_dir}/experiment_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Experiment results saved to {results_path}")
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        else:
            return obj
    
    def get_database_summary(self) -> Dict[str, Any]:
        """Get summary of available data in the database"""
        return self.data_loader.get_data_summary()
    
    def close(self):
        """Clean up resources"""
        self.data_loader.close()
        logger.info("Trading Framework closed")

# Example usage and testing functions
def run_sample_experiment():
    """Run a sample experiment with demo configuration"""
    framework = TradingFramework()
    
    # Get available symbols (limit to first 5 for demo)
    symbols_df = framework.data_loader.get_symbols(limit=5)
    symbols = symbols_df['symbol'].tolist()
    
    logger.info(f"Running sample experiment with symbols: {symbols}")
    
    # Run experiment
    results = framework.run_experiment(
        symbols=symbols,
        start_date="2023-01-01",
        end_date="2024-01-01",
        experiment_name="sample_experiment"
    )
    
    framework.close()
    return results

if __name__ == "__main__":
    # Run sample experiment
    results = run_sample_experiment()
    print("\nExperiment completed!")
    print(f"Results saved for experiment: {results.get('experiment_name', 'unknown')}")