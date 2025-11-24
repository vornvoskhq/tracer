"""
Experiment configuration templates for different trading strategies and scenarios
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import json

@dataclass
class TechnicalIndicatorConfig:
    """Configuration for technical indicators"""
    rsi: Dict[str, Union[int, float]] = field(default_factory=lambda: {
        "period": 14,
        "oversold_threshold": 30,
        "overbought_threshold": 70
    })
    macd: Dict[str, int] = field(default_factory=lambda: {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
    })
    bollinger_bands: Dict[str, Union[int, float]] = field(default_factory=lambda: {
        "period": 20,
        "std_dev": 2
    })
    momentum: Dict[str, Union[int, float]] = field(default_factory=lambda: {
        "window": 5,
        "std_window": 20,
        "std_multiplier": 1.5
    })
    roc_periods: List[int] = field(default_factory=lambda: [5, 10])
    volume_roc_periods: List[int] = field(default_factory=lambda: [5])

@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters"""
    interpretability_sample_size: int = 1000
    lime_sample_size: int = 100
    lime_max_samples: int = 100
    min_valid_target_ratio: float = 0.8
    max_correlation_threshold: float = 0.95
    rolling_sharpe_window: int = 60
    trading_days_per_year: int = 252

@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters"""
    dpi: int = 300
    figure_size: List[int] = field(default_factory=lambda: [12, 8])

@dataclass
class ExperimentConfig:
    """Configuration for a trading experiment"""
    name: str
    description: str
    
    # Data selection
    symbols: List[str]
    start_date: str
    end_date: str
    
    # Model parameters
    buy_threshold: float = 0.02
    sell_threshold: float = 0.02
    forecast_horizon: int = 5
    buy_forecast_horizon: Optional[int] = None
    sell_forecast_horizon: Optional[int] = None
    confidence_threshold: float = 0.65
    
    # Target strategy
    target_strategy: str = 'dual'  # 'dual' or 'buy_only'
    
    # Feature engineering
    ema_periods: List[int] = None
    volume_window: int = 20
    
    # Model training
    test_size: float = 0.2
    class_weight: str = 'balanced'
    penalty: str = 'l1'
    solver: str = 'liblinear'
    max_iter: int = 1000
    tol: float = 1e-4
    model_type: str = 'logistic'
    
    # Ensemble specific
    base_models: Optional[List[str]] = None
    weights: Optional[List[float]] = None
    
    # Equal-weight specific parameters (optional)
    ew_correlation_signs: bool = False
    ew_calibrate_prior: bool = False
    ew_activation_threshold: float = 0.0
    ew_temperature: float = 1.0
    ew_buy_aggregation: str = 'fraction'
    ew_sell_aggregation: str = 'fraction'
    
    # XGBoost specific parameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Backtesting
    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005
    max_position_size: float = 0.1
    # Thresholding
    threshold_strategy: str = 'absolute'  # 'absolute' or 'percentile'
    threshold_percentile: float = 0.7
    threshold_window: int = 60
    # Trade frequency/gating
    auto_threshold_fallback: bool = True
    fallback_percentile: float = 0.7
    fallback_window: int = 60
    buy_consecutive_days: int = 1
    sell_consecutive_days: int = 1
    hysteresis_margin: float = 0.0
    trade_cooldown_days: int = 0
    min_holding_days: int = 0
    
    # New configurable parameters
    technical_indicators: TechnicalIndicatorConfig = field(default_factory=TechnicalIndicatorConfig)
    enabled_features: List[str] = field(default_factory=lambda: [
        "price_above_ema9", "price_above_ema30", "price_momentum_strong",
        "ema9_above_ema30", "ema_spread_accelerating",
        "macd_above_signal", "macd_above_zero", "signal_above_zero", "macd_positive_hist",
        "volume_above_average", "volume_ratio", "ha_uptrend",
        "rsi_oversold", "rsi_overbought", "rsi_bullish",
        "price_roc_5", "price_roc_10", "volume_roc_5",
        "ema_separation_accel", "price_ema_separation_accel"
    ])
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    def __post_init__(self):
        if self.ema_periods is None:
            self.ema_periods = [9, 21, 50]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'description': self.description,
            'symbols': self.symbols,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold,
            'forecast_horizon': self.forecast_horizon,
            'buy_forecast_horizon': self.buy_forecast_horizon,
            'sell_forecast_horizon': self.sell_forecast_horizon,
            'confidence_threshold': self.confidence_threshold,
            'ema_periods': self.ema_periods,
            'volume_window': self.volume_window,
            'test_size': self.test_size,
            'class_weight': self.class_weight,
            'penalty': self.penalty,
            'solver': self.solver,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'model_type': self.model_type,
             'target_strategy': getattr(self, 'target_strategy', 'dual'),
             'base_models': getattr(self, 'base_models', None),
             'weights': getattr(self, 'weights', None),
            # Equal-weight params
            'ew_correlation_signs': self.ew_correlation_signs,
            'ew_calibrate_prior': self.ew_calibrate_prior,
            'ew_activation_threshold': self.ew_activation_threshold,
            'ew_temperature': self.ew_temperature,
            'ew_buy_aggregation': self.ew_buy_aggregation,
            'ew_sell_aggregation': self.ew_sell_aggregation,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'initial_capital': self.initial_capital,
            'commission': self.commission,
             'slippage': getattr(self, 'slippage', 0.0005),
            'max_position_size': self.max_position_size,
             'threshold_strategy': self.threshold_strategy,
             'threshold_percentile': self.threshold_percentile,
             'threshold_window': self.threshold_window,
             'auto_threshold_fallback': getattr(self, 'auto_threshold_fallback', True),
             'fallback_percentile': getattr(self, 'fallback_percentile', 0.7),
             'fallback_window': getattr(self, 'fallback_window', 60),
             'buy_consecutive_days': getattr(self, 'buy_consecutive_days', 1),
             'sell_consecutive_days': getattr(self, 'sell_consecutive_days', 1),
             'hysteresis_margin': getattr(self, 'hysteresis_margin', 0.0),
             'trade_cooldown_days': getattr(self, 'trade_cooldown_days', 0),
             'min_holding_days': getattr(self, 'min_holding_days', 0),
             'threshold_strategy': self.threshold_strategy,
             'threshold_percentile': self.threshold_percentile,
             'threshold_window': self.threshold_window,
            'technical_indicators': {
                'rsi': self.technical_indicators.rsi,
                'macd': self.technical_indicators.macd,
                'bollinger_bands': self.technical_indicators.bollinger_bands,
                'momentum': self.technical_indicators.momentum,
                'roc_periods': self.technical_indicators.roc_periods,
                'volume_roc_periods': self.technical_indicators.volume_roc_periods
            },
            'enabled_features': self.enabled_features,
            'analysis': {
                'interpretability_sample_size': self.analysis.interpretability_sample_size,
                'lime_sample_size': self.analysis.lime_sample_size,
                'lime_max_samples': self.analysis.lime_max_samples,
                'min_valid_target_ratio': self.analysis.min_valid_target_ratio,
                'max_correlation_threshold': self.analysis.max_correlation_threshold,
                'rolling_sharpe_window': self.analysis.rolling_sharpe_window,
                'trading_days_per_year': self.analysis.trading_days_per_year
            },
            'visualization': {
                'dpi': self.visualization.dpi,
                'figure_size': self.visualization.figure_size
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary"""
        # Handle nested configuration objects
        data_copy = data.copy()
        
        # Parse technical_indicators if present
        if 'technical_indicators' in data_copy:
            tech_data = data_copy.pop('technical_indicators')
            data_copy['technical_indicators'] = TechnicalIndicatorConfig(
                rsi=tech_data.get('rsi', TechnicalIndicatorConfig().rsi),
                macd=tech_data.get('macd', TechnicalIndicatorConfig().macd),
                bollinger_bands=tech_data.get('bollinger_bands', TechnicalIndicatorConfig().bollinger_bands),
                momentum=tech_data.get('momentum', TechnicalIndicatorConfig().momentum),
                roc_periods=tech_data.get('roc_periods', TechnicalIndicatorConfig().roc_periods),
                volume_roc_periods=tech_data.get('volume_roc_periods', TechnicalIndicatorConfig().volume_roc_periods)
            )
        
        # Parse analysis if present
        if 'analysis' in data_copy:
            analysis_data = data_copy.pop('analysis')
            data_copy['analysis'] = AnalysisConfig(
                interpretability_sample_size=analysis_data.get('interpretability_sample_size', 1000),
                lime_sample_size=analysis_data.get('lime_sample_size', 100),
                lime_max_samples=analysis_data.get('lime_max_samples', 100),
                min_valid_target_ratio=analysis_data.get('min_valid_target_ratio', 0.8),
                max_correlation_threshold=analysis_data.get('max_correlation_threshold', 0.95),
                rolling_sharpe_window=analysis_data.get('rolling_sharpe_window', 60),
                trading_days_per_year=analysis_data.get('trading_days_per_year', 252)
            )
        
        # Parse visualization if present
        if 'visualization' in data_copy:
            viz_data = data_copy.pop('visualization')
            data_copy['visualization'] = VisualizationConfig(
                dpi=viz_data.get('dpi', 300),
                figure_size=viz_data.get('figure_size', [12, 8])
            )
        
        # Optional top-level additions for Sprint 2
        data_copy['target_strategy'] = data_copy.get('target_strategy', data_copy.get('target', 'dual'))
        data_copy['base_models'] = data_copy.get('base_models')
        data_copy['weights'] = data_copy.get('weights')
        return cls(**data_copy)  # includes slippage if present
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

# Predefined experiment configurations
EXPERIMENT_CONFIGS = {
    'quick_test': ExperimentConfig(
        name='quick_test',
        description='Quick test with major ETFs for development and debugging',
        symbols=['SPY', 'QQQ', 'IWM'],
        start_date='2023-01-01',
        end_date='2024-01-01',
        buy_threshold=0.02,
        sell_threshold=0.02,
        forecast_horizon=5,
        confidence_threshold=0.6,
        ema_periods=[9, 21, 50],
        initial_capital=50000.0
    ),
    
    'conservative_strategy': ExperimentConfig(
        name='conservative_strategy',
        description='Conservative strategy with lower thresholds and longer horizon',
        symbols=['SPY', 'QQQ', 'DIA', 'IWM', 'VTI'],
        start_date='2022-01-01',
        end_date='2024-01-01',
        buy_threshold=0.015,  # 1.5% threshold
        sell_threshold=0.015,
        forecast_horizon=10,   # Longer horizon
        confidence_threshold=0.7,  # Higher confidence
        ema_periods=[9, 21, 50, 100],
        commission=0.0005,     # Lower commission
        max_position_size=0.05  # Smaller positions
    ),
    
    'aggressive_strategy': ExperimentConfig(
        name='aggressive_strategy',
        description='Aggressive strategy with higher thresholds and shorter horizon',
        symbols=['QQQ', 'TQQQ', 'ARKK', 'SPXL'],
        start_date='2023-01-01',
        end_date='2024-01-01',
        buy_threshold=0.03,    # 3% threshold
        sell_threshold=0.03,
        forecast_horizon=3,    # Shorter horizon
        confidence_threshold=0.55,  # Lower confidence for more trades
        ema_periods=[5, 13, 21],    # Faster EMAs
        commission=0.002,      # Higher commission for frequent trading
        max_position_size=0.15  # Larger positions
    ),
    
    'tech_stocks': ExperimentConfig(
        name='tech_stocks',
        description='Technology stocks analysis',
        symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'],
        start_date='2022-01-01',
        end_date='2024-01-01',
        buy_threshold=0.025,
        sell_threshold=0.025,
        forecast_horizon=7,
        confidence_threshold=0.65,
        ema_periods=[9, 21, 50],
        initial_capital=200000.0
    ),
    
    'dividend_stocks': ExperimentConfig(
        name='dividend_stocks',
        description='Dividend-focused stocks with conservative approach',
        symbols=['JNJ', 'PG', 'KO', 'PFE', 'VZ', 'T', 'XOM'],
        start_date='2022-01-01',
        end_date='2024-01-01',
        buy_threshold=0.01,    # Lower threshold for stable stocks
        sell_threshold=0.015,
        forecast_horizon=15,   # Longer horizon for stable stocks
        confidence_threshold=0.75,  # Higher confidence
        ema_periods=[21, 50, 100, 200],  # Longer EMAs
        max_position_size=0.08
    ),
    
    'sector_rotation': ExperimentConfig(
        name='sector_rotation',
        description='Sector ETFs for rotation strategy',
        symbols=['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB'],
        start_date='2021-01-01',
        end_date='2024-01-01',
        buy_threshold=0.02,
        sell_threshold=0.02,
        forecast_horizon=10,
        confidence_threshold=0.6,
        ema_periods=[9, 21, 50],
        initial_capital=300000.0,
        max_position_size=0.12  # Larger positions for sector rotation
    ),
    
    'small_cap_growth': ExperimentConfig(
        name='small_cap_growth',
        description='Small cap growth stocks with higher volatility parameters',
        symbols=['IWM', 'VB', 'VTWO', 'SCHA'],
        start_date='2023-01-01',
        end_date='2024-01-01',
        buy_threshold=0.04,    # Higher threshold for volatile stocks
        sell_threshold=0.04,
        forecast_horizon=5,
        confidence_threshold=0.6,
        ema_periods=[5, 13, 21, 50],
        commission=0.0015,     # Higher commission for small caps
        max_position_size=0.08
    ),
    
    'crypto_related': ExperimentConfig(
        name='crypto_related',
        description='Cryptocurrency-related stocks and ETFs',
        symbols=['COIN', 'MSTR', 'RIOT', 'MARA', 'BITO'],
        start_date='2023-01-01',
        end_date='2024-01-01',
        buy_threshold=0.05,    # Very high threshold for crypto volatility
        sell_threshold=0.05,
        forecast_horizon=3,    # Short horizon for high volatility
        confidence_threshold=0.5,  # Lower confidence for more signals
        ema_periods=[5, 10, 21],   # Very fast EMAs
        commission=0.002,
        max_position_size=0.05  # Smaller positions due to high risk
    ),
    
    'dual_signal': ExperimentConfig(
        name='dual_signal',
        description='Dual logistic regression model as described in convo.txt - separate buy/sell models with crossover signals',
        symbols=['SPY', 'QQQ'],
        start_date='2023-01-01',
        end_date='2024-01-01',
        buy_threshold=0.02,    # 2% gain threshold for buy signal success
        sell_threshold=0.02,   # 2% loss threshold for sell signal success
        forecast_horizon=5,    # 5-day forecast horizon (reduced for smaller datasets)
        confidence_threshold=0.65,  # 65% confidence threshold as discussed in convo.txt
        ema_periods=[9, 21, 50],    # EMA periods from codex.txt (adapted for data size)
        volume_window=20,
        test_size=0.2,
        class_weight='balanced',
        penalty='l1',          # L1 regularization for feature selection
        solver='liblinear',    # Solver that supports L1 penalty
        initial_capital=100000.0,
        commission=0.001,      # 0.1% commission
        max_position_size=0.1  # Max 10% per position
    )
}

def list_available_configs() -> Dict[str, str]:
    """List all available experiment configurations"""
    return {name: config.description for name, config in EXPERIMENT_CONFIGS.items()}

def get_config(name: str) -> ExperimentConfig:
    """Get a specific experiment configuration"""
    if name not in EXPERIMENT_CONFIGS:
        available = list(EXPERIMENT_CONFIGS.keys())
        raise ValueError(f"Configuration '{name}' not found. Available: {available}")
    return EXPERIMENT_CONFIGS[name]

def create_custom_config(
    name: str,
    symbols: List[str],
    start_date: str,
    end_date: str,
    **kwargs
) -> ExperimentConfig:
    """Create a custom experiment configuration"""
    return ExperimentConfig(
        name=name,
        description=f"Custom configuration: {name}",
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )

def save_config_templates():
    """Save all predefined configurations to files"""
    import os
    os.makedirs('results/configs', exist_ok=True)
    
    for name, config in EXPERIMENT_CONFIGS.items():
        filepath = f'results/configs/{name}.json'
        config.save_to_file(filepath)
        print(f"Saved {name} configuration to {filepath}")

if __name__ == "__main__":
    # Save all configuration templates
    save_config_templates()
    
    # List available configurations
    print("\nAvailable Experiment Configurations:")
    print("=" * 50)
    for name, description in list_available_configs().items():
        print(f"{name:20} - {description}")