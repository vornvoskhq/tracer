"""
Configuration settings for the stock analysis framework
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', '5432'))
    database: str = os.getenv('DB_NAME', 'stock_data')
    username: str = os.getenv('DB_USER', 'postgres')
    password: str = os.getenv('DB_PASSWORD', '')
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class ModelConfig:
    """ML model configuration"""
    # Model selection
    model_type: str = 'logistic'
    target_strategy: str = 'dual'  # 'dual', 'buy_only', 'buy', 'sell', 'dual_signal', 'breakout', 'reversal'
    target_params: Dict[str, Any] = None  # optional params for target strategy
    base_models: List[str] = None  # for ensemble
    weights: List[float] = None    # for ensemble
    
    # Equal-weight specific parameters
    ew_correlation_signs: bool = True
    ew_calibrate_prior: bool = True
    ew_activation_threshold: float = 0.0
    ew_temperature: float = 0.75
    ew_buy_aggregation: str = 'fraction'
    ew_sell_aggregation: str = 'any_deactivated'
    
    # Target definition
    buy_threshold: float = 0.02  # 2% gain for buy signal
    sell_threshold: float = 0.02  # 2% loss for sell signal
    forecast_horizon: int = 5   # days to look ahead
    
    # Model parameters
    test_size: float = 0.2
    random_state: int = 42
    class_weight: str = 'balanced'
    penalty: str = 'l1'
    solver: str = 'liblinear'
    max_iter: int = 1000
    C: float = 1.0  # regularization strength for logistic
    
    # Optional hooks (Phase 4)
    calibration: str = 'none'   # 'none', 'platt'|'sigmoid', 'isotonic'
    tuning: str = 'none'        # 'none', 'simple'
    
    # Feature engineering
    ema_periods: List[int] = None
    volume_window: int = 20
    
    def __post_init__(self):
        if self.ema_periods is None:
            # Use shorter EMAs for smaller datasets
            self.ema_periods = [9, 21, 50]  # Removed 200-day EMA

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1% commission
    slippage: float = 0.0005   # 0.05% slippage
    confidence_threshold: float = 0.65  # minimum probability for signal (absolute strategy)
    max_position_size: float = 0.1  # max 10% of capital per position
    # Thresholding strategy: 'absolute' uses confidence_threshold; 'percentile' uses rolling quantile per symbol
    threshold_strategy: str = 'absolute'
    threshold_percentile: float = 0.7  # if percentile strategy, quantile in [0,1]
    threshold_window: int = 60         # rolling window size for percentile strategy
    # Optional fallback if absolute yields zero signals
    auto_threshold_fallback: bool = True
    fallback_percentile: float = 0.7
    fallback_window: int = 60
    # Trade frequency controls
    buy_consecutive_days: int = 1
    sell_consecutive_days: int = 1
    hysteresis_margin: float = 0.0  # require prob difference > margin to avoid flip-flops
    trade_cooldown_days: int = 0    # min days between trades per symbol
    min_holding_days: int = 0       # min days to hold a position before selling (unless forced close)

    
@dataclass
class VisualizationConfig:
    """Visualization settings"""
    figure_size: tuple = (15, 10)
    dpi: int = 300
    style: str = 'seaborn-v0_8'
    color_palette: List[str] = None
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Global configuration instances
db_config = DatabaseConfig()
model_config = ModelConfig()
backtest_config = BacktestConfig()
viz_config = VisualizationConfig()