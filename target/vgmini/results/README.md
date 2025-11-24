# Stock Trading ML Framework

A comprehensive machine learning framework for stock market analysis, signal generation, and backtesting. This framework implements the technical analysis features from your codex and the ML approach discussed in your conversation with DeepSeek.

## Features

- **Technical Indicator Engine**: Implements crossover signals, MACD, EMA, Heikin-Ashi, and other indicators from `codex.txt`
- **Dual ML Models**: Separate logistic regression models for buy and sell signals as discussed in `convo.txt`
- **Comprehensive Backtesting**: Full backtesting engine with performance metrics
- **SHAP Analysis**: Model interpretability with SHAP feature importance
- **Rich Visualizations**: Interactive charts, performance dashboards, and signal analysis
- **PostgreSQL Integration**: Direct connection to your stock database
- **Extensible Architecture**: Easy to add new features, models, and analysis methods

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PostgreSQL    │───▶│   Data Loader    │───▶│ Feature Engine  │
│   Database      │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Visualization   │◀───│  Main Framework  │◀───│   ML Models     │
│   Engine        │    │                  │    │ (Buy/Sell)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌──────────────────┐
                       │ Backtest Engine  │
                       └──────────────────┘
```

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables (create .env file)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stock_data
DB_USER=postgres
DB_PASSWORD=your_password
```

### 2. Basic Usage

```python
from main_framework import TradingFramework

# Initialize framework
framework = TradingFramework()

# Run experiment
results = framework.run_experiment(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date="2023-01-01",
    end_date="2024-01-01",
    experiment_name="my_first_experiment"
)

# View results
print(f"Total Return: {results['backtest_results']['aggregate']['avg_total_return']:.2%}")
framework.close()
```

### 3. Run Example Experiments

```bash
python example_usage.py
```

## Configuration

### Model Configuration (`config.py`)

```python
@dataclass
class ModelConfig:
    buy_threshold: float = 0.02      # 2% gain for buy signal success
    sell_threshold: float = 0.03     # 3% loss for sell signal success  
    forecast_horizon: int = 10       # Days to look ahead
    ema_periods: List[int] = [9, 30, 50, 200]  # EMA periods
    # ... other parameters
```

### Backtest Configuration

```python
@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    commission: float = 0.001        # 0.1% commission
    confidence_threshold: float = 0.65  # Min probability for signals
    max_position_size: float = 0.1   # Max 10% per position
```

## Features from Codex.txt

The framework implements all technical indicators from your codex:

### Price vs Moving Averages
- `price_above_ema9`: Low price above 9-period EMA
- `price_above_ema30`: Low price above 30-period EMA  
- `ema9_above_ema30`: 9-period EMA above 30-period EMA

### MACD Crossovers
- `macd_above_signal`: MACD line above signal line
- `macd_above_zero`: MACD above zero line
- `macd_positive_hist`: MACD histogram positive

### Additional Signals
- `ha_uptrend`: Heikin-Ashi uptrend signal
- `volume_above_average`: Volume above 20-day average
- `price_momentum_strong`: Strong price momentum indicator

## ML Approach (from convo.txt)

### Dual Model Architecture
- **Buy Model**: Predicts probability of >10 days positive growth
- **Sell Model**: Predicts probability of >10 days negative growth (-3%)

### Target Definition
```python
# Buy signal success: future return > buy_threshold (2%)
target_buy = (future_return > 0.02).astype(int)

# Sell signal success: future return < -sell_threshold (-3%)  
target_sell = (future_return < -0.03).astype(int)
```

### Signal Generation
```python
# Generate signals only when:
# 1. Probability > confidence_threshold (65%)
# 2. One model significantly outweighs the other
buy_signal = (buy_prob > 0.65) & (buy_prob > sell_prob)
sell_signal = (sell_prob > 0.65) & (sell_prob > buy_prob)
```

## Database Schema

The framework expects your PostgreSQL database to have these tables:

```sql
-- Symbols table
symbols (id, symbol, name, created_at, updated_at)

-- OHLC data table  
ohlc_data (id, symbol_id, timestamp, open, high, low, close, volume, interval, is_final)

-- Optional metadata tables
ohlc_metadata (ohlc_id, source_time, data_quality, has_gap)
data_gaps (id, symbol_id, interval, start_time, end_time, expected_points, actual_points)
```

## Output Files

### Models
- `models/trained_model_YYYYMMDD_HHMMSS.pkl`: Trained ML models

### Visualizations  
- `visualizations/{experiment_name}/feature_importance.png`: Feature importance charts
- `visualizations/{experiment_name}/{symbol}_performance.png`: Backtest performance
- `visualizations/{experiment_name}/{symbol}_signals.html`: Interactive signal charts
- `visualizations/{experiment_name}/{symbol}_dashboard.png`: Performance dashboard

### Results
- `experiment_results/{experiment_name}_YYYYMMDD_HHMMSS.json`: Complete results

### SHAP Analysis
- `visualizations/shap_buy_model.png`: SHAP importance for buy model
- `visualizations/shap_sell_model.png`: SHAP importance for sell model

## Key Results and Metrics

### Performance Metrics
- **Total Return**: Portfolio return vs buy-and-hold
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Trade Length**: Days held per trade

### Model Metrics  
- **AUC Score**: Model discrimination ability
- **Feature Importance**: Logistic regression coefficients
- **SHAP Values**: Feature contribution to individual predictions

### Trade Analysis
- **Signal Summary**: Total buy/sell signals generated
- **Trade Distribution**: Histogram of trade returns
- **Monthly Returns**: Performance by month/year

## Customization

### Adding New Features

```python
# In feature_engineering.py
def calculate_custom_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
    df['my_indicator'] = your_calculation_here
    return df

# Add to crossover signals
def create_crossover_signals(self, df: pd.DataFrame) -> pd.DataFrame:
    # ... existing signals ...
    df['my_signal'] = (df['my_indicator'] > threshold).astype(int)
    return df
```

### Custom Models

```python
# In ml_models.py  
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(self, X, y):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model
```

### Custom Backtesting Rules

```python
# In backtesting.py
def custom_position_sizing(self, portfolio_value, signal_strength):
    # Implement your position sizing logic
    return position_size
```

## Example Experiments

### 1. Basic Experiment
```python
python -c "from example_usage import basic_experiment; basic_experiment()"
```

### 2. Custom Configuration
```python
from main_framework import TradingFramework

config_overrides = {
    'buy_threshold': 0.03,
    'sell_threshold': 0.025,
    'forecast_horizon': 15
}

framework = TradingFramework(config_overrides)
results = framework.run_experiment(['AAPL'], "2023-01-01", "2024-01-01", "custom_test")
```

### 3. Feature Analysis
```python
# Focus on model interpretability
python -c "from example_usage import feature_analysis_experiment; feature_analysis_experiment()"
```

## Troubleshooting

### Common Issues

1. **Database Connection**: Check your `.env` file and database credentials
2. **Insufficient Data**: Ensure symbols have enough historical data (>200 days recommended)
3. **Memory Issues**: Reduce the number of symbols or date range for large experiments
4. **Missing Features**: Some features require minimum data periods (e.g., 200-day EMA)

### Logging

Check `trading_framework.log` for detailed execution logs:

```python
import logging
logging.basicConfig(level=logging.DEBUG)  # For verbose logging
```

## Performance Optimization

### For Large Datasets
- Use date ranges to limit data size
- Process symbols in batches
- Consider using sample data for SHAP analysis
- Enable database indexing on `symbol_id` and `timestamp`

### Memory Management
- The framework automatically handles feature calculation in chunks
- SHAP analysis uses sampling for performance
- Visualizations are created incrementally

## Contributing

To extend the framework:

1. **New Indicators**: Add to `feature_engineering.py`
2. **New Models**: Add to `ml_models.py`  
3. **New Visualizations**: Add to `visualization.py`
4. **New Backtesting Rules**: Add to `backtesting.py`

## License

This framework is provided as-is for educational and research purposes.

## Support

For issues or questions:
1. Check the logs in `trading_framework.log`
2. Review the example experiments in `example_usage.py`
3. Ensure your database schema matches the expected format