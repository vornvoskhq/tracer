# Trading Framework Architecture Documentation

## Overview

This document provides a comprehensive overview of the trading framework's architecture, detailing each source file, class, function, and their relationships with configuration parameters and output structures.

## Source Directory Structure

### Core Framework Files

#### `src/main_framework.py` - Central Orchestration Engine

**Purpose**: Primary orchestrator that coordinates all framework components through a lazy-loading architecture.

**Key Classes**:

**TradingFramework Class**
- **Purpose**: Main entry point that manages the complete trading experiment lifecycle
- **Configuration Handling**: Accepts config_overrides dictionary to customize model_config and backtest_config parameters
- **Data Structures**: Maintains lazy-loaded component references (_data_loader, _feature_engineer, _ml_models, _viz_engine)
- **Output Paths**: Creates and manages experiment result directories under `results/{experiment_name}/`

**Key Methods**:
- **run_experiment()**: Orchestrates the complete experiment pipeline
  - **Input Parameters**: symbols list, start_date, end_date, experiment_name
  - **Configuration Dependencies**: Uses global model_config and backtest_config
  - **Output Structure**: Returns experiment_results dictionary containing backtest_results, model_results, shap_analysis
  - **File Outputs**: 
    - `results/{experiment_name}/experiment_results.json`
    - `results/{experiment_name}/models/trained_model.pkl`
    - `results/{experiment_name}/visualizations/*.png`
    - `results/{experiment_name}/analysis/*.png`

- **_apply_config_overrides()**: Modifies global configuration objects
  - **Parameters Handled**: buy_threshold, sell_threshold, forecast_horizon, confidence_threshold, max_position_size, commission, initial_capital
  - **Data Flow**: Updates model_config and backtest_config attributes dynamically

**Property-Based Lazy Loading**:
- **data_loader**: Initializes DataLoader for market data retrieval
- **feature_engineer**: Creates FeatureEngineer with custom configuration support
- **ml_models**: Instantiates StockSignalPredictor for model training
- **viz_engine**: Sets up VisualizationEngine for chart generation

#### `src/ml_models.py` - Machine Learning Core

**Purpose**: Implements dual logistic regression models for buy/sell signal prediction with comprehensive model evaluation and interpretability.

**Key Classes**:

**StockSignalPredictor Class**
- **Purpose**: Manages training and prediction for separate buy and sell models
- **Configuration Dependencies**: Uses model_config for penalty, solver, class_weight, max_iter, random_state parameters
- **Data Structures**: Maintains buy_model, sell_model, scaler, feature_names, and explainer objects

**Key Methods**:
- **prepare_data()**: Processes feature matrix and target vectors
  - **Input Structure**: DataFrame with feature columns and target_buy/target_sell columns
  - **Output Structure**: Returns X (feature matrix), y_buy (buy targets), y_sell (sell targets)
  - **Data Validation**: Removes rows with missing values, logs signal distribution statistics

- **train_models()**: Executes model training with chronological data splitting
  - **Configuration Parameters**: test_size, penalty, solver, class_weight, max_iter, random_state
  - **Model Selection Logic**: Uses DummyClassifier when insufficient signal diversity detected
  - **Output Structure**: Returns training results dictionary with train_auc, test_auc, feature_importance
  - **File Output**: Saves models to `results/{experiment_name}/models/trained_model.pkl`

- **predict_signals()**: Generates trading signals with confidence filtering
  - **Input Parameters**: Feature matrix X, confidence_threshold
  - **Configuration Dependency**: Uses confidence_threshold from config
  - **Output Structure**: Dictionary containing buy_probability, sell_probability, buy_signals, sell_signals, signal_strength, signal_direction
  - **Signal Logic**: Requires probability > confidence_threshold AND relative strength comparison

- **get_comprehensive_feature_importance()**: Analyzes feature importance using multiple methods
  - **Methods Used**: Linear coefficients, permutation importance, LIME analysis
  - **Output Structure**: Nested dictionary with buy_model and sell_model importance analyses
  - **File Dependencies**: Integrates with ModelInterpreter for LIME-based analysis

**AlternativeModels Class**
- **Purpose**: Provides alternative ML algorithms for comparison
- **Methods**: train_lasso_regression(), train_random_forest()
- **Configuration**: Uses model_config.random_state for reproducibility

#### `src/feature_engineering.py` - Feature Generation Engine

**Purpose**: Transforms raw market data into engineered features for ML model training, with support for asymmetric forecast horizons.

**Key Classes**:

**FeatureEngineer Class**
- **Purpose**: Comprehensive feature engineering pipeline with configurable technical indicators
- **Configuration Handling**: Supports both traditional config objects and nested dictionary configurations
- **Data Structures**: Processes OHLCV data and generates 20+ technical features

**Key Methods**:
- **process_features()**: Main feature engineering pipeline
  - **Input Structure**: DataFrame with OHLCV columns (open, high, low, close, volume, timestamp)
  - **Configuration Dependencies**: ema_periods, volume_window, technical_indicators settings
  - **Output Structure**: Enhanced DataFrame with engineered features and target variables
  - **Feature Categories**: Price features, EMA features, MACD features, volume features, momentum features

- **create_target_variables()**: Generates buy/sell targets with asymmetric horizon support
  - **Asymmetric Logic**: Supports different forecast horizons for buy vs sell signals
  - **Configuration Parameters**: buy_forecast_horizon, sell_forecast_horizon, buy_threshold, sell_threshold
  - **Data Validation**: Ensures temporal integrity, prevents lookahead bias
  - **Output Structure**: Adds target_buy, target_sell, target_return columns

- **calculate_technical_indicators()**: Computes technical analysis indicators
  - **RSI Configuration**: technical_indicators.rsi.period, oversold_threshold, overbought_threshold
  - **MACD Configuration**: technical_indicators.macd.fast_period, slow_period, signal_period
  - **Momentum Configuration**: technical_indicators.momentum.window, std_window, std_multiplier
  - **Output Features**: rsi, rsi_oversold, rsi_overbought, macd, macd_signal, macd_hist

- **get_feature_columns()**: Manages feature selection based on configuration
  - **Configuration Parameter**: enabled_features list
  - **Feature Grouping**: Organizes features by category (price_features, ema_features, etc.)
  - **Dynamic Features**: Generates ROC features based on technical_indicators.roc_periods
  - **Output Structure**: Dictionary mapping feature categories to feature name lists

**Configuration Helper Methods**:
- **_get_config_value()**: Safely navigates nested configuration dictionaries
  - **Path Navigation**: Supports dot notation for nested access (e.g., "technical_indicators.rsi.period")
  - **Fallback Logic**: Returns default values when configuration paths don't exist

#### `src/backtesting.py` - Trading Simulation Engine

**Purpose**: Simulates trading strategies with realistic transaction costs and position management.

**Key Classes**:

**BacktestEngine Class**
- **Purpose**: Executes trading simulations with comprehensive performance metrics
- **Configuration Dependencies**: initial_capital, commission, max_position_size from backtest_config
- **Data Structures**: Maintains portfolio state, trade log, and performance metrics

**Key Methods**:
- **run_backtest()**: Executes complete trading simulation
  - **Input Structure**: DataFrame with signals, prices, and timestamps
  - **Configuration Parameters**: initial_capital, commission, max_position_size, confidence_threshold
  - **Output Structure**: Dictionary containing total_return, sharpe_ratio, max_drawdown, volatility, win_rate, total_trades, trade_log
  - **File Output**: Performance data integrated into experiment results

- **calculate_performance_metrics()**: Computes comprehensive trading statistics
  - **Metrics Calculated**: Total return, annualized return, Sharpe ratio, maximum drawdown, volatility, win rate
  - **Configuration Dependency**: trading_days_per_year from analysis config (default 252)
  - **Data Structures**: Processes daily returns, drawdown series, trade outcomes

- **execute_trade()**: Processes individual trading transactions
  - **Position Management**: Enforces max_position_size limits
  - **Cost Calculation**: Applies commission fees to all transactions
  - **Trade Logging**: Records entry/exit prices, quantities, timestamps, profit/loss

#### `src/visualization.py` - Chart Generation Engine

**Purpose**: Creates comprehensive visualizations for trading analysis and model interpretability.

**Key Classes**:

**VisualizationEngine Class**
- **Purpose**: Generates trading charts, performance dashboards, and model analysis plots
- **Configuration Dependencies**: visualization.dpi, visualization.figure_size
- **Output Paths**: Saves charts to `results/{experiment_name}/visualizations/`

**Key Methods**:
- **create_trading_dashboard()**: Generates comprehensive trading analysis dashboard
  - **Input Structure**: DataFrame with prices, signals, portfolio values
  - **Chart Components**: Price charts with signals, portfolio performance, drawdown analysis
  - **File Output**: `results/{experiment_name}/visualizations/{symbol}_dashboard.png`
  - **Configuration**: Uses visualization.dpi for image quality

- **plot_performance_comparison()**: Creates performance comparison charts
  - **Input Structure**: Multiple backtest results for comparison
  - **Metrics Visualized**: Returns, Sharpe ratios, drawdowns across strategies
  - **File Output**: `results/{experiment_name}/visualizations/{symbol}_performance.png`

- **plot_shap_summary()**: Generates model interpretability plots
  - **Input Structure**: SHAP values, feature names, feature matrix
  - **Fallback Logic**: Uses LIME-based plots when SHAP unavailable
  - **File Output**: `results/{experiment_name}/analysis/interpretability_{model_type}_model.png`
  - **Configuration**: Handles visualization.figure_size for plot dimensions

- **create_signal_analysis()**: Analyzes signal quality and distribution
  - **Signal Metrics**: Signal frequency, accuracy, timing analysis
  - **Interactive Output**: `results/{experiment_name}/visualizations/{symbol}_signals.html`

#### `src/data_loader.py` - Market Data Management

**Purpose**: Handles market data retrieval, caching, and validation with multiple data source support.

**Key Classes**:

**DataLoader Class**
- **Purpose**: Manages market data acquisition with intelligent caching
- **Data Sources**: Yahoo Finance (yfinance), with extensible architecture for additional sources
- **Caching Strategy**: Implements file-based caching to minimize API calls

**Key Methods**:
- **load_data()**: Primary data retrieval interface
  - **Input Parameters**: symbol, start_date, end_date
  - **Output Structure**: DataFrame with OHLCV columns (open, high, low, close, volume, timestamp)
  - **Caching Logic**: Checks for existing data, fetches missing periods incrementally
  - **Data Validation**: Ensures data quality, removes invalid records

- **get_multiple_symbols()**: Batch data loading for multiple symbols
  - **Input Structure**: List of symbols, date range parameters
  - **Output Structure**: Dictionary mapping symbols to OHLCV DataFrames
  - **Error Handling**: Continues processing if individual symbols fail

- **validate_data()**: Ensures data quality and completeness
  - **Validation Checks**: Missing values, data gaps, price anomalies
  - **Output Structure**: Cleaned DataFrame with validation warnings logged

#### `src/data_integrity.py` - Data Quality Assurance

**Purpose**: Comprehensive data validation and quality assurance for trading data.

**Key Classes**:

**DataIntegrityValidator Class**
- **Purpose**: Validates data quality across multiple dimensions
- **Validation Categories**: Completeness, consistency, accuracy, temporal integrity

**Key Methods**:
- **run_comprehensive_validation()**: Executes complete data quality assessment
  - **Input Structure**: DataFrame with market data and features
  - **Validation Checks**: Missing values, outliers, temporal consistency, feature correlations
  - **Configuration Parameter**: analysis.max_correlation_threshold for correlation validation
  - **Output Structure**: Validation report with warnings and recommendations

#### `src/model_interpretability.py` - Model Analysis Engine

**Purpose**: Provides model interpretability through SHAP and LIME analysis for understanding feature importance.

**Key Classes**:

**ModelInterpreter Class**
- **Purpose**: Generates model explanations using multiple interpretability methods
- **Methods Supported**: Linear coefficients, permutation importance, LIME analysis
- **Configuration Dependencies**: analysis.lime_sample_size, analysis.lime_max_samples

**SHAPCompatibilityWrapper Class**
- **Purpose**: Provides SHAP-like interface using LIME for Python 3.12 compatibility
- **Fallback Strategy**: Uses LIME when SHAP unavailable, maintains consistent API
- **Configuration Parameters**: Uses analysis.lime_max_samples for explanation generation

**Key Methods**:
- **get_feature_importance_summary()**: Comprehensive feature importance analysis
  - **Methods Used**: Linear coefficients, permutation importance, LIME explanations
  - **Configuration**: analysis.lime_sample_size controls sample size for analysis
  - **Output Structure**: Dictionary with multiple importance measures per feature

#### `src/config.py` - Configuration Management

**Purpose**: Centralized configuration management with default values and validation.

**Configuration Objects**:

**ModelConfig Class**
- **Parameters**: buy_threshold, sell_threshold, forecast_horizon, confidence_threshold, test_size, class_weight, penalty, solver, max_iter, tol, random_state
- **Default Values**: Provides sensible defaults for all ML model parameters
- **Validation**: Ensures parameter consistency and valid ranges

**BacktestConfig Class**
- **Parameters**: initial_capital, commission, max_position_size, trading_days_per_year
- **Purpose**: Controls backtesting simulation parameters
- **Default Values**: $100,000 capital, 0.1% commission, 10% max position size

#### `src/experiment_configs.py` - Experiment Configuration Framework

**Purpose**: Defines experiment configuration structure with support for complex nested configurations.

**Key Classes**:

**ExperimentConfig Class**
- **Purpose**: Complete experiment specification with all parameters
- **Configuration Sections**: Data selection, model parameters, feature engineering, backtesting, technical indicators, analysis, visualization
- **Serialization**: Supports JSON serialization for experiment persistence
- **Validation**: Ensures configuration consistency and completeness

**TechnicalIndicatorConfig Class**
- **Parameters**: RSI settings (period, thresholds), MACD settings (periods), Bollinger Bands, momentum parameters
- **Purpose**: Configures all technical indicator calculations
- **Default Values**: Standard technical analysis parameters

**AnalysisConfig Class**
- **Parameters**: interpretability_sample_size, lime_sample_size, correlation_threshold, trading_days_per_year
- **Purpose**: Controls analysis and interpretability settings

**VisualizationConfig Class**
- **Parameters**: dpi, figure_size
- **Purpose**: Controls chart generation quality and dimensions

## Data Flow Architecture

### Input Data Structure
- **Market Data**: OHLCV format with timestamp indexing
- **Configuration**: Nested dictionary structure supporting all framework parameters
- **Symbols**: List of ticker symbols for multi-asset analysis

### Processing Pipeline
1. **Data Loading**: Raw market data retrieval and caching
2. **Feature Engineering**: Technical indicator calculation and feature generation
3. **Model Training**: Dual model training with chronological data splitting
4. **Signal Generation**: Probability-based signal generation with confidence filtering
5. **Backtesting**: Trading simulation with realistic transaction costs
6. **Analysis**: Performance metrics and model interpretability
7. **Visualization**: Comprehensive chart generation and dashboard creation

### Output Structure
- **Experiment Results**: JSON format with complete experiment metadata
- **Model Artifacts**: Serialized models and scalers for future use
- **Visualizations**: PNG charts and HTML interactive plots
- **Analysis Reports**: Feature importance and model interpretability data

## Configuration Parameter Mapping

### Model Training Parameters
- **buy_threshold, sell_threshold**: Target return thresholds for signal labeling
- **forecast_horizon**: Days ahead for target calculation
- **buy_forecast_horizon, sell_forecast_horizon**: Asymmetric horizon support
- **confidence_threshold**: Minimum probability for signal generation
- **test_size**: Train/test split ratio
- **penalty, solver, max_iter, tol**: Logistic regression hyperparameters

### Feature Engineering Parameters
- **ema_periods**: EMA calculation periods
- **technical_indicators**: Nested configuration for all technical indicators
- **enabled_features**: List of features to include in model training
- **volume_window**: Rolling window for volume calculations

### Backtesting Parameters
- **initial_capital**: Starting portfolio value
- **commission**: Transaction cost percentage
- **max_position_size**: Maximum position size as portfolio percentage

### Analysis Parameters
- **interpretability_sample_size**: Sample size for SHAP analysis
- **lime_sample_size**: Sample size for LIME explanations
- **rolling_sharpe_window**: Window for rolling Sharpe ratio calculation

This architecture provides a comprehensive, configurable framework for systematic trading strategy development and evaluation with full transparency into model behavior and performance characteristics.