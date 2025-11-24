#!/usr/bin/env python3
"""
Debug script to analyze feature calculations and equal-weight model behavior
Tests a few specific symbols to understand why all scores are identical
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from src.data_loader import StockDataLoader
from src.feature_engineering import FeatureEngineer
from src.ml_models import StockSignalPredictor, EqualWeightSignalClassifier
from src.yaml_config import load_merged_experiment_yaml
from src.experiment_configs import ExperimentConfig

def analyze_symbol_features(symbol, feature_data, feature_names):
    """Analyze features for a specific symbol"""
    print(f"\n{'='*60}")
    print(f"SYMBOL: {symbol}")
    print(f"{'='*60}")
    
    if len(feature_data) == 0:
        print("❌ No feature data available")
        return
        
    # Get latest feature values
    latest_features = feature_data[-1]  # Most recent row
    
    print(f"📊 Feature Analysis (Latest Values):")
    print(f"{'Feature':<30} {'Value':<10} {'Activated':<10}")
    print("-" * 50)
    
    activated_count = 0
    total_features = len(latest_features)
    
    for i, (feature_name, value) in enumerate(zip(feature_names, latest_features)):
        activated = "✅ YES" if value > 0.0 else "❌ NO"
        if value > 0.0:
            activated_count += 1
            
        print(f"{feature_name:<30} {value:<10.3f} {activated:<10}")
    
    activation_rate = activated_count / total_features if total_features > 0 else 0
    print(f"\n📈 Summary:")
    print(f"   Total Features: {total_features}")
    print(f"   Activated: {activated_count}")
    print(f"   Activation Rate: {activation_rate:.1%}")
    print(f"   Expected Probability: ~{activation_rate:.3f}")
    
    return activation_rate

def test_equal_weight_model():
    """Test the equal weight model on a few symbols with detailed diagnostics"""
    
    print("🔍 VGMini Equal-Weight Model Debug Analysis")
    print("=" * 80)
    
    # Load experiment config
    try:
        config_data = load_merged_experiment_yaml('equal_weight_dual_enhanced')
        config = ExperimentConfig.from_dict(config_data)
        print(f"✅ Loaded config for: {config.name}")
        print(f"   Features: {len(config.enabled_features)} enabled")
        print(f"   Confidence threshold: {config_data.get('confidence_threshold', 'Not set')}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    # Test symbols - mix of different types
    test_symbols = ['AAPL', 'SPY', 'QQQ', 'XLE', 'TSLA']
    print(f"\n🎯 Testing symbols: {test_symbols}")
    
    # Initialize components
    data_loader = StockDataLoader()
    feature_engineer = FeatureEngineer()
    
    all_features = []
    all_symbols = []
    activation_rates = {}
    
    print(f"\n📊 Loading and processing symbol data...")
    
    # Get symbol IDs first
    symbols_df = data_loader.get_symbols()
    print(f"   Available symbols in DB: {len(symbols_df)}")
    
    for symbol in test_symbols:
        try:
            print(f"\n⏳ Processing {symbol}...")
            
            # Get symbol ID
            symbol_info = symbols_df[symbols_df['symbol'] == symbol]
            if len(symbol_info) == 0:
                print(f"❌ Symbol {symbol} not found in database")
                continue
                
            symbol_id = symbol_info.iloc[0]['id']
            
            # Load OHLC data for symbol (match ranking function exactly)
            start_date_obj = pd.to_datetime(config.start_date).date()
            end_date_obj = pd.to_datetime(config.end_date).date() 
            
            data = data_loader.get_ohlc_data([symbol_id], start_date=start_date_obj.isoformat(), end_date=end_date_obj.isoformat())
            if data is None or len(data) < 50:
                print(f"❌ Insufficient data for {symbol} ({len(data) if data is not None else 0} records)")
                continue
                
            print(f"   Loaded {len(data)} records")
            
            # Process exactly like ranking function does
            df = data_loader.validate_data_quality(data)
            if len(df) < 50:
                print(f"❌ Insufficient data after cleaning for {symbol}")
                continue
            
            # Engineer features exactly like ranking function 
            fdf = df.copy()
            fdf = feature_engineer.calculate_ema(fdf, [9, 30])  # Use default EMA periods
            fdf = feature_engineer.calculate_macd(fdf)
            fdf = feature_engineer.calculate_heikin_ashi(fdf)
            fdf = feature_engineer.calculate_volume_indicators(fdf)
            fdf = feature_engineer.calculate_additional_indicators(fdf)
            fdf = feature_engineer.create_crossover_signals(fdf)
            fdf = feature_engineer.create_structure_signals(fdf)  # Add market structure features
            
            print(f"   Engineered features: {len(fdf)} records")
            
            # Take the last row as current state (like ranking does)
            if len(fdf) == 0:
                print(f"❌ No features after engineering for {symbol}")
                continue
                
            last_row = fdf.iloc[-1]  # Most recent feature values
            
            # Filter to enabled features only
            enabled_feature_cols = [col for col in config.enabled_features if col in fdf.columns]
            if len(enabled_feature_cols) == 0:
                print(f"❌ No enabled features found for {symbol}")
                print(f"   Available: {[col for col in fdf.columns if 'target' not in col]}")
                continue
                
            feature_values = [last_row[col] for col in enabled_feature_cols]
            activation_rate = analyze_symbol_features(symbol, [feature_values], enabled_feature_cols)
            
            activation_rates[symbol] = activation_rate
            all_features.append(feature_values)
            all_symbols.append(symbol)
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            continue
    
    if len(all_features) == 0:
        print("\n❌ No feature data collected!")
        return
    
    print(f"\n🧮 EQUAL-WEIGHT MODEL TESTING")
    print("=" * 60)
    
    # Convert to numpy arrays
    X = np.array(all_features)
    print(f"📊 Combined dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Create and test equal-weight model
    model_config = config_data.get('model', {})
    ew_model = EqualWeightSignalClassifier(
        mode='buy',
        correlation_signs=model_config.get('ew_correlation_signs', True),
        activation_threshold=model_config.get('ew_activation_threshold', 0.0),
        temperature=model_config.get('ew_temperature', 1.0),
        calibrate_prior=model_config.get('ew_calibrate_prior', True),
        aggregation=model_config.get('ew_buy_aggregation', 'fraction')
    )
    
    print(f"\n⚙️  Model Configuration:")
    print(f"   Correlation signs: {ew_model.correlation_signs}")
    print(f"   Activation threshold: {ew_model.activation_threshold}")
    print(f"   Temperature: {ew_model.temperature}")
    print(f"   Calibrate prior: {ew_model.calibrate_prior}")
    print(f"   Aggregation: {ew_model.aggregation}")
    
    # Fit model (no targets needed for equal-weight)
    ew_model.fit(X, None)
    print(f"\n✅ Model fitted")
    print(f"   Signs: {ew_model.signs_}")
    print(f"   Intercept: {ew_model.intercept_}")
    
    # Test predictions on latest data for each symbol
    print(f"\n🎯 SYMBOL PREDICTIONS")
    print("=" * 60)
    
    for symbol in activation_rates.keys():
        try:
            # Get latest features for this symbol
            symbol_mask = np.array(all_symbols) == symbol
            symbol_features = X[symbol_mask]
            
            if len(symbol_features) == 0:
                continue
                
            latest_features = symbol_features[-1:] # Most recent row
            
            # Get prediction
            proba = ew_model.predict_proba(latest_features)
            buy_prob = proba[0, 1]  # Probability of class 1 (buy)
            
            print(f"\n{symbol}:")
            print(f"   Activation rate: {activation_rates[symbol]:.1%}")
            print(f"   Model probability: {buy_prob:.3f}")
            print(f"   Expected vs Actual: {activation_rates[symbol]:.3f} vs {buy_prob:.3f}")
            print(f"   Difference: {abs(activation_rates[symbol] - buy_prob):.3f}")
            
        except Exception as e:
            print(f"❌ Error predicting {symbol}: {e}")
    
    print(f"\n🔍 DIAGNOSIS")
    print("=" * 60)
    
    unique_rates = set(activation_rates.values())
    if len(unique_rates) == 1:
        print("❌ ALL SYMBOLS HAVE IDENTICAL ACTIVATION RATES!")
        print(f"   Rate: {list(unique_rates)[0]:.1%}")
        print("   This suggests market-wide bearish conditions OR feature calculation issues")
    else:
        print("✅ Symbols show different activation rates")
        print("   This suggests the model should produce different scores")
    
    # Check feature variance
    feature_vars = np.var(X, axis=0)
    print(f"\n📊 Feature Variance Analysis:")
    zero_var_features = np.sum(feature_vars == 0)
    print(f"   Features with zero variance: {zero_var_features}/{len(feature_vars)}")
    print(f"   Average feature variance: {np.mean(feature_vars):.6f}")
    
    if zero_var_features > len(feature_vars) * 0.5:
        print("❌ Too many features have zero variance - they're constant across all data!")
    
    return activation_rates

if __name__ == "__main__":
    try:
        activation_rates = test_equal_weight_model()
        print(f"\n✅ Debug analysis completed!")
    except Exception as e:
        print(f"\n❌ Error in debug analysis: {e}")
        import traceback
        traceback.print_exc()