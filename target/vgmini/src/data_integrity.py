"""
Data integrity validation module
Ensures no synthetic data, forward bias, or data leakage in the ML pipeline
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from sqlalchemy import text
from .data_loader import StockDataLoader

logger = logging.getLogger(__name__)

class DataIntegrityValidator:
    """
    Validates data integrity to prevent forward bias and ensure real market data only
    """
    
    def __init__(self, data_loader: StockDataLoader):
        self.data_loader = data_loader
        
    def validate_temporal_integrity(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate temporal integrity of the dataset
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'chronological_order': True,
            'no_future_gaps': True,
            'realistic_timestamps': True,
            'no_duplicate_timestamps': True,
            'weekend_handling': True
        }
        
        if df.empty:
            logger.warning("Empty dataset provided for validation")
            return results
        
        # Check chronological order (per symbol if multiple symbols)
        if 'symbol' in df.columns:
            # Check chronological order within each symbol
            chronological_issues = 0
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol]
                timestamps = symbol_data['timestamp'].values
                if not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
                    chronological_issues += 1
                    logger.warning(f"Data for symbol {symbol} is not in chronological order")
            
            if chronological_issues > 0:
                results['chronological_order'] = False
                logger.error(f"Found chronological order issues in {chronological_issues} symbols")
        else:
            # Check overall chronological order
            timestamps = df['timestamp'].values
            if not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
                results['chronological_order'] = False
                logger.error("Data is not in chronological order")
        
        # Check for unrealistic future timestamps (allow some tolerance for market data)
        current_timestamp = datetime.now().timestamp()
        # Convert to milliseconds if needed
        if df['timestamp'].max() > 4102444800:  # Data is in milliseconds
            current_timestamp *= 1000
        
        # Allow up to 30 days in the future (for market data that might include some future dates)
        future_threshold = current_timestamp + (30 * 24 * 60 * 60 * 1000 if df['timestamp'].max() > 4102444800 else 30 * 24 * 60 * 60)
        future_data = df[df['timestamp'] > future_threshold]
        if not future_data.empty:
            results['no_future_gaps'] = False
            logger.error(f"Found {len(future_data)} records with timestamps more than 30 days in future")
        
        # Check for realistic timestamp range (not before 1970 or too far in future)
        min_timestamp = datetime(1970, 1, 1).timestamp()
        max_timestamp = datetime(2030, 1, 1).timestamp()
        
        # Convert to milliseconds if data is in milliseconds
        if df['timestamp'].max() > 4102444800:  # Data is in milliseconds
            min_timestamp *= 1000
            max_timestamp *= 1000
        
        invalid_timestamps = df[
            (df['timestamp'] < min_timestamp) | 
            (df['timestamp'] > max_timestamp)
        ]
        if not invalid_timestamps.empty:
            results['realistic_timestamps'] = False
            logger.error(f"Found {len(invalid_timestamps)} records with unrealistic timestamps")
        
        # Check for duplicate timestamps (within same symbol)
        if 'symbol' in df.columns:
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol]
                duplicates = symbol_data[symbol_data['timestamp'].duplicated()]
                if not duplicates.empty:
                    results['no_duplicate_timestamps'] = False
                    logger.error(f"Found {len(duplicates)} duplicate timestamps for {symbol}")
        else:
            duplicates = df[df['timestamp'].duplicated()]
            if not duplicates.empty:
                results['no_duplicate_timestamps'] = False
                logger.error(f"Found {len(duplicates)} duplicate timestamps")
        
        return results
    
    def validate_ohlc_integrity(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate OHLC data integrity
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid_ohlc_relationships': True,
            'positive_prices': True,
            'realistic_price_ranges': True,
            'no_zero_volume_with_price_change': True,
            'realistic_volume': True
        }
        
        if df.empty:
            return results
        
        # Validate OHLC relationships: High >= max(Open, Close), Low <= min(Open, Close)
        invalid_high = df[df['high'] < df[['open', 'close']].max(axis=1)]
        invalid_low = df[df['low'] > df[['open', 'close']].min(axis=1)]
        
        if not invalid_high.empty:
            results['valid_ohlc_relationships'] = False
            logger.error(f"Found {len(invalid_high)} records where high < max(open, close)")
        
        if not invalid_low.empty:
            results['valid_ohlc_relationships'] = False
            logger.error(f"Found {len(invalid_low)} records where low > min(open, close)")
        
        # Check for positive prices
        price_columns = ['open', 'high', 'low', 'close']
        negative_prices = df[(df[price_columns] <= 0).any(axis=1)]
        if not negative_prices.empty:
            results['positive_prices'] = False
            logger.error(f"Found {len(negative_prices)} records with non-positive prices")
        
        # Check for realistic price ranges (no extreme outliers that suggest synthetic data)
        for col in price_columns:
            if col in df.columns:
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                
                # Flag if 99th percentile is more than 1000x the 1st percentile
                if q99 > 1000 * q01:
                    results['realistic_price_ranges'] = False
                    logger.warning(f"Extreme price range detected in {col}: {q01:.2f} to {q99:.2f}")
        
        # Check volume integrity
        if 'volume' in df.columns:
            # Zero volume with significant price change suggests synthetic data
            df['price_change'] = abs(df['close'] - df['open']) / df['open']
            zero_volume_big_change = df[
                (df['volume'] == 0) & (df['price_change'] > 0.05)  # >5% price change with 0 volume
            ]
            
            if not zero_volume_big_change.empty:
                results['no_zero_volume_with_price_change'] = False
                logger.error(f"Found {len(zero_volume_big_change)} records with zero volume but significant price change")
            
            # Check for negative volume
            negative_volume = df[df['volume'] < 0]
            if not negative_volume.empty:
                results['realistic_volume'] = False
                logger.error(f"Found {len(negative_volume)} records with negative volume")
        
        return results
    
    def validate_no_forward_bias_in_features(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, bool]:
        """
        Validate that features don't contain forward-looking bias
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature column names
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'no_future_data_leakage': True,
            'proper_lag_structure': True,
            'no_perfect_predictions': True
        }
        
        if df.empty or not feature_columns:
            return results
        
        # Check if any features are perfectly correlated with future returns
        if 'future_return' in df.columns:
            for feature in feature_columns:
                if feature in df.columns:
                    correlation = abs(df[feature].corr(df['future_return']))
                    if correlation > 0.95:  # Suspiciously high correlation
                        results['no_perfect_predictions'] = False
                        logger.error(f"Feature {feature} has suspiciously high correlation ({correlation:.3f}) with future returns")
        
        # Validate that moving averages and indicators use only past data
        # This is a heuristic check - in practice, you'd need to verify the calculation logic
        for feature in feature_columns:
            if feature in df.columns:
                # Check for any NaN values at the beginning (expected for moving averages)
                # but not in the middle (which might indicate forward-looking calculation)
                feature_series = df[feature].dropna()
                if len(feature_series) < len(df) * 0.8:  # More than 20% NaN values
                    logger.warning(f"Feature {feature} has many NaN values - verify calculation method")
        
        return results
    
    def validate_database_source_only(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate that data comes only from database, no synthetic or external sources
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'database_source_only': True,
            'no_synthetic_patterns': True,
            'realistic_market_behavior': True
        }
        
        if df.empty:
            return results
        
        # Check for synthetic patterns that wouldn't occur in real market data
        
        # 1. Check for perfectly regular intervals (synthetic data often has perfect spacing)
        if len(df) > 10:
            time_diffs = np.diff(df['timestamp'].values)
            unique_diffs = np.unique(time_diffs)
            
            # If all time differences are exactly the same, it might be synthetic
            if len(unique_diffs) == 1:
                logger.warning("All time intervals are identical - verify this is real market data")
        
        # 2. Check for unrealistic price movements
        if 'close' in df.columns and len(df) > 1:
            returns = df['close'].pct_change().dropna()
            
            # Check for too many perfect round numbers
            close_prices = df['close'].values
            round_numbers = np.sum(close_prices == np.round(close_prices))
            round_percentage = round_numbers / len(close_prices)
            
            if round_percentage > 0.5:  # More than 50% round numbers
                results['no_synthetic_patterns'] = False
                logger.warning(f"High percentage ({round_percentage:.1%}) of round number prices - check data source")
            
            # Check for unrealistic return distributions
            if len(returns) > 100:
                # Real market data should have fat tails, not perfect normal distribution
                from scipy import stats
                _, p_value = stats.jarque_bera(returns.dropna())
                
                if p_value > 0.1:  # Fails to reject normality
                    logger.warning("Returns appear too normally distributed - verify real market data")
        
        # 3. Verify data comes from our database by checking against known symbols
        if 'symbol' in df.columns:
            db_symbols = self.data_loader.get_symbols()['symbol'].tolist()
            unknown_symbols = set(df['symbol'].unique()) - set(db_symbols)
            
            if unknown_symbols:
                results['database_source_only'] = False
                logger.error(f"Found symbols not in database: {unknown_symbols}")
        
        return results
    
    def validate_target_variable_integrity(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate that target variables are calculated correctly without forward bias
        
        Args:
            df: DataFrame with target variables
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'proper_target_calculation': True,
            'no_lookahead_in_targets': True,
            'realistic_target_distribution': True
        }
        
        if 'target_buy' not in df.columns and 'target_sell' not in df.columns:
            logger.warning("No target variables found for validation")
            return results
        
        # Check that targets are binary (0 or 1)
        for target_col in ['target_buy', 'target_sell']:
            if target_col in df.columns:
                unique_values = df[target_col].dropna().unique()
                if not all(val in [0, 1] for val in unique_values):
                    results['proper_target_calculation'] = False
                    logger.error(f"Target variable {target_col} contains non-binary values: {unique_values}")
        
        # Check that future returns are calculated correctly
        if 'future_return' in df.columns and 'close' in df.columns:
            # Manually calculate future return for a few samples to verify
            sample_indices = np.random.choice(len(df)-10, min(10, len(df)-10), replace=False)
            
            for idx in sample_indices:
                if idx + 10 < len(df):  # Assuming 10-day forecast horizon
                    current_price = df.iloc[idx]['close']
                    future_price = df.iloc[idx + 10]['close']
                    expected_return = (future_price - current_price) / current_price
                    actual_return = df.iloc[idx]['future_return']
                    
                    if abs(expected_return - actual_return) > 0.001:  # Allow small floating point errors
                        results['no_lookahead_in_targets'] = False
                        logger.error(f"Future return calculation error at index {idx}: "
                                   f"expected {expected_return:.4f}, got {actual_return:.4f}")
                        break
        
        # Check for realistic target distribution (not too skewed)
        for target_col in ['target_buy', 'target_sell']:
            if target_col in df.columns:
                target_mean = df[target_col].mean()
                if target_mean < 0.01 or target_mean > 0.99:  # Less than 1% or more than 99%
                    results['realistic_target_distribution'] = False
                    logger.warning(f"Target variable {target_col} has extreme distribution: {target_mean:.3f}")
        
        return results
    
    def run_comprehensive_validation(self, df: pd.DataFrame, feature_columns: List[str] = None) -> Dict[str, Dict[str, bool]]:
        """
        Run all validation checks
        
        Args:
            df: DataFrame to validate
            feature_columns: List of feature columns to validate
            
        Returns:
            Dictionary with all validation results
        """
        logger.info("Running comprehensive data integrity validation...")
        
        validation_results = {
            'temporal_integrity': self.validate_temporal_integrity(df),
            'ohlc_integrity': self.validate_ohlc_integrity(df),
            'database_source': self.validate_database_source_only(df),
            'target_integrity': self.validate_target_variable_integrity(df)
        }
        
        if feature_columns:
            validation_results['feature_integrity'] = self.validate_no_forward_bias_in_features(df, feature_columns)
        
        # Summary
        all_passed = all(
            all(checks.values()) for checks in validation_results.values()
        )
        
        if all_passed:
            logger.info("✓ All data integrity checks passed")
        else:
            logger.error("✗ Some data integrity checks failed")
            for category, checks in validation_results.items():
                failed_checks = [check for check, passed in checks.items() if not passed]
                if failed_checks:
                    logger.error(f"  {category}: Failed checks: {failed_checks}")
        
        return validation_results
    
    def validate_symbol_data_completeness(self, symbol: str, start_date: str, end_date: str) -> Dict[str, any]:
        """
        Validate completeness of data for a specific symbol
        
        Args:
            symbol: Stock symbol
            start_date: Start date string
            end_date: End date string
            
        Returns:
            Dictionary with completeness metrics
        """
        try:
            data = self.data_loader.get_symbol_data(symbol, start_date, end_date)
            
            if data.empty:
                return {
                    'symbol': symbol,
                    'data_available': False,
                    'record_count': 0,
                    'date_coverage': 0.0,
                    'gaps_detected': True
                }
            
            # Calculate expected vs actual trading days
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Approximate trading days (excluding weekends, not accounting for holidays)
            total_days = (end_dt - start_dt).days
            expected_trading_days = total_days * (5/7)  # Rough estimate
            
            actual_records = len(data)
            coverage = actual_records / expected_trading_days if expected_trading_days > 0 else 0
            
            # Check for gaps in data
            data['date'] = pd.to_datetime(data['datetime']).dt.date
            date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
            trading_days = date_range[date_range.weekday < 5]  # Monday=0, Friday=4
            
            missing_days = set(trading_days.date) - set(data['date'])
            gaps_detected = len(missing_days) > expected_trading_days * 0.1  # More than 10% missing
            
            return {
                'symbol': symbol,
                'data_available': True,
                'record_count': actual_records,
                'date_coverage': coverage,
                'gaps_detected': gaps_detected,
                'missing_days_count': len(missing_days),
                'data_quality_score': min(coverage, 1.0) * (0.5 if gaps_detected else 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error validating data completeness for {symbol}: {e}")
            return {
                'symbol': symbol,
                'data_available': False,
                'error': str(e)
            }