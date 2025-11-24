"""
Feature engineering module for technical indicators and crossover signals
Based on the features defined in codex.txt
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from .config import model_config

# Make ta library optional
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("Warning: 'ta' library not available. Using fallback technical indicator implementations.")

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Creates technical indicators and crossover signals for ML model"""
    
    def __init__(self, config=None):
        if config and isinstance(config, dict):
            # Handle custom config dictionary
            self.custom_config = config
            self.config = model_config  # Keep reference to original config
        else:
            self.custom_config = None
            self.config = config or model_config
    
    def _get_config_value(self, path, default=None):
        """Safely get configuration value from custom config or default config.
        Supports nested dicts and simple attribute objects (e.g., dataclasses).
        """
        if self.custom_config:
            current = self.custom_config
            for key in path.split('.'):
                if isinstance(current, dict) and key in current:
                    current = current[key]
                elif hasattr(current, key):
                    current = getattr(current, key)
                else:
                    # Fall back to default config attribute (last segment)
                    return getattr(self.config, path.split('.')[-1], default)
            return current
        else:
            # Use default config attribute
            return getattr(self.config, path.split('.')[-1], default)
        
    def calculate_ema(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate Exponential Moving Averages"""
        df = df.copy()
        for period in periods:
            if TA_AVAILABLE:
                df[f'ema_{period}'] = ta.trend.EMAIndicator(
                    close=df['close'], window=period
                ).ema_indicator()
            else:
                # Fallback: Use pandas ewm (exponentially weighted moving average)
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df
    
    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators"""
        df = df.copy()
        
        # MACD parameters from config
        fast_period = self._get_config_value('technical_indicators.macd.fast_period', 12)
        slow_period = self._get_config_value('technical_indicators.macd.slow_period', 26)
        signal_period = self._get_config_value('technical_indicators.macd.signal_period', 9)
        
        if TA_AVAILABLE:
            macd_indicator = ta.trend.MACD(
                close=df['close'], 
                window_fast=fast_period, 
                window_slow=slow_period, 
                window_sign=signal_period
            )
            
            df['macd'] = macd_indicator.macd()
            df['macd_signal'] = macd_indicator.macd_signal()
            df['macd_hist'] = macd_indicator.macd_diff()
        else:
            # Fallback: Manual MACD calculation
            ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
            ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    def calculate_heikin_ashi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Heikin-Ashi candlesticks"""
        df = df.copy()
        
        # Initialize first Heikin-Ashi values
        df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['ha_open'] = np.nan
        df['ha_high'] = np.nan
        df['ha_low'] = np.nan
        
        # Calculate Heikin-Ashi values
        for i in range(len(df)):
            if i == 0:
                df.loc[i, 'ha_open'] = (df.loc[i, 'open'] + df.loc[i, 'close']) / 2
            else:
                df.loc[i, 'ha_open'] = (df.loc[i-1, 'ha_open'] + df.loc[i-1, 'ha_close']) / 2
            
            df.loc[i, 'ha_high'] = max(df.loc[i, 'high'], df.loc[i, 'ha_open'], df.loc[i, 'ha_close'])
            df.loc[i, 'ha_low'] = min(df.loc[i, 'low'], df.loc[i, 'ha_open'], df.loc[i, 'ha_close'])
        
        return df
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        df = df.copy()
        
        # Volume moving average
        df['volume_ma'] = df['volume'].rolling(window=self.config.volume_window).mean()
        
        # Volume ratio (current volume vs average)
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Above average volume flag
        df['volume_above_average'] = (df['volume'] > df['volume_ma']).astype(int)
        
        return df
    
    def calculate_additional_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional technical indicators"""
        df = df.copy()
        
        # RSI
        rsi_period = self._get_config_value('technical_indicators.rsi.period', 14)
        if TA_AVAILABLE:
            df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=rsi_period).rsi()
        else:
            # Fallback: Manual RSI calculation
            df['rsi'] = self._calculate_rsi_fallback(df['close'], rsi_period)
        
        # ATR (Average True Range) with safe window sizing
        atr_window = int(self._get_config_value('technical_indicators.atr_window', 14))
        atr_window = max(2, min(atr_window, len(df))) if len(df) > 0 else 2
        try:
            if len(df) >= 2:
                if TA_AVAILABLE:
                    df['atr'] = ta.volatility.AverageTrueRange(
                        high=df['high'], low=df['low'], close=df['close'], window=atr_window
                    ).average_true_range()
                else:
                    # Fallback: Manual ATR calculation
                    df['atr'] = self._calculate_atr_fallback(df, atr_window)
            else:
                df['atr'] = 0.0
        except Exception:
            df['atr'] = 0.0
        
        # Bollinger Bands with configured period
        bb_period = int(self._get_config_value('technical_indicators.bollinger_bands.period', 20))
        bb_std = float(self._get_config_value('technical_indicators.bollinger_bands.std_dev', 2))
        bb_period = max(2, min(bb_period, len(df))) if len(df) > 0 else 2
        try:
            if TA_AVAILABLE:
                bb_indicator = ta.volatility.BollingerBands(close=df['close'], window=bb_period, window_dev=bb_std)
                df['bb_upper'] = bb_indicator.bollinger_hband()
                df['bb_lower'] = bb_indicator.bollinger_lband()
                df['bb_middle'] = bb_indicator.bollinger_mavg()
            else:
                # Fallback: Manual Bollinger Bands calculation
                df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
                bb_std_values = df['close'].rolling(window=bb_period).std()
                df['bb_upper'] = df['bb_middle'] + (bb_std_values * bb_std)
                df['bb_lower'] = df['bb_middle'] - (bb_std_values * bb_std)
        except Exception:
            df['bb_upper'] = np.nan
            df['bb_lower'] = np.nan
            df['bb_middle'] = np.nan
        
        # Price position within Bollinger Bands
        denom = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
        df['bb_position'] = (df['close'] - df['bb_lower']) / denom
        
        return df
    
    def create_crossover_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create crossover signals based on codex.txt features
        """
        df = df.copy()
        
        # Get available EMA columns
        ema_cols = [col for col in df.columns if col.startswith('ema_') and col.split('_')[1].isdigit()]
        ema_periods = sorted([int(col.split('_')[1]) for col in ema_cols])
        
        # Price vs Moving Averages (adapt to available EMAs)
        if 'ema_9' in df.columns:
            df['price_above_ema9'] = (df['low'] > df['ema_9']).astype(int)
        
        # Use the second shortest EMA if ema_30 not available
        if 'ema_30' in df.columns:
            df['price_above_ema30'] = (df['low'] > df['ema_30']).astype(int)
            df['ema9_above_ema30'] = (df['ema_9'] > df['ema_30']).astype(int)
        elif len(ema_periods) >= 2:
            second_ema = f'ema_{ema_periods[1]}'
            df['price_above_ema30'] = (df['low'] > df[second_ema]).astype(int)
            df['ema9_above_ema30'] = (df['ema_9'] > df[second_ema]).astype(int)
        
        # MACD Crossovers (from codex.txt)
        df['macd_above_signal'] = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_above_zero'] = (df['macd'] > 0).astype(int)
        df['signal_above_zero'] = (df['macd_signal'] > 0).astype(int)
        df['macd_positive_hist'] = (df['macd_hist'] > 0).astype(int)
        
        # Heikin-Ashi Trend (from codex.txt)
        df['ha_uptrend'] = (df['ha_close'] > df['ha_open']).astype(int)
        
        # Additional crossover signals (adapt to available EMAs)
        if len(ema_periods) >= 3:
            third_ema = f'ema_{ema_periods[2]}'
            df['price_above_ema50'] = (df['close'] > df[third_ema]).astype(int)
            if 'ema_9' in df.columns:
                df['ema9_above_ema50'] = (df['ema_9'] > df[third_ema]).astype(int)
            if len(ema_periods) >= 2:
                second_ema = f'ema_{ema_periods[1]}'
                df['ema30_above_ema50'] = (df[second_ema] > df[third_ema]).astype(int)
        
        if len(ema_periods) >= 4:
            fourth_ema = f'ema_{ema_periods[3]}'
            df['price_above_ema200'] = (df['close'] > df[fourth_ema]).astype(int)
            if len(ema_periods) >= 3:
                third_ema = f'ema_{ema_periods[2]}'
                df['ema50_above_ema200'] = (df[third_ema] > df[fourth_ema]).astype(int)
        
        # Momentum and trend strength signals
        df['price_momentum_strong'] = self._calculate_price_momentum(df)
        # Continuous EMA spread metrics and acceleration
        df['ema_spread'] = self._compute_ema_spread(df)
        df['ema_spread_accel_value'] = self._compute_ema_spread_accel_value(df)
        df['ema_spread_accelerating'] = self._calculate_ema_spread_acceleration(df)
        
        # Rate-of-Change metrics from config
        roc_periods = self._get_config_value('technical_indicators.roc_periods', [5, 10])
        volume_roc_periods = self._get_config_value('technical_indicators.volume_roc_periods', [5])
        
        for period in roc_periods:
            df[f'price_roc_{period}'] = self._calculate_rate_of_change(df['close'], period)
        
        for period in volume_roc_periods:
            df[f'volume_roc_{period}'] = self._calculate_rate_of_change(df['volume'], period)
        
        # Accelerating separation metrics
        df['ema_separation_accel'] = self._calculate_ema_separation_acceleration(df)
        df['price_ema_separation_accel'] = self._calculate_price_ema_separation_acceleration(df)
        
        # RSI signals
        # RSI thresholds from config
        rsi_oversold_threshold = self._get_config_value('technical_indicators.rsi.oversold_threshold', 30)
        rsi_overbought_threshold = self._get_config_value('technical_indicators.rsi.overbought_threshold', 70)
        df['rsi_oversold'] = (df['rsi'] < rsi_oversold_threshold).astype(int)
        df['rsi_overbought'] = (df['rsi'] > rsi_overbought_threshold).astype(int)
        df['rsi_bullish'] = (df['rsi'] > 50).astype(int)
        
        return df
    
    def create_structure_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create 9 binary market structure indicators to solve identical scores problem.
        These features distinguish between consolidation, breakout, and trend quality patterns.
        """
        df = df.copy()
        
        # Detect swing points for structure analysis
        window = 5
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(df) - window):
            # Swing high detection
            if all(df.iloc[i]['high'] >= df.iloc[j]['high'] for j in range(i-window, i)) and \
               all(df.iloc[i]['high'] >= df.iloc[j]['high'] for j in range(i+1, i+window+1)):
                swing_highs.append(i)
            
            # Swing low detection  
            if all(df.iloc[i]['low'] <= df.iloc[j]['low'] for j in range(i-window, i)) and \
               all(df.iloc[i]['low'] <= df.iloc[j]['low'] for j in range(i+1, i+window+1)):
                swing_lows.append(i)
        
        # Initialize the 9 binary breakout indicators
        structure_features = [
            'above_resistance', 'near_resistance', 'breakout_confirmed',
            'higher_highs_higher_lows', 'mixed_signals', 'consolidation_zone',
            'clean_uptrend', 'fresh_breakout', 'volume_confirmed_move'
        ]
        
        for feature in structure_features:
            df[feature] = 0
        
        # Calculate indicators for each bar
        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            
            # Get recent swing points (last 40 bars for context)
            lookback = min(40, i)
            recent_highs = [idx for idx in swing_highs if i - lookback <= idx < i]
            recent_lows = [idx for idx in swing_lows if i - lookback <= idx < i]
            
            if recent_highs and recent_lows:
                resistance = df.iloc[recent_highs[-1]]['high']
                support = df.iloc[recent_lows[-1]]['low']
                
                # 1. Above Resistance - Price trading above recent swing high
                df.loc[i, 'above_resistance'] = int(current_price > resistance * 1.005)
                
                # 2. Near Resistance - Price within 2% of resistance level
                df.loc[i, 'near_resistance'] = int(abs(current_price - resistance) / resistance < 0.02)
                
                # 3. Breakout Confirmed - Price above multiple recent highs
                if len(recent_highs) >= 3:
                    max_recent = max([df.iloc[idx]['high'] for idx in recent_highs[-3:]])
                    df.loc[i, 'breakout_confirmed'] = int(current_price > max_recent * 1.01)
                
                # 4 & 5. Trend Structure Quality
                if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                    last_high = df.iloc[recent_highs[-1]]['high']
                    prev_high = df.iloc[recent_highs[-2]]['high']
                    last_low = df.iloc[recent_lows[-1]]['low']
                    prev_low = df.iloc[recent_lows[-2]]['low']
                    
                    hh = last_high > prev_high  # Higher high
                    hl = last_low > prev_low    # Higher low
                    lh = last_high < prev_high  # Lower high
                    ll = last_low < prev_low    # Lower low
                    
                    # 4. Higher Highs Higher Lows - Clean uptrend pattern
                    df.loc[i, 'higher_highs_higher_lows'] = int(hh and hl)
                    
                    # 5. Mixed Signals - Consolidation/choppy patterns
                    df.loc[i, 'mixed_signals'] = int((hh and ll) or (lh and hl))
                
                # 6. Consolidation Zone - Price in narrow range
                price_range = resistance - support
                range_pct = price_range / support if support > 0 else 0
                in_range = support * 1.02 < current_price < resistance * 0.98
                df.loc[i, 'consolidation_zone'] = int(range_pct < 0.06 and in_range)
                
                # 7. Clean Uptrend - Structure + EMAs aligned
                if 'ema_9' in df.columns and 'ema_30' in df.columns:
                    ema_aligned = current_price > df.iloc[i]['ema_9'] > df.iloc[i]['ema_30']
                    df.loc[i, 'clean_uptrend'] = int(df.iloc[i]['higher_highs_higher_lows'] and ema_aligned)
                
                # 8. Fresh Breakout - Recent breakout activity (last 5 bars)
                breakout_window = min(5, i)
                if breakout_window > 0:
                    recent_breakouts = sum(df.iloc[max(0, i-breakout_window):i]['above_resistance'])
                    df.loc[i, 'fresh_breakout'] = int(recent_breakouts > 0)
            
            # 9. Volume Confirmed Move - Volume-backed significant moves
            if 'volume' in df.columns and i >= 10:
                avg_volume = df.iloc[max(0, i-10):i]['volume'].mean()
                high_volume = df.iloc[i]['volume'] > avg_volume * 1.5
                if i > 0:
                    significant_move = abs(df.iloc[i]['close'] - df.iloc[i-1]['close']) / df.iloc[i-1]['close'] > 0.015
                    df.loc[i, 'volume_confirmed_move'] = int(high_volume and significant_move)
        
        return df
    
    def _calculate_rsi_fallback(self, close_prices: pd.Series, period: int) -> pd.Series:
        """Fallback RSI calculation when ta library is not available"""
        delta = close_prices.diff()
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Default to neutral when insufficient data
    
    def _calculate_atr_fallback(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Fallback ATR calculation when ta library is not available"""
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr.fillna(0)
    
    def _calculate_price_momentum(self, df: pd.DataFrame, window: int = None) -> pd.Series:
        """Calculate price momentum strength"""
        # Get momentum parameters from config
        if window is None:
            window = self._get_config_value('technical_indicators.momentum.window', 5)
        std_window = self._get_config_value('technical_indicators.momentum.std_window', 20)
        std_multiplier = self._get_config_value('technical_indicators.momentum.std_multiplier', 1.5)
        
        price_change = df['close'].pct_change(window)
        momentum_threshold = price_change.rolling(window=std_window).std() * std_multiplier
        return (price_change > momentum_threshold).astype(int)
    
    def _calculate_ema_spread_acceleration(self, df: pd.DataFrame) -> pd.Series:
        """Calculate EMA spread acceleration (binary increasing flag)."""
        if 'ema_9' not in df.columns:
            return pd.Series(0, index=df.index)
        
        ema_cols = [col for col in df.columns if col.startswith('ema_') and col.split('_')[1].isdigit()]
        ema_periods = sorted([int(col.split('_')[1]) for col in ema_cols])
        
        if len(ema_periods) >= 2:
            short_ema = f'ema_{ema_periods[0]}'
            long_ema = f'ema_{ema_periods[1]}'
            spread = df[short_ema] - df[long_ema]
            accel = spread.diff().diff()
            return (accel > 0).astype(int)
        else:
            return pd.Series(0, index=df.index)

    def _compute_ema_spread(self, df: pd.DataFrame) -> pd.Series:
        """Continuous EMA spread (short - long)."""
        ema_cols = [col for col in df.columns if col.startswith('ema_') and col.split('_')[1].isdigit()]
        ema_periods = sorted([int(col.split('_')[1]) for col in ema_cols])
        if len(ema_periods) >= 2:
            short_ema = f'ema_{ema_periods[0]}'
            long_ema = f'ema_{ema_periods[1]}'
            return (df[short_ema] - df[long_ema]).fillna(0)
        return pd.Series(0, index=df.index)

    def _compute_ema_spread_accel_value(self, df: pd.DataFrame) -> pd.Series:
        """Continuous acceleration of EMA spread (second difference)."""
        ema_cols = [col for col in df.columns if col.startswith('ema_') and col.split('_')[1].isdigit()]
        ema_periods = sorted([int(col.split('_')[1]) for col in ema_cols])
        if len(ema_periods) >= 2:
            short_ema = f'ema_{ema_periods[0]}'
            long_ema = f'ema_{ema_periods[1]}'
            spread = df[short_ema] - df[long_ema]
            return spread.diff().diff().fillna(0)
        return pd.Series(0, index=df.index)
    
    def _calculate_rate_of_change(self, series: pd.Series, periods: int) -> pd.Series:
        """Calculate Rate of Change (ROC) indicator"""
        roc = ((series - series.shift(periods)) / series.shift(periods)) * 100
        return roc.fillna(0)
    
    def _calculate_ema_separation_acceleration(self, df: pd.DataFrame) -> pd.Series:
        """Calculate acceleration of EMA separation (distance between EMAs)"""
        ema_cols = [col for col in df.columns if col.startswith('ema_') and col.split('_')[1].isdigit()]
        ema_periods = sorted([int(col.split('_')[1]) for col in ema_cols])
        
        if len(ema_periods) >= 2:
            short_ema = f'ema_{ema_periods[0]}'
            long_ema = f'ema_{ema_periods[1]}'
            
            # Calculate percentage separation
            separation = ((df[short_ema] - df[long_ema]) / df[long_ema]) * 100
            
            # Calculate acceleration of separation
            separation_velocity = separation.diff()
            separation_acceleration = separation_velocity.diff()
            
            # Return binary signal: 1 if separation is accelerating, 0 otherwise
            return (separation_acceleration > 0).astype(int)
        else:
            return pd.Series(0, index=df.index)
    
    def _calculate_price_ema_separation_acceleration(self, df: pd.DataFrame) -> pd.Series:
        """Calculate acceleration of price separation from EMA"""
        ema_cols = [col for col in df.columns if col.startswith('ema_') and col.split('_')[1].isdigit()]
        ema_periods = sorted([int(col.split('_')[1]) for col in ema_cols])
        
        if len(ema_periods) >= 1:
            primary_ema = f'ema_{ema_periods[0]}'
            
            # Calculate percentage separation between price and EMA
            price_ema_separation = ((df['close'] - df[primary_ema]) / df[primary_ema]) * 100
            
            # Calculate acceleration of this separation
            separation_velocity = price_ema_separation.diff()
            separation_acceleration = separation_velocity.diff()
            
            # Return binary signal: 1 if price is accelerating away from EMA upward
            return (separation_acceleration > 0).astype(int)
        else:
            return pd.Series(0, index=df.index)
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for buy/sell signals based on future returns
        STRICT FORWARD BIAS PREVENTION: Only uses data available at prediction time
        """
        df = df.copy()
        
        # CRITICAL: Ensure we're working with chronologically sorted data
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Check if asymmetric horizons are configured
        buy_horizon = self._get_config_value('buy_forecast_horizon', None)
        sell_horizon = self._get_config_value('sell_forecast_horizon', None)
        
        # Special handling for asymmetric horizon experiments
        if hasattr(self, 'custom_config') and self.custom_config:
            experiment_name = self.custom_config.get('name', '')
            if any(name in experiment_name for name in ['dual_signal_v4', 'dual_signal_v5', 'buy_only_v1']):
                buy_horizon = 7
                sell_horizon = 3
                logger.info(f"Detected asymmetric experiment ({experiment_name}) - forcing asymmetric horizons: BUY={buy_horizon}, SELL={sell_horizon}")
        
        if buy_horizon is not None and sell_horizon is not None:
            # Asymmetric horizons - different horizons for buy and sell signals
            logger.info(f"Using asymmetric forecast horizons: BUY={buy_horizon} days, SELL={sell_horizon} days")
            
            # Calculate future prices for different horizons
            df['future_close_buy'] = df['close'].shift(-buy_horizon)
            df['future_close_sell'] = df['close'].shift(-sell_horizon)
            
            # Validate future timestamps
            future_timestamps_buy = df['timestamp'].shift(-buy_horizon)
            future_timestamps_sell = df['timestamp'].shift(-sell_horizon)
            current_timestamps = df['timestamp']
            
            valid_future_buy = future_timestamps_buy > current_timestamps
            valid_future_sell = future_timestamps_sell > current_timestamps
            
            df.loc[~valid_future_buy, 'future_close_buy'] = np.nan
            df.loc[~valid_future_sell, 'future_close_sell'] = np.nan
            
            # Calculate future returns for each horizon
            df['future_return_buy'] = (df['future_close_buy'] - df['close']) / df['close']
            df['future_return_sell'] = (df['future_close_sell'] - df['close']) / df['close']
            
            # Create binary targets with different horizons
            df['target_buy'] = np.nan
            df['target_sell'] = np.nan
            
            buy_valid_mask = ~df['future_return_buy'].isna()
            sell_valid_mask = ~df['future_return_sell'].isna()
            
            df.loc[buy_valid_mask, 'target_buy'] = (df.loc[buy_valid_mask, 'future_return_buy'] > self.config.buy_threshold).astype(int)
            df.loc[sell_valid_mask, 'target_sell'] = (df.loc[sell_valid_mask, 'future_return_sell'] < -self.config.sell_threshold).astype(int)
            
            # Store the return values for analysis
            df['target_return'] = df['future_return_buy'].fillna(df['future_return_sell'])
            
        else:
            # Traditional symmetric horizon
            horizon = self.config.forecast_horizon
            logger.info(f"Using symmetric forecast horizon: {horizon} days")
            
            # Calculate future returns using ONLY future data (no lookahead bias)
            df['future_close'] = df['close'].shift(-horizon)
            
            # Validate that future_close is actually in the future
            future_timestamps = df['timestamp'].shift(-horizon)
            current_timestamps = df['timestamp']
            
            # Ensure future timestamps are actually later than current timestamps
            valid_future = future_timestamps > current_timestamps
            df.loc[~valid_future, 'future_close'] = np.nan
            
            # Calculate future returns only where we have valid future data
            df['future_return'] = (df['future_close'] - df['close']) / df['close']
            
            # Create binary targets - only where future data exists
            df['target_buy'] = np.nan
            df['target_sell'] = np.nan
            df['target_return'] = np.nan
            
            valid_mask = ~df['future_return'].isna()
            df.loc[valid_mask, 'target_buy'] = (df.loc[valid_mask, 'future_return'] > self.config.buy_threshold).astype(int)
            df.loc[valid_mask, 'target_sell'] = (df.loc[valid_mask, 'future_return'] < -self.config.sell_threshold).astype(int)
            df.loc[valid_mask, 'target_return'] = df.loc[valid_mask, 'future_return']
        
        # Log validation info - handle both symmetric and asymmetric cases
        if buy_horizon is not None and sell_horizon is not None:
            # Asymmetric case - count valid targets from both buy and sell
            buy_valid_targets = buy_valid_mask.sum()
            sell_valid_targets = sell_valid_mask.sum()
            valid_targets = max(buy_valid_targets, sell_valid_targets)  # Use the higher count
            logger.info(f"Asymmetric targets: BUY valid={buy_valid_targets}, SELL valid={sell_valid_targets}")
        else:
            # Symmetric case
            valid_targets = valid_mask.sum()
        
        total_records = len(df)
        logger.info(f"Target variables created: {valid_targets}/{total_records} valid targets "
                   f"({valid_targets/total_records:.1%} coverage)")
        
        if valid_targets < total_records * 0.8:
            logger.warning(f"Low target coverage ({valid_targets/total_records:.1%}) - "
                          f"check forecast horizon and data availability")
        
        return df
    
    def get_feature_columns(self) -> Dict[str, List[str]]:
        """Get lists of feature columns by category"""
        # Check if enabled_features is configured
        enabled_features = self._get_config_value('enabled_features', None)
        
        if enabled_features:
            # Use configured enabled features
            # Group them by category for backward compatibility
            all_possible_features = {
                'price_features': [
                    'price_above_ema9', 'price_above_ema30', 'price_above_ema50', 
                    'price_above_ema200', 'price_momentum_strong'
                ],
                'ema_features': [
                    'ema9_above_ema30', 'ema9_above_ema50', 'ema30_above_ema50', 
                    'ema50_above_ema200', 'ema_spread_accelerating'
                ],
                'macd_features': [
                    'macd_above_signal', 'macd_above_zero', 'signal_above_zero',
                    'macd_positive_hist'
                ],
                'volume_features': [
                    'volume_above_average', 'volume_ratio'
                ],
                'trend_features': [
                    'ha_uptrend'
                ],
                'momentum_features': [
                    'rsi_oversold', 'rsi_overbought', 'rsi_bullish'
                ],
                'rate_of_change_features': [],  # Will be populated dynamically
                'separation_features': [
                    'ema_separation_accel', 'price_ema_separation_accel',
                    'ema_spread', 'ema_spread_accel_value'
                ],
                'structure_features': [
                    'above_resistance', 'near_resistance', 'breakout_confirmed',
                    'higher_highs_higher_lows', 'mixed_signals', 'consolidation_zone',
                    'clean_uptrend', 'fresh_breakout', 'volume_confirmed_move'
                ]
            }
            
            # Add dynamic ROC features based on config
            roc_periods = self._get_config_value('technical_indicators.roc_periods', [5, 10])
            volume_roc_periods = self._get_config_value('technical_indicators.volume_roc_periods', [5])
            
            for period in roc_periods:
                all_possible_features['rate_of_change_features'].append(f'price_roc_{period}')
            for period in volume_roc_periods:
                all_possible_features['rate_of_change_features'].append(f'volume_roc_{period}')
            
            # Filter to only enabled features
            filtered_features = {}
            for category, features in all_possible_features.items():
                filtered_features[category] = [f for f in features if f in enabled_features]
            
            return filtered_features
        
        else:
            # Fallback to original behavior for backward compatibility
            base_features = {
                'price_features': [
                    'price_above_ema9', 'price_above_ema30', 'price_momentum_strong'
                ],
                'ema_features': [
                    'ema9_above_ema30', 'ema_spread_accelerating'
                ],
                'macd_features': [
                    'macd_above_signal', 'macd_above_zero', 'signal_above_zero',
                    'macd_positive_hist'
                ],
                'volume_features': [
                    'volume_above_average', 'volume_ratio'
                ],
                'trend_features': [
                    'ha_uptrend'
                ],
                'momentum_features': [
                    'rsi_oversold', 'rsi_overbought', 'rsi_bullish'
                ],
                'rate_of_change_features': [
                    'price_roc_5', 'price_roc_10', 'volume_roc_5'
                ],
                'separation_features': [
                    'ema_separation_accel', 'price_ema_separation_accel',
                    'ema_spread', 'ema_spread_accel_value'
                ]
            }
            
            # Add optional features based on configured EMA periods
            ema_periods = self.config.ema_periods
            if len(ema_periods) >= 3:
                base_features['price_features'].append('price_above_ema50')
                base_features['ema_features'].extend(['ema9_above_ema50', 'ema30_above_ema50'])
            
            if len(ema_periods) >= 4:
                base_features['price_features'].append('price_above_ema200')
                base_features['ema_features'].append('ema50_above_ema200')
            
            return base_features
    
    def process_symbol_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline for a single symbol
        STRICT NO FORWARD BIAS: All calculations use only historical data available at each point in time
        """
        logger.info(f"Processing features for symbol data with {len(df)} records")
        
        # CRITICAL: Ensure chronological order and validate data integrity
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Validate no forward bias in input data
        self._validate_chronological_order(df)
        
        # Calculate all indicators using ONLY past data
        df = self.calculate_ema(df, self.config.ema_periods)
        df = self.calculate_macd(df)
        df = self.calculate_heikin_ashi(df)
        df = self.calculate_volume_indicators(df)
        df = self.calculate_additional_indicators(df)
        
        # Create crossover signals (these use current and past data only)
        df = self.create_crossover_signals(df)
        
        # Create structure signals (market structure context for breakout detection)
        df = self.create_structure_signals(df)
        
        # Create target variables (uses future data, but properly labeled as such)
        df = self.create_target_variables(df)
        
        # STRICT FILTERING: Remove rows where we can't calculate features without forward bias
        max_period = max(self.config.ema_periods)
        
        # Remove initial rows where indicators can't be calculated
        initial_valid_idx = max_period
        df_features = df.iloc[initial_valid_idx:].copy()
        
        # Remove final rows where targets can't be calculated (no future data available)
        # This ensures we never use future information for prediction
        if len(df_features) > self.config.forecast_horizon:
            df_final = df_features.iloc[:-self.config.forecast_horizon].copy()
        else:
            logger.warning(f"Insufficient data after removing forecast horizon. "
                          f"Need at least {max_period + self.config.forecast_horizon} records, "
                          f"got {len(df)}")
            df_final = df_features.copy()
        
        # Final validation
        self._validate_no_forward_bias(df_final)
        
        logger.info(f"Feature engineering complete. Final dataset: {len(df_final)} records "
                   f"(removed {len(df) - len(df_final)} records for temporal integrity)")
        
        return df_final
    
    def _validate_chronological_order(self, df: pd.DataFrame):
        """Validate that data is in chronological order"""
        if len(df) < 2:
            return
            
        timestamps = df['timestamp'].values
        if not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
            raise ValueError("Data is not in chronological order - this would cause forward bias")
    
    def _validate_no_forward_bias(self, df: pd.DataFrame):
        """Validate that no forward bias exists in the processed data"""
        # Check that all feature calculations are valid
        feature_columns = self.get_all_feature_names()
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Ensure no features have perfect correlation with targets (sign of data leakage)
        if 'target_buy' in df.columns and 'target_sell' in df.columns:
            for feature in available_features:
                if feature in df.columns:
                    # Check correlation with future targets
                    buy_corr = abs(df[feature].corr(df['target_buy']))
                    sell_corr = abs(df[feature].corr(df['target_sell']))
                    
                    if buy_corr > 0.95 or sell_corr > 0.95:
                        logger.error(f"FORWARD BIAS DETECTED: Feature {feature} has suspiciously high "
                                   f"correlation with targets (buy: {buy_corr:.3f}, sell: {sell_corr:.3f})")
                        raise ValueError(f"Forward bias detected in feature {feature}")
        
        logger.debug("Forward bias validation passed")
    
    def get_all_feature_names(self) -> List[str]:
        """Get all feature column names"""
        feature_groups = self.get_feature_columns()
        all_features = []
        for features in feature_groups.values():
            all_features.extend(features)
        return all_features