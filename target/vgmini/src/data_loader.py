"""
Data loading and preprocessing module for stock analysis framework
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from typing import List, Optional, Tuple
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .config import db_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataLoader:
    """Handles loading and preprocessing of stock data from PostgreSQL database"""
    
    def __init__(self):
        self.engine = create_engine(db_config.connection_string)
        
    def get_symbols(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Get list of available symbols"""
        query = "SELECT id, symbol, name FROM symbols ORDER BY symbol"
        if limit:
            query += f" LIMIT {limit}"
        
        return pd.read_sql(query, self.engine)
    
    def get_ohlc_data(self, symbol_ids: List[int], 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      interval: str = '1d') -> pd.DataFrame:
        """
        Load OHLC data for given symbols
        
        Args:
            symbol_ids: List of symbol IDs to load
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (default '1d')
        """
        # Convert symbol_ids to comma-separated string
        symbol_ids_str = ','.join(map(str, symbol_ids))
        
        query = f"""
        SELECT 
            o.symbol_id,
            s.symbol,
            o.timestamp,
            o.open,
            o.high,
            o.low,
            o.close,
            o.volume,
            o.interval
        FROM ohlc_data o
        JOIN symbols s ON o.symbol_id = s.id
        WHERE o.symbol_id IN ({symbol_ids_str})
        AND o.interval = :interval
        """
        
        params = {'interval': interval}
        
        if start_date:
            # Convert date to timestamp (handle both seconds and milliseconds)
            start_timestamp = pd.to_datetime(start_date).timestamp()
            # Check if database uses milliseconds by looking at existing data
            check_query = "SELECT MAX(timestamp) as max_ts FROM ohlc_data LIMIT 1"
            check_result = pd.read_sql(check_query, self.engine)
            if not check_result.empty and check_result.iloc[0]['max_ts'] > 4102444800:
                # Database uses milliseconds
                start_timestamp *= 1000
                end_timestamp_multiplier = 1000
            else:
                end_timestamp_multiplier = 1
                
            query += f" AND o.timestamp >= {int(start_timestamp)}"
            
        if end_date:
            # Include the full end_date day by advancing to end-of-day (23:59:59)
            end_dt = pd.to_datetime(end_date)
            end_dt_eod = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            end_timestamp = end_dt_eod.timestamp()
            if 'end_timestamp_multiplier' in locals():
                end_timestamp *= end_timestamp_multiplier
            else:
                # Check again if we didn't check for start_date
                check_query = "SELECT MAX(timestamp) as max_ts FROM ohlc_data LIMIT 1"
                check_result = pd.read_sql(check_query, self.engine)
                if not check_result.empty and check_result.iloc[0]['max_ts'] > 4102444800:
                    end_timestamp *= 1000
            query += f" AND o.timestamp <= {int(end_timestamp)}"
            
        query += " ORDER BY o.symbol_id, o.timestamp"
        
        df = pd.read_sql(text(query), self.engine, params=params)
        
        # Convert timestamp to datetime (handle both seconds and milliseconds)
        # Check if timestamps are in milliseconds (> year 2100 when interpreted as seconds)
        if not df.empty and df['timestamp'].max() > 4102444800:  # Jan 1, 2100 in seconds
            # Timestamps are in milliseconds
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            logger.debug("Converted timestamps from milliseconds")
        else:
            # Timestamps are in seconds
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            logger.debug("Converted timestamps from seconds")
            
        df = df.sort_values(['symbol_id', 'datetime']).reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} OHLC records for {len(symbol_ids)} symbols")
        return df
    
    def get_symbol_data(self, symbol: str, 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """Get data for a single symbol by symbol name"""
        # Get symbol ID
        symbol_query = "SELECT id FROM symbols WHERE symbol = :symbol"
        result = pd.read_sql(text(symbol_query), self.engine, params={'symbol': symbol})
        
        if result.empty:
            raise ValueError(f"Symbol {symbol} not found in database")
            
        symbol_id = result.iloc[0]['id']
        return self.get_ohlc_data([symbol_id], start_date, end_date)
    
    def validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the loaded data
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Cleaned DataFrame
        """
        initial_count = len(df)
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        # Remove rows where high < low (data errors)
        df = df[df['high'] >= df['low']]
        
        # Remove rows where close is outside high/low range
        df = df[(df['close'] >= df['low']) & (df['close'] <= df['high'])]
        df = df[(df['open'] >= df['low']) & (df['open'] <= df['high'])]
        
        # Remove rows with zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        df = df[(df[price_cols] > 0).all(axis=1)]
        
        # Remove rows with negative volume
        df = df[df['volume'] >= 0]
        
        cleaned_count = len(df)
        removed_count = initial_count - cleaned_count
        
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} invalid records during data validation")
            
        return df.reset_index(drop=True)
    
    def get_data_summary(self, interval: Optional[str] = None) -> dict:
        """Get summary statistics about the database. If interval is provided, date_range is computed for that interval only."""
        queries = {
            'total_symbols': "SELECT COUNT(*) as count FROM symbols",
            'total_records': "SELECT COUNT(*) as count FROM ohlc_data",
            'date_range': (
                "SELECT MIN(timestamp) as min_timestamp, MAX(timestamp) as max_timestamp FROM ohlc_data"
                if not interval else
                "SELECT MIN(timestamp) as min_timestamp, MAX(timestamp) as max_timestamp FROM ohlc_data WHERE interval = :interval"
            ),
            'intervals': "SELECT DISTINCT interval FROM ohlc_data ORDER BY interval"
        }
        
        summary = {}
        for key, query in queries.items():
            if key == 'date_range' and interval:
                result = pd.read_sql(text(query), self.engine, params={'interval': interval})
            else:
                result = pd.read_sql(query, self.engine)
            if key == 'date_range':
                min_ts = result.iloc[0]['min_timestamp']
                max_ts = result.iloc[0]['max_timestamp']
                
                if min_ts is not None and max_ts is not None:
                    # Check if timestamps are in milliseconds
                    if max_ts > 4102444800:  # Jan 1, 2100 in seconds
                        # Timestamps are in milliseconds
                        summary[key] = {
                            'start_date': pd.to_datetime(min_ts, unit='ms'),
                            'end_date': pd.to_datetime(max_ts, unit='ms')
                        }
                    else:
                        # Timestamps are in seconds
                        summary[key] = {
                            'start_date': pd.to_datetime(min_ts, unit='s'),
                            'end_date': pd.to_datetime(max_ts, unit='s')
                        }
                else:
                    summary[key] = {
                        'start_date': None,
                        'end_date': None
                    }
            elif key == 'intervals':
                summary[key] = result['interval'].tolist()
            else:
                summary[key] = result.iloc[0]['count']
                
        return summary
    
    def close(self):
        """Close database connection"""
        self.engine.dispose()