"""
Backtesting engine for evaluating trading strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from .config import backtest_config

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Backtesting engine for evaluating ML-based trading strategies
    """
    
    def __init__(self, config=None):
        self.config = config or backtest_config
        self.trades = []
        self.portfolio_history = []
        
    def run_backtest(self, df: pd.DataFrame, signals: Dict[str, np.ndarray]) -> Dict[str, any]:
        """
        Run backtest on historical data with ML signals
        
        Args:
            df: DataFrame with OHLC data and datetime
            signals: Dictionary with buy_signals, sell_signals, and probabilities
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest with {len(df)} data points")
        
        # Initialize portfolio
        portfolio = {
            'cash': self.config.initial_capital,
            'positions': {},  # symbol -> {'shares': int, 'entry_price': float, 'entry_date': datetime}
            'total_value': self.config.initial_capital
        }
        
        self.trades = []
        self.portfolio_history = []
        
        # Trade gating state
        consecutive_buy = 0
        consecutive_sell = 0
        last_buy_date = None  # apply cooldown to buys only to avoid blocking exits
        
        # Process each day
        for i in range(len(df)):
            current_data = df.iloc[i]
            current_date = current_data['datetime']
            current_price = current_data['close']
            symbol = current_data.get('symbol', 'STOCK')
            
            # Process signals
            raw_buy_signal = signals['buy_signals'][i] if i < len(signals['buy_signals']) else 0
            raw_sell_signal = signals['sell_signals'][i] if i < len(signals['sell_signals']) else 0
            buy_prob = float(signals['buy_probability'][i]) if i < len(signals['buy_probability']) else 0.0
            sell_prob = float(signals['sell_probability'][i]) if i < len(signals['sell_probability']) else 0.0
            
            # Hysteresis to avoid flip-flopping
            margin = float(getattr(self.config, 'hysteresis_margin', 0.0) or 0.0)
            buy_ok = raw_buy_signal and (buy_prob - sell_prob >= margin)
            sell_ok = raw_sell_signal and (sell_prob - buy_prob >= margin)
            
            # Consecutive day confirmation
            if buy_ok:
                consecutive_buy += 1
            else:
                consecutive_buy = 0
            if sell_ok:
                consecutive_sell += 1
            else:
                consecutive_sell = 0
            
            # Apply gating rules
            can_buy = (
                buy_ok
                and (symbol not in portfolio['positions'])
                and (consecutive_buy >= int(getattr(self.config, 'buy_consecutive_days', 1)))
            )
            # Cooldown (buys only)
            cd_days = int(getattr(self.config, 'trade_cooldown_days', 0) or 0)
            if can_buy and last_buy_date is not None:
                if (current_date - last_buy_date).days < cd_days:
                    can_buy = False
            
            can_sell = (
                sell_ok
                and (symbol in portfolio['positions'])
                and (consecutive_sell >= int(getattr(self.config, 'sell_consecutive_days', 1)))
            )
            # Minimum holding period for sells
            if can_sell:
                min_hold = int(getattr(self.config, 'min_holding_days', 0) or 0)
                if min_hold > 0:
                    entry_date = portfolio['positions'][symbol]['entry_date']
                    if (current_date - entry_date).days < min_hold:
                        can_sell = False
            
            # Execute trades
            if can_buy:
                self._execute_buy(portfolio, symbol, current_price, current_date, buy_prob)
                last_buy_date = current_date
            elif can_sell:
                self._execute_sell(portfolio, symbol, current_price, current_date, sell_prob)
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(portfolio, current_price, symbol)
            portfolio['total_value'] = portfolio_value
            
            # Record portfolio state
            self.portfolio_history.append({
                'date': current_date,
                'cash': portfolio['cash'],
                'positions_value': portfolio_value - portfolio['cash'],
                'total_value': portfolio_value,
                'price': current_price
            })
        
        # Close any remaining positions at the end
        if portfolio['positions']:
            final_price = df.iloc[-1]['close']
            final_date = df.iloc[-1]['datetime']
            for symbol in list(portfolio['positions'].keys()):
                self._execute_sell(portfolio, symbol, final_price, final_date, 0.5, force_close=True)
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(df)
        
        if len(self.trades) > 0:
            logger.info(f"Backtest completed. Total trades: {len(self.trades)}")
            logger.info(f"Final portfolio value: ${results['final_value']:,.2f}")
            logger.info(f"Total return: {results['total_return']:.2%}")
        else:
            logger.debug(f"Backtest completed. No trades executed - portfolio unchanged")
        
        return results
    
    def _execute_buy(self, portfolio: Dict, symbol: str, price: float, date, probability: float):
        """Execute a buy order"""
        # Calculate position size (percentage of portfolio)
        portfolio_value = portfolio['total_value']
        max_position_value = portfolio_value * self.config.max_position_size
        
        # Adjust position size based on signal confidence
        confidence_multiplier = min(probability / 0.5, 2.0)  # Scale with confidence
        position_value = max_position_value * confidence_multiplier * 0.5  # Conservative sizing
        
        # Calculate shares to buy
        cost_per_share = price * (1 + self.config.commission + self.config.slippage)
        shares = int(position_value / cost_per_share)
        total_cost = shares * cost_per_share
        
        if shares > 0 and total_cost <= portfolio['cash']:
            portfolio['cash'] -= total_cost
            portfolio['positions'][symbol] = {
                'shares': shares,
                'entry_price': price,
                'entry_date': date,
                'entry_probability': probability
            }
            
            logger.debug(f"BUY: {shares} shares of {symbol} at ${price:.2f} on {date}")
    
    def _execute_sell(self, portfolio: Dict, symbol: str, price: float, date, probability: float, force_close: bool = False):
        """Execute a sell order"""
        if symbol not in portfolio['positions']:
            return
        
        position = portfolio['positions'][symbol]
        shares = position['shares']
        
        # Calculate proceeds after costs
        proceeds_per_share = price * (1 - self.config.commission - self.config.slippage)
        total_proceeds = shares * proceeds_per_share
        
        # Record trade
        entry_price = position['entry_price']
        entry_date = position['entry_date']
        
        trade = {
            'symbol': symbol,
            'entry_date': entry_date,
            'exit_date': date,
            'entry_price': entry_price,
            'exit_price': price,
            'shares': shares,
            'entry_probability': position['entry_probability'],
            'exit_probability': probability,
            'days_held': (date - entry_date).days,
            'return_pct': (price - entry_price) / entry_price,
            'profit_loss': total_proceeds - (shares * entry_price * (1 + self.config.commission + self.config.slippage)),
            'trade_type': 'FORCED_CLOSE' if force_close else 'SIGNAL'
        }
        
        self.trades.append(trade)
        
        # Update portfolio
        portfolio['cash'] += total_proceeds
        del portfolio['positions'][symbol]
        
        logger.debug(f"SELL: {shares} shares of {symbol} at ${price:.2f} on {date} "
                    f"(Return: {trade['return_pct']:.2%})")
    
    def _calculate_portfolio_value(self, portfolio: Dict, current_price: float, symbol: str) -> float:
        """Calculate total portfolio value"""
        cash = portfolio['cash']
        positions_value = 0
        
        for pos_symbol, position in portfolio['positions'].items():
            if pos_symbol == symbol:
                positions_value += position['shares'] * current_price
            else:
                # For simplicity, assume other positions maintain their entry price
                # In a real backtest, you'd need price data for all symbols
                positions_value += position['shares'] * position['entry_price']
        
        return cash + positions_value
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate comprehensive performance metrics"""
        portfolio_df = pd.DataFrame(self.portfolio_history)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # Basic metrics
        initial_value = self.config.initial_capital
        final_value = portfolio_df['total_value'].iloc[-1] if not portfolio_df.empty else initial_value
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate daily returns
        portfolio_df['daily_return'] = portfolio_df['total_value'].pct_change()
        
        # Benchmark (buy and hold)
        initial_price = df['close'].iloc[0]
        final_price = df['close'].iloc[-1]
        benchmark_return = (final_price - initial_price) / initial_price
        
        # Risk metrics
        daily_returns = portfolio_df['daily_return'].dropna()
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        
        # Drawdown calculation
        portfolio_df['cummax'] = portfolio_df['total_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['total_value'] - portfolio_df['cummax']) / portfolio_df['cummax']
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Trade statistics
        trade_stats = {}
        if not trades_df.empty:
            winning_trades = trades_df[trades_df['return_pct'] > 0]
            losing_trades = trades_df[trades_df['return_pct'] <= 0]
            
            trade_stats = {
                'total_trades': len(trades_df),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
                'avg_return': trades_df['return_pct'].mean(),
                'avg_winning_return': winning_trades['return_pct'].mean() if len(winning_trades) > 0 else 0,
                'avg_losing_return': losing_trades['return_pct'].mean() if len(losing_trades) > 0 else 0,
                'avg_days_held': trades_df['days_held'].mean(),
                'total_profit_loss': trades_df['profit_loss'].sum(),
                'profit_factor': abs(winning_trades['profit_loss'].sum() / losing_trades['profit_loss'].sum()) if len(losing_trades) > 0 and losing_trades['profit_loss'].sum() != 0 else float('inf')
            }
        
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'excess_return': total_return - benchmark_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_history': portfolio_df,
            'trades': trades_df,
            **trade_stats
        }
    
    def get_trade_summary(self) -> pd.DataFrame:
        """Get summary of all trades"""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def get_monthly_returns(self) -> pd.DataFrame:
        """Calculate monthly returns"""
        if not self.portfolio_history:
            return pd.DataFrame()
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df.set_index('date', inplace=True)
        
        monthly_values = portfolio_df['total_value'].resample('M').last()
        monthly_returns = monthly_values.pct_change().dropna()
        
        return monthly_returns.to_frame('monthly_return')