"""
Visualization module for stock analysis framework
Creates charts for performance, signals, and SHAP analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import logging
from .config import viz_config

# Import SHAP with Python 3.12 compatibility
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    from .model_interpretability import ModelInterpreter

logger = logging.getLogger(__name__)

class VisualizationEngine:
    """Creates comprehensive visualizations for the trading framework"""
    
    def __init__(self, config=None):
        self.config = config or viz_config
        plt.style.use(self.config.style)
        
    def plot_backtest_performance(self, results: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive backtest performance visualization
        
        Args:
            results: Backtest results dictionary
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
        fig.suptitle('Backtest Performance Analysis', fontsize=16, fontweight='bold')
        
        portfolio_df = results['portfolio_history']
        trades_df = results['trades']
        
        # Portfolio value over time
        ax1 = axes[0, 0]
        ax1.plot(portfolio_df['date'], portfolio_df['total_value'], 
                label='Portfolio Value', color=self.config.color_palette[0], linewidth=2)
        
        # Add benchmark (buy and hold)
        initial_price = portfolio_df['price'].iloc[0]
        initial_capital = 100000.0  # Default initial capital
        benchmark_values = initial_capital * (portfolio_df['price'] / initial_price)
        ax1.plot(portfolio_df['date'], benchmark_values, 
                label='Buy & Hold', color=self.config.color_palette[1], linewidth=2, alpha=0.7)
        
        ax1.set_title('Portfolio Value vs Buy & Hold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown chart
        ax2 = axes[0, 1]
        portfolio_df['cummax'] = portfolio_df['total_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['total_value'] - portfolio_df['cummax']) / portfolio_df['cummax']
        
        ax2.fill_between(portfolio_df['date'], portfolio_df['drawdown'], 0, 
                        color=self.config.color_palette[3], alpha=0.7)
        ax2.set_title(f'Drawdown (Max: {results["max_drawdown"]:.2%})')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # Trade distribution
        ax3 = axes[1, 0]
        if not trades_df.empty:
            returns = trades_df['return_pct'] * 100
            ax3.hist(returns, bins=30, alpha=0.7, color=self.config.color_palette[2], edgecolor='black')
            ax3.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
            ax3.set_title('Trade Return Distribution')
            ax3.set_xlabel('Return (%)')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No trades executed', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Trade Return Distribution')
        
        # Performance metrics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        metrics_data = [
            ['Total Return', f"{results['total_return']:.2%}"],
            ['Benchmark Return', f"{results['benchmark_return']:.2%}"],
            ['Excess Return', f"{results['excess_return']:.2%}"],
            ['Sharpe Ratio', f"{results['sharpe_ratio']:.2f}"],
            ['Max Drawdown', f"{results['max_drawdown']:.2%}"],
            ['Volatility', f"{results['volatility']:.2%}"],
            ['Total Trades', f"{results.get('total_trades', 0)}"],
            ['Win Rate', f"{results.get('win_rate', 0):.2%}"]
        ]
        
        table = ax4.table(cellText=metrics_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Performance chart saved to {save_path}")
        
        return fig
    
    def plot_signals_on_price_chart(self, df: pd.DataFrame, signals: Dict, 
                                   symbol: str = "STOCK", save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive price chart with buy/sell signals
        
        Args:
            df: DataFrame with OHLC data
            signals: Dictionary with signal data
            symbol: Stock symbol for title
            save_path: Optional path to save the figure
            
        Returns:
            Plotly Figure object
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} Price with Signals', 'Signal Probabilities', 'Volume'),
            row_heights=[0.6, 0.25, 0.15]
        )
        
        # Price chart with candlesticks
        fig.add_trace(
            go.Candlestick(
                x=df['datetime'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add buy signals
        buy_signals = signals.get('buy_signals', [])
        if len(buy_signals) > 0:
            buy_dates = df[buy_signals.astype(bool)]['datetime']
            buy_prices = df[buy_signals.astype(bool)]['close']
            
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='green'),
                    name='Buy Signal'
                ),
                row=1, col=1
            )
        
        # Add sell signals
        sell_signals = signals.get('sell_signals', [])
        if len(sell_signals) > 0:
            sell_dates = df[sell_signals.astype(bool)]['datetime']
            sell_prices = df[sell_signals.astype(bool)]['close']
            
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='red'),
                    name='Sell Signal'
                ),
                row=1, col=1
            )
        
        # Signal probabilities
        if 'buy_probability' in signals and 'sell_probability' in signals:
            fig.add_trace(
                go.Scatter(
                    x=df['datetime'],
                    y=signals['buy_probability'],
                    name='Buy Probability',
                    line=dict(color='green', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['datetime'],
                    y=signals['sell_probability'],
                    name='Sell Probability',
                    line=dict(color='red', width=2)
                ),
                row=2, col=1
            )
            
            # Add confidence threshold line
            fig.add_hline(y=0.65, line_dash="dash", line_color="gray", 
                         annotation_text="Confidence Threshold", row=2, col=1)
        
        # Volume
        fig.add_trace(
            go.Bar(
                x=df['datetime'],
                y=df['volume'],
                name='Volume',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Trading Signals Analysis',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Probability", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Signal chart saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(self, importance_dict: Dict[str, Dict[str, float]], 
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance for both buy and sell models
        
        Args:
            importance_dict: Dictionary with buy_model and sell_model importance
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        # Buy model importance
        buy_importance = importance_dict['buy_model']
        buy_features = list(buy_importance.keys())
        buy_values = list(buy_importance.values())
        
        # Sort by absolute importance
        buy_sorted_idx = np.argsort([abs(v) for v in buy_values])[::-1]
        buy_features_sorted = [buy_features[i] for i in buy_sorted_idx]
        buy_values_sorted = [buy_values[i] for i in buy_sorted_idx]
        
        colors_buy = ['green' if v > 0 else 'red' for v in buy_values_sorted]
        
        axes[0].barh(range(len(buy_features_sorted)), buy_values_sorted, color=colors_buy, alpha=0.7)
        axes[0].set_yticks(range(len(buy_features_sorted)))
        axes[0].set_yticklabels(buy_features_sorted)
        axes[0].set_xlabel('Coefficient Value')
        axes[0].set_title('Buy Model Feature Importance')
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Sell model importance
        sell_importance = importance_dict['sell_model']
        sell_features = list(sell_importance.keys())
        sell_values = list(sell_importance.values())
        
        # Sort by absolute importance
        sell_sorted_idx = np.argsort([abs(v) for v in sell_values])[::-1]
        sell_features_sorted = [sell_features[i] for i in sell_sorted_idx]
        sell_values_sorted = [sell_values[i] for i in sell_sorted_idx]
        
        colors_sell = ['green' if v > 0 else 'red' for v in sell_values_sorted]
        
        axes[1].barh(range(len(sell_features_sorted)), sell_values_sorted, color=colors_sell, alpha=0.7)
        axes[1].set_yticks(range(len(sell_features_sorted)))
        axes[1].set_yticklabels(sell_features_sorted)
        axes[1].set_xlabel('Coefficient Value')
        axes[1].set_title('Sell Model Feature Importance')
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Feature importance chart saved to {save_path}")
        
        return fig
    
    def plot_shap_summary(self, shap_values: np.ndarray, feature_names: List[str], 
                         X: np.ndarray, model_type: str = 'buy', 
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create SHAP summary plot (with Python 3.12 compatibility)
        
        Args:
            shap_values: SHAP values array
            feature_names: List of feature names
            X: Feature matrix
            model_type: 'buy' or 'sell'
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        plt.figure(figsize=(12, 8))
        
        if SHAP_AVAILABLE:
            # Use SHAP if available
            shap.summary_plot(shap_values, X, feature_names=feature_names, 
                             plot_type="bar", show=False)
        else:
            # Check if LIME values are all zeros (should be handled upstream now)
            mean_importance = np.abs(shap_values).mean(axis=0)
            if np.all(mean_importance == 0):
                logger.info(f"No feature importance data available for {model_type} model visualization")
                # Create a placeholder plot indicating no data
                plt.text(0.5, 0.5, f'No feature importance data available\nfor {model_type} model visualization', 
                        ha='center', va='center', transform=plt.gca().transAxes, 
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.axis('off')
            else:
                # Create LIME-based alternative plot
                sorted_indices = np.argsort(mean_importance)[::-1][:15]  # Top 15 features
                
                sorted_features = [feature_names[i] for i in sorted_indices]
                sorted_importance = mean_importance[sorted_indices]
                
                # Create horizontal bar plot
                colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))
                bars = plt.barh(range(len(sorted_features)), sorted_importance, color=colors, alpha=0.7)
                
                plt.yticks(range(len(sorted_features)), sorted_features)
                plt.xlabel('Mean |Feature Importance|')
                plt.gca().invert_yaxis()
                
                # Add value labels
                for i, (bar, importance) in enumerate(zip(bars, sorted_importance)):
                    plt.text(importance + 0.001, i, f'{importance:.3f}', 
                            va='center', ha='left', fontsize=8)
        
        plt.title(f'Feature Importance - {model_type.title()} Model', 
                 fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return plt.gcf()
    
    def plot_monthly_returns_heatmap(self, monthly_returns: pd.DataFrame, 
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create monthly returns heatmap
        
        Args:
            monthly_returns: DataFrame with monthly returns
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Prepare data for heatmap
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_returns['Year'] = monthly_returns.index.year
        monthly_returns['Month'] = monthly_returns.index.month
        
        # Create pivot table
        heatmap_data = monthly_returns.pivot_table(
            values='monthly_return', 
            index='Year', 
            columns='Month', 
            aggfunc='mean'
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(heatmap_data * 100, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=0, ax=ax, cbar_kws={'label': 'Monthly Return (%)'})
        
        ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        # Set month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_labels)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Monthly returns heatmap saved to {save_path}")
        
        return fig
    
    def create_dashboard_summary(self, results: Dict, symbol: str = "PORTFOLIO", 
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive dashboard summary
        
        Args:
            results: Complete results dictionary
            symbol: Symbol name for title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle(f'{symbol} Trading Strategy Dashboard', fontsize=20, fontweight='bold')
        
        # Portfolio performance (large chart)
        ax1 = fig.add_subplot(gs[0, :2])
        portfolio_df = results['portfolio_history']
        ax1.plot(portfolio_df['date'], portfolio_df['total_value'], 
                linewidth=3, color=self.config.color_palette[0])
        ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        
        # Key metrics
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.axis('off')
        
        metrics_text = f"""
        Total Return: {results['total_return']:.2%}
        Sharpe Ratio: {results['sharpe_ratio']:.2f}
        Max Drawdown: {results['max_drawdown']:.2%}
        
        Total Trades: {results.get('total_trades', 0)}
        Win Rate: {results.get('win_rate', 0):.2%}
        Avg Return/Trade: {results.get('avg_return', 0):.2%}
        """
        
        ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Drawdown
        ax3 = fig.add_subplot(gs[1, :2])
        portfolio_df['cummax'] = portfolio_df['total_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['total_value'] - portfolio_df['cummax']) / portfolio_df['cummax']
        ax3.fill_between(portfolio_df['date'], portfolio_df['drawdown'], 0, 
                        color=self.config.color_palette[3], alpha=0.7)
        ax3.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # Trade distribution
        ax4 = fig.add_subplot(gs[1, 2:])
        trades_df = results['trades']
        if not trades_df.empty:
            returns = trades_df['return_pct'] * 100
            ax4.hist(returns, bins=20, alpha=0.7, color=self.config.color_palette[2], edgecolor='black')
            ax4.axvline(returns.mean(), color='red', linestyle='--', linewidth=2)
            ax4.set_title('Trade Return Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Return (%)')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        ax5 = fig.add_subplot(gs[2, :])
        if len(portfolio_df) > 60:  # Need enough data for rolling calculation
            portfolio_df['daily_return'] = portfolio_df['total_value'].pct_change()
            rolling_sharpe = portfolio_df['daily_return'].rolling(60).mean() / portfolio_df['daily_return'].rolling(60).std() * np.sqrt(252)
            ax5.plot(portfolio_df['date'], rolling_sharpe, linewidth=2, color=self.config.color_palette[4])
            ax5.set_title('Rolling 60-Day Sharpe Ratio', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Sharpe Ratio')
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax5.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Dashboard summary saved to {save_path}")
        
        return fig