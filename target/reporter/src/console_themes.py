"""Console theme management for dystopian aesthetic."""

from typing import Dict, Any


class DystopianTheme:
    """Dystopian console theme with post-soviet/slum aesthetics."""
    
    # Dystopian symbols and decorators
    SYMBOLS = {
        'analysis': '',       # No emoji
        'loading': '',        # No emoji  
        'generating': '',     # No emoji
        'success': '▓',       # Block instead of checkmark
        'error': '▓',         # Block instead of X
        'warning': '▓',       # Block instead of warning
        'info': '▓',          # Block
        'bullet': '▪',        # Small block
        'separator': '▓' * 50,
        'border': '▒' * 80
    }
    
    # Dystopian messages
    MESSAGES = {
        'analyzing': 'PROCESSING MARKET DATA',
        'loading_data': 'EXTRACTING FINANCIAL METRICS', 
        'loading_prompt': 'INITIALIZING ANALYSIS PROTOCOL',
        'generating': 'COMPUTING ECONOMIC PROJECTIONS',
        'saved': 'DATA ARCHIVED TO TERMINAL',
        'completed': 'ANALYSIS CYCLE COMPLETE'
    }
    
    @staticmethod
    def format_header(symbol: str, model: str) -> str:
        """Format analysis header with dystopian theme."""
        return f"\n▓ SURVEILLANCE: {symbol.upper()} | {model} | DECLINE DETECTED"

    @staticmethod
    def format_footer() -> str:
        """Format analysis footer."""
        return f"{DystopianTheme.SYMBOLS['separator']}\n▓ END TRANSMISSION"

    @staticmethod
    def get_progress_message(action: str) -> str:
        """Get dystopian progress message."""
        return f"{DystopianTheme.SYMBOLS['info']} {DystopianTheme.MESSAGES.get(action, action.upper())}"

    @staticmethod
    def format_stock_list_header(count: int) -> str:
        """Format stock list with dystopian theme."""
        return f"""
{DystopianTheme.SYMBOLS['border']}
☢ MARKET SURVEILLANCE DATA: TOP {count} ENTITIES
▓ ECONOMIC DETERIORATION METRICS AVAILABLE
{DystopianTheme.SYMBOLS['border']}"""

    @staticmethod
    def format_stock_entry(rank: int, symbol: str, score: Any, buy_signal: bool) -> str:
        """Format individual stock entry."""
        signal_icon = "▓" if buy_signal else "░"
        score_display = f"{score:.3f}" if score is not None else "NULL"
        return f"  {rank:2d}. {symbol:6s} {signal_icon} METRIC: {score_display}"


class NormalTheme:
    """Standard console theme."""
    
    SYMBOLS = {
        'analysis': '📈',
        'loading': '📊', 
        'generating': '🤖',
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️',
        'bullet': '•',
        'separator': '-' * 50,
        'border': '=' * 80
    }
    
    @staticmethod
    def format_header(symbol: str, model: str) -> str:
        return f"\n📈 {symbol.upper()} Analysis ({model}):"

    @staticmethod
    def format_footer() -> str:
        return NormalTheme.SYMBOLS['separator']

    @staticmethod
    def get_progress_message(action: str) -> str:
        return f"🔍 {action}..."

    @staticmethod
    def format_stock_list_header(count: int) -> str:
        return f"🏆 Top {count} stocks from VGMini rankings:"

    @staticmethod
    def format_stock_entry(rank: int, symbol: str, score: Any, buy_signal: bool) -> str:
        signal_icon = "🟢" if buy_signal else "🔴"
        score_display = f"{score:.3f}" if score is not None else "None"
        return f"  {rank:2d}. {symbol:6s} {signal_icon} Score: {score_display}"