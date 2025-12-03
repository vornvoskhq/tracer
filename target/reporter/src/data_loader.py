"""Data loader for VGMini stock data."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class VGMiniDataLoader:
    """Load stock data from VGMini outputs."""
    
    def __init__(self, vgmini_root: str = "../vgmini"):
        self.vgmini_root = Path(vgmini_root)
        if not str(vgmini_root).endswith('/results'):
            # If just the vgmini directory path is provided, append results
            self.results_dir = self.vgmini_root / "results"
        else:
            # If the full results path is provided, use it directly
            self.results_dir = self.vgmini_root
        
    def get_latest_ranking_file(self) -> Optional[Path]:
        """Find the most recent ranking file from VGMini."""
        if not self.results_dir.exists():
            return None
            
        # Look for ranking files
        ranking_files = list(self.results_dir.glob("*_ranking.txt"))
        
        if not ranking_files:
            return None
            
        # Return the most recently modified
        return max(ranking_files, key=lambda p: p.stat().st_mtime)
    
    def parse_ranking_file(self, ranking_file: Path) -> List[Dict[str, Any]]:
        """Parse a VGMini ranking file to extract stock symbols and scores."""
        stocks = []
        
        try:
            with open(ranking_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('📊') or line.startswith('#'):
                        continue
                        
                    # Parse various ranking formats
                    parts = line.split()
                    if len(parts) >= 2:
                        # Extract rank number and symbol
                        rank_part = parts[0].rstrip('.')
                        if rank_part.isdigit():
                            rank = int(rank_part)
                            symbol = parts[1]
                            
                            # Try to extract score and buy signal from different formats
                            score = None
                            buy_signal = False
                            
                            # Look for numeric score (could be anywhere in the line)
                            for part in parts[2:]:  # Skip rank and symbol
                                try:
                                    # Try to parse as float
                                    potential_score = float(part)
                                    if 0 <= potential_score <= 1:  # Reasonable score range
                                        score = potential_score
                                        break
                                except ValueError:
                                    continue
                            
                            # Look for buy signal indicators
                            line_lower = line.lower()
                            if 'buy' in line_lower or 'true' in line_lower:
                                buy_signal = True
                            
                            stocks.append({
                                'rank': rank,
                                'symbol': symbol,
                                'score': score,
                                'buy_signal': buy_signal
                            })
        except Exception as e:
            print(f"Error parsing ranking file {ranking_file}: {e}")
            
        return stocks
    
    def get_top_stocks(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the top N stocks from the latest ranking."""
        ranking_file = self.get_latest_ranking_file()
        
        if ranking_file is None:
            print("No ranking file found in VGMini results directory")
            return []
            
        print(f"Using ranking file: {ranking_file}")
        stocks = self.parse_ranking_file(ranking_file)
        
        # Return top N stocks
        return stocks[:n]
    
    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """
        Gather all available data for a specific stock symbol.
        
        This is a placeholder - in a real implementation, this would
        integrate with VGMini's data sources to get:
        - Technical indicators
        - Price history
        - Volume data
        - Any fundamental data available
        """
        # For now, return basic structure
        # TODO: Integrate with VGMini data sources
        return {
            'symbol': symbol,
            'data_sources': ['ranking'],
            'ranking_data': self._get_stock_ranking_data(symbol),
            'note': 'Full data integration with VGMini pending - currently showing ranking data only'
        }
    
    def _get_stock_ranking_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ranking-specific data for a stock."""
        stocks = self.get_top_stocks(n=50)  # Get more stocks to find the specific one
        
        for stock in stocks:
            if stock['symbol'] == symbol:
                return stock
        
        return None
    
    def format_stock_data_for_llm(self, stock_data: Dict[str, Any]) -> str:
        """Format stock data into a string suitable for LLM analysis."""
        symbol = stock_data.get('symbol', 'UNKNOWN')
        ranking_data = stock_data.get('ranking_data')
        
        output_lines = [
            f"Stock Symbol: {symbol}",
            "",
            "=== RANKING DATA ==="
        ]
        
        if ranking_data:
            output_lines.extend([
                f"Rank: {ranking_data.get('rank', 'N/A')}",
                f"Score: {ranking_data.get('score', 'N/A')}",
                f"Buy Signal: {ranking_data.get('buy_signal', 'N/A')}",
            ])
        else:
            output_lines.append("No ranking data available")
        
        output_lines.extend([
            "",
            "=== TECHNICAL DATA ===",
            "(Technical indicators integration pending)",
            "",
            "=== FUNDAMENTAL DATA ===", 
            "(Fundamental data integration pending)",
            "",
            f"Note: {stock_data.get('note', 'Data collection in progress')}"
        ])
        
        return "\n".join(output_lines)