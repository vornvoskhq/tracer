"""
limiter.py - Swing High/Low Detection & Market Structure Analysis

Identifies major swing highs and lows for stocks to determine:
- Higher Highs / Higher Lows (bullish trend)
- Lower Highs / Lower Lows (bearish trend)
- Consolidation patterns
- Breakout/breakdown from supply/demand zones

Output format compatible with stock-chart2 for visualization.
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import yfinance as yf


@dataclass
class LimiterConfig:
    """Configuration for swing point detection"""
    lookback_days: int = 90  # How far back to analyze
    swing_window: int = 5    # Window size for swing detection (left/right bars)
    major_threshold: float = 0.02  # 2% move to be considered "major"
    consolidation_range: float = 0.05  # 5% range for consolidation detection
    volume_confirm: bool = True  # Require volume confirmation
    
    def to_dict(self):
        return asdict(self)


@dataclass
class SwingPoint:
    """Represents a swing high or low"""
    date: str
    price: float
    type: str  # "high" or "low"
    volume: float
    strength: float  # 0-1, how significant the swing is
    
    def to_dict(self):
        return asdict(self)


@dataclass
class MarketStructure:
    """Analysis of market structure for a symbol"""
    symbol: str
    last_major_high: Optional[SwingPoint]
    last_major_low: Optional[SwingPoint]
    previous_major_high: Optional[SwingPoint]
    previous_major_low: Optional[SwingPoint]
    current_price: float
    trend: str  # "bullish", "bearish", "consolidating", "breakout", "breakdown"
    trend_strength: float  # 0-1
    structure_summary: str
    support_level: float
    resistance_level: float
    distance_to_support_pct: float
    distance_to_resistance_pct: float
    all_swing_highs: List[SwingPoint]
    all_swing_lows: List[SwingPoint]
    
    def to_dict(self):
        d = asdict(self)
        # Convert SwingPoint objects to dicts
        if self.last_major_high:
            d['last_major_high'] = self.last_major_high.to_dict()
        if self.last_major_low:
            d['last_major_low'] = self.last_major_low.to_dict()
        if self.previous_major_high:
            d['previous_major_high'] = self.previous_major_high.to_dict()
        if self.previous_major_low:
            d['previous_major_low'] = self.previous_major_low.to_dict()
        d['all_swing_highs'] = [s.to_dict() for s in self.all_swing_highs]
        d['all_swing_lows'] = [s.to_dict() for s in self.all_swing_lows]
        return d


def detect_swing_highs(df: pd.DataFrame, window: int = 5) -> List[int]:
    """
    Detect swing highs using a window-based approach.
    A swing high is a high that is higher than 'window' bars before and after.
    """
    swing_highs = []
    highs = df['High'].values
    
    for i in range(window, len(highs) - window):
        left_max = max(highs[i-window:i])
        right_max = max(highs[i+1:i+window+1])
        
        if highs[i] > left_max and highs[i] > right_max:
            swing_highs.append(i)
    
    return swing_highs


def detect_swing_lows(df: pd.DataFrame, window: int = 5) -> List[int]:
    """
    Detect swing lows using a window-based approach.
    A swing low is a low that is lower than 'window' bars before and after.
    """
    swing_lows = []
    lows = df['Low'].values
    
    for i in range(window, len(lows) - window):
        left_min = min(lows[i-window:i])
        right_min = min(lows[i+1:i+window+1])
        
        if lows[i] < left_min and lows[i] < right_min:
            swing_lows.append(i)
    
    return swing_lows


def calculate_swing_strength(df: pd.DataFrame, idx: int, swing_type: str, 
                            window: int = 10, volume_confirm: bool = True) -> float:
    """
    Calculate the strength/significance of a swing point (0-1).
    Factors:
    - Price movement magnitude
    - Volume relative to average
    - Time since last swing
    """
    strength = 0.5  # Base strength
    
    # Factor 1: Price movement magnitude
    if swing_type == "high":
        price = df.loc[idx, 'High']
        recent_range = df.loc[max(0, idx-window):idx, 'High'].max() - df.loc[max(0, idx-window):idx, 'Low'].min()
    else:
        price = df.loc[idx, 'Low']
        recent_range = df.loc[max(0, idx-window):idx, 'High'].max() - df.loc[max(0, idx-window):idx, 'Low'].min()
    
    if recent_range > 0:
        # Higher range = more significant
        strength += 0.2 * min(recent_range / price, 0.1) * 10
    
    # Factor 2: Volume confirmation
    if volume_confirm and 'Volume' in df.columns:
        avg_volume = df.loc[max(0, idx-20):idx, 'Volume'].mean()
        if avg_volume > 0:
            volume_ratio = df.loc[idx, 'Volume'] / avg_volume
            strength += 0.3 * min(volume_ratio / 2, 1)  # Cap at 2x average
    
    return min(strength, 1.0)


def analyze_market_structure(symbol: str, cfg: LimiterConfig) -> Optional[MarketStructure]:
    """
    Analyze market structure for a symbol.
    Returns MarketStructure object with all relevant information.
    """
    try:
        # Download historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=cfg.lookback_days)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty or len(df) < cfg.swing_window * 2:
            print(f"  ⚠ {symbol}: Insufficient data")
            return None
        
        # Reset index to get dates as a column
        df = df.reset_index()
        
        # Detect all swing points
        swing_high_indices = detect_swing_highs(df, cfg.swing_window)
        swing_low_indices = detect_swing_lows(df, cfg.swing_window)
        
        # Create SwingPoint objects for highs
        all_swing_highs = []
        for idx in swing_high_indices:
            strength = calculate_swing_strength(df, idx, "high", cfg.swing_window, cfg.volume_confirm)
            swing = SwingPoint(
                date=df.loc[idx, 'Date'].strftime('%Y-%m-%d'),
                price=float(df.loc[idx, 'High']),
                type="high",
                volume=float(df.loc[idx, 'Volume']),
                strength=strength
            )
            all_swing_highs.append(swing)
        
        # Create SwingPoint objects for lows
        all_swing_lows = []
        for idx in swing_low_indices:
            strength = calculate_swing_strength(df, idx, "low", cfg.swing_window, cfg.volume_confirm)
            swing = SwingPoint(
                date=df.loc[idx, 'Date'].strftime('%Y-%m-%d'),
                price=float(df.loc[idx, 'Low']),
                type="low",
                volume=float(df.loc[idx, 'Volume']),
                strength=strength
            )
            all_swing_lows.append(swing)
        
        # Filter for "major" swings based on threshold
        current_price = float(df.iloc[-1]['Close'])
        
        # Get major highs (price significantly above average)
        if all_swing_highs:
            avg_high = sum(s.price for s in all_swing_highs) / len(all_swing_highs)
            major_highs = [s for s in all_swing_highs if (s.price - avg_high) / avg_high >= cfg.major_threshold]
            major_highs.sort(key=lambda s: s.date, reverse=True)
        else:
            major_highs = []
        
        # Get major lows (price significantly below average)
        if all_swing_lows:
            avg_low = sum(s.price for s in all_swing_lows) / len(all_swing_lows)
            major_lows = [s for s in all_swing_lows if (avg_low - s.price) / avg_low >= cfg.major_threshold]
            major_lows.sort(key=lambda s: s.date, reverse=True)
        else:
            major_lows = []
        
        # Identify last major high/low and previous ones
        last_major_high = major_highs[0] if major_highs else (all_swing_highs[-1] if all_swing_highs else None)
        previous_major_high = major_highs[1] if len(major_highs) > 1 else (all_swing_highs[-2] if len(all_swing_highs) > 1 else None)
        
        last_major_low = major_lows[0] if major_lows else (all_swing_lows[-1] if all_swing_lows else None)
        previous_major_low = major_lows[1] if len(major_lows) > 1 else (all_swing_lows[-2] if len(all_swing_lows) > 1 else None)
        
        # Determine support and resistance levels
        resistance_level = last_major_high.price if last_major_high else current_price * 1.05
        support_level = last_major_low.price if last_major_low else current_price * 0.95
        
        # Calculate distances
        distance_to_resistance_pct = ((resistance_level - current_price) / current_price) * 100
        distance_to_support_pct = ((current_price - support_level) / current_price) * 100
        
        # Determine trend and structure
        trend, trend_strength, structure_summary = determine_trend_structure(
            current_price, last_major_high, last_major_low, 
            previous_major_high, previous_major_low, cfg
        )
        
        return MarketStructure(
            symbol=symbol,
            last_major_high=last_major_high,
            last_major_low=last_major_low,
            previous_major_high=previous_major_high,
            previous_major_low=previous_major_low,
            current_price=current_price,
            trend=trend,
            trend_strength=trend_strength,
            structure_summary=structure_summary,
            support_level=support_level,
            resistance_level=resistance_level,
            distance_to_support_pct=distance_to_support_pct,
            distance_to_resistance_pct=distance_to_resistance_pct,
            all_swing_highs=all_swing_highs,
            all_swing_lows=all_swing_lows
        )
        
    except Exception as e:
        print(f"  ✗ {symbol}: Error - {e}")
        return None


def determine_trend_structure(current_price: float, 
                              last_high: Optional[SwingPoint],
                              last_low: Optional[SwingPoint],
                              prev_high: Optional[SwingPoint],
                              prev_low: Optional[SwingPoint],
                              cfg: LimiterConfig) -> Tuple[str, float, str]:
    """
    Determine trend and market structure based on swing points.
    Returns (trend, strength, summary)
    """
    
    # Handle cases where we don't have enough data
    if not last_high or not last_low:
        return "insufficient_data", 0.0, "Not enough swing points detected"
    
    # Check for higher highs and higher lows (bullish)
    if prev_high and prev_low:
        hh = last_high.price > prev_high.price  # Higher high
        hl = last_low.price > prev_low.price    # Higher low
        lh = last_high.price < prev_high.price  # Lower high
        ll = last_low.price < prev_low.price    # Lower low
        
        # Strong uptrend
        if hh and hl:
            strength = 0.8 + (0.2 * min((last_high.price - prev_high.price) / prev_high.price, 0.1) * 10)
            return "bullish", strength, f"Higher High (${last_high.price:.2f}) and Higher Low (${last_low.price:.2f}) - Strong Uptrend"
        
        # Strong downtrend
        elif lh and ll:
            strength = 0.8 + (0.2 * min((prev_low.price - last_low.price) / prev_low.price, 0.1) * 10)
            return "bearish", strength, f"Lower High (${last_high.price:.2f}) and Lower Low (${last_low.price:.2f}) - Strong Downtrend"
        
        # Mixed signals - check current price position
        elif hh and ll:
            # Making higher highs but lower lows = volatile/consolidating
            range_pct = (last_high.price - last_low.price) / last_low.price
            if range_pct < cfg.consolidation_range:
                return "consolidating", 0.4, f"Consolidating between ${last_low.price:.2f} and ${last_high.price:.2f}"
            else:
                return "volatile", 0.3, f"Mixed signals - Higher High but Lower Low"
        
        elif lh and hl:
            # Lower highs but higher lows = converging/consolidating
            return "consolidating", 0.5, f"Converging pattern - potential breakout or breakdown"
    
    # Check for breakout or breakdown
    if last_high and prev_high:
        breakout_pct = (last_high.price - prev_high.price) / prev_high.price
        if breakout_pct > cfg.major_threshold and current_price > last_high.price * 0.98:
            return "breakout", 0.85, f"Breaking out above resistance at ${last_high.price:.2f}"
    
    if last_low and prev_low:
        breakdown_pct = (prev_low.price - last_low.price) / prev_low.price
        if breakdown_pct > cfg.major_threshold and current_price < last_low.price * 1.02:
            return "breakdown", 0.85, f"Breaking down below support at ${last_low.price:.2f}"
    
    # Default: determine by price position relative to last high/low
    mid_point = (last_high.price + last_low.price) / 2
    
    if current_price > mid_point:
        strength = (current_price - mid_point) / (last_high.price - mid_point) * 0.7
        return "bullish", strength, f"Price above mid-range, testing resistance at ${last_high.price:.2f}"
    else:
        strength = (mid_point - current_price) / (mid_point - last_low.price) * 0.7
        return "bearish", strength, f"Price below mid-range, testing support at ${last_low.price:.2f}"


def run_limiter(shortlist_path: str, out_dir: str, cfg: Optional[LimiterConfig] = None) -> str:
    """
    Main function to run limiter analysis on a shortlist.
    Generates JSON output compatible with stock-chart2 visualization.
    """
    if cfg is None:
        cfg = LimiterConfig()
    
    print(f"\n{'='*80}")
    print(f"Running Limiter Analysis")
    print(f"{'='*80}")
    print(f"Shortlist: {shortlist_path}")
    print(f"Output: {out_dir}")
    print(f"Config: lookback={cfg.lookback_days}d, window={cfg.swing_window}, threshold={cfg.major_threshold*100}%")
    print(f"{'='*80}\n")
    
    # Read shortlist
    symbols = []
    with open(shortlist_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                symbols.append(parts[1])
    
    print(f"Analyzing {len(symbols)} symbols...")
    
    # Analyze each symbol
    results = []
    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] {symbol}...", end='')
        structure = analyze_market_structure(symbol, cfg)
        if structure:
            results.append(structure)
            print(f" ✓ {structure.trend} ({structure.trend_strength:.2f})")
        else:
            print(f" ✗ Failed")
    
    print(f"\n{'='*80}")
    print(f"Completed: {len(results)}/{len(symbols)} symbols analyzed")
    print(f"{'='*80}\n")
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate timestamp for CSV
    date_value = datetime.now().strftime('%Y-%m-%d')
    
    # Write JSON output (main format)
    out_json = os.path.join(out_dir, "limiter.json")
    json_output = {
        "metadata": {
            "source": "vgmini_limiter_analysis",
            "generated_at": datetime.now().isoformat(),
            "config": cfg.to_dict(),
            "total_symbols": len(results)
        },
        "results": [r.to_dict() for r in results],
        "summary": {
            "bullish": len([r for r in results if r.trend == "bullish"]),
            "bearish": len([r for r in results if r.trend == "bearish"]),
            "consolidating": len([r for r in results if r.trend == "consolidating"]),
            "breakout": len([r for r in results if r.trend == "breakout"]),
            "breakdown": len([r for r in results if r.trend == "breakdown"])
        }
    }
    
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2)
    
    # Write CSV output (for easy viewing)
    out_csv = os.path.join(out_dir, f"limiter_{date_value}.csv")
    csv_data = []
    for r in results:
        csv_data.append({
            'symbol': r.symbol,
            'current_price': r.current_price,
            'trend': r.trend,
            'trend_strength': r.trend_strength,
            'last_high': r.last_major_high.price if r.last_major_high else None,
            'last_high_date': r.last_major_high.date if r.last_major_high else None,
            'last_low': r.last_major_low.price if r.last_major_low else None,
            'last_low_date': r.last_major_low.date if r.last_major_low else None,
            'resistance': r.resistance_level,
            'support': r.support_level,
            'dist_to_resistance_%': r.distance_to_resistance_pct,
            'dist_to_support_%': r.distance_to_support_pct,
            'structure_summary': r.structure_summary
        })
    
    df_out = pd.DataFrame(csv_data)
    df_out.to_csv(out_csv, index=False)
    
    print(f"Limiter outputs:\n - {out_json}\n - {out_csv}")
    print(f"\nTrend Distribution:")
    print(f"  Bullish: {json_output['summary']['bullish']}")
    print(f"  Bearish: {json_output['summary']['bearish']}")
    print(f"  Consolidating: {json_output['summary']['consolidating']}")
    print(f"  Breakout: {json_output['summary']['breakout']}")
    print(f"  Breakdown: {json_output['summary']['breakdown']}")
    
    return out_json


def main():
    ap = argparse.ArgumentParser(description='Detect swing highs/lows and analyze market structure for shortlist symbols')
    ap.add_argument('--shortlist', default='vgmini/results/decorrelated/equal_weight_dual.txt',
                   help='Path to shortlist file')
    ap.add_argument('--out-dir', default='vgmini/results/limiter',
                   help='Output directory for results')
    ap.add_argument('--lookback', type=int, default=90,
                   help='Days to look back for analysis')
    ap.add_argument('--window', type=int, default=5,
                   help='Window size for swing detection')
    ap.add_argument('--threshold', type=float, default=0.02,
                   help='Threshold for major swing (0.02 = 2%%)')
    ap.add_argument('--consolidation-range', type=float, default=0.05,
                   help='Range for consolidation detection (0.05 = 5%%)')
    ap.add_argument('--no-volume-confirm', action='store_true',
                   help='Disable volume confirmation')
    
    args = ap.parse_args()
    
    cfg = LimiterConfig(
        lookback_days=args.lookback,
        swing_window=args.window,
        major_threshold=args.threshold,
        consolidation_range=args.consolidation_range,
        volume_confirm=not args.no_volume_confirm
    )
    
    run_limiter(args.shortlist, args.out_dir, cfg)


if __name__ == "__main__":
    main()
