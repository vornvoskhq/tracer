import os
import sys
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .render_charts import parse_shortlist
from .decorrelate import _resolve_existing

# Avoid introducing SQLAlchemy: reuse existing cluster2 PostgreSQL helpers (psycopg2-based)
# Ensure repo root on sys.path so 'cluster2' package is importable when running from vgmini/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
from cluster2.src.db_utils import get_db_connection, fetch_all_symbols, fetch_ohlc_data


@dataclass
class TrailStopConfig:
    atr_period: int = 14
    atr_mult_stop: float = 2.0
    atr_mult_limit: float = 3.0
    min_bars: int = 50
    require_price_above_ema: bool = True
    require_ema9_above_ema30: bool = True


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = _ema(series, 12)
    ema26 = _ema(series, 26)
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    # df columns: high, low, close (sorted by datetime)
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        (df['high'] - df['low']).abs(),
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    # Wilder smoothing alternative:
    # atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr


def _trend_state(df: pd.DataFrame) -> Dict[str, float]:
    # Assumes df sorted by datetime
    closes = df['close']
    ema9 = _ema(closes, 9)
    ema30 = _ema(closes, 30)
    macd_line, macd_sig, macd_hist = _macd(closes)

    last = df.iloc[-1]
    last_close = float(last['close'])
    last_ema9 = float(ema9.iloc[-1]) if pd.notna(ema9.iloc[-1]) else math.nan
    last_ema30 = float(ema30.iloc[-1]) if pd.notna(ema30.iloc[-1]) else math.nan
    last_macd = float(macd_line.iloc[-1]) if pd.notna(macd_line.iloc[-1]) else math.nan
    last_sig = float(macd_sig.iloc[-1]) if pd.notna(macd_sig.iloc[-1]) else math.nan
    last_hist = float(macd_hist.iloc[-1]) if pd.notna(macd_hist.iloc[-1]) else math.nan

    trend_positive = (
        (not math.isnan(last_ema9) and not math.isnan(last_ema30) and last_ema9 >= last_ema30) and
        (not math.isnan(last_ema30) and last_close >= last_ema30)
    )

    return {
        'ema9': last_ema9,
        'ema30': last_ema30,
        'macd': last_macd,
        'macd_signal': last_sig,
        'macd_hist': last_hist,
        'trend_positive': bool(trend_positive),
    }


def _fetch_symbol_df(symbol: str, interval: str = '1d') -> pd.DataFrame:
    conn = get_db_connection()
    try:
        # Resolve symbol id
        syms = fetch_all_symbols(conn)
        sym_map = {s['symbol']: s['id'] for s in syms}
        if symbol not in sym_map:
            return pd.DataFrame()
        sid = sym_map[symbol]
        df = fetch_ohlc_data(conn, [sid], interval)
        if df.empty:
            return pd.DataFrame()
        # Ensure expected columns and datetime ordering
        df = df.reset_index().rename(columns={'timestamp': 'datetime'})
        df = df.sort_values('datetime').reset_index(drop=True)
        return df
    finally:
        try:
            conn.close()
        except Exception:
            pass


def compute_trailing_levels_for_symbol(symbol: str, cfg: TrailStopConfig) -> Optional[Dict]:
    df = _fetch_symbol_df(symbol, '1d')
    if df.empty:
        return {
            'symbol': symbol,
            'error': 'no_data'
        }
    # Basic clean
    df = df.sort_values('datetime').reset_index(drop=True)
    if len(df) < max(cfg.min_bars, cfg.atr_period + 5):
        return {
            'symbol': symbol,
            'error': f'not_enough_data: {len(df)} rows'
        }
    # Trend metrics
    trend = _trend_state(df)
    last_row = df.iloc[-1]
    last_dt = pd.to_datetime(last_row['datetime']).date().isoformat()
    entry = float(last_row['close'])
    # ATR
    atr_series = _atr(df, cfg.atr_period)
    last_atr = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else math.nan
    if not math.isfinite(last_atr) or last_atr <= 0:
        return {
            'symbol': symbol,
            'date': last_dt,
            'entry_price': entry,
            'error': 'atr_unavailable'
        }
    # Optional trend gating
    if cfg.require_price_above_ema and (not math.isfinite(trend['ema30']) or entry < trend['ema30']):
        trend['trend_positive'] = False
    if cfg.require_ema9_above_ema30 and (not math.isfinite(trend['ema9']) or not math.isfinite(trend['ema30']) or trend['ema9'] < trend['ema30']):
        trend['trend_positive'] = False

    stop = max(entry - cfg.atr_mult_stop * last_atr, 0.0)
    limit = entry + cfg.atr_mult_limit * last_atr
    stop_pct = (entry - stop) / entry if entry > 0 else math.nan
    limit_pct = (limit - entry) / entry if entry > 0 else math.nan

    return {
        'symbol': symbol,
        'date': last_dt,
        'entry_price': round(entry, 4),
        'atr': round(last_atr, 4),
        'atr_period': cfg.atr_period,
        'atr_mult_stop': cfg.atr_mult_stop,
        'atr_mult_limit': cfg.atr_mult_limit,
        'stop_loss_price': round(stop, 4),
        'stop_loss_pct': round(stop_pct * 100.0, 3) if math.isfinite(stop_pct) else None,
        'take_profit_price': round(limit, 4),
        'take_profit_pct': round(limit_pct * 100.0, 3) if math.isfinite(limit_pct) else None,
        'ema9': trend['ema9'],
        'ema30': trend['ema30'],
        'macd': trend['macd'],
        'macd_signal': trend['macd_signal'],
        'macd_hist': trend['macd_hist'],
        'trend_positive': trend['trend_positive'],
    }


def run_trailstop(shortlist_path: str, out_dir: str, cfg: Optional[TrailStopConfig] = None) -> str:
    if cfg is None:
        cfg = TrailStopConfig()

    sl_path = _resolve_existing(shortlist_path)
    syms = parse_shortlist(sl_path)
    if not syms:
        raise RuntimeError(f"No symbols parsed from shortlist {sl_path}")

    out_dir = _resolve_existing(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    results: List[Dict] = []
    for sym in syms:
        try:
            row = compute_trailing_levels_for_symbol(sym, cfg)
        except Exception as e:
            row = {'symbol': sym, 'error': f'exception: {e}'}
        results.append(row)

    df = pd.DataFrame(results)
    # Determine date for filename: use most common 'date' present or today
    date_value = None
    if 'date' in df.columns and df['date'].dropna().any():
        try:
            date_value = df['date'].dropna().mode().iloc[0]
        except Exception:
            date_value = None
    if not date_value:
        from datetime import datetime
        date_value = datetime.now().date().isoformat()

    out_csv = os.path.join(out_dir, f"trailstop_{date_value}.csv")
    df.to_csv(out_csv, index=False)

    # Use fixed filename for JSON to avoid clutter
    out_json = os.path.join(out_dir, "trailstop.json")
    
    # Include metadata with generation date
    from datetime import datetime
    json_output = {
        "metadata": {
            "generated_date": date_value,
            "generated_timestamp": datetime.now().isoformat(),
            "source": "vgmini_trailstop_analysis",
            "shortlist_path": shortlist_path,
            "config": cfg.__dict__ if cfg else None
        },
        "results": results,
        "total_symbols": len(results)
    }
    
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2)

    # Generate orderer-compatible format with fixed filename
    out_orderer = os.path.join(out_dir, "orderer_orders.json")
    orderer_data = generate_orderer_format(json_output)
    with open(out_orderer, 'w', encoding='utf-8') as f:
        json.dump(orderer_data, f, indent=2)

    print(f"Trailstop outputs:\n - {out_csv}\n - {out_json}\n - {out_orderer}")
    return out_csv


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Compute ATR-based trailing stops for shortlist symbols')
    ap.add_argument('--shortlist', default='vgmini/results/decorrelated/equal_weight_dual.txt')
    ap.add_argument('--out-dir', default='vgmini/results/trailstop')
    ap.add_argument('--atr-period', type=int, default=14)
    ap.add_argument('--atr-mult-stop', type=float, default=2.0)
    ap.add_argument('--atr-mult-limit', type=float, default=3.0)
    ap.add_argument('--min-bars', type=int, default=50)
    ap.add_argument('--no-require-price-above-ema', action='store_true')
    ap.add_argument('--no-require-ema9-above-ema30', action='store_true')
    args = ap.parse_args()

    cfg = TrailStopConfig(
        atr_period=args.atr_period,
        atr_mult_stop=args.atr_mult_stop,
        atr_mult_limit=args.atr_mult_limit,
        min_bars=args.min_bars,
        require_price_above_ema=not args.no_require_price_above_ema,
        require_ema9_above_ema30=not args.no_require_ema9_above_ema30,
    )

    run_trailstop(args.shortlist, args.out_dir, cfg)


def generate_orderer_format(trailstop_data: Dict) -> Dict:
    """
    Convert vgmini trailstop results into a format compatible with orderer.
    
    The orderer expects:
    - symbol: stock symbol
    - quantity: number of shares to buy
    - trail_amount: dollar amount to trail below ASK for trailing stop
    - stop_offset: dollar amount below ASK for hard stop loss
    
    We calculate these from the ATR-based levels in trailstop results.
    """
    orders = []
    
    # Extract metadata and results from new format
    if 'metadata' in trailstop_data:
        source_metadata = trailstop_data['metadata']
        results = trailstop_data.get('results', [])
    else:
        # Handle old format (list of results)
        source_metadata = {}
        results = trailstop_data if isinstance(trailstop_data, list) else []
    
    metadata = {
        "source": "vgmini_trailstop",
        "description": "Orders generated from vgmini trailing stop analysis",
        "generated_date": source_metadata.get('generated_date'),
        "generated_timestamp": source_metadata.get('generated_timestamp')
    }
    
    for result in results:
        # Skip symbols with errors or negative trend
        if 'error' in result:
            continue
        if not result.get('trend_positive', False):
            continue
            
        symbol = result['symbol']
        entry_price = result['entry_price']
        stop_loss_price = result['stop_loss_price']
        
        # Calculate trail amount and stop offset based on ATR levels
        # trail_amount: how much below current ASK the trailing stop should be
        # This should be the difference between entry and stop loss
        trail_amount = entry_price - stop_loss_price
        
        # stop_offset: hard stop loss level below ASK
        # Use the same value as trail_amount for consistency
        stop_offset = trail_amount
        
        # Default quantity of 1 share (can be overridden by user)
        quantity = 1
        
        # Calculate position sizing based on risk (optional enhancement)
        risk_per_share = trail_amount
        stop_loss_pct = result.get('stop_loss_pct', 0) / 100.0 if result.get('stop_loss_pct') else 0
        
        order = {
            "symbol": symbol,
            "quantity": quantity,
            "trail_amount": round(trail_amount, 2),
            "stop_offset": round(stop_offset, 2),
            "metadata": {
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": result.get('take_profit_price'),
                "stop_loss_pct": result.get('stop_loss_pct'),
                "take_profit_pct": result.get('take_profit_pct'),
                "atr": result.get('atr'),
                "atr_period": result.get('atr_period'),
                "trend_positive": result.get('trend_positive'),
                "date": result.get('date'),
                "risk_per_share": round(risk_per_share, 2)
            }
        }
        orders.append(order)
    
    return {
        "metadata": metadata,
        "orders": orders,
        "total_orders": len(orders)
    }
