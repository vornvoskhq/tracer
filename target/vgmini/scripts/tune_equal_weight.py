#!/usr/bin/env python3
"""
Tuner for equal_weight_dual parameters using DB-loaded candles.

It evaluates grids of parameters on the last N bars for a given symbol, using
VGMini's FeatureEngineer and StockSignalPredictor (loading a trained model from
results/<experiment>/models/trained_model.pkl so the feature order/scaler match).

Usage:
  python -m vgmini.scripts.tune_equal_weight \
      --experiment equal_weight_dual \
      --symbol SPY \
      --lookback 200 \
      --temp 0.5 0.6 0.7 0.75 0.8 1.0 \
      --act 0.0 0.05 0.1 \
      --hyst 0.00 0.03 0.05 0.07

Outputs a ranked table of parameter sets with noise metrics and a recommended
config snippet.
"""
import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Add project to path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

SRC_DIR = BASE_DIR / "vgmini" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# Robust imports (match infer_cli behavior)
try:
    from vgmini.src.config import model_config as global_model_config
    from vgmini.src.feature_engineering import FeatureEngineer
    from vgmini.src.ml_models import StockSignalPredictor
    from vgmini.src.data_loader import StockDataLoader
except Exception:
    try:
        from config import model_config as global_model_config  # type: ignore
        from feature_engineering import FeatureEngineer  # type: ignore
        from ml_models import StockSignalPredictor  # type: ignore
        from data_loader import StockDataLoader  # type: ignore
    except Exception as _e:
        print(json.dumps({"error": f"Import failure: {_e}", "sys_path": sys.path[:5]}), file=sys.stderr)
        sys.exit(1)


def load_trained(experiment: str):
    base = BASE_DIR
    res = base / 'results' / experiment / 'experiment_results.json'
    if not res.exists():
        raise FileNotFoundError(f"Run the experiment first: {res} not found")
    with res.open('r') as f:
        meta = json.load(f)
    model_path = meta.get('model_results', {}).get('model_path')
    if model_path:
        mp = Path(model_path)
        model_path = mp if mp.is_absolute() else (base / mp)
    else:
        model_path = base / 'results' / experiment / 'models' / 'trained_model.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"Model pickle not found at {model_path}")
    cfg_model = meta.get('config', {}).get('model_config', {})
    cfg_bt = meta.get('config', {}).get('backtest_config', {})
    return str(model_path), cfg_model, cfg_bt


def get_last_n(loader: StockDataLoader, symbol: str, n: int) -> pd.DataFrame:
    df_all = loader.get_symbol_data(symbol)
    if df_all is None or df_all.empty:
        raise RuntimeError(f"No data for {symbol}")
    df_all = df_all.sort_values('datetime').reset_index(drop=True)
    if len(df_all) > n:
        df = df_all.iloc[-n:].copy()
    else:
        df = df_all.copy()
    # Important: reset index so FeatureEngineer.loc[i] works (it expects RangeIndex)
    return df.reset_index(drop=True)


def evaluate(params, df_feat: pd.DataFrame, desired_cols: list, predictor: StockSignalPredictor):
    # Temporarily override predictor config knobs
    # We do not retrain; we only control inference behavior that affects probabilities
    # For equal_weight in our code, temperature/activation_threshold live inside the model
    # but we can emulate via post-processing using fraction logic on binary features.
    # Simplest approach: rebuild a temporary equal-weight proba from raw features:
    # p_buy = sigmoid(((frac_on - 0.5)/temp)) ; p_sell analogous on fraction-off
    temp = params['temp']
    act = params['act']
    hyst = params['hyst']

    # Build X
    df_ready = pd.DataFrame(index=df_feat.index)
    for c in desired_cols:
        df_ready[c] = df_feat.get(c, 0.0)
    X = df_ready[desired_cols].fillna(0).values

    # Base predictor probabilities (for reference)
    base = predictor.predict_signals(X, confidence_threshold=0.0)
    bp = np.array(base['buy_probability'], dtype=float)
    sp = np.array(base['sell_probability'], dtype=float)

    # Also compute a simple fraction-based proba using the configured features
    # Detect binary features (0/1) and compute fraction on for each row
    Xbin = (X > act).astype(float)
    frac_on = Xbin.mean(axis=1)
    # sigmoid helper
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    # No prior calibration shift here (intercept=0)
    buy_p = sigmoid((frac_on - 0.5) / max(1e-6, temp))
    # For sell, fraction-off
    frac_off = 1.0 - frac_on
    sell_p = sigmoid((frac_off - 0.5) / max(1e-6, temp))

    # Apply hysteresis when deciding signals
    buy_sig = (buy_p > (sell_p + hyst)).astype(int)
    sell_sig = (sell_p > (buy_p + hyst)).astype(int)

    # Metrics to reduce noise
    buys = int(buy_sig.sum())
    sells = int(sell_sig.sum())
    buy_gt_sell = int((buy_p > sell_p).sum())
    sell_gt_buy = int((sell_p > buy_p).sum())
    # Continuation: sum of runs length >= 2
    def runs(sig):
        total = 0
        i = 0
        n = len(sig)
        while i < n:
            if sig[i]:
                j = i
                while j+1 < n and sig[j+1]:
                    j += 1
                if j - i + 1 >= 2:
                    total += (j - i + 1)
                i = j + 1
            else:
                i += 1
        return total
    cont_buy = runs(buy_sig)
    cont_sell = runs(sell_sig)

    # Penalize noisy sells in uptrends: noise score = sells + sell_gt_buy + cont_sell*0.5
    # Basic long-only PnL proxy using close-to-close returns
    prices = df_feat['close'].astype(float).values if 'close' in df_feat.columns else None
    pnl = 0.0
    sharpe = 0.0
    max_dd = 0.0
    trades = 0
    if prices is not None and len(prices) >= 2:
        rets = np.zeros(len(prices))
        rets[1:] = (prices[1:] - prices[:-1]) / np.maximum(prices[:-1], 1e-12)
        pos = np.zeros(len(prices), dtype=int)
        in_pos = False
        for i in range(len(buy_sig)):
            if in_pos:
                if sell_sig[i]:
                    in_pos = False
                else:
                    pos[i] = 1
            else:
                if buy_sig[i]:
                    in_pos = True
                    pos[i] = 1
                    trades += 1
        port_rets = rets * np.roll(pos, 1)  # hold from previous close into next close
        eq = np.cumprod(1.0 + port_rets)
        pnl = float(eq[-1] - 1.0)
        std = float(np.std(port_rets[1:]) + 1e-12)
        mean = float(np.mean(port_rets[1:]))
        sharpe = float((mean / std) * np.sqrt(252.0)) if std > 0 else 0.0
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.maximum(peak, 1e-12)
        max_dd = float(np.min(dd)) if len(dd) else 0.0

    noise = sells + sell_gt_buy + 0.5 * cont_sell

    return {
        'temp': temp,
        'act': act,
        'hyst': hyst,
        'buys': buys,
        'sells': sells,
        'buy_gt_sell': buy_gt_sell,
        'sell_gt_buy': sell_gt_buy,
        'cont_buy': cont_buy,
        'cont_sell': cont_sell,
        'noise': float(noise),
        'buy_max': float(buy_p.max()),
        'sell_max': float(sell_p.max()),
        'buy_avg': float(buy_p.mean()),
        'sell_avg': float(sell_p.mean()),
        'pnl': float(pnl),
        'sharpe': float(sharpe),
        'max_dd': float(max_dd),
        'trades': int(trades),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', default='equal_weight_dual')
    ap.add_argument('--symbol', default='SPY')
    ap.add_argument('--lookback', type=int, default=200)
    ap.add_argument('--temp', type=float, nargs='+', default=[0.5, 0.6, 0.7, 0.75, 0.8, 1.0])
    ap.add_argument('--act', type=float, nargs='+', default=[0.0, 0.05, 0.1])
    ap.add_argument('--hyst', type=float, nargs='+', default=[0.00, 0.03, 0.05, 0.07])
    args = ap.parse_args()

    # Load
    model_path, cfg_model, cfg_bt = load_trained(args.experiment)

    loader = StockDataLoader()
    df = get_last_n(loader, args.symbol, args.lookback)

    # Use experiment cfg for feature calc parameters (EMA, MACD windows, etc.)
    fe_cfg = {}
    if isinstance(cfg_model, dict):
        fe_cfg.update(cfg_model)
    if isinstance(cfg_bt, dict):
        fe_cfg.update(cfg_bt)
    fe = FeatureEngineer(config=fe_cfg)

    # Build features exactly like infer_cli does
    df_feat = df.copy()
    df_feat = fe.calculate_ema(df_feat, getattr(fe.config, 'ema_periods', [9, 21, 50]))
    df_feat = fe.calculate_macd(df_feat)
    df_feat = fe.calculate_heikin_ashi(df_feat)
    df_feat = fe.calculate_volume_indicators(df_feat)
    df_feat = fe.calculate_additional_indicators(df_feat)
    df_feat = fe.create_crossover_signals(df_feat)

    # Load predictor and align feature order
    predictor = StockSignalPredictor()
    predictor.load_models(model_path)
    desired_cols = predictor.feature_names

    # Grid search
    results = []
    for t in args.temp:
        for a in args.act:
            for h in args.hyst:
                res = evaluate({'temp': t, 'act': a, 'hyst': h}, df_feat, desired_cols, predictor)
                results.append(res)

    # Rank by composite: prioritize low noise, then higher cont_buy, then higher Sharpe, then fewer sells
    results.sort(key=lambda r: (r['noise'], -r['cont_buy'], -r.get('sharpe', 0.0), r['sells']))

    # Print top 10
    print("\nTop parameter sets (low noise, high continuation, higher Sharpe):")
    print("temp  act   hyst  |  buys  sells  cont_buy  cont_sell  pnl     sharpe  max_dd   buy_max  sell_max  buy_avg  sell_avg  noise")
    for r in results[:10]:
        print(f"{r['temp']:<4.2f}  {r['act']:<4.2f}  {r['hyst']:<4.2f}  |  {r['buys']:<4d}  {r['sells']:<5d}  {r['cont_buy']:<9d}  {r['cont_sell']:<9d}  {r.get('pnl',0):<7.3f}  {r.get('sharpe',0):<6.2f}  {r.get('max_dd',0):<7.3f}  {r['buy_max']:<7.3f}  {r['sell_max']:<8.3f}  {r['buy_avg']:<7.3f}  {r['sell_avg']:<8.3f}  {r['noise']:<5.1f}")

    # Recommend first result
    if results:
        best = results[0]
        print("\nRecommended config snippet:")
        print("model:")
        print("  type: equal_weight")
        print("  ew_correlation_signs: false")
        print("  ew_calibrate_prior: false")
        print("  ew_activation_threshold:", best['act'])
        print("  ew_temperature:", best['temp'])
        print("  ew_buy_aggregation: fraction")
        print("  ew_sell_aggregation: fraction")
        print("hysteresis_margin:", best['hyst'])


if __name__ == '__main__':
    main()
