#!/usr/bin/env python3
"""
Serverless CLI for on-demand VGMini inference.
Reads candles from stdin (JSON), runs feature engineering and model inference for the
specified experiment, and prints per-candle signals as JSON to stdout.

Usage:
  python -m vgmini.infer_cli --experiment <id> --symbol <SYM> --timeframe 1d < candles.json

Input JSON schema:
{
  "candles": [
    {"timestamp": 1672531200, "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.7, "volume": 123456},
    ...
  ]
}

Output JSON schema:
{
  "experiment": "...",
  "symbol": "...",
  "timeframe": "1d",
  "signals": [
    {"timestamp": 1672531200, "buy_signal": 1, "sell_signal": 0,
     "buy_probability": 0.72, "sell_probability": 0.18,
     "success_buy": true, "success_sell": null},
    ...
  ]
}
"""
import sys
import json
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project paths are on sys.path based on VGM_BASEDIR or this file location
try:
    BASE_DIR = Path(os.environ.get("VGM_BASEDIR") or Path(__file__).resolve().parents[1])
    # Insert vgmini package root (BASE_DIR) and repository root (parent of BASE_DIR)
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    PARENT_DIR = BASE_DIR.parent
    if str(PARENT_DIR) not in sys.path:
        sys.path.insert(0, str(PARENT_DIR))
except Exception as _e:
    print(json.dumps({"warn": f"Path setup failed: {_e}"}), file=sys.stderr)

# Local imports
try:
    from vgmini.src.feature_engineering import FeatureEngineer
    from vgmini.src.ml_models import StockSignalPredictor
except Exception as _imp_err:
    # Fallback if module path resolution fails; try relative imports when executed as module
    try:
        from .src.feature_engineering import FeatureEngineer  # type: ignore
        from .src.ml_models import StockSignalPredictor  # type: ignore
    except Exception as _rel_err:
        print(json.dumps({"error": f"Import failure: abs={str(_imp_err)}; rel={str(_rel_err)}", "sys_path": sys.path[:5]}), file=sys.stderr)
        sys.exit(1)


def load_experiment_artifacts(experiment_id: str):
    # Resolve project base directory (vgmini package root)
    base_dir = Path(os.environ.get("VGM_BASEDIR") or Path(__file__).resolve().parents[1])
    results_path = base_dir / "results" / experiment_id / "experiment_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Experiment results not found: {results_path}")
    with results_path.open("r") as f:
        results = json.load(f)
    # Prefer explicit path from results, else fallback to trained_model.pkl in models dir
    model_path_value = results.get("model_results", {}).get("model_path")
    model_path = None
    if model_path_value:
        p = Path(model_path_value)
        model_path = p if p.is_absolute() else (base_dir / p)
    else:
        # Fallback path
        default_pkl = base_dir / "results" / experiment_id / "models" / "trained_model.pkl"
        if default_pkl.exists():
            model_path = default_pkl
        else:
            # Try any .pkl in models dir
            models_dir = base_dir / "results" / experiment_id / "models"
            if models_dir.exists():
                pickles = sorted(models_dir.glob("*.pkl"))
                if pickles:
                    model_path = pickles[-1]
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"Model file not found. Searched: results.model_path={model_path_value}, default={base_dir / 'results' / experiment_id / 'models' / 'trained_model.pkl'}")
    model_config = results.get("config", {}).get("model_config", {})
    backtest_config = results.get("config", {}).get("backtest_config", {})
    return results, str(model_path), model_config, backtest_config


def build_dataframe(candles: list, symbol: str) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    # Normalize timestamp units to seconds if in ms
    if not df.empty and df['timestamp'].max() > 4_102_444_800:
        df['timestamp'] = (df['timestamp'] // 1000).astype(int)
    df['symbol'] = symbol
    # Create datetime for possible downstream use
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    # Enforce column types
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Sort chronological
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--timeframe', default='1d')
    parser.add_argument('--threshold', type=float, default=None)
    # Accept but ignore stock-chart2 threshold arguments for compatibility
    parser.add_argument('--buy-threshold', type=float, default=None, help='Ignored for compatibility')
    parser.add_argument('--sell-threshold', type=float, default=None, help='Ignored for compatibility')
    args = parser.parse_args()

    try:
        payload = sys.stdin.read()
        data = json.loads(payload)
        candles = data.get('candles', [])
        if not candles:
            print(json.dumps({"error": "No candles provided"}), file=sys.stderr)
            sys.exit(2)

        # Load experiment artifacts
        results, model_path, model_cfg, bt_cfg = load_experiment_artifacts(args.experiment)

        # Build DataFrame
        df = build_dataframe(candles, args.symbol)

        # Feature engineering
        # Mimic FeatureEngineer usage in framework: construct with config
        try:
            from vgmini.src.experiment_configs import ExperimentConfig
        except ImportError:
            from .src.experiment_configs import ExperimentConfig  # type: ignore
        # Build a custom config dict for FeatureEngineer that clamps windows for small inputs
        n = len(df)
        ema_src = model_cfg.get('ema_periods', [9, 21, 50]) if isinstance(model_cfg, dict) else [9, 21, 50]
        ema_adj = sorted({max(1, min(int(p), n)) for p in ema_src}) or [1]
        volume_win_adj = max(1, min(int(model_cfg.get('volume_window', 20)) if isinstance(model_cfg, dict) else 20, n))
        # Technical indicators with safe window sizes
        ti = {
            'rsi': {
                'period': max(2, min(14, n)),
                'oversold_threshold': 30,
                'overbought_threshold': 70,
            },
            'macd': {
                'fast_period': max(2, min(12, n)),
                'slow_period': max(3, min(26, n)),
                'signal_period': max(2, min(9, n)),
            },
            'bollinger_bands': {
                'period': max(2, min(20, n)),
                'std_dev': 2,
            },
            'momentum': {
                'window': max(1, min(5, n)),
                'std_window': max(2, min(20, n)),
                'std_multiplier': 1.5,
            },
            'roc_periods': [p for p in [5, 10] if p <= n] or [1],
            'volume_roc_periods': [p for p in [5] if p <= n] or [1],
        }
        custom_cfg = {}
        if isinstance(model_cfg, dict):
            custom_cfg.update(model_cfg)
        if isinstance(bt_cfg, dict):
            custom_cfg.update(bt_cfg)
        custom_cfg.update({
            'ema_periods': ema_adj,
            'volume_window': volume_win_adj,
            'technical_indicators': ti,
        })
        fe = FeatureEngineer(config=custom_cfg)
        # Also set thresholds and horizons on the underlying config object used by FeatureEngineer
        try:
            if isinstance(model_cfg, dict):
                if 'buy_threshold' in model_cfg: fe.config.buy_threshold = float(model_cfg['buy_threshold'])
                if 'sell_threshold' in model_cfg: fe.config.sell_threshold = float(model_cfg['sell_threshold'])
                if 'forecast_horizon' in model_cfg: fe.config.forecast_horizon = int(model_cfg['forecast_horizon'])
            # Apply clamped EMA and volume window so methods that read fe.config use them
            fe.config.ema_periods = ema_adj
            fe.config.volume_window = volume_win_adj
        except Exception:
            pass
        # Build features using the same steps as training (without trimming rows)
        def _build_features(fe_obj, df_in):
            df2 = df_in.copy()
            df2 = fe_obj.calculate_ema(df2, fe_obj.config.ema_periods)
            df2 = fe_obj.calculate_macd(df2)
            df2 = fe_obj.calculate_heikin_ashi(df2)
            df2 = fe_obj.calculate_volume_indicators(df2)
            df2 = fe_obj.calculate_additional_indicators(df2)
            df2 = fe_obj.create_crossover_signals(df2)
            return df2
        df_feat = _build_features(fe, df)

        # Prepare features for model
        available_features = fe.get_all_feature_names()
        predictor = StockSignalPredictor()
        predictor.load_models(model_path)
        # Build feature matrix aligned to model's expected features
        desired_cols = predictor.feature_names if predictor.feature_names else available_features
        # Create a dataframe with exactly the desired columns in order, filling missing with 0.0
        df_ready = pd.DataFrame(index=df_feat.index)
        for col in desired_cols:
            if col in df_feat.columns:
                df_ready[col] = df_feat[col]
            else:
                df_ready[col] = 0.0
        X = df_ready[desired_cols].fillna(0).values

        # Predict base signals
        conf_thr = float(args.threshold if args.threshold is not None else bt_cfg.get('confidence_threshold', 0.5))
        base = predictor.predict_signals(X, confidence_threshold=conf_thr)

        # Compute targets (success labels) with same horizons
        df_targets = fe.create_target_variables(df_feat.copy())
        # Align sizes if needed
        out = []
        # Prepare timestamps from df_feat (fallback to original df)
        if 'timestamp' in df_feat.columns:
            ts_series = df_feat['timestamp'].astype('int64')
        else:
            ts_series = df['timestamp'].astype('int64') if 'timestamp' in df.columns else pd.Series([0] * len(df_ready), index=df_ready.index)
        for i in range(len(df_ready)):
            ts_val = int(ts_series.iloc[i]) if i < len(ts_series) else 0
            buy_sig = int(base['buy_signals'][i]) if i < len(base['buy_signals']) else 0
            sell_sig = int(base['sell_signals'][i]) if i < len(base['sell_signals']) else 0
            buy_p = float(base['buy_probability'][i]) if i < len(base['buy_probability']) else 0.0
            sell_p = float(base['sell_probability'][i]) if i < len(base['sell_probability']) else 0.0
            sb = None
            ss = None
            if 'target_buy' in df_targets.columns and i < len(df_targets):
                tb = df_targets.iloc[i]['target_buy']
                sb = (bool(int(tb)) if not pd.isna(tb) else None)
            if 'target_sell' in df_targets.columns and i < len(df_targets):
                tsell = df_targets.iloc[i]['target_sell']
                ss = (bool(int(tsell)) if not pd.isna(tsell) else None)
            out.append({
                'timestamp': ts_val,
                'buy_signal': buy_sig,
                'sell_signal': sell_sig,
                'buy_probability': buy_p,
                'sell_probability': sell_p,
                'success_buy': sb,
                'success_sell': ss,
            })

        result = {
            'experiment': args.experiment,
            'symbol': args.symbol,
            'timeframe': args.timeframe,
            'signals': out,
        }
        print(json.dumps(result))
        sys.exit(0)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(json.dumps({"error": str(e), "trace": tb}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
