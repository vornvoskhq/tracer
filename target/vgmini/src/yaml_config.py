"""
YAML configuration loader with deep-merge semantics for experiments.
- Loads configs/global.yaml and configs/experiments/<name>.yaml
- Merges them (experiment overrides global) using a deep merge
- Produces a dictionary compatible with ExperimentConfig.from_dict
"""
from __future__ import annotations
from typing import Any, Dict
import os
import json

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Please install pyyaml.") from e


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dictionaries (override wins). Lists are replaced by default.
    This keeps behavior simple and predictable for Sprint 1.
    """
    result: Dict[str, Any] = dict(base)  # shallow copy
    for k, v in (override or {}).items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_yaml_file(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must be a mapping at the root")
    return data


def load_merged_experiment_yaml(exp_name: str) -> Dict[str, Any]:
    """Load and merge global.yaml with a specific experiment YAML.

    Returns a normalized flat dictionary compatible with ExperimentConfig.from_dict.
    """
    global_path = os.path.join("configs", "global.yaml")
    exp_path = os.path.join("configs", "experiments", f"{exp_name}.yaml")

    if not os.path.exists(exp_path):
        raise FileNotFoundError(
            f"Experiment YAML not found: {exp_path}. Create it under configs/experiments/"
        )

    global_cfg = _load_yaml_file(global_path)
    exp_cfg = _load_yaml_file(exp_path)

    merged = _deep_merge(global_cfg, exp_cfg)

    # Normalize into ExperimentConfig-compatible dict
    # Expect structure like:
    # {
    #   name, description,
    #   universe: { symbols: [...] },
    #   dates: { start: "YYYY-MM-DD", end: "YYYY-MM-DD" },
    #   model: {...},
    #   backtest: {...},
    #   features: { enabled: [...], technical_indicators: {...} },
    #   analysis: {...}, visualization: {...}
    # }

    name = merged.get("name", exp_name)
    description = merged.get("description", f"Experiment {exp_name}")

    # Universe (support symbols or cluster via configs/clusters.yaml)
    universe = merged.get("universe", {}) or {}
    symbols = universe.get("symbols", [])
    cluster_name = universe.get("cluster")
    # Prefer cluster if explicitly specified in the experiment's universe block
    exp_universe = exp_cfg.get("universe", {}) if isinstance(exp_cfg, dict) else {}
    prefer_cluster = bool(exp_universe.get("cluster"))
    if cluster_name:
        # Force cluster preference if specified in experiment YAML
        # Temporarily ignore inherited symbols so we can try both sources
        _orig_symbols = symbols
        symbols = []
        # Try YAML clusters first
        clusters_path = os.path.join("configs", "clusters.yaml")
        try:
            clusters_cfg = _load_yaml_file(clusters_path)
            clusters = clusters_cfg.get("clusters", {}) if isinstance(clusters_cfg, dict) else {}
            cluster_symbols = clusters.get(cluster_name)
            if isinstance(cluster_symbols, list) and cluster_symbols:
                symbols = cluster_symbols
        except Exception:
            pass
        
        # Also try cluster-analysis.json at project root
        if not symbols:
            json_path = os.path.join("cluster-analysis.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as jf:
                        data = json.load(jf)
                    # Expected structure: {"clusters": {"name": [symbols...]}} or {"name": [symbols...]}
                    clusters = {}
                    if isinstance(data, dict) and 'clusters' in data and isinstance(data['clusters'], dict):
                        clusters = data['clusters']
                    elif isinstance(data, dict):
                        clusters = data
                    cluster_symbols = clusters.get(cluster_name)
                    if isinstance(cluster_symbols, list) and cluster_symbols:
                        symbols = cluster_symbols
                except Exception:
                    pass
        # If still unresolved, restore original symbols
        if not symbols:
            symbols = _orig_symbols

    # Dates
    dates = merged.get("dates", {}) or {}
    start_date = dates.get("start")
    end_date = dates.get("end")

    # Features
    features = merged.get("features", {}) or {}
    enabled_features = features.get("enabled", [])
    technical_indicators = features.get("technical_indicators", {})

    # Model/backtest/analysis/viz
    model = merged.get("model", {}) or {}
    backtest = merged.get("backtest", {}) or {}
    analysis = merged.get("analysis", {}) or {}
    visualization = merged.get("visualization", {}) or {}

    # Target strategy / ensemble additions
    target_strategy = merged.get("target_strategy", model.get("target_strategy"))
    target_params = merged.get("target_params", model.get("target_params"))
    base_models = model.get("base_models")
    weights = model.get("weights")

    # Build flat config dict compatible with ExperimentConfig
    cfg: Dict[str, Any] = {
        "name": name,
        "description": description,
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
        # Model targets/horizon
        "buy_threshold": model.get("buy_threshold"),
        "sell_threshold": model.get("sell_threshold"),
        "forecast_horizon": model.get("forecast_horizon"),
        "buy_forecast_horizon": model.get("buy_forecast_horizon"),
        "sell_forecast_horizon": model.get("sell_forecast_horizon"),
        "confidence_threshold": backtest.get("confidence_threshold", model.get("confidence_threshold")),
        # Feature engineering
        "ema_periods": model.get("ema_periods"),
        "volume_window": model.get("volume_window"),
        # Training/model selection
        "test_size": model.get("test_size"),
        "class_weight": model.get("class_weight"),
        "penalty": model.get("penalty"),
        "solver": model.get("solver"),
        "max_iter": model.get("max_iter"),
        "tol": model.get("tol"),
        "model_type": model.get("type", model.get("model_type")),
        "target_strategy": target_strategy,
        "target_params": target_params,
        "calibration": model.get("calibration"),
        "tuning": model.get("tuning"),
        "C": model.get("C"),
        "base_models": base_models,
        "weights": weights, 
        # Equal-weight knobs
        "ew_correlation_signs": model.get("ew_correlation_signs"),
        "ew_calibrate_prior": model.get("ew_calibrate_prior"),
        "ew_activation_threshold": model.get("ew_activation_threshold"),
        "ew_temperature": model.get("ew_temperature"),
        "ew_buy_aggregation": model.get("ew_buy_aggregation"),
        "ew_sell_aggregation": model.get("ew_sell_aggregation"),
        # XGBoost
        "n_estimators": model.get("n_estimators"),
        "max_depth": model.get("max_depth"),
        "learning_rate": model.get("learning_rate"),
        "subsample": model.get("subsample"),
        "colsample_bytree": model.get("colsample_bytree"),
        # Backtest
        "initial_capital": backtest.get("initial_capital"),
        "commission": backtest.get("commission"),
        "slippage": backtest.get("slippage"),
        "max_position_size": backtest.get("max_position_size"),
        "threshold_strategy": backtest.get("threshold_strategy"),
        "threshold_percentile": backtest.get("threshold_percentile"),
        "threshold_window": backtest.get("threshold_window"),
        "auto_threshold_fallback": backtest.get("auto_threshold_fallback"),
        "fallback_percentile": backtest.get("fallback_percentile"),
        "fallback_window": backtest.get("fallback_window"),
        "buy_consecutive_days": backtest.get("buy_consecutive_days"),
        "sell_consecutive_days": backtest.get("sell_consecutive_days"),
        "hysteresis_margin": backtest.get("hysteresis_margin"),
        "trade_cooldown_days": backtest.get("trade_cooldown_days"),
        "min_holding_days": backtest.get("min_holding_days"),
        # Features/analysis/viz
        "technical_indicators": technical_indicators,
        "enabled_features": enabled_features,
        "analysis": analysis,
        "visualization": visualization,
    }

    # Remove None values so ExperimentConfig defaults can apply
    cfg = {k: v for k, v in cfg.items() if v is not None}
    return cfg


def list_yaml_experiments() -> Dict[str, Dict[str, Any]]:
    """Discover YAML experiments under configs/experiments and return minimal metadata.
    Returns mapping: name -> {path, description}
    """
    base = os.path.join("configs", "experiments")
    result: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(base):
        return result
    for fname in sorted(os.listdir(base)):
        if not fname.endswith(".yaml"):
            continue
        path = os.path.join(base, fname)
        try:
            data = _load_yaml_file(path)
            name = data.get("name", os.path.splitext(fname)[0])
            desc = data.get("description", "No description")
            result[name] = {"path": path, "description": desc}
        except Exception:
            # Skip unreadable files
            continue
    return result
