"""
Core base classes and simple registries aligning with the refactor roadmap.

These are intentionally lightweight and do not change the current execution path
(the existing pipeline in TradingFramework remains the source of truth).
They provide extendable abstractions that future work can integrate with.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple, Type
import json
import pickle
import numpy as np
import pandas as pd

# ----------------------------
# Simple registries
# ----------------------------
features_registry: Dict[str, Type["FeatureBase"]] = {}
models_registry: Dict[str, Type["ModelBase"]] = {}
targets_registry: Dict[str, Type["TargetStrategyBase"]] = {}


def register_feature(name: str):
    def decorator(cls: Type["FeatureBase"]) -> Type["FeatureBase"]:
        features_registry[name] = cls
        return cls
    return decorator


def register_model(name: str):
    def decorator(cls: Type["ModelBase"]) -> Type["ModelBase"]:
        models_registry[name] = cls
        return cls
    return decorator


def register_target(name: str):
    def decorator(cls: Type["TargetStrategyBase"]) -> Type["TargetStrategyBase"]:
        targets_registry[name] = cls
        return cls
    return decorator


# ----------------------------
# Base classes
# ----------------------------
@dataclass
class FeatureBase:
    """Base class for a single feature computation.
    Subclasses should implement compute() and may define dependencies.
    """
    name: str
    dependencies: List[str] = field(default_factory=list)
    lag: int = 0  # how many rows are required in the past

    def compute(self, df: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        raise NotImplementedError


class ModelBase:
    """Base class for prediction models. Minimal interface used by the roadmap.
    """
    supports_dual: bool = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ModelBase":
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "ModelBase":
        with open(path, "rb") as f:
            return pickle.load(f)


class TargetStrategyBase:
    """Base class for generating targets from a price series.
    Concrete strategies should implement generate_targets().
    """
    def generate_targets(
        self,
        df: pd.DataFrame,
        buy_threshold: float,
        sell_threshold: float,
        buy_horizon: Optional[int] = None,
        sell_horizon: Optional[int] = None,
        symmetric_horizon: Optional[int] = None,
        **params,
    ) -> pd.DataFrame:
        raise NotImplementedError


# ----------------------------
# Concrete target strategies
# ----------------------------
@register_target("dual")
@register_target("dual_signal")
class DualSignalTargets(TargetStrategyBase):
    def generate_targets(
        self,
        df: pd.DataFrame,
        buy_threshold: float,
        sell_threshold: float,
        buy_horizon: Optional[int] = None,
        sell_horizon: Optional[int] = None,
        symmetric_horizon: Optional[int] = 5,
        **params,
    ) -> pd.DataFrame:
        df = df.copy().sort_values("datetime").reset_index(drop=True)
        horizon = int(symmetric_horizon or 5)
        # future close
        future_close = df["close"].shift(-horizon)
        # validate time strictly increases
        future_ts = df["timestamp"].shift(-horizon)
        current_ts = df["timestamp"]
        valid = future_ts > current_ts
        future_close[~valid] = np.nan
        ret = (future_close - df["close"]) / df["close"]
        df["target_buy"] = np.where(ret.notna() & (ret > buy_threshold), 1, 0)
        df["target_sell"] = np.where(ret.notna() & (ret < -sell_threshold), 1, 0)
        return df


@register_target("buy_only")
@register_target("buy")
class BuyOnlyTargets(TargetStrategyBase):
    def generate_targets(
        self,
        df: pd.DataFrame,
        buy_threshold: float,
        sell_threshold: float,
        buy_horizon: Optional[int] = None,
        sell_horizon: Optional[int] = None,
        symmetric_horizon: Optional[int] = 5,
        **params,
    ) -> pd.DataFrame:
        df = df.copy().sort_values("datetime").reset_index(drop=True)
        horizon = int(symmetric_horizon or 5)
        future_close = df["close"].shift(-horizon)
        future_ts = df["timestamp"].shift(-horizon)
        current_ts = df["timestamp"]
        valid = future_ts > current_ts
        future_close[~valid] = np.nan
        ret = (future_close - df["close"]) / df["close"]
        df["target_buy"] = np.where(ret.notna() & (ret > buy_threshold), 1, 0)
        # No sell target in buy-only; keep zeros
        df["target_sell"] = 0
        return df


# ----------------------------
# Additional target strategies (Sprint 4)
# ----------------------------
@register_target("sell")
class SellOnlyTargets(TargetStrategyBase):
    def generate_targets(
        self,
        df: pd.DataFrame,
        buy_threshold: float,
        sell_threshold: float,
        buy_horizon: Optional[int] = None,
        sell_horizon: Optional[int] = None,
        symmetric_horizon: Optional[int] = 5,
        **params,
    ) -> pd.DataFrame:
        df = df.copy().sort_values("datetime").reset_index(drop=True)
        horizon = int(symmetric_horizon or 5)
        future_close = df["close"].shift(-horizon)
        future_ts = df["timestamp"].shift(-horizon)
        current_ts = df["timestamp"]
        valid = future_ts > current_ts
        future_close[~valid] = np.nan
        ret = (future_close - df["close"]) / df["close"]
        df["target_buy"] = 0
        df["target_sell"] = np.where(ret.notna() & (ret < -sell_threshold), 1, 0)
        return df

@register_target("breakout")
class BreakoutTargets(TargetStrategyBase):
    def generate_targets(self, df: pd.DataFrame,
                         buy_threshold: float, sell_threshold: float,
                         buy_horizon: Optional[int] = None,
                         sell_horizon: Optional[int] = None,
                         symmetric_horizon: Optional[int] = 5,
                         **params) -> pd.DataFrame:
        df = df.copy().sort_values("datetime").reset_index(drop=True)
        lookback = int(params.get("lookback_window", 20))
        confirm = int(params.get("confirm_window", 1))
        # breakout: close breaks above rolling max by buy_threshold
        rolling_max = df['close'].rolling(lookback, min_periods=max(2, lookback//2)).max()
        cond = (df['close'] > (rolling_max * (1 + float(params.get("breakout_threshold", buy_threshold)))))
        # confirmation by next confirm_window bars
        signal = cond.shift(1).rolling(confirm).max() if confirm > 1 else cond
        df['target_buy'] = signal.fillna(False).astype(int)
        df['target_sell'] = 0
        return df

@register_target("reversal")
class ReversalTargets(TargetStrategyBase):
    def generate_targets(self, df: pd.DataFrame,
                         buy_threshold: float, sell_threshold: float,
                         buy_horizon: Optional[int] = None,
                         sell_horizon: Optional[int] = None,
                         symmetric_horizon: Optional[int] = 5,
                         **params) -> pd.DataFrame:
        df = df.copy().sort_values("datetime").reset_index(drop=True)
        lookback = int(params.get("lookback_window", 10))
        confirm = int(params.get("confirm_window", 1))
        # reversal: close below rolling min then next bar closes above previous close by threshold
        rolling_min = df['close'].rolling(lookback, min_periods=max(2, lookback//2)).min()
        touched = df['close'] < rolling_min
        # bullish reversal confirmation
        rev_cond = (df['close'].shift(-1) > df['close'] * (1 + float(params.get("reversal_threshold", buy_threshold))))
        signal = (touched & rev_cond).rolling(confirm).max() if confirm > 1 else (touched & rev_cond)
        df['target_buy'] = signal.fillna(False).astype(int)
        df['target_sell'] = 0
        return df

# ----------------------------
# Example model wrappers (not wired into pipeline yet)
# ----------------------------
@register_model("logistic")
class LogisticModel(ModelBase):
    def __init__(self, **kwargs):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticModel":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


@register_model("xgboost")
class XGBoostModel(ModelBase):
    def __init__(self, **kwargs):
        try:
            import xgboost as xgb
        except Exception as e:
            raise RuntimeError("XGBoost is not installed. Add it to your environment.") from e
        params = dict(kwargs)
        params.setdefault("eval_metric", "logloss")
        self.model = xgb.XGBClassifier(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


# ----------------------------
# Experiment orchestration skeleton
# ----------------------------
class ExperimentBase:
    """Skeleton for a registry-driven experiment orchestrator.
    Not used by the current CLI, but provided to align with the roadmap.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run(self) -> Dict[str, Any]:
        # Placeholder; a future refactor can wire FeatureBase, TargetStrategyBase, and ModelBase here.
        return {"status": "not-implemented", "config": self.config}
