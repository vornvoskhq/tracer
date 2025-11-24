"""
Sprint 1 refactor scaffolding: base classes and simple registries.
These are lightweight and do not replace the current pipeline yet; they
provide a foundation for Sprint 2/3 integration.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Type

# Registries
features_registry: Dict[str, Type["FeatureBase"]] = {}
models_registry: Dict[str, Type["ModelBase"]] = {}
targets_registry: Dict[str, Type["TargetStrategyBase"]] = {}


def register_feature(name: str):
    def _wrap(cls: Type[FeatureBase]):
        features_registry[name] = cls
        return cls
    return _wrap


def register_model(name: str):
    def _wrap(cls: Type[ModelBase]):
        models_registry[name] = cls
        return cls
    return _wrap


def register_target(name: str):
    def _wrap(cls: Type[TargetStrategyBase]):
        targets_registry[name] = cls
        return cls
    return _wrap


class FeatureBase:
    """Abstract feature interface.
    Subclasses should implement compute(df, params) and declare dependencies.
    """
    name: str = "feature"
    dependencies: List[str] = []

    def compute(self, df, params: Optional[Dict[str, Any]] = None):  # pragma: no cover
        raise NotImplementedError


class ModelBase:
    """Abstract model interface."""
    supports_dual: bool = True

    def fit(self, X, y):  # pragma: no cover
        raise NotImplementedError

    def predict_scores(self, X):  # pragma: no cover
        raise NotImplementedError

    def feature_importance(self) -> Dict[str, float]:  # pragma: no cover
        return {}

    def save(self, path: str):  # pragma: no cover
        pass

    def load(self, path: str):  # pragma: no cover
        pass


class TargetStrategyBase:
    """Abstract target strategy interface."""

    def generate_targets(self, df, params: Optional[Dict[str, Any]] = None):  # pragma: no cover
        raise NotImplementedError


class ExperimentBase:
    """Abstract experiment orchestrator.
    Sprint 1 provides the interface only; current pipeline uses TradingFramework.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run(self) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError
