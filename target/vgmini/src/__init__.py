"""
Stock Trading ML Framework (package initializer)

This module intentionally avoids importing heavy submodules at import time
(e.g., main_framework, config) to prevent side effects and speed up tools
that import vgmini.src for lightweight utilities like infer_cli.

Public APIs are exposed via lazy imports using __getattr__ (PEP 562).
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

__version__ = "1.0.0"
__author__ = "Trading Framework Team"

# Names we want to expose from submodules
_EXPORTED = {
    'TradingFramework': ('vgmini.src.main_framework', 'TradingFramework'),
    'model_config': ('vgmini.src.config', 'model_config'),
    'backtest_config': ('vgmini.src.config', 'backtest_config'),
    'db_config': ('vgmini.src.config', 'db_config'),
    'EXPERIMENT_CONFIGS': ('vgmini.src.experiment_configs', 'EXPERIMENT_CONFIGS'),
    'get_config': ('vgmini.src.experiment_configs', 'get_config'),
    'create_custom_config': ('vgmini.src.experiment_configs', 'create_custom_config'),
}

__all__ = list(_EXPORTED.keys())

if TYPE_CHECKING:
    # For type checkers only; avoids runtime imports
    from .main_framework import TradingFramework  # noqa: F401
    from .config import model_config, backtest_config, db_config  # noqa: F401
    from .experiment_configs import EXPERIMENT_CONFIGS, get_config, create_custom_config  # noqa: F401


def __getattr__(name: str) -> Any:
    """Lazily import selected symbols on first access.

    This keeps import-time side effects to a minimum while preserving the
    public API surface of vgmini.src.
    """
    target = _EXPORTED.get(name)
    if not target:
        raise AttributeError(f"module 'vgmini.src' has no attribute {name!r}")
    mod_name, attr = target
    import importlib
    mod = importlib.import_module(mod_name)
    value = getattr(mod, attr)
    globals()[name] = value  # cache for subsequent access
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
