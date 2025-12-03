import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


# There are no built-in prompt presets or model defaults in code. All presets,
# the default model, pricing information, and UI state must be defined in
# app_config.json.
DEFAULT_PRESETS: Dict[str, Dict[str, str]] = {}


DEFAULT_CONFIG: Dict[str, Any] = {}


def _config_path() -> Path:
    """
    Return the path to the app_config.json file at the repository root.
    """
    root = Path(__file__).resolve().parent.parent
    return root / "app_config.json"


def load_llm_config() -> Dict[str, Any]:
    """
    Load application configuration from app_config.json, falling back to defaults.

    The returned dict always contains at least the following keys:
      - model: str
      - temperature: float
      - max_tokens: int | None
      - default_prompt_preset: str
      - presets: dict[preset_id, {label, template}]
      - models: list[str]
      - verbose_logging: bool
      - ui: dict (may be empty)
    """
    config = deepcopy(DEFAULT_CONFIG)

    path = _config_path()
    if not path.exists():
        # For this tool, configuration is required. Failing fast is preferable
        # to silently running with surprising defaults.
        raise FileNotFoundError(
            f"Required configuration file app_config.json not found at: {path}"
        )

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            # Shallow merge for top-level keys we care about
            for key in (
                "model",
                "temperature",
                "max_tokens",
                "default_prompt_preset",
                "presets",
                "models",
                "model_prices",
                "ui",
            ):
                if key in raw:
                    config[key] = raw[key]
    except Exception as exc:
        # If the config exists but cannot be parsed, treat this as a hard error
        # rather than silently falling back to defaults.
        raise RuntimeError(f"Failed to parse app_config.json: {exc}") from exc

    # Load presets directly from configuration; there are no built-in prompt defaults.
    presets = config.get("presets") or {}
    if not isinstance(presets, dict) or not presets:
        raise RuntimeError(
            "No LLM presets defined in app_config.json['presets']. "
            "At least one preset with a non-empty template is required."
        )

    merged_presets: Dict[str, Dict[str, str]] = {}

    # Normalize templates so that any stored newline escape sequences become
    # real newlines for easier editing in the LLM config dialog.
    #
    # This is conservative on purpose:
    #   - It preserves real newlines already present in strings.
    #   - It only converts explicit "\n" and "\r\n" escape sequences.
    def _normalize_template(t: str) -> str:
        # Collapse Windows-style escape sequences to "\n"
        t = t.replace("\\r\\n", "\\n")
        # Turn explicit "\n" escapes into real newlines
        t = t.replace("\\n", "\n")
        return t

    # Use only presets from config; there are no code-defined defaults.
    for pid, pconf in presets.items():
        if not isinstance(pconf, dict):
            continue
        label = str(pconf.get("label", pid))
        raw_template = str(pconf.get("template", "")).strip()
        if not raw_template:
            raise RuntimeError(
                f"Preset '{pid}' in app_config.json['presets'] must define a non-empty 'template'."
            )
        template = _normalize_template(raw_template)
        merged_presets[pid] = {"label": label, "template": template}

    config["presets"] = merged_presets

    # Ensure default_prompt_preset is valid and refers to a real preset.
    default_preset = config.get("default_prompt_preset")
    if not isinstance(default_preset, str) or default_preset not in merged_presets:
        raise RuntimeError(
            "app_config.json['default_prompt_preset'] must be set to the ID of an existing preset."
        )

    # Keep models and model_prices in sync:
    # - model_prices is the authoritative source
    # - models is derived from its keys for the UI combo box
    prices = config.get("model_prices")
    if isinstance(prices, dict) and prices:
        # Deterministic ordering for stable UI
        derived_models = sorted(str(m) for m in prices.keys())
        config["models"] = derived_models

    return config


def save_llm_config(config: Dict[str, Any]) -> None:
    """
    Persist application configuration to app_config.json.

    Only known keys are written; unknown keys are ignored to keep the file tidy.
    """
    path = _config_path()
    to_save: Dict[str, Any] = {}

    model = config.get("model")
    if isinstance(model, str):
        to_save["model"] = model

    temperature = config.get("temperature")
    if isinstance(temperature, (int, float)):
        to_save["temperature"] = float(temperature)

    max_tokens = config.get("max_tokens")
    if isinstance(max_tokens, int):
        to_save["max_tokens"] = max_tokens
    else:
        to_save["max_tokens"] = None

    default_preset = config.get("default_prompt_preset")
    if isinstance(default_preset, str):
        to_save["default_prompt_preset"] = default_preset

    presets = config.get("presets") or {}
    cleaned_presets: Dict[str, Dict[str, str]] = {}
    if isinstance(presets, dict):
        for pid, pconf in presets.items():
            if not isinstance(pconf, dict):
                continue
            label = str(pconf.get("label", pid))
            template = str(pconf.get("template", DEFAULT_PRESETS.get(pid, {}).get("template", "{code}")))
            cleaned_presets[pid] = {"label": label, "template": template}
    to_save["presets"] = cleaned_presets

    # Optional per-model price table (authoritative list of models).
    model_prices = config.get("model_prices")
    if isinstance(model_prices, dict):
        cleaned_prices: Dict[str, Dict[str, float]] = {}
        cleaned_models: list[str] = []
        for model_id, price_info in model_prices.items():
            if not isinstance(model_id, str) or not isinstance(price_info, dict):
                continue
            try:
                inp = float(price_info.get("input"))  # type: ignore[arg-type]
                out = float(price_info.get("output"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            cleaned_prices[model_id] = {"input": inp, "output": out}
            cleaned_models.append(model_id)
        if cleaned_prices:
            # model_prices is authoritative; models is derived for the UI.
            to_save["model_prices"] = cleaned_prices
            to_save["models"] = sorted(cleaned_models)

    # Verbose logging flag
    verbose_logging = config.get("verbose_logging")
    if isinstance(verbose_logging, bool):
        to_save["verbose_logging"] = verbose_logging

    # Column / row visibility toggles
    show_caller_column = config.get("show_caller_column")
    if isinstance(show_caller_column, bool):
        to_save["show_caller_column"] = show_caller_column

    show_phase_column = config.get("show_phase_column")
    if isinstance(show_phase_column, bool):
        to_save["show_phase_column"] = show_phase_column

    hide_import_rows = config.get("hide_import_rows")
    if isinstance(hide_import_rows, bool):
        to_save["hide_import_rows"] = hide_import_rows

    # Optional per-model price table (USD per 1M tokens) for cost estimation.
    model_prices = config.get("model_prices")
    if isinstance(model_prices, dict):
        cleaned_prices: Dict[str, Dict[str, float]] = {}
        for model_id, price_info in model_prices.items():
            if not isinstance(model_id, str) or not isinstance(price_info, dict):
                continue
            try:
                inp = float(price_info.get("input"))  # type: ignore[arg-type]
                out = float(price_info.get("output"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            cleaned_prices[model_id] = {"input": inp, "output": out}
        if cleaned_prices:
            to_save["model_prices"] = cleaned_prices

    # UI state (splitter sizes, dialog sizes, etc.)
    ui_state = config.get("ui")
    if isinstance(ui_state, dict):
        to_save["ui"] = ui_state

    path.write_text(json.dumps(to_save, indent=2, sort_keys=True), encoding="utf-8")