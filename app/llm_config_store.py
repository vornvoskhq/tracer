import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


# Default prompt presets and LLM configuration. This provides sane defaults when
# no app_config.json file exists yet.
DEFAULT_PRESETS: Dict[str, Dict[str, str]] = {
    "concise-tech": {
        "label": "Concise technical summary",
        "template": (
            "You are an expert Python engineer. Summarize the purpose and behavior "
            "of the following function in concise, technical prose. Focus on:\n"
            "- Overall purpose\n"
            "- Key inputs and outputs\n"
            "- Important side effects (I/O, network, database, etc.)\n"
            "- Non-obvious edge cases or constraints\n\n"
            "Function source or trace context:\n"
            "```python\n"
            "{code}\n"
            "```"
        ),
    },
    "onboarding": {
        "label": "High-level explanation (onboarding)",
        "template": (
            "You are helping onboard a new engineer to this codebase. Explain what "
            "the following function or execution path does in clear, approachable language. "
            "Focus on:\n"
            "- What problem it solves in the overall system\n"
            "- How it fits into the execution flow\n"
            "- Any assumptions or preconditions\n"
            "- Gotchas or areas where changes are risky\n\n"
            "Function source or trace context:\n"
            "```python\n"
            "{code}\n"
            "```"
        ),
    },
    "behavior-io": {
        "label": "Behavior + inputs/outputs only",
        "template": (
            "Summarize the behavior of the following function or execution path, focusing strictly on:\n"
            "- Inputs (parameters and important global state)\n"
            "- Outputs (return values and changes to state)\n"
            "- Invariants the function relies on\n\n"
            "Avoid restating the code line-by-line.\n\n"
            "Function source or trace context:\n"
            "```python\n"
            "{code}\n"
            "```"
        ),
    },
    "bugs-edges": {
        "label": "Potential bugs / edge cases",
        "template": (
            "Review the following function or execution path for potential bugs and edge cases. "
            "Provide a short explanation that covers:\\n"
            "- Any obvious or likely bugs\\n"
            "- Edge cases that might fail (e.g., empty inputs, None, large values)\\n"
            "- Error handling or lack thereof\\n"
            "- Suggestions for tests that should be added\\n\\n"
            "Function source or trace context:\\n"
            "```python\\n"
            "{code}\\n"
            "```"
        ),
    },
    "refactor-ai-mess": {
        "label": "Refactor AI-generated / overcomplex code",
        "template": (
            "You are an expert Python engineer. Review the following function or execution slice, which was likely "
            "generated or heavily edited by AI. Provide a concise analysis that covers:\n"
            "- Where the structure is overcomplicated, redundant, or over-abstracted\n"
            "- Any dead code, unused branches, or unnecessary configuration flags\n"
            "- Any use of synthetic or fallback data paths that should be removed or made explicit "
            "(for technical analysis software, there must be no hidden fake data).\n\n"
            "Then propose a refactor plan:\n"
            "- How to simplify while preserving behavior and data integrity\n"
            "- Which pieces can safely be deleted vs. just reorganized\n"
            "- A short checklist of steps to get from the current version to a cleaner one\n\n"
            "Function source or trace context:\n"
            "```python\n"
            "{code}\n"
            "```"
        ),
    },
    "entrypoints": {
        "label": "Suggest entry points",
        "template": (
            "You are helping a developer understand a new, black-box Python codebase. "
            "You will be given snippets from many .py files in this project.\\\\n\\\\n"
            "Your tasks:\\\\n"
            "- Identify which file(s) are most likely to act as entrypoints (top-level scripts or main modules).\\\\n"
            "- For each candidate entrypoint, briefly explain why it is likely an entrypoint "
            "(e.g., has an if __name__ == '__main__' block, defines a main() that parses CLI args, "
            "or is clearly the starting script).\\\\n"
            "- If there appears to be a thin wrapper script that quickly hands off control to a deeper module "
            "or framework, explain that layering and suggest which deeper file is the 'real' place to start reading.\\\\n"
            "- Provide a short recommendation section: 'If you want to understand this application, start by reading: ...'.\\\\n\\\\n"
            "Project file snippets:\\\\n"
            "```text\\\\n"
            "{code}\\\\n"
            "```"
        ),
    },
}


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": "openai/gpt-4o-mini",
    "temperature": 0.10,
    "max_tokens": 512,
    "default_prompt_preset": "concise-tech",
    "presets": DEFAULT_PRESETS,
    # The list of LLM model IDs is user-configurable and stored only in
    # app_config.json["models"]. We keep this empty here so there is a single
    # source of truth on disk.
    "models": [],
    # Whether to log full LLM context (including file contents) to the LLM log.
    # When False, entrypoint logs only include instructions + file list.
    "verbose_logging": False,
    # Column / row visibility settings in the trace view. These are toggled via
    # the Config menu and persisted at the top level next to verbose_logging.
    "show_caller_column": True,
    "show_phase_column": False,
    "hide_import_rows": False,
    # Optional UI state; these keys may or may not be present in user configs.
    # They are included here only to document expected structure.
    "ui": {
        "main_splitter_sizes": [600, 600],
        "left_splitter_sizes": [400, 200],
        # Default column widths for the left execution/I-O tree.
        "left_tree_column_widths": [14, 40, 40, 70, 70, 140, 210, 170, 60],
        "llm_dialog_size": [800, 600],
    },
}


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
    if path.exists():
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
                    "verbose_logging",
                    "show_caller_column",
                    "show_phase_column",
                    "hide_import_rows",
                    "ui",
                ):
                    if key in raw:
                        config[key] = raw[key]
        except Exception:
            # On any error, fall back to defaults.
            pass

    # Ensure presets at least contain the defaults; user-defined presets can override.
    presets = config.get("presets") or {}
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

    # Start with defaults
    for pid, pconf in DEFAULT_PRESETS.items():
        tmpl = _normalize_template(str(pconf.get("template", "{code}")))
        merged_presets[pid] = {"label": str(pconf.get("label", pid)), "template": tmpl}

    # Overlay any presets from config (allowing user overrides / additions)
    if isinstance(presets, dict):
        for pid, pconf in presets.items():
            if not isinstance(pconf, dict):
                continue
            # Keep only label/template keys to avoid surprises
            label = str(pconf.get("label", pid))
            raw_template = str(pconf.get("template", DEFAULT_PRESETS.get(pid, {}).get("template", "{code}")))
            template = _normalize_template(raw_template)
            merged_presets[pid] = {"label": label, "template": template}

    config["presets"] = merged_presets

    # Ensure default_prompt_preset is valid
    default_preset = config.get("default_prompt_preset")
    if default_preset not in merged_presets:
        # Fallback to the first preset ID
        default_preset = next(iter(merged_presets.keys()))
        config["default_prompt_preset"] = default_preset

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

    # Optional list of known models for the settings dialog.
    models = config.get("models")
    if isinstance(models, list):
        cleaned_models = [str(m) for m in models if isinstance(m, str) and m.strip()]
        if cleaned_models:
            to_save["models"] = cleaned_models

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

    # UI state (splitter sizes, dialog sizes, etc.)
    ui_state = config.get("ui")
    if isinstance(ui_state, dict):
        to_save["ui"] = ui_state

    path.write_text(json.dumps(to_save, indent=2, sort_keys=True), encoding="utf-8")