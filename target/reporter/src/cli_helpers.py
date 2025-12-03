"""CLI helper functions for displaying configuration information."""

from typing import Dict, Any
from .config import Config


def list_available_models(config: Config) -> None:
    """List all available LLM models with pricing and recommendations."""
    models = config.available_models
    
    if not models:
        print("⚠ No models configured")
        return
    
    print("⚙ Available LLM Models:")
    print()
    
    # Show current default first
    default_key = config.default_model
    if default_key in models:
        model = models[default_key]
        print(f"▓ CURRENT DEFAULT: {default_key}")
        _display_model_info(default_key, model, is_default=True)
        print()
    
    # Show recommended models
    recommended = {k: v for k, v in models.items() if v.get('recommended', False) and k != default_key}
    if recommended:
        print("⚙ RECOMMENDED MODELS:")
        for key, model in recommended.items():
            _display_model_info(key, model)
        print()
    
    # Show other models
    others = {k: v for k, v in models.items() if not v.get('recommended', False) and k != default_key}
    if others:
        print("⚙ OTHER MODELS:")
        for key, model in others.items():
            _display_model_info(key, model)


def _display_model_info(key: str, model: Dict[str, Any], is_default: bool = False) -> None:
    """Display information for a single model."""
    display_name = model.get('display_name', key)
    input_cost = model.get('input_cost', 0)
    output_cost = model.get('output_cost', 0)
    
    # Calculate cost for typical analysis (1K input, 500 output tokens)
    typical_cost = (1000/1_000_000 * input_cost) + (500/1_000_000 * output_cost)
    
    prefix = "  ▓" if is_default else "  •"
    print(f"{prefix} {key}: {display_name}")
    print(f"    Cost: ${input_cost:.2f}/${output_cost:.2f} per 1M tokens (~${typical_cost:.4f}/analysis)")


def list_length_templates(config: Config) -> None:
    """List all available length templates."""
    templates = config.length_templates
    
    if not templates:
        print("⚠ No length templates configured")
        return
    
    print("⚙ Available Length Templates:")
    print()
    
    # Show current default first
    default_key = config.default_length
    if default_key in templates:
        template = templates[default_key]
        print(f"▓ CURRENT DEFAULT: {default_key}")
        _display_length_info(default_key, template, is_default=True)
        print()
    
    # Show other templates
    others = {k: v for k, v in templates.items() if k != default_key}
    if others:
        print("⚙ OTHER TEMPLATES:")
        for key, template in others.items():
            _display_length_info(key, template)


def _display_length_info(key: str, template: Dict[str, Any], is_default: bool = False) -> None:
    """Display information for a single length template."""
    display_name = template.get('display_name', key)
    max_chars = template.get('max_chars')
    instruction = template.get('instruction', '')
    
    prefix = "  ▓" if is_default else "  •"
    chars_info = f"({max_chars} chars max)" if max_chars else "(no limit)"
    
    print(f"{prefix} {key}: {display_name} {chars_info}")
    if instruction:
        print(f"    {instruction[:100]}{'...' if len(instruction) > 100 else ''}")


def display_current_config(config: Config) -> None:
    """Display current configuration summary."""
    model_config = config.get_model_config(config.default_model)
    print(f"▓ CONFIG: {config.default_model} | {config.default_length} | {config.console_theme} | {config.default_system_prompt}")