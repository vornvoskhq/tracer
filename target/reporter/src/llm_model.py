"""
LLM Model Management

Provides comprehensive management of LLM models, configurations, and cost tracking.
Handles model settings, API interactions, and cost calculations in one place.
"""

import os
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from time import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast, Tuple

import yaml

# Type variable for generic function typing
F = TypeVar('F', bound=Callable[..., Any])

# Global cost tracker
total_cost: float = 0.0

def load_api_key(env_paths: Optional[List[Union[str, Path]]] = None) -> str:
    """Load API key from environment or .env file.
    
    Args:
        env_paths: Optional list of paths to check for .env files
        
    Returns:
        The API key if found, empty string otherwise
    """
    # Check environment variable first
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if api_key:
        return api_key
        
    # Default paths to check for .env file
    if env_paths is None:
        env_paths = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parent.parent / ".env",
            Path.home() / ".env"
        ]
    
    # Try each path
    for env_path in env_paths:
        env_path = Path(env_path)
        if not env_path.exists():
            continue
            
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "OPENROUTER_API_KEY":
                    return value.strip('"\'').strip()
        except (OSError, UnicodeDecodeError):
            continue
            
    return ""

def track_llm_cost(model_id: str, input_tokens: int, output_tokens: int, manager: 'ModelManager') -> float:
    """Track the cost of an LLM call and update the total cost.
    
    Args:
        model_id: Short name of the model
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        manager: ModelManager instance
        
    Returns:
        The calculated cost in USD
    """
    model_config = manager.get_model(model_id)
    if not model_config:
        print(f"⚠️  Warning: Model '{model_id}' not found in config")
        return 0.0
        
    cost = model_config.get_cost(input_tokens, output_tokens)
    global total_cost
    total_cost += cost
    
    # Format token counts
    def format_tokens(num):
        if num >= 1000:
            return f"{num/1000:.1f}K"
        return str(num)
    
    # Print cost information in a more readable format
    print(f"💲 Cost: ${cost:.6f} | "
          f"Model: {model_config.display_name} | "
          f"Tokens: {format_tokens(input_tokens)} in, {format_tokens(output_tokens)} out | "
          f"Total: ${total_cost:.6f}")
    return cost

from typing import TypeVar, Callable, Awaitable, Any, TypeVar, cast

F = TypeVar('F', bound=Callable[..., Awaitable[Any]])

def llm_cost_estimator(func: F) -> F:
    """Decorator to track and log the cost of LLM API calls.
    
    The decorated async function must return a tuple of (result, input_tokens, output_tokens).
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time()
        try:
            # Await the async function call
            result, input_tokens, output_tokens = await func(*args, **kwargs)
            
            # Get the model_id from kwargs or the first argument (self)
            model_id = kwargs.get('model_id')
            if not model_id and len(args) > 0 and hasattr(args[0], 'model_manager'):
                model_id = getattr(args[0], 'model_manager', None) and args[0].model_manager._default_model
                
            if model_id and len(args) > 0 and hasattr(args[0], 'model_manager'):
                track_llm_cost(model_id, input_tokens, output_tokens, args[0].model_manager)
                
            print(f"⏱️  Time: {time() - start_time:.2f}s")
            
            # Return the full tuple with token counts
            return (result, input_tokens, output_tokens)
            
        except Exception as e:
            print(f"⚠️  Error in llm_cost_estimator: {str(e)}")
            # Re-raise the exception to maintain the same behavior as the original function
            raise
        
    return cast(F, wrapper)


@dataclass
class ModelConfig:
    """Configuration for an LLM model.
    
    Attributes:
        full_name: Full model name/identifier used by the API
        display_name: Human-readable name for display
        input_cost: Cost per million input tokens (default: 0.0)
        output_cost: Cost per million output tokens (default: 0.0)
        recommended: Whether this model is recommended for use (default: False)
        max_tokens: Maximum tokens for this model (optional)
        temperature: Default temperature setting (default: 0.3)
        context_window: Model's context window size in tokens (optional)
        default_system_prompt: Default system prompt filename (optional)
    """
    full_name: str
    display_name: str
    input_cost: float = 0.0
    output_cost: float = 0.0
    recommended: bool = False
    max_tokens: Optional[int] = None
    temperature: float = 0.3
    context_window: Optional[int] = None
    default_system_prompt: str = "default_analysis.txt"
    
    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost for a given number of input and output tokens.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        return (self.input_cost * input_tokens / 1_000_000 + 
                self.output_cost * output_tokens / 1_000_000)
                
    @property
    def total_cost(self) -> float:
        """Get the total cost of all LLM calls."""
        return total_cost
        
    def reset_cost(self) -> None:
        """Reset the total cost counter."""
        global total_cost
        total_cost = 0.0


class ModelManager:
    """Manages model configurations and provides access to model settings.
    
    This class handles loading model configurations, managing default models,
    and providing a unified interface for model-related operations.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the model manager with the given config path.
        
        Args:
            config_path: Path to the configuration YAML file. If not provided,
                        will look for config.yaml in the project root.
        """
        self._total_cost: float = 0.0
        self.config_path = self._resolve_config_path(config_path)
        self._models: Dict[str, ModelConfig] = {}
        self._default_model: Optional[str] = None
        self._roles: Dict[str, str] = {}  # role_name -> model_id
        self._config: Dict[str, Any] = {}
        self._load_config()
        
    @property
    def default_model(self) -> str:
        """Get the default model ID."""
        if not self._default_model and self._models:
            self._default_model = next(iter(self._models.keys()))
        return self._default_model or ""
    
    @default_model.setter
    def default_model(self, model_id: str) -> None:
        """Set the default model ID."""
        if model_id not in self._models:
            raise ValueError(f"Model '{model_id}' not found in configuration")
        self._default_model = model_id
        
    @property
    def available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get a dictionary of all available models with their configurations."""
        return {
            model_id: {
                'full_name': config.full_name,
                'display_name': config.display_name,
                'input_cost': config.input_cost,
                'output_cost': config.output_cost,
                'recommended': config.recommended,
                'max_tokens': config.max_tokens,
                'temperature': config.temperature,
                'context_window': config.context_window
            }
            for model_id, config in self._models.items()
        }
    
    def _resolve_config_path(self, config_path: Optional[Union[str, Path]]) -> Path:
        """Resolve the path to the configuration file.
        
        Args:
            config_path: Explicit path to config file, or None to search
            
        Returns:
            Path to the configuration file
            
        Raises:
            FileNotFoundError: If no config file is found and no path was provided
        """
        if config_path is not None:
            path = Path(config_path).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            return path
            
        # Look for config.yaml in common locations
        possible_paths = [
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent / "config.yaml",
            Path.home() / ".config" / "reporter" / "config.yaml"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
                
        # If no config file found, create a default one
        default_path = Path.cwd() / "config.yaml"
        self._create_default_config(default_path)
        return default_path
        
    def _create_default_config(self, path: Path) -> None:
        """Create a default configuration file if none exists."""
        default_config = {
            "models": {
                "gpt-4": {
                    "full_name": "openai/gpt-4",
                    "display_name": "GPT-4",
                    "input_cost": 30.0,
                    "output_cost": 60.0,
                    "recommended": True,
                    "max_tokens": 8000
                },
                "claude-2": {
                    "full_name": "anthropic/claude-2",
                    "display_name": "Claude 2",
                    "input_cost": 11.02,
                    "output_cost": 32.68,
                    "recommended": True,
                    "max_tokens": 100000
                }
            },
            "default_model": "gpt-4",
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(default_config, f, default_flow_style=False)
        print(f"⚠ Created default config file at {path}")
        
    def _load_config(self) -> None:
        """Load model configurations from the YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"⚠️  Warning: Failed to load config file: {e}")
            self._config = {}
            
        llm_config = self._config.get('llm', {})
            
        # Load model configurations
        models_config = llm_config.get('available_models', {})
        for model_id, model_config in models_config.items():
            self._models[model_id] = ModelConfig(
                full_name=model_config.get('full_name', ''),
                display_name=model_config.get('display_name', model_id),
                input_cost=float(model_config.get('input_cost', 0.0)),
                output_cost=float(model_config.get('output_cost', 0.0)),
                recommended=bool(model_config.get('recommended', False)),
                max_tokens=model_config.get('max_tokens'),
                temperature=float(model_config.get('temperature', 0.3)),
                context_window=model_config.get('context_window'),
                default_system_prompt=model_config.get('default_system_prompt', 'default_analysis.txt')
            )
            
        # Load role assignments
        self._roles = llm_config.get('roles', {})
        
        # Set default model
        self._default_model = llm_config.get('default_model')
        if self._default_model not in self._models and self._models:
            self._default_model = next(iter(self._models.keys()))
                
    def get_model_config(self, model_id: Optional[str] = None, role: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a specific model, role, or the default model.
        
        Args:
            model_id: Specific model ID to get config for
            role: Role name to get model config for (takes precedence over model_id)
            
        Returns:
            Dictionary with model configuration
            
        Raises:
            ValueError: If the specified model or role is not found
        """
        # Resolve model ID from role if provided
        if role:
            if role not in self._roles:
                raise ValueError(f"Role '{role}' not found in configuration")
            model_id = self._roles[role]
        
        # Fall back to provided model_id or default
        model_id = model_id or self.default_model
        
        if model_id not in self._models:
            raise ValueError(f"Model '{model_id}' not found in configuration")
            
        model = self._models[model_id]
        return {
            'model_id': model_id,
            'full_name': model.full_name,
            'display_name': model.display_name,
            'input_cost': model.input_cost,
            'output_cost': model.output_cost,
            'max_tokens': model.max_tokens,
            'temperature': model.temperature,
            'context_window': model.context_window,
            'default_system_prompt': model.default_system_prompt
        }
        
    def get_model_for_role(self, role: str) -> Dict[str, Any]:
        """Get the model configuration for a specific role.
        
        Args:
            role: Name of the role to get model for
            
        Returns:
            Dictionary with model configuration
            
        Raises:
            ValueError: If the role is not found
        """
        return self.get_model_config(role=role)
        
    def list_roles(self) -> Dict[str, str]:
        """Get a dictionary of all defined roles and their assigned models.
        
        Returns:
            Dictionary mapping role names to model IDs
        """
        return self._roles.copy()
        
    def get_default_system_prompt(self, model_id: Optional[str] = None) -> str:
        """Get the default system prompt filename for a model."""
        return self.get_model_config(model_id).get('default_system_prompt', 'default_analysis.txt')
        
    def get_temperature(self, model_id: Optional[str] = None) -> float:
        """Get the temperature setting for a model."""
        return float(self.get_model_config(model_id).get('temperature', 0.3))
        
    def get_max_tokens(self, model_id: Optional[str] = None) -> int:
        """Get the max tokens setting for a model."""
        return int(self.get_model_config(model_id).get('max_tokens', 2000))
        
    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return self._models.get(model_id)
    
    def get_default_model(self) -> Optional[ModelConfig]:
        """Get the default model configuration."""
        if self._default_model:
            return self.get_model(self._default_model)
        return None
    
    def list_models(self) -> List[str]:
        """Get a list of all available model IDs."""
        return list(self._models.keys())
    
    def get_recommended_models(self) -> List[str]:
        """Get a list of recommended model IDs."""
        return [
            model_id for model_id, model in self._models.items() 
            if model.recommended
        ]
    
    def get_model_display_name(self, model_id: str) -> str:
        """Get the display name for a model."""
        if model := self.get_model(model_id):
            return model.display_name
        return model_id
    
    def get_model_full_name(self, model_id: str) -> str:
        """Get the full name/identifier for a model."""
        if model := self.get_model(model_id):
            return model.full_name
        return model_id
    
    def get_model_costs(self, model_id: str) -> Dict[str, float]:
        """Get the input and output costs for a model."""
        if model := self.get_model(model_id):
            return {
                'input_cost': model.input_cost,
                'output_cost': model.output_cost
            }
        return {'input_cost': 0.0, 'output_cost': 0.0}
    
    def calculate_cost(
        self, 
        model_id: str, 
        input_tokens: int, 
        output_tokens: int
    ) -> float:
        """Calculate the cost for a given number of tokens."""
        if model := self.get_model(model_id):
            return model.get_cost(input_tokens, output_tokens)
        return 0.0
    
    async def generate(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None, 
        temperature: Optional[float] = None
    ):
        """Generate a response using the specified model.
        
        Args:
            model: Model ID to use
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Temperature setting
            
        Returns:
            Mock response object with usage and content
        """
        # This is a mock implementation for now
        # In a real implementation, this would call the actual LLM API
        
        # For now, return an error since we don't have a real implementation
        raise NotImplementedError("ModelManager.generate() is not implemented. Use a proper LLM client instead.")
