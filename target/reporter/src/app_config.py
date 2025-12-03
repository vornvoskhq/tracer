"""
Application configuration management.

This module handles application-level configurations including VGMini integration
and other non-LLM-specific settings.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

@dataclass
class VGMiniConfig:
    """Configuration for VGMini integration."""
    ranking_output_path: str = 'vgmini/results'
    # Add other VGMini-specific settings here

@dataclass
class AppConfig:
    """Main application configuration."""
    vgmini: VGMiniConfig = field(default_factory=VGMiniConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AppConfig':
        """Create AppConfig from dictionary."""
        vgmini_config = config_dict.get('vgmini', {})
        return cls(
            vgmini=VGMiniConfig(
                ranking_output_path=vgmini_config.get('ranking_output_path', 'vgmini/results')
            )
        )

def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load application configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
        
    Returns:
        AppConfig instance with loaded configuration
    """
    import yaml
    from pathlib import Path
    
    if config_path is None:
        # Default config locations to check
        config_paths = [
            Path('config/app_config.yaml'),
            Path('~/.config/mithra/app_config.yaml').expanduser(),
            Path('/etc/mithra/app_config.yaml')
        ]
        
        for path in config_paths:
            if path.exists():
                config_path = str(path)
                break
        else:
            # No config file found, return default config
            return AppConfig()
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
            return AppConfig.from_dict(config_dict)
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return AppConfig()
