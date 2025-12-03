"""
Agent discovery system.

This module provides automatic discovery and registration of agents
in the agents package and subpackages.
"""

import importlib
import pkgutil
from pathlib import Path
from typing import List

from .base_agent import BaseAgent
from .registry import AgentRegistry


def discover_agents(package_path: str = "agents") -> None:
    """Automatically discover and register all agent classes.
    
    This function scans the agents package and all subpackages,
    importing modules and registering any agent classes found.
    
    Args:
        package_path: The package path to scan for agents
    """
    try:
        # Get the package
        package = importlib.import_module(package_path)
        package_file = Path(package.__file__).parent
        
        # Import all subpackages
        for module_info in pkgutil.iter_modules([str(package_file)]):
            module_name = module_info.name
            
            # Skip __pycache__ and private modules
            if module_name.startswith('_') or module_name == '__pycache__':
                continue
            
            # Import the module
            full_module_name = f"{package_path}.{module_name}"
            try:
                importlib.import_module(full_module_name)
            except ImportError as e:
                # Log but don't fail - some modules might have import issues
                print(f"Warning: Could not import module {full_module_name}: {e}")
                
    except ImportError as e:
        print(f"Warning: Could not import package {package_path}: {e}")


def discover_agents_in_subpackages(package_path: str = "agents") -> None:
    """Discover agents in all subpackages recursively.
    
    Args:
        package_path: The base package path to scan
    """
    try:
        # Get the package
        package = importlib.import_module(package_path)
        package_file = Path(package.__file__).parent
        
        # Recursively discover in all subpackages
        for module_info in pkgutil.walk_packages([str(package_file)], prefix=f"{package_path}."):
            module_name = module_info.name
            
            # Skip __pycache__ and private modules
            if module_name.startswith('_') or '__pycache__' in module_name:
                continue
                
            # Only import packages (not individual modules)
            if module_info.ispkg:
                try:
                    importlib.import_module(module_name)
                except ImportError as e:
                    print(f"Warning: Could not import package {module_name}: {e}")
                    
    except ImportError as e:
        print(f"Warning: Could not import package {package_path}: {e}")


def get_discovered_agents() -> List[str]:
    """Get list of discovered agent names.
    
    Returns:
        List of agent names that have been discovered and registered
    """
    return AgentRegistry.get_agent_names()


# Auto-discover agents when this module is imported
discover_agents()
discover_agents_in_subpackages()
