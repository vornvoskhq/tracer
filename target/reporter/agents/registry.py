"""
Agent registry for discovery and management of agents.

This module provides a central registry system for all available agents
in the framework.
"""

from typing import Dict, List, Type, Optional
from .base_agent import BaseAgent


class AgentRegistry:
    """Central registry for all available agents.
    
    This class manages the registration, discovery, and instantiation
    of agents in the system.
    """
    
    _agents: Dict[str, Type[BaseAgent]] = {}
    _instances: Dict[str, BaseAgent] = {}
    
    @classmethod
    def register(cls, agent_class: Type[BaseAgent]) -> None:
        """Register an agent class.
        
        Args:
            agent_class: The agent class to register
            
        Raises:
            ValueError: If agent name is already registered or invalid
        """
        # Validate the agent class
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(f"Agent class {agent_class.__name__} must inherit from BaseAgent")
        
        if not agent_class.name:
            raise ValueError(f"Agent class {agent_class.__name__} must define a 'name' class variable")
        
        name = agent_class.name
        
        # Check for duplicates
        if name in cls._agents:
            existing_class = cls._agents[name]
            if existing_class != agent_class:
                raise ValueError(f"Agent name '{name}' is already registered by {existing_class.__name__}")
        
        # Register the agent
        cls._agents[name] = agent_class
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister an agent.
        
        Args:
            name: The name of the agent to unregister
            
        Returns:
            True if agent was unregistered, False if not found
        """
        if name in cls._agents:
            del cls._agents[name]
        
        # Remove any cached instances
        if name in cls._instances:
            del cls._instances[name]
        
        return name in cls._agents
    
    @classmethod
    def get_agent_class(cls, name: str) -> Type[BaseAgent]:
        """Get an agent class by name.
        
        Args:
            name: The name of the agent
            
        Returns:
            The agent class
            
        Raises:
            ValueError: If agent is not found
        """
        if name not in cls._agents:
            available = list(cls._agents.keys())
            raise ValueError(f"Agent '{name}' not found. Available agents: {available}")
        
        return cls._agents[name]
    
    @classmethod
    def create_agent(cls, name: str, config_path: Optional[str] = None, **kwargs) -> BaseAgent:
        """Create an agent instance.
        
        Args:
            name: The name of the agent to create
            config_path: Optional configuration file path
            **kwargs: Additional arguments for agent initialization
            
        Returns:
            An instance of the requested agent
            
        Raises:
            ValueError: If agent is not found or creation fails
        """
        agent_class = cls.get_agent_class(name)
        
        try:
            return agent_class(config_path=config_path, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to create agent '{name}': {str(e)}")
    
    @classmethod
    def get_agent_instance(cls, name: str, config_path: Optional[str] = None, **kwargs) -> BaseAgent:
        """Get or create an agent instance (cached).
        
        This method caches agent instances to avoid creating multiple
        instances of the same agent.
        
        Args:
            name: The name of the agent
            config_path: Optional configuration file path
            **kwargs: Additional arguments for agent initialization
            
        Returns:
            An instance of the requested agent
        """
        # Create cache key
        cache_key = f"{name}:{config_path or 'default'}:{hash(tuple(sorted(kwargs.items())))}"
        
        # Return cached instance if available
        if cache_key in cls._instances:
            return cls._instances[cache_key]
        
        # Create new instance and cache it
        instance = cls.create_agent(name, config_path, **kwargs)
        cls._instances[cache_key] = instance
        
        return instance
    
    @classmethod
    def list_agents(cls) -> Dict[str, Dict[str, str]]:
        """List all registered agents with their metadata.
        
        Returns:
            Dictionary mapping agent names to their metadata
        """
        return {
            name: {
                "name": agent_class.name,
                "description": agent_class.description,
                "version": agent_class.version,
                "class_name": agent_class.__name__,
                "module": agent_class.__module__
            }
            for name, agent_class in cls._agents.items()
        }
    
    @classmethod
    def get_agent_names(cls) -> List[str]:
        """Get a list of all registered agent names.
        
        Returns:
            List of agent names
        """
        return list(cls._agents.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an agent is registered.
        
        Args:
            name: The name of the agent to check
            
        Returns:
            True if agent is registered, False otherwise
        """
        return name in cls._agents
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached agent instances."""
        cls._instances.clear()
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered agents and instances."""
        cls._agents.clear()
        cls._instances.clear()


def register_agent(agent_class: Type[BaseAgent]) -> Type[BaseAgent]:
    """Decorator for registering agents.
    
    This decorator can be used to automatically register agent classes
    when they are imported.
    
    Args:
        agent_class: The agent class to register
        
    Returns:
        The same agent class (for decorator chaining)
    """
    AgentRegistry.register(agent_class)
    return agent_class
