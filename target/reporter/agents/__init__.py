"""
Agents package for autonomous analysis and toolchain execution.

This package provides a framework for building extensible agents that can
autonomously create and execute toolchains for various analysis tasks.
"""

from .base_agent import BaseAgent
from .types import AgentRequest, AgentResponse, ChainPlan, ChainStep
from .registry import AgentRegistry

__all__ = [
    'BaseAgent',
    'AgentRequest', 
    'AgentResponse',
    'ChainPlan',
    'ChainStep',
    'AgentRegistry'
]

# Auto-discover and register all agents
try:
    from . import discovery
except ImportError:
    # Discovery might fail if dependencies are missing
    pass
