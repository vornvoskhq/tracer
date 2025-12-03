"""
Tools package for stock analysis and processing.

This package contains various tools that can be used independently or chained together
for more complex workflows.
"""

"""
Tools for the reporter system.

This package provides various tools that can be used within the reporter system.
Each tool follows a consistent interface defined by the base Tool class.
"""

# Export base classes and types
from .base_tool import (
    Tool,
    ToolResult,
    ToolError,
    ToolStatus
)

# Import and expose the main tools
from .stock_identity import StockIdentity, get_company_profile
from .summarizer import Summarizer

# For backward compatibility and easier imports
__all__ = [
    # Base classes
    'Tool',
    'ToolResult',
    'ToolError',
    'ToolStatus',
    
    # Tools
    'StockIdentity',
    'Summarizer',
    
    # Legacy
    'get_company_profile',
]
