"""
Base classes for tool implementation.

This module provides the base classes and types for creating consistent, 
well-typed tools that can be used within the toolchain.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar, Type, ClassVar
from enum import Enum
import inspect

T = TypeVar('T')  # Return type of the tool

class ToolError(Exception):
    """Base exception for tool-related errors."""
    pass

class ToolStatus(str, Enum):
    """Status of a tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"

@dataclass
class ToolResult(Generic[T]):
    """Container for tool execution results.
    
    Attributes:
        status: The status of the tool execution
        data: The result data (if successful)
        error: Error message (if failed)
        metadata: Additional metadata about the execution
    """
    status: ToolStatus
    data: Optional[T] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Whether the tool execution was successful."""
        return self.status == ToolStatus.SUCCESS
    
    @classmethod
    def success_result(cls, data: T, **metadata) -> 'ToolResult[T]':
        """Create a successful result."""
        return cls(status=ToolStatus.SUCCESS, data=data, metadata=metadata)
    
    @classmethod
    def error_result(cls, error: str, status: ToolStatus = ToolStatus.ERROR, **metadata) -> 'ToolResult[Any]':
        """Create an error result."""
        return cls(status=status, error=error, metadata=metadata)

class Tool(ABC):
    """Base class for all tools.
    
    Subclasses should implement the `_execute` method and can optionally
    override other methods to customize behavior.
    """
    
    # Class variables that can be overridden by subclasses
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    version: ClassVar[str] = "1.0"
    
    def __init_subclass__(cls, **kwargs):
        """Validate that required class variables are set in subclasses."""
        super().__init_subclass__(**kwargs)
        if not cls.name:
            raise ToolError(f"Tool class {cls.__name__} must define a 'name' class variable")
    
    @property
    def tool_info(self) -> Dict[str, Any]:
        """Get information about this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "parameters": self.get_parameters_info()
        }
    
    def get_parameters_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about the tool's parameters.
        
        Returns:
            Dictionary mapping parameter names to their metadata (type, required, description)
        """
        params = {}
        sig = inspect.signature(self._execute)
        
        for name, param in list(sig.parameters.items())[1:]:  # Skip 'self'
            param_info = {
                "type": param.annotation if param.annotation != inspect.Parameter.empty else Any,
                "required": param.default == inspect.Parameter.empty,
                "default": param.default if param.default != inspect.Parameter.empty else None
            }
            
            # Try to get parameter description from docstring
            doc = inspect.getdoc(self._execute) or ""
            param_doc = ""
            in_params = False
            
            for line in doc.split('\n'):
                line = line.strip()
                if line.startswith('Args:'):
                    in_params = True
                    continue
                if in_params and ':' in line and (line.startswith('Returns:') or line.startswith('Raises:') or not line):
                    in_params = False
                    break
                if in_params and f"{name}:" in line:
                    param_doc = line.split(':', 1)[1].strip()
            
            if param_doc:
                param_info["description"] = param_doc
                
            params[name] = param_info
            
        return params
    
    async def execute(self, **kwargs) -> ToolResult[Any]:
        """Execute the tool with the given parameters.
        
        This is the main entry point that handles common functionality like
        parameter validation before delegating to the tool-specific _execute method.
        """
        try:
            # Validate parameters
            sig = inspect.signature(self._execute)
            try:
                # Skip 'self' parameter
                bound_args = sig.bind_partial(**kwargs)
                bound_args.apply_defaults()
            except TypeError as e:
                return ToolResult.error_result(
                    f"Invalid parameters: {str(e)}",
                    ToolStatus.VALIDATION_ERROR
                )
            
            # Execute the tool
            try:
                result = await self._execute(**bound_args.arguments)
                return ToolResult.success_result(result)
                
            except Exception as e:
                return ToolResult.error_result(
                    f"Tool execution failed: {str(e)}",
                    ToolStatus.EXECUTION_ERROR
                )
                
        except Exception as e:
            return ToolResult.error_result(
                f"Unexpected error in tool execution: {str(e)}",
                ToolStatus.ERROR
            )
    
    @abstractmethod
    async def _execute(self, **kwargs) -> Any:
        """Tool-specific implementation.
        
        Subclasses must implement this method to provide the tool's functionality.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            The result of the tool execution
            
        Raises:
            ToolError: If the tool encounters an error condition
        """
        pass
