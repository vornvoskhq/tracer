"""
Base agent class for autonomous analysis and toolchain execution.

This module provides the abstract base class that all agents should inherit from.
"""

import asyncio
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, ClassVar
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.toolchain import Toolchain
from tools import Tool, ToolResult, ToolStatus
from .types import (
    AgentRequest, AgentResponse, ChainPlan, ChainStep, 
    AgentExecution, AgentStatus
)


class BaseAgent(ABC):
    """Base class for all agents in the system.
    
    Subclasses must implement the `process` method and define class variables
    for name, description, and version.
    """
    
    # Class variables that must be defined by subclasses
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    version: ClassVar[str] = "1.0"
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the agent.
        
        Args:
            config_path: Path to configuration file
        """
        # Initialize toolchain
        self.toolchain = Toolchain(config_path)
        
        # Execution tracking
        self.execution_history: List[AgentExecution] = []
        self.current_status: AgentStatus = AgentStatus.IDLE
        
        # Agent-specific setup
        self._setup_tools()
    
    def _setup_tools(self):
        """Set up tools available to this agent.
        
        Subclasses can override this to register specific tools.
        """
        # Default tools - subclasses can extend this
        pass
    
    @abstractmethod
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process an agent request.
        
        This is the main entry point for agent execution. Subclasses must
        implement this method to handle their specific logic.
        
        Args:
            request: The agent request to process
            
        Returns:
            AgentResponse containing the results
        """
        pass
    
    def get_available_tools(self) -> Dict[str, Tool]:
        """Get tools available to this agent.
        
        Returns:
            Dictionary mapping tool names to Tool instances
        """
        return self.toolchain.tools
    
    def get_tool_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available tools.
        
        Returns:
            Dictionary with tool metadata
        """
        return {
            name: tool.tool_info 
            for name, tool in self.toolchain.tools.items()
        }
    
    async def execute_chain(self, plan: ChainPlan) -> Dict[str, Any]:
        """Execute a planned toolchain.
        
        Args:
            plan: The execution plan to run
            
        Returns:
            Dictionary containing execution results
        """
        self.current_status = AgentStatus.EXECUTING
        
        try:
            # Clear any existing steps and add the planned steps
            self.toolchain.steps = []
            self.toolchain.results = {}
            
            # Add each step to the toolchain
            for step in plan.steps:
                self.toolchain.add(step.tool, **step.parameters)
            
            # Execute the chain
            results = await self.toolchain.run()
            
            self.current_status = AgentStatus.COMPLETED
            return results
            
        except Exception as e:
            self.current_status = AgentStatus.ERROR
            raise
    
    def _record_execution(self, execution: AgentExecution):
        """Record an execution in the history.
        
        Args:
            execution: The execution to record
        """
        self.execution_history.append(execution)
        
        # Keep only the last 100 executions to prevent memory bloat
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def get_execution_history(self, limit: int = 10) -> List[AgentExecution]:
        """Get recent execution history.
        
        Args:
            limit: Maximum number of executions to return
            
        Returns:
            List of recent executions
        """
        return self.execution_history[-limit:]
    
    def get_status(self) -> AgentStatus:
        """Get the current agent status.
        
        Returns:
            Current agent status
        """
        return self.current_status
    
    def validate_request(self, request: AgentRequest) -> bool:
        """Validate an agent request.
        
        Args:
            request: The request to validate
            
        Returns:
            True if request is valid for this agent
        """
        # Basic validation - subclasses can override for specific requirements
        return bool(request.goal and request.goal.strip())
    
    async def _safe_execute(self, request: AgentRequest) -> AgentResponse:
        """Safely execute a request with error handling.
        
        Args:
            request: The request to execute
            
        Returns:
            AgentResponse with error handling
        """
        start_time = datetime.now()
        
        try:
            # Validate request
            if not self.validate_request(request):
                return AgentResponse(
                    success=False,
                    data=None,
                    message="Invalid request for this agent",
                    execution_plan=ChainPlan(steps=[]),
                    execution_results={},
                    execution_time=0,
                    agent_name=self.name,
                    request_id=request.request_id
                )
            
            # Process the request
            response = await self.process(request)
            
            # Record execution
            execution = AgentExecution(
                agent_name=self.name,
                request=request,
                plan=response.execution_plan,
                results=response.execution_results,
                status=AgentStatus.COMPLETED if response.success else AgentStatus.ERROR,
                start_time=start_time,
                end_time=datetime.now()
            )
            self._record_execution(execution)
            
            return response
            
        except Exception as e:
            # Record failed execution
            execution = AgentExecution(
                agent_name=self.name,
                request=request,
                plan=ChainPlan(steps=[]),
                results={},
                status=AgentStatus.ERROR,
                start_time=start_time,
                end_time=datetime.now(),
                error_message=str(e)
            )
            self._record_execution(execution)
            
            return AgentResponse(
                success=False,
                data=None,
                message=f"Agent execution failed: {str(e)}",
                execution_plan=ChainPlan(steps=[]),
                execution_results={},
                execution_time=(datetime.now() - start_time).total_seconds(),
                agent_name=self.name,
                request_id=request.request_id
            )
    
    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute a request safely.
        
        This is the main public interface that should be called by clients.
        
        Args:
            request: The request to execute
            
        Returns:
            AgentResponse with the results
        """
        return await self._safe_execute(request)
