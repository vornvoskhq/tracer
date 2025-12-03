"""
Agent types and data structures.

This module defines the core types used throughout the agent system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum


class AgentStatus(str, Enum):
    """Status of agent execution."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentRequest:
    """Request sent to an agent."""
    
    # Core request data
    goal: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Optional metadata
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=lowest, 10=highest
    
    # Agent-specific hints
    preferred_tools: List[str] = field(default_factory=list)
    exclude_tools: List[str] = field(default_factory=list)
    max_steps: int = 10


@dataclass
class ChainStep:
    """Single step in an agent's execution plan."""
    
    tool: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    required_data: List[str] = field(default_factory=list)  # Data dependencies
    output_data: List[str] = field(default_factory=list)    # What this step produces
    
    def __post_init__(self):
        """Validate the step after creation."""
        if not self.tool:
            raise ValueError("ChainStep must specify a tool")


@dataclass
class ChainPlan:
    """Execution plan for an agent."""
    
    steps: List[ChainStep]
    description: Optional[str] = None
    estimated_duration: Optional[float] = None  # in seconds
    confidence: float = 1.0  # 0.0 to 1.0
    
    def __post_init__(self):
        """Validate the plan after creation."""
        if not self.steps:
            raise ValueError("ChainPlan must have at least one step")
        
        # Validate step dependencies
        for i, step in enumerate(self.steps):
            for dep in step.required_data:
                # Check if dependency is provided by previous steps
                dep_found = False
                for prev_step in self.steps[:i]:
                    if dep in prev_step.output_data:
                        dep_found = True
                        break
                if not dep_found:
                    raise ValueError(f"Step {i} requires '{dep}' but no previous step provides it")


@dataclass
class AgentExecution:
    """Record of an agent execution."""
    
    agent_name: str
    request: AgentRequest
    plan: ChainPlan
    results: Dict[str, Any]
    status: AgentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Duration of execution in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


@dataclass
class AgentResponse:
    """Response from an agent."""
    
    # Core response data
    success: bool
    data: Any
    message: str
    
    # Execution metadata
    execution_plan: ChainPlan
    execution_results: Dict[str, Any]
    execution_time: float
    
    # Optional metadata
    agent_name: str
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for serialization."""
        return {
            'success': self.success,
            'data': self.data,
            'message': self.message,
            'execution_plan': {
                'steps': [
                    {
                        'tool': step.tool,
                        'parameters': step.parameters,
                        'description': step.description
                    }
                    for step in self.execution_plan.steps
                ],
                'description': self.execution_plan.description,
                'confidence': self.execution_plan.confidence
            },
            'execution_results': self.execution_results,
            'execution_time': self.execution_time,
            'agent_name': self.agent_name,
            'request_id': self.request_id,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'warnings': self.warnings
        }
