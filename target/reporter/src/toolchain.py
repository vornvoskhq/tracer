#!/usr/bin/env python3
"""
Toolchain

A system for chaining and executing tools in sequence.
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent directory to path for tools imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the tools
from tools import StockIdentity, Summarizer, Tool, ToolResult, ToolError, ToolStatus
from .llm_model import ModelManager


class StepResult:
    """Placeholder for a step result that gets populated after execution."""
    
    def __init__(self, step_name: str, toolchain: 'Toolchain'):
        self.step_name = step_name
        self.toolchain = toolchain
        self._result = None
    
    @property
    def result(self):
        """Get the result after execution."""
        if self._result is None:
            self._result = self.toolchain.results.get(self.step_name)
        return self._result
    
    def __getattr__(self, name):
        """Delegate attribute access to the actual result."""
        if self.result is None:
            raise AttributeError(f"Step '{self.step_name}' has not been executed yet")
        return getattr(self.result, name)


class Toolchain:
    """Manages the execution of multiple tools in sequence."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the toolchain with default settings."""
        self.model_manager = ModelManager(config_path)
        self.tools: Dict[str, Tool] = {
            'stock_identity': StockIdentity(self.model_manager),
            'summarizer': Summarizer(self.model_manager)
        }
        self.steps: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}
        self._step_counter = 0
    
    def get_tool(self, tool_name: str) -> Tool:
        """Get a tool by name."""
        if tool_name not in self.tools:
            raise KeyError(f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}")
        return self.tools[tool_name]
    
    def add(self, tool: Any, **kwargs) -> StepResult:
        """Add a tool to the pipeline.
        
        Args:
            tool: The tool instance to add
            **kwargs: Additional parameters for the tool
            
        Returns:
            StepResult object for accessing the result after execution
        """
        step_name = f"step_{self._step_counter}"
        self._step_counter += 1
        
        # Store the step configuration
        self.steps.append({
            'name': step_name,
            'tool': tool,
            'kwargs': kwargs
        })
        
        # Return a placeholder that will be populated after execution
        return StepResult(step_name, self)
    
    def _process_templates(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Process template strings in kwargs with actual results."""
        processed = {}
        for key, value in kwargs.items():
            if isinstance(value, str) and '{' in value:
                processed[key] = self._replace_template_variables(value)
            else:
                processed[key] = value
        return processed
    
    def _replace_template_variables(self, template: str) -> str:
        """Replace template variables with actual result values."""
        result = template
        for step_name, step_result in self.results.items():
            if step_result.success and step_result.data:
                # Get the actual data from the result
                data = self._extract_data_from_result(step_result)
                if isinstance(data, dict):
                    for k, v in data.items():
                        result = result.replace(f"{{{step_name}.{k}}}", str(v))
                else:
                    result = result.replace(f"{{{step_name}}}", str(data))
        return result
    
    def _extract_data_from_result(self, result: ToolResult) -> Any:
        """Extract the actual data from a ToolResult, handling nested structures."""
        if hasattr(result.data, 'data'):
            # Nested ToolResult
            return result.data.data
        return result.data
    
    async def _execute_step(self, step: Dict[str, Any]) -> ToolResult:
        """Execute a single step with template processing."""
        # Process parameters with template substitution
        processed_kwargs = self._process_templates(step['kwargs'])
        
        # Execute the tool
        if isinstance(step['tool'], str):
            return await self.execute_tool(step['tool'], **processed_kwargs)
        else:
            return await step['tool'].execute(**processed_kwargs)
    
    async def run(self) -> Dict[str, Any]:
        """Execute all added steps in sequence."""
        self.results = {}
        for step in self.steps:
            self.results[step['name']] = await self._execute_step(step)
        return self.results
    
    async def execute_tool(
        self, 
        tool_name: str, 
        **kwargs
    ) -> ToolResult[Any]:
        """Execute a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Parameters to pass to the tool
            
        Returns:
            ToolResult containing the execution result or error
        """
        try:
            tool = self.get_tool(tool_name)
            result = await tool.execute(**kwargs)
            return result
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to execute tool '{tool_name}': {str(e)}",
                metadata={"tool": tool_name, "params": kwargs}
            )


async def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Chain multiple tools to analyze a stock')
    parser.add_argument('symbol', type=str, help='Stock symbol to analyze (e.g., AAPL)')
    parser.add_argument('--instruction', type=str, 
                        help='Custom instruction for the summary (optional)')
    
    args = parser.parse_args()
    
    try:
        toolchain = Toolchain()
        
        # Demonstrate the new add-run pattern
        stock_identity = toolchain.add('stock_identity', symbol=args.symbol)
        summary = toolchain.add('summarizer', 
                              text="{step_0.business_summary}",
                              instruction=args.instruction or "Provide a concise summary of the following business description:")
        
        await toolchain.run()
        
        # Print the results
        if stock_identity.result and stock_identity.result.success:
            identity_data = stock_identity.result.data
            print(f"Stock Analysis: {identity_data.get('company_name', 'N/A')} ({args.symbol})")
            if identity_data.get('sector') and identity_data.get('industry'):
                print(f"Sector/Industry: {identity_data.get('sector', 'N/A')}/{identity_data.get('industry', 'N/A')}")
        
        if summary.result and summary.result.success:
            print("Business Summary:")
            # Handle the nested result structure
            summary_data = summary.result.data
            if hasattr(summary_data, 'data') and isinstance(summary_data.data, dict):
                print(summary_data.data.get('summary', 'No summary available'))
            else:
                print(str(summary_data))
        
        # Check for errors
        if stock_identity.result and not stock_identity.result.success:
            print(f"\n❌ Error fetching stock identity: {stock_identity.result.error}", file=sys.stderr)
            sys.exit(1)
        
        if summary.result and not summary.result.success:
            print(f"\n⚠️  Warning: Summary generation failed: {summary.result.error}", file=sys.stderr)
        
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
