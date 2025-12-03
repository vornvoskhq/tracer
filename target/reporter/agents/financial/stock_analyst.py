"""
StockAnalyst agent for comprehensive stock analysis.

This agent specializes in analyzing individual stocks using multiple tools
and generating comprehensive reports.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tools import StockIdentity, Summarizer
from ..base_agent import BaseAgent
from ..types import (
    AgentRequest, AgentResponse, ChainPlan, ChainStep, 
    AgentStatus
)
from ..registry import register_agent


@register_agent
class StockAnalyst(BaseAgent):
    """Agent for comprehensive stock analysis.
    
    This agent analyzes individual stocks using a combination of tools
    to provide comprehensive analysis including company identity,
    business summary, and qualitative insights.
    """
    
    name = "stock_analyst"
    description = "Analyzes individual stocks using multiple tools and generates comprehensive reports"
    version = "1.0"
    
    def _setup_tools(self):
        """Set up tools available to the StockAnalyst agent."""
        # Add stock analysis tools to the toolchain
        self.toolchain.tools.update({
            'stock_identity': StockIdentity(self.toolchain.model_manager),
            'summarizer': Summarizer(self.toolchain.model_manager)
        })
    
    def validate_request(self, request: AgentRequest) -> bool:
        """Validate a stock analysis request.
        
        Args:
            request: The request to validate
            
        Returns:
            True if request is valid for stock analysis
        """
        # Must have a goal
        if not request.goal or not request.goal.strip():
            return False
        
        # Extract symbol from request
        symbol = request.parameters.get('symbol')
        if symbol:
            # Validate symbol format (basic check)
            symbol = symbol.upper().strip()
            if not symbol.isalpha() or len(symbol) > 5:
                return False
            return True
            
        # Check if goal contains a stock symbol (simple pattern matching)
        goal_lower = request.goal.lower()
        if any(word in goal_lower for word in ['stock', 'analyze', 'analysis']):
            # Try to extract symbol from goal (basic implementation)
            words = request.goal.upper().split()
            for word in words:
                if 1 <= len(word) <= 5 and word.isalpha():
                    request.parameters['symbol'] = word
                    return True
        
        return False
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process a stock analysis request.
        
        Args:
            request: The stock analysis request
            
        Returns:
            AgentResponse with analysis results
        """
        import time
        start_time = time.time()
        
        try:
            # Extract and validate symbol from request
            symbol = request.parameters.get('symbol')
            if not symbol:
                raise ValueError("Stock symbol is required for analysis")
            
            symbol = symbol.upper().strip()
            
            # Additional symbol validation
            if not symbol.isalpha() or len(symbol) > 5:
                raise ValueError(f"Invalid stock symbol format: {symbol}")
            
            # Plan the analysis chain
            plan = self._plan_analysis_chain(request, symbol)
            
            # Execute the chain
            results = await self.execute_chain(plan)
            
            # Validate execution results
            if not results:
                raise ValueError("No results returned from toolchain execution")
            
            # Check for critical failures
            critical_failures = []
            for step_name, result in results.items():
                if not result.success:
                    critical_failures.append(f"{step_name}: {result.error}")
            
            if critical_failures:
                raise ValueError(f"Critical tool failures: {'; '.join(critical_failures)}")
            
            # Format the response
            response_data = self._format_analysis_results(results, symbol)
            
            # Validate response data
            if not response_data.get('analysis'):
                raise ValueError("Failed to generate analysis data")
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                success=True,
                data=response_data,
                message=f"Successfully analyzed {symbol}",
                execution_plan=plan,
                execution_results=results,
                execution_time=execution_time,
                agent_name=self.name,
                request_id=request.request_id,
                confidence=0.9
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return AgentResponse(
                success=False,
                data=None,
                message=f"Stock analysis failed: {str(e)}",
                execution_plan=ChainPlan(steps=[]),
                execution_results={},
                execution_time=execution_time,
                agent_name=self.name,
                request_id=request.request_id,
                confidence=0.0
            )
    
    def _plan_analysis_chain(self, request: AgentRequest, symbol: str) -> ChainPlan:
        """Plan the analysis chain based on request and symbol.
        
        Args:
            request: The agent request
            symbol: The stock symbol to analyze
            
        Returns:
            ChainPlan with the analysis steps
        """
        steps = []
        
        # Step 1: Get stock identity
        identity_step = ChainStep(
            tool='stock_identity',
            parameters={'symbol': symbol},
            description=f"Get company identity for {symbol}",
            output_data=['company_info', 'business_summary', 'sector', 'industry']
        )
        steps.append(identity_step)
        
        # Step 2: Summarize business information
        instruction = request.parameters.get('instruction', request.context.get('instruction'))
        if not instruction:
            instruction = "Provide a comprehensive analysis of the following business description, including key strengths, market position, and potential risks:"
        
        summary_step = ChainStep(
            tool='summarizer',
            parameters={
                'text': '{step_0.business_summary}',
                'instruction': instruction
            },
            description="Generate comprehensive business analysis",
            required_data=['business_summary'],
            output_data=['analysis_summary']
        )
        steps.append(summary_step)
        
        return ChainPlan(
            steps=steps,
            description=f"Comprehensive analysis of {symbol}",
            confidence=0.9
        )
    
    def _format_analysis_results(self, results: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Format the analysis results into a structured response.
        
        Args:
            results: Raw results from toolchain execution
            symbol: The analyzed stock symbol
            
        Returns:
            Formatted analysis data
        """
        formatted_data = {
            'symbol': symbol,
            'analysis': {},
            'timestamp': results.get('timestamp'),
            'tools_used': []
        }
        
        # Process stock identity results
        if 'step_0' in results and results['step_0'].success:
            identity_data = results['step_0'].data
            formatted_data['analysis']['company_info'] = {
                'company_name': identity_data.get('company_name', 'N/A'),
                'sector': identity_data.get('sector', 'N/A'),
                'industry': identity_data.get('industry', 'N/A'),
                'business_summary': identity_data.get('business_summary', 'N/A')
            }
            formatted_data['tools_used'].append('stock_identity')
        
        # Process summary results
        if 'step_1' in results and results['step_1'].success:
            summary_data = results['step_1'].data
            
            # Handle nested result structure - extract clean text
            analysis_text = None
            
            # Case 1: summary_data is a ToolResult with nested data
            if hasattr(summary_data, 'data'):
                summary_content = summary_data.data
                # Case 2: Nested ToolResult
                if hasattr(summary_content, 'data'):
                    nested_data = summary_content.data
                    if isinstance(nested_data, dict):
                        analysis_text = nested_data.get('summary', str(nested_data))
                    else:
                        analysis_text = str(nested_data)
                # Case 3: Direct dictionary with summary key
                elif isinstance(summary_content, dict):
                    analysis_text = summary_content.get('summary', str(summary_content))
                else:
                    analysis_text = str(summary_content)
            # Case 4: Direct dictionary
            elif isinstance(summary_data, dict):
                analysis_text = summary_data.get('summary', str(summary_data))
            else:
                analysis_text = str(summary_data)
            
            # Clean up the analysis text
            if analysis_text:
                # Remove dictionary formatting if present
                if analysis_text.startswith("{'summary':") and analysis_text.endswith("'}"):
                    # Extract the content inside the quotes
                    import re
                    match = re.search(r"'summary':\s*'([^']*)'", analysis_text)
                    if match:
                        analysis_text = match.group(1)
                
                # Remove any remaining dictionary artifacts
                analysis_text = analysis_text.replace("{'summary':", "").replace("'}", "").strip()
                if analysis_text.startswith("'") and analysis_text.endswith("'"):
                    analysis_text = analysis_text[1:-1]
            
            formatted_data['analysis']['business_analysis'] = analysis_text
            formatted_data['tools_used'].append('summarizer')
        
        # Add metadata
        formatted_data['execution_summary'] = {
            'total_steps': len(results),
            'successful_steps': sum(1 for r in results.values() if r.success),
            'failed_steps': sum(1 for r in results.values() if not r.success)
        }
        
        return formatted_data
