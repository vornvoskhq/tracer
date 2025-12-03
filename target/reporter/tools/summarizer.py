#!/usr/bin/env python3
"""
Text Summarization Tool

A tool that takes text and generates a summary based on instructions.
"""

from typing import Dict, Any, Optional
import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.base_tool import Tool, ToolResult, ToolStatus
from src.llm_model import ModelManager
from src.llm_client import StockAnalysisLLM

class Summarizer(Tool):
    """A tool for generating text summaries using LLMs."""
    
    name: str = "summarizer"
    description: str = "Generates a summary of the input text based on the given instructions."
    version: str = "1.0"
    
    def __init__(self, model_manager: ModelManager):
        """Initialize the summarizer with a model manager."""
        self.model_manager = model_manager
        self.llm_client = None  # Will be initialized when needed
    
    async def _execute(self, text: str, instruction: Optional[str] = None, **kwargs) -> ToolResult:
        """
        Generate a summary of the input text.
        
        Args:
            text: The text to summarize
            instruction: Optional instructions for the summary
            **kwargs: Additional arguments (model, max_tokens, etc.)
            
        Returns:
            ToolResult with the summary or an error message
        """
        try:
            if not text or not text.strip():
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error="No text provided for summarization",
                    metadata={"input_text_length": 0}
                )
            
            # Get model configuration
            model_config = self.model_manager.get_model_for_role('summarizer')
            model_name = model_config.get('full_name', 'z-ai/glm-4.5-air')
            max_tokens = kwargs.get('max_tokens', 500)
            
            # Prepare the system prompt with clear instructions and placeholder for stock_data
            system_prompt = """You are a helpful assistant that summarizes text. 
            Provide a clear and concise summary of the following business description. 
            Focus on the company's main business activities, products, and market position.
            
            Business description to summarize:
            {stock_data}"""
            
            # Add any additional instructions if provided
            if instruction:
                system_prompt += f"\n\nAdditional instructions: {instruction}"
            
            # Initialize LLM client if not already done
            if self.llm_client is None:
                self.llm_client = StockAnalysisLLM()
            
            # Use the client with context manager to ensure proper cleanup
            try:
                async with self.llm_client as client:
                    response = await client.analyze_stock(
                        stock_data=text,
                        system_prompt=system_prompt,
                        stock_symbol=kwargs.get('symbol', ''),
                        role='summarizer'  # Specify the role for this tool
                    )
            except Exception as e:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error=f"Error in LLM request: {str(e)}",
                    metadata={"model": model_name, "input_text_length": len(text), "role": 'summarizer'}
                )
            
            # Check if we got a valid response
            if not response or not isinstance(response, tuple) or len(response) != 3:
                error_msg = f"Unexpected response format from LLM. Type: {type(response).__name__}"
                if isinstance(response, tuple):
                    error_msg += f", Length: {len(response)}"
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error=error_msg,
                    metadata={
                        "model": model_name,
                        "input_text_length": len(text),
                        "response_type": str(type(response)),
                        "response_length": str(len(response)) if hasattr(response, '__len__') else 'N/A'
                    }
                )
            
            summary, input_tokens, output_tokens = response
            
            # Check for error in response
            if not summary or (isinstance(summary, str) and summary.startswith("Error")):
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error=str(summary) if summary else "Empty response from LLM",
                    metadata={
                        "model": model_name,
                        "input_text_length": len(text),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens
                    }
                )
            
            # Ensure summary is a string
            summary_text = str(summary).strip()
            if not summary_text:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error="Empty summary generated",
                    metadata={
                        "model": model_name,
                        "input_text_length": len(text),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens
                    }
                )
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"summary": summary_text},
                metadata={
                    "model": model_name,
                    "input_text_length": len(text),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Error processing LLM response: {str(e)}",
                metadata={
                    "model": model_name,
                    "input_text_length": len(text)
                    }
                )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to generate summary: {str(e)}",
                metadata={"input_text_length": len(text) if text else 0}
            )


async def main():
    """Command-line interface for the summarization tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate a summary of text using an LLM')
    parser.add_argument('--text', type=str, help='Text to summarize (or use --file)')
    parser.add_argument('--file', type=str, help='File containing text to summarize')
    parser.add_argument('--instruction', type=str, default="Provide a concise summary of the text.",
                      help='Instructions for the summary')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                      help='Model to use for summarization')
    
    args = parser.parse_args()
    
    # Read text from file or command line
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        # Read from stdin if no text or file provided
        text = sys.stdin.read()
    
    if not text.strip():
        print("Error: No text provided to summarize", file=sys.stderr)
        sys.exit(1)
    
    try:
        model_manager = ModelManager()
        summarizer = Summarizer(model_manager)
        result = await summarizer.execute(
            text=text,
            instruction=args.instruction,
            model=args.model
        )
        
        if result.success:
            print("\nSummary:")
            print("=" * 80)
            print(result.data['summary'])
            print("\nMetadata:", result.metadata)
        else:
            print(f"Error: {result.error}", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())