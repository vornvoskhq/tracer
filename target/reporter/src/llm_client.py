"""LLM client for making requests to OpenRouter API."""

import asyncio
import json
import os
import time
import aiohttp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.llm_model import ModelManager, ModelConfig, track_llm_cost, llm_cost_estimator, load_api_key


def _load_api_key_from_env_file() -> str:
    """Load OPENROUTER_API_KEY from .env file."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]
    
    for env_path in candidates:
        if not env_path.exists():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "OPENROUTER_API_KEY":
                    return value.strip().strip('"').strip("'")
        except OSError:
            continue
    return ""


def _log_console_run(
    *,
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    prompt_tokens: int,
    completion_tokens: int,
    estimated_cost: Optional[float],
    duration_s: Optional[float] = None,
) -> None:
    """Log LLM run to console."""
    max_tok_display = str(max_tokens) if isinstance(max_tokens, int) and max_tokens > 0 else "-"
    cost_display = f"${estimated_cost:.6f}" if estimated_cost is not None else "NA"
    dur_display = f"{duration_s:.2f}s" if isinstance(duration_s, (int, float)) else "-"

    print(
        f"LLM | model={model} | temp={temperature:.2f} | max_tok={max_tok_display} | "
        f"in={prompt_tokens} | out={completion_tokens} | cost={cost_display} | dur={dur_display}"
    )


class StockAnalysisLLM:
    """Client for interacting with OpenRouter API."""
    
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the LLM client.
        
        Args:
            config_path: Path to model configuration file
        """
        self.api_key = load_api_key()
        self.model_manager = ModelManager(config_path)
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None

    @llm_cost_estimator
    async def analyze_stock(
        self,
        stock_data: str,
        system_prompt: str,
        stock_symbol: str = "UNKNOWN",
        role: str = 'default'
    ) -> str:
        """
        Analyze stock data using the provided system prompt.
        
        Args:
            stock_data: Raw stock data to analyze
            system_prompt: System prompt template with {stock_data} placeholder
            stock_symbol: Stock symbol for logging purposes
        
        Returns:
            Analysis response from LLM
        """
        if not self.api_key:
            return "OpenRouter API key not configured (set OPENROUTER_API_KEY)."
            
        if not self.session:
            self.session = aiohttp.ClientSession()

        # Format the prompt with stock data
        user_prompt = system_prompt.format(stock_data=stock_data)
        
        # Get model configuration for the specified role
        try:
            model_config = self.model_manager.get_model_for_role(role)
        except ValueError:
            # Fall back to default if role not found
            model_config = self.model_manager.get_model_config()
        model_name = model_config.get('full_name', 'openai/gpt-3.5-turbo')
        display_name = model_config.get('display_name', model_name)
        
        # Log model configuration in a clean, single line
        print(f"\n🔧 Using {display_name} (role: {role})")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/your-repo",  # Update this
            "X-Title": "Stock Analysis Tool"
        }
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are an expert financial analyst specializing in stock market analysis."},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": model_config.get('temperature', 0.7),
            "max_tokens": model_config.get('max_tokens', 1000)
        }
        
        try:
            async with self.session.post(
                self.OPENROUTER_API_URL,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return f"Error from OpenRouter API (HTTP {response.status}): {error_text}"
                    
                result = await response.json()
                
                if 'choices' in result and result['choices'] and 'message' in result['choices'][0]:
                    content = result['choices'][0]['message'].get('content', '').strip()
                    usage = result.get('usage', {})
                    input_tokens = usage.get('prompt_tokens', 0)
                    output_tokens = usage.get('completion_tokens', 0)
                    return (content, input_tokens, output_tokens)  # Return as tuple
                else:
                    error_detail = result.get('error', {}).get('message', 'No valid response from model')
                    return (f"Error from model: {error_detail}", 0, 0)
                    return f"Error from model: {error_detail}", 0, 0
                    
        except Exception as e:
            error_msg = f"Error making request to OpenRouter API: {str(e)}"
            print(f"⚠️  {error_msg}", file=sys.stderr)
            return (error_msg, 0, 0)  # Explicitly return as tuple