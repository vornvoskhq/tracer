"""Report generation and formatting for stock analysis."""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import Config
from data_loader import VGMiniDataLoader
from llm_client import StockAnalysisLLM
from console_themes import DystopianTheme, NormalTheme


class StockReportGenerator:
    """Generate stock analysis reports using LLM."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = VGMiniDataLoader(config.ranking_output_path)
        
        # Get full model name from config
        model_config = config.get_model_config(config.default_model)
        full_model_name = model_config.get('full_name', config.default_model)
        
        # Initialize LLM client
        self.llm_client = StockAnalysisLLM(
            model=full_model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            price_table=config.model_prices
        )
        
    def load_system_prompt(self, prompt_filename: str) -> str:
        """Load system prompt from file."""
        prompt_path = self.config.get_prompt_path(prompt_filename)
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
            
        try:
            return prompt_path.read_text(encoding='utf-8')
        except Exception as e:
            raise ValueError(f"Error reading prompt file {prompt_path}: {e}")
    
    async def generate_report(self, symbol: str, prompt_filename: str) -> str:
        """Generate a stock analysis report for the given symbol."""
        
        # Check LLM configuration
        if not self.llm_client.is_configured():
            return "⚡ LLM not configured. Please set OPENROUTER_API_KEY environment variable."
        
        # Get theme for messages
        theme = DystopianTheme if self.config.console_theme == "dystopian" else NormalTheme
        
        print(f"{theme.get_progress_message('analyzing')} {symbol}...")
        
        # Load stock data
        print(theme.get_progress_message('loading_data'))
        stock_data = self.data_loader.get_stock_data(symbol)
        formatted_data = self.data_loader.format_stock_data_for_llm(stock_data)
        
        # Load system prompt
        print(theme.get_progress_message('loading_prompt') + f": {prompt_filename}")
        try:
            system_prompt = self.load_system_prompt(prompt_filename)
            # Apply length configuration to prompt
            system_prompt = self._apply_length_config(system_prompt)
        except (FileNotFoundError, ValueError) as e:
            return f"⚡ Error loading prompt: {e}"
        
        # Generate analysis
        print(theme.get_progress_message('generating'))
        analysis = await self.llm_client.analyze_stock(
            stock_data=formatted_data,
            system_prompt=system_prompt,
            stock_symbol=symbol
        )
        
        return analysis
    
    def _apply_length_config(self, system_prompt: str) -> str:
        """Apply summary length configuration to the prompt."""
        length_key = self.config.default_length
        length_config = self.config.get_length_config(length_key)
        
        instruction = length_config.get('instruction', 'Provide a comprehensive analysis with detailed explanations.')
        return system_prompt + f"\n\n{instruction}"
    
    def save_report(self, symbol: str, analysis: str) -> str:
        """Save report to file and return the file path."""
        
        # Ensure reports directory exists
        reports_dir = Path(self.config.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        filename = f"{symbol.upper()}.txt"
        filepath = reports_dir / filename
        
        # Create report content with header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_config = self.config.get_model_config(self.config.default_model)
        length_config = self.config.get_length_config(self.config.default_length)
        
        report_content = f"""Stock Analysis Report: {symbol.upper()}
Generated: {timestamp}
Model: {self.config.default_model} ({model_config.get('display_name', 'Unknown')})
Length: {self.config.default_length} ({length_config.get('display_name', 'Unknown')})

{'=' * 60}

{analysis}

{'=' * 60}
End of Report
"""
        
        # Write to file
        try:
            filepath.write_text(report_content, encoding='utf-8')
            return str(filepath)
        except Exception as e:
            raise IOError(f"Error saving report to {filepath}: {e}")
    
    def display_console_report(self, symbol: str, analysis: str) -> None:
        """Display report in console with formatting."""
        
        # Get theme
        theme = DystopianTheme if self.config.console_theme == "dystopian" else NormalTheme
        
        if self.config.console_format == "detailed":
            if self.config.console_theme == "dystopian":
                print(theme.format_header(symbol, self.config.default_model))
                print(f"▓ TEMPORAL MARKER: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                length_config = self.config.get_length_config(self.config.default_length)
                length_display = length_config.get('display_name', self.config.default_length.upper())
                print(f"▓ ANALYSIS LENGTH: {length_display}")
                print(theme.SYMBOLS['border'])
                print()
                print(analysis)
                print()
                print(theme.format_footer())
            else:
                # Standard detailed format
                print("\n" + "=" * 80)
                print(f"📈 STOCK ANALYSIS REPORT: {symbol.upper()}")
                print("=" * 80)
                print(f"🤖 Model: {self.config.default_model}")
                print(f"⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 80)
                print()
                print(analysis)
                print()
                print("=" * 80)
        else:
            # Compact format
            print(theme.format_header(symbol, self.config.default_model))
            print(analysis)
            print(theme.format_footer())
    
    async def generate_and_save_report(
        self, 
        symbol: str, 
        prompt_filename: str,
        save_to_file: bool = True,
        display_console: bool = True
    ) -> Optional[str]:
        """Generate report and optionally save/display it."""
        
        # Generate the analysis
        analysis = await self.generate_report(symbol, prompt_filename)
        
        # Check if analysis generation failed
        if analysis.startswith("❌"):
            print(analysis)
            return None
        
        # Display in console if requested
        if display_console:
            self.display_console_report(symbol, analysis)
        
        # Save to file if requested
        saved_path = None
        if save_to_file:
            try:
                saved_path = self.save_report(symbol, analysis)
                theme = DystopianTheme if self.config.console_theme == "dystopian" else NormalTheme
                if self.config.console_theme == "dystopian":
                    print(f"▓ {theme.get_progress_message('saved')}: {saved_path}")
                else:
                    print(f"💾 Report saved to: {saved_path}")
            except IOError as e:
                if self.config.console_theme == "dystopian":
                    print(f"☢ Warning: Data archival failed - {e}")
                else:
                    print(f"⚠️  Warning: Could not save report - {e}")
        
        return saved_path