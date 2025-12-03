#!/usr/bin/env python3
"""
Stock Analysis Reporter CLI

Analyzes stocks from VGMini rankings using LLMs to provide qualitative analysis.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import Config
from src.data_loader import VGMiniDataLoader  
from src.report_generator import StockReportGenerator
from src.console_themes import DystopianTheme, NormalTheme
from src.cli_helpers import list_available_models, list_length_templates, display_current_config


def list_available_prompts(config: Config) -> None:
    """List all available prompt files."""
    prompts_dir = config.get_prompt_path("")
    
    if not prompts_dir.exists():
        print("⚠ Prompts directory not found")
        return
    
    prompt_files = [f.name for f in prompts_dir.glob("*.txt") if f.is_file()]
    
    if not prompt_files:
        print("⚠ No prompt files found")
        return
        
    print("⚙ Available prompts:")
    for prompt_file in sorted(prompt_files):
        print(f"  • {prompt_file}")


def list_top_stocks(config: Config, n: int = 10) -> None:
    """List top N stocks from VGMini rankings."""
    data_loader = VGMiniDataLoader(config.ranking_output_path)
    stocks = data_loader.get_top_stocks(n)
    
    # Get theme
    theme = DystopianTheme if config.console_theme == "dystopian" else NormalTheme
    
    if not stocks:
        print("⚡ No stocks found in VGMini rankings")
        return
        
    print(theme.format_stock_list_header(len(stocks)))
    for stock in stocks:
        print(theme.format_stock_entry(
            stock['rank'], 
            stock['symbol'], 
            stock.get('score'), 
            stock.get('buy_signal', False)
        ))


async def main():
    parser = argparse.ArgumentParser(
        description="Analyze stocks using LLMs based on VGMini rankings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s AAPL                           # Analyze AAPL with default settings
  %(prog)s AAPL --prompt technical_focus.txt  # Use technical analysis prompt
  %(prog)s --list-stocks                      # Show top ranked stocks
  %(prog)s --list-prompts                     # Show available prompts
  %(prog)s --list-models                      # Show available LLM models
  %(prog)s --list-lengths                     # Show length templates
  %(prog)s --top 3                           # Analyze top 3 ranked stocks
  
Configuration is managed through config.yaml - no command-line overrides needed.
        """
    )
    
    # Main action arguments
    parser.add_argument(
        "symbol", 
        nargs="?",
        help="Stock symbol to analyze (e.g., AAPL, TSLA)"
    )
    
    parser.add_argument(
        "--prompt", 
        default=None,
        help="System prompt file to use (default from config)"
    )
    
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model to use (default from config)"
    )
    
    # Listing actions
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available prompt files"
    )
    
    parser.add_argument(
        "--list-stocks",
        action="store_true", 
        help="List top ranked stocks from VGMini"
    )
    
    parser.add_argument(
        "--top",
        type=int,
        metavar="N",
        help="Analyze top N stocks from rankings"
    )
    
    # Output options
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save report to file"
    )
    
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Don't display report in console"
    )
    
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available LLM models and pricing"
    )
    
    parser.add_argument(
        "--list-lengths",
        action="store_true", 
        help="List available length templates"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = Config(args.config)
        
        # No CLI overrides needed - everything is in config
        
        # Handle listing actions
        if args.list_prompts:
            list_available_prompts(config)
            return
            
        if args.list_stocks:
            list_top_stocks(config, 20)
            return
            
        if args.list_models:
            list_available_models(config)
            return
            
        if args.list_lengths:
            list_length_templates(config)
            return
        
        # Set up report generator
        report_gen = StockReportGenerator(config)
        
        # Show current configuration for transparency
        display_current_config(config)
        
        prompt_filename = args.prompt or config.default_system_prompt
        
        # Handle top N stocks analysis
        if args.top:
            data_loader = VGMiniDataLoader(config.ranking_output_path)
            top_stocks = data_loader.get_top_stocks(args.top)
            
            if not top_stocks:
                print("❌ No stocks found in VGMini rankings")
                return
                
            print(f"🚀 Analyzing top {len(top_stocks)} stocks...")
            
            for i, stock in enumerate(top_stocks, 1):
                symbol = stock['symbol']
                print(f"\n[{i}/{len(top_stocks)}] Analyzing {symbol}...")
                
                await report_gen.generate_and_save_report(
                    symbol=symbol,
                    prompt_filename=prompt_filename,
                    save_to_file=not args.no_save,
                    display_console=not args.no_console
                )
            
            print(f"\n✅ Completed analysis of {len(top_stocks)} stocks")
            return
        
        # Handle single stock analysis
        if not args.symbol:
            parser.error("Stock symbol required (or use --list-stocks, --list-prompts, or --top)")
        
        symbol = args.symbol.upper()
        
        await report_gen.generate_and_save_report(
            symbol=symbol,
            prompt_filename=prompt_filename,
            save_to_file=not args.no_save,
            display_console=not args.no_console
        )
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())