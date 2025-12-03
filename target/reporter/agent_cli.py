#!/usr/bin/env python3
"""
Agent CLI - Command-line interface for the agent system.

This demonstrates how to use the StockAnalyst agent as a replacement
for the traditional reporter_cli functionality.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agents import AgentRegistry, AgentRequest


async def analyze_stock(symbol: str, instruction: str = None, config: str = None):
    """Analyze a stock using the StockAnalyst agent.
    
    Args:
        symbol: Stock symbol to analyze
        instruction: Optional custom instruction for analysis
        config: Optional config file path
    """
    try:
        # Get the StockAnalyst agent
        analyst = AgentRegistry.create_agent('stock_analyst', config_path=config)
        
        # Create the request
        parameters = {'symbol': symbol}
        if instruction:
            parameters['instruction'] = instruction
        
        request = AgentRequest(
            goal=f"Analyze {symbol} stock",
            parameters=parameters
        )
        
        print(f"🤖 Analyzing {symbol} using {analyst.name} agent...")
        print(f"📋 Goal: {request.goal}")
        
        # Execute the analysis
        response = await analyst.execute(request)
        
        # Display results
        if response.success:
            print(f"\n✅ Analysis completed successfully!")
            print(f"⏱️  Execution time: {response.execution_time:.2f}s")
            print(f"🎯 Confidence: {response.confidence:.1%}")
            
            data = response.data
            print(f"\n📊 Analysis for {data['symbol']}:")
            print("=" * 50)
            
            # Company information
            if 'company_info' in data['analysis']:
                info = data['analysis']['company_info']
                print(f"🏢 Company: {info['company_name']}")
                print(f"🏭 Sector: {info['sector']}")
                print(f"⚙️  Industry: {info['industry']}")
            
            # Business analysis
            if 'business_analysis' in data['analysis']:
                print(f"\n📈 Business Analysis:")
                print("-" * 30)
                analysis_text = data['analysis']['business_analysis']
                # Truncate very long analyses for console display
                if len(analysis_text) > 500:
                    print(analysis_text[:500] + "...")
                else:
                    print(analysis_text)
            
            # Execution details
            print(f"\n🔧 Tools used: {', '.join(data['tools_used'])}")
            summary = data['execution_summary']
            print(f"📋 Steps: {summary['successful_steps']}/{summary['total_steps']} successful")
            
            if response.warnings:
                print(f"\n⚠️  Warnings:")
                for warning in response.warnings:
                    print(f"  - {warning}")
        
        else:
            print(f"\n❌ Analysis failed: {response.message}")
            if response.execution_time > 0:
                print(f"⏱️  Failed after {response.execution_time:.2f}s")
            return 1
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return 1
    
    return 0


async def list_agents():
    """List all available agents."""
    print("🤖 Available Agents:")
    print("=" * 30)
    
    agents = AgentRegistry.list_agents()
    for name, info in agents.items():
        print(f"📋 {name}")
        print(f"   Description: {info['description']}")
        print(f"   Version: {info['version']}")
        print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Agent-based stock analysis system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s AAPL                           # Analyze AAPL with default settings
  %(prog)s AAPL --instruction "Focus on risks"  # Custom analysis instruction
  %(prog)s --list-agents                  # List available agents
        """
    )
    
    # Main analysis arguments
    parser.add_argument(
        'symbol', 
        nargs='?',
        help='Stock symbol to analyze (e.g., AAPL, TSLA)'
    )
    
    parser.add_argument(
        '--instruction',
        help='Custom instruction for the analysis'
    )
    
    parser.add_argument(
        '--config',
        help='Path to configuration file'
    )
    
    # Utility arguments
    parser.add_argument(
        '--list-agents',
        action='store_true',
        help='List all available agents'
    )
    
    args = parser.parse_args()
    
    # Handle list agents
    if args.list_agents:
        asyncio.run(list_agents())
        return
    
    # Validate main arguments
    if not args.symbol:
        parser.error("Stock symbol is required (or use --list-agents)")
    
    # Run the analysis
    exit_code = asyncio.run(analyze_stock(
        symbol=args.symbol.upper(),
        instruction=args.instruction,
        config=args.config
    ))
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
