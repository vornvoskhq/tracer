import asyncio
import yfinance as yf
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, ClassVar

from src.llm_model import ModelManager
from .base_tool import Tool, ToolResult, ToolStatus


@dataclass
class CompanyProfile:
    """Container for company profile information."""
    symbol: str
    company_name: str
    sector: str
    industry: str
    business_summary: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert the profile to a dictionary."""
        return {
            'symbol': self.symbol,
            'company_name': self.company_name,
            'sector': self.sector,
            'industry': self.industry,
            'business_summary': self.business_summary
        }
    
    def __str__(self) -> str:
        """String representation of the company profile."""
        return (
            f"Company: {self.company_name} ({self.symbol})\n"
            f"Sector/Industry: {self.sector}/{self.industry}\n"
            f"\nBusiness Summary:\n{self.business_summary}"
        )


class StockIdentity(Tool):
    """Tool for retrieving company profile information."""
    
    # Class variables
    name: ClassVar[str] = "stock_identity"
    description: ClassVar[str] = "Retrieves company profile information including name, sector, and business summary."
    version: ClassVar[str] = "1.0"
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        """Initialize the StockIdentity tool.
        
        Args:
            model_manager: Optional ModelManager instance. If not provided,
                         a new one will be created.
        """
        self.model_manager = model_manager or ModelManager()
    
    def get_parameters_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about the tool's parameters.
        
        Returns:
            Dictionary mapping parameter names to their metadata (type, required, description)
        """
        return {
            "symbol": {
                "type": str,
                "required": True,
                "description": "Stock symbol (e.g., 'AAPL')"
            }
        }
    
    async def _execute(self, symbol: str) -> Dict[str, Any]:
        """Fetch company profile for the given stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Dictionary containing company profile information
            
        Raises:
            ValueError: If the symbol is invalid or data cannot be fetched
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Create and return profile as dict
            profile = CompanyProfile(
                symbol=symbol.upper(),
                company_name=info.get('longName', 'N/A'),
                sector=info.get('sector', 'N/A'),
                industry=info.get('industry', 'N/A'),
                business_summary=info.get('longBusinessSummary', 'N/A')
            )
            
            return profile.to_dict()
            
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}")
    
    def display_profile(self, profile_data: Dict[str, Any]) -> None:
        """Display company profile in a formatted string.
        
        Args:
            profile_data: Company profile data as returned by _execute
        """
        profile = CompanyProfile(**profile_data)
        print(profile)


# For backward compatibility
def get_company_profile(symbol: str) -> List[str]:
    """Legacy function that returns a list for backward compatibility."""
    tool = StockIdentity()
    try:
        result = asyncio.run(tool.execute(symbol=symbol))
        if not result.success:
            return [f"Error: {result.error}"]
            
        profile = CompanyProfile(**result.data)
        return [
            profile.company_name,
            f"{profile.sector}/{profile.industry}",
            profile.business_summary
        ]
    except Exception as e:
        return [f"Error: {str(e)}"]


async def main():
    """Example usage of the StockIdentity tool."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python -m tools.stock_identity <symbol>")
        sys.exit(1)
        
    symbol = sys.argv[1]
    tool = StockIdentity()
    
    try:
        result = await tool.execute(symbol=symbol)
        if not result.success:
            print(f"Error: {result.error}", file=sys.stderr)
            sys.exit(1)
            
        tool.display_profile(result.data)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
