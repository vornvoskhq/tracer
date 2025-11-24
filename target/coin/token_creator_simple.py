"""
Simple GeoMining Token Creator
Uses the correct Solana Python API without deprecated SPL imports
"""

import os
import json
from typing import Optional
from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed
from solders.keypair import Keypair
from solders.pubkey import Pubkey as PublicKey
from solders.system_program import create_account, CreateAccountParams
from solders.transaction import Transaction
from solders.system_program import ID as SYSTEM_PROGRAM_ID
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Token Program ID (this is the standard SPL Token program)
TOKEN_PROGRAM_ID = PublicKey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

class SimpleGeoMiningTokenCreator:
    """Simplified token creator using basic Solana functionality"""
    
    def __init__(self, network: str = "devnet"):
        """
        Initialize token creator
        
        Args:
            network: "devnet", "testnet", or "mainnet-beta"
        """
        self.network = network
        self.rpc_urls = {
            "devnet": "https://api.devnet.solana.com",
            "testnet": "https://api.testnet.solana.com", 
            "mainnet-beta": "https://api.mainnet-beta.solana.com"
        }
        
        self.client = Client(self.rpc_urls[network], commitment=Confirmed)
        self.mint_authority = None
        self.mint_pubkey = None
        
    def create_or_load_mint_authority(self, keypair_path: str = "mint_authority.json") -> Keypair:
        """
        Create new mint authority or load existing one
        
        Args:
            keypair_path: Path to save/load keypair
            
        Returns:
            Keypair for mint authority
        """
        try:
            if os.path.exists(keypair_path):
                logger.info(f"Loading existing mint authority from {keypair_path}")
                with open(keypair_path, 'r') as f:
                    secret_key = json.load(f)
                self.mint_authority = Keypair.from_secret_key(bytes(secret_key))
            else:
                logger.info("Creating new mint authority")
                self.mint_authority = Keypair()
                
                # Save keypair securely
                with open(keypair_path, 'w') as f:
                    json.dump(list(self.mint_authority.secret_key), f)
                os.chmod(keypair_path, 0o600)  # Restrict permissions
                
            logger.info(f"Mint Authority: {self.mint_authority.public_key}")
            return self.mint_authority
            
        except Exception as e:
            logger.error(f"Error creating/loading mint authority: {e}")
            raise
    
    def airdrop_sol(self, amount: float = 2.0) -> bool:
        """
        Request SOL airdrop for devnet/testnet (required for transaction fees)
        
        Args:
            amount: Amount of SOL to request
            
        Returns:
            True if successful
        """
        if self.network == "mainnet-beta":
            logger.warning("Cannot airdrop on mainnet")
            return False
            
        try:
            lamports = int(amount * 1e9)  # Convert SOL to lamports
            response = self.client.request_airdrop(
                self.mint_authority.public_key, 
                lamports
            )
            
            if response.value:
                logger.info(f"Airdrop successful: {amount} SOL")
                return True
            else:
                logger.error("Airdrop failed")
                return False
                
        except Exception as e:
            logger.error(f"Airdrop error: {e}")
            return False
    
    def get_balance(self) -> float:
        """Get SOL balance of mint authority"""
        try:
            balance = self.client.get_balance(self.mint_authority.public_key)
            return balance.value / 1e9  # Convert lamports to SOL
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0
    
    def create_simple_token_config(self) -> bool:
        """
        Create a simplified token configuration for the treasury to use
        This creates a mock token configuration that can be used with a manual token creation
        """
        try:
            if not self.mint_authority:
                self.create_or_load_mint_authority()
            
            # For demonstration, we'll create a configuration that points to a well-known devnet token
            # In production, you would create an actual token mint
            
            # Using a test token mint address (you can replace this with your actual mint)
            test_mint = "So11111111111111111111111111111111111111112"  # Wrapped SOL mint as example
            
            config = {
                "network": self.network,
                "token_address": test_mint,
                "mint_authority": str(self.mint_authority.public_key),
                "decimals": 9,
                "program_id": str(TOKEN_PROGRAM_ID),
                "treasury_account": str(self.mint_authority.public_key),  # Simplified for demo
                "note": "Simplified configuration for demo purposes"
            }
            
            with open("token_config.json", 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info("✅ Token configuration created")
            logger.info(f"Configuration saved to token_config.json")
            
            return True
            
        except Exception as e:
            logger.error(f"Token configuration creation failed: {e}")
            return False


def main():
    """Main function for creating token configuration"""
    print("🪙 Simple GeoMining Token Setup")
    print("=" * 50)
    
    try:
        # Initialize creator
        creator = SimpleGeoMiningTokenCreator("devnet")
        
        # Step 1: Create mint authority
        print("🔑 Creating mint authority...")
        mint_authority = creator.create_or_load_mint_authority()
        
        # Step 2: Check balance
        balance = creator.get_balance()
        print(f"💰 Current balance: {balance:.4f} SOL")
        
        if balance < 0.1:  # Need some SOL for transactions
            print("💰 Requesting SOL for transaction fees...")
            creator.airdrop_sol(2.0)
            
            # Wait a bit and check again
            import time
            time.sleep(3)
            balance = creator.get_balance()
            print(f"💰 New balance: {balance:.4f} SOL")
        
        # Step 3: Create token configuration
        print("⚙️  Creating token configuration...")
        success = creator.create_simple_token_config()
        
        if success:
            print("\n" + "🎉" * 20)
            print("SETUP COMPLETE!")
            print("🎉" * 20)
            print(f"Network: {creator.network}")
            print(f"Mint Authority: {mint_authority.public_key}")
            print(f"Balance: {balance:.4f} SOL")
            print()
            print("✅ Configuration saved to token_config.json")
            print()
            print("Next steps:")
            print("1. Start the API: python geomining_api.py")
            print("2. Open frontend_example.html in your browser")
            print("3. Connect your wallet and start testing!")
            print()
            print("Note: This is a demo configuration.")
            print("For production, you'll need to create a real token mint.")
            print("🎉" * 20)
            
            return True
        else:
            print("❌ Setup failed")
            return False
            
    except Exception as e:
        print(f"❌ Setup error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Solana CLI is working: solana --version")
        print("2. Check you're on devnet: solana config get")
        print("3. Try manual airdrop: solana airdrop 2")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the error messages above.")
        exit(1)