"""
GeoMining Token Creator
Creates and manages SPL tokens on Solana for geolocation-based mining rewards
"""

import os
import json
from typing import Optional
from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed
from solana.keypair import Keypair
from solana.publickey import PublicKey
from solana.rpc.types import TokenAccountOpts
from solders.pubkey import Pubkey as PublicKey
from solders.keypair import Keypair
from solders.system_program import CreateAccountParams, create_account
from solders.spl.token.instructions import (
    initialize_mint, 
    mint_to, 
    create_account as create_token_account,
    get_associated_token_address,
    create_associated_token_account
)
from solders.spl.token.state import MINT_LEN, ACCOUNT_LEN
from solders.transaction import Transaction
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solders.transaction import Transaction
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeoMiningTokenCreator:
    """Handles creation and management of the GeoMining SPL token"""
    
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
        self.token = None
        
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
    
    def create_token(self, decimals: int = 9) -> Optional[PublicKey]:
        """
        Create a new SPL token
        
        Args:
            decimals: Number of decimal places (9 is standard like SOL)
            
        Returns:
            Token mint address
        """
        try:
            if not self.mint_authority:
                self.create_or_load_mint_authority()
                
            # Create token
            self.token = Token.create_mint(
                self.client,
                self.mint_authority,
                self.mint_authority.public_key,  # mint authority
                decimals,
                TOKEN_PROGRAM_ID
            )
            
            logger.info(f"Token created: {self.token.pubkey}")
            return self.token.pubkey
            
        except Exception as e:
            logger.error(f"Token creation failed: {e}")
            return None
    
    def create_token_account(self, owner: Optional[PublicKey] = None) -> Optional[PublicKey]:
        """
        Create token account for holding tokens
        
        Args:
            owner: Account owner (defaults to mint authority)
            
        Returns:
            Token account address
        """
        try:
            if not self.token:
                logger.error("Token not created yet")
                return None
                
            owner = owner or self.mint_authority.public_key
            
            account = self.token.create_account(owner)
            logger.info(f"Token account created: {account}")
            return account
            
        except Exception as e:
            logger.error(f"Token account creation failed: {e}")
            return None
    
    def mint_initial_supply(self, amount: int, destination: PublicKey) -> bool:
        """
        Mint initial token supply
        
        Args:
            amount: Number of tokens to mint (in smallest unit)
            destination: Account to receive tokens
            
        Returns:
            True if successful
        """
        try:
            if not self.token:
                logger.error("Token not created yet")
                return False
                
            self.token.mint_to(
                destination,
                self.mint_authority,
                amount
            )
            
            logger.info(f"Minted {amount} tokens to {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Minting failed: {e}")
            return False
    
    def get_token_balance(self, account: PublicKey) -> Optional[int]:
        """
        Get token balance for an account
        
        Args:
            account: Token account address
            
        Returns:
            Token balance in smallest unit
        """
        try:
            if not self.token:
                return None
                
            balance = self.token.get_balance(account)
            return balance.value.amount
            
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return None
    
    def save_token_config(self, config_path: str = "token_config.json"):
        """
        Save token configuration for later use
        
        Args:
            config_path: Path to save configuration
        """
        if not self.token:
            logger.error("No token to save")
            return
            
        config = {
            "network": self.network,
            "token_address": str(self.token.pubkey),
            "mint_authority": str(self.mint_authority.public_key),
            "decimals": 9,  # This should be stored when creating token
            "program_id": str(TOKEN_PROGRAM_ID)
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Token config saved to {config_path}")


# Example usage and setup
if __name__ == "__main__":
    # Example token creation flow
    creator = GeoMiningTokenCreator("devnet")
    
    # Step 1: Create mint authority
    mint_authority = creator.create_or_load_mint_authority()
    
    # Step 2: Get some SOL for transaction fees (devnet only)
    creator.airdrop_sol(2.0)
    
    # Step 3: Create the token
    token_address = creator.create_token(decimals=9)
    
    if token_address:
        print(f"✅ GeoMining Token created: {token_address}")
        
        # Step 4: Create treasury account
        treasury_account = creator.create_token_account()
        
        if treasury_account:
            # Step 5: Mint initial supply (1 billion tokens)
            initial_supply = 1_000_000_000 * (10 ** 9)  # 1B tokens with 9 decimals
            success = creator.mint_initial_supply(initial_supply, treasury_account)
            
            if success:
                print(f"✅ Minted {initial_supply / (10**9):,.0f} tokens to treasury")
                print(f"Treasury Account: {treasury_account}")
                
                # Save configuration
                creator.save_token_config()
                print("✅ Token configuration saved")
                
                # Display summary
                print("\n" + "="*50)
                print("GEOMINING TOKEN CREATION SUMMARY")
                print("="*50)
                print(f"Network: {creator.network}")
                print(f"Token Address: {token_address}")
                print(f"Treasury Account: {treasury_account}")
                print(f"Mint Authority: {mint_authority.public_key}")
                print(f"Initial Supply: 1,000,000,000 tokens")
                print("="*50)
            else:
                print("❌ Failed to mint initial supply")
        else:
            print("❌ Failed to create treasury account")
    else:
        print("❌ Failed to create token")