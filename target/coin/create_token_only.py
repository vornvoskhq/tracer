#!/usr/bin/env python3
"""
Simple GeoMining Token Creator
Run this after setting up Solana CLI manually
"""

import sys
import json
from token_creator_simple import SimpleGeoMiningTokenCreator

def main():
    print("🪙 GeoMining Token Creation")
    print("=" * 40)
    
    try:
        # Initialize creator
        print("📋 Initializing simplified token setup...")
        creator = SimpleGeoMiningTokenCreator("devnet")
        
        # Create mint authority
        print("🔑 Creating mint authority...")
        mint_authority = creator.create_or_load_mint_authority()
        print(f"Mint Authority: {mint_authority.public_key}")
        
        # Check balance
        balance = creator.get_balance()
        print(f"💰 Current balance: {balance:.4f} SOL")
        
        if balance < 0.1:  # Need some SOL for transactions
            print("💰 Requesting SOL for transaction fees...")
            airdrop_success = creator.airdrop_sol(2.0)
            
            if not airdrop_success:
                print("⚠️  Airdrop failed. You may need SOL in your wallet for transaction fees.")
                print("You can request SOL manually with: solana airdrop 2")
                
                response = input("Continue anyway? (y/N): ").strip().lower()
                if response != 'y':
                    sys.exit(1)
            
            # Wait a moment for airdrop to process
            import time
            time.sleep(3)
            balance = creator.get_balance()
            print(f"💰 New balance: {balance:.4f} SOL")
        
        # Create simplified token configuration
        print("⚙️  Creating token configuration...")
        success = creator.create_simple_token_config()
        
        if success:
            print("\n" + "🎉" * 20)
            print("DEMO SETUP COMPLETE!")
            print("🎉" * 20)
            print(f"Network: devnet")
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
            print("📝 Note: This creates a demo configuration.")
            print("   The reward system will work for testing purposes.")
            print("   For production, you'll need to create a real token mint.")
            print("🎉" * 20)
            
            return True
        else:
            print("❌ Failed to create configuration")
            
    except Exception as e:
        print(f"❌ Token creation failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Solana CLI is installed and in PATH")
        print("2. Check you're on devnet: solana config get")
        print("3. Ensure you have SOL: solana balance")
        print("4. Try manual airdrop: solana airdrop 2")
        return False
    
    return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Token creation failed. Please check the error messages above.")
        sys.exit(1)