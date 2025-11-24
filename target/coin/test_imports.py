#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing imports...")
    
    try:
        print("  ✓ Testing solana.rpc...")
        from solana.rpc.api import Client
        from solana.rpc.commitment import Confirmed
        
        print("  ✓ Testing solders...")
        from solders.keypair import Keypair
        from solders.pubkey import Pubkey as PublicKey
        
        print("  ✓ Testing token creator...")
        from token_creator_simple import SimpleGeoMiningTokenCreator
        
        print("  ✓ Testing treasury manager...")
        from treasury_manager import GeoMiningTreasury, Location, RewardConfig
        
        print("  ✓ Testing API...")
        from geomining_api import app
        
        print("  ✓ Testing other dependencies...")
        import fastapi
        import uvicorn
        import pydantic
        import geopy
        import haversine
        
        print("✅ All imports successful!")
        
        # Test basic functionality
        print("\n🧪 Testing basic functionality...")
        
        # Test PublicKey creation
        test_pubkey = PublicKey.from_string("11111111111111111111111111111111")
        print(f"  ✓ PublicKey creation: {test_pubkey}")
        
        # Test Keypair creation
        test_keypair = Keypair()
        print(f"  ✓ Keypair creation: {test_keypair.pubkey()}")
        
        # Test token creator
        creator = SimpleGeoMiningTokenCreator("devnet")
        print(f"  ✓ Token creator: {creator.network}")
        
        # Test location and reward config (without requiring config file)
        from datetime import datetime
        location = Location(37.7749, -122.4194, datetime.utcnow())
        reward_config = RewardConfig()
        print(f"  ✓ Location and RewardConfig: {location.latitude}, {reward_config.base_reward}")
        
        print("✅ Basic functionality test passed!")
        print("📝 Note: Treasury manager will work once token_config.json is created")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ General error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🎉 All tests passed! You can now run:")
        print("  python create_token_only.py")
        print("  python geomining_api.py")
    else:
        print("\n❌ Some tests failed. Check error messages above.")