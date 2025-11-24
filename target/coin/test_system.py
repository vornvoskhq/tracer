"""
GeoMining System Test Suite
Tests the complete token system functionality
"""

import json
import time
import asyncio
from datetime import datetime, timedelta
from treasury_manager import GeoMiningTreasury, Location, RewardConfig
from token_creator import GeoMiningTokenCreator

def test_reward_calculation():
    """Test the reward calculation logic"""
    print("🧪 Testing reward calculation...")
    
    treasury = GeoMiningTreasury()
    
    # Test first check-in
    user_id = "test_user_001"
    location1 = Location(
        latitude=37.7749,
        longitude=-122.4194,
        timestamp=datetime.utcnow()
    )
    
    amount, reason, valid = treasury.calculate_reward(user_id, location1)
    print(f"✅ First check-in: {amount} tokens - {reason} (Valid: {valid})")
    assert valid == True
    assert amount == treasury.reward_config.base_reward
    
    # Simulate storing the first reward
    from treasury_manager import UserReward
    first_reward = UserReward(
        user_id=user_id,
        wallet_address="test_wallet",
        amount=amount,
        reward_type=reason,
        location=location1
    )
    treasury.user_history[user_id] = [first_reward]
    
    # Test second check-in too soon
    location2 = Location(
        latitude=37.7849,  # 1km north
        longitude=-122.4194,
        timestamp=datetime.utcnow() + timedelta(minutes=30)  # 30 minutes later
    )
    
    amount, reason, valid = treasury.calculate_reward(user_id, location2)
    print(f"✅ Too soon check-in: {amount} tokens - {reason} (Valid: {valid})")
    assert valid == False
    
    # Test valid second check-in
    location3 = Location(
        latitude=37.7849,  # 1km north
        longitude=-122.4194,
        timestamp=datetime.utcnow() + timedelta(hours=2)  # 2 hours later
    )
    
    amount, reason, valid = treasury.calculate_reward(user_id, location3)
    print(f"✅ Valid second check-in: {amount} tokens - {reason} (Valid: {valid})")
    assert valid == True
    assert amount > treasury.reward_config.base_reward  # Should have bonuses
    
    print("✅ Reward calculation tests passed!")

def test_distance_calculation():
    """Test distance calculation between locations"""
    print("🧪 Testing distance calculation...")
    
    treasury = GeoMiningTreasury()
    
    # Test known distance (approximately 1km)
    loc1 = Location(37.7749, -122.4194, datetime.utcnow())  # San Francisco
    loc2 = Location(37.7849, -122.4194, datetime.utcnow())  # 1km north
    
    distance = treasury.calculate_distance_km(loc1, loc2)
    print(f"✅ Distance: {distance:.2f} km")
    assert 0.9 <= distance <= 1.1  # Allow some margin for precision
    
    # Test zero distance
    distance_zero = treasury.calculate_distance_km(loc1, loc1)
    print(f"✅ Same location distance: {distance_zero:.2f} km")
    assert distance_zero < 0.001
    
    print("✅ Distance calculation tests passed!")

def test_streak_calculation():
    """Test consecutive day streak calculation"""
    print("🧪 Testing streak calculation...")
    
    treasury = GeoMiningTreasury()
    user_id = "streak_test_user"
    
    # Create rewards for consecutive days
    rewards = []
    base_time = datetime.utcnow().replace(hour=12, minute=0, second=0, microsecond=0)
    
    for i in range(5):  # 5 consecutive days
        reward_time = base_time - timedelta(days=i)
        location = Location(37.7749 + i*0.001, -122.4194, reward_time)
        
        reward = UserReward(
            user_id=user_id,
            wallet_address="test_wallet",
            amount=10.0,
            reward_type="test",
            location=location,
            timestamp=reward_time
        )
        rewards.append(reward)
    
    treasury.user_history[user_id] = rewards
    
    # Test streak calculation
    current_time = base_time
    streak = treasury._calculate_streak(rewards, current_time)
    print(f"✅ Calculated streak: {streak} days")
    assert streak == 5
    
    print("✅ Streak calculation tests passed!")

def test_token_config_loading():
    """Test loading token configuration"""
    print("🧪 Testing configuration loading...")
    
    # Create a test config
    test_config = {
        "network": "devnet",
        "token_address": "11111111111111111111111111111111",
        "mint_authority": "11111111111111111111111111111111",
        "decimals": 9,
        "program_id": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
    }
    
    with open("test_config.json", "w") as f:
        json.dump(test_config, f)
    
    try:
        # Test loading config
        treasury = GeoMiningTreasury("test_config.json")
        assert treasury.config["network"] == "devnet"
        assert treasury.config["decimals"] == 9
        print("✅ Configuration loading test passed!")
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        
    finally:
        # Cleanup
        import os
        if os.path.exists("test_config.json"):
            os.remove("test_config.json")

def test_api_requests():
    """Test API endpoint functionality (requires running server)"""
    print("🧪 Testing API endpoints...")
    
    try:
        import httpx
        
        base_url = "http://localhost:8000"
        
        # Test health check
        with httpx.Client() as client:
            response = client.get(f"{base_url}/")
            if response.status_code == 200:
                print("✅ API health check passed!")
                
                # Test wallet connection
                wallet_data = {
                    "wallet_address": "11111111111111111111111111111111"
                }
                
                response = client.post(f"{base_url}/wallet/connect", json=wallet_data)
                if response.status_code == 200:
                    print("✅ Wallet connection endpoint test passed!")
                else:
                    print(f"⚠️  Wallet connection test failed: {response.status_code}")
                
                # Test reward preview
                response = client.get(f"{base_url}/mining/preview", params={
                    "user_id": "test_user",
                    "latitude": 37.7749,
                    "longitude": -122.4194
                })
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Reward preview: {data['potential_reward']} tokens")
                else:
                    print(f"⚠️  Reward preview test failed: {response.status_code}")
            else:
                print("⚠️  API server not running. Start with: python geomining_api.py")
                
    except ImportError:
        print("⚠️  httpx not installed. Install with: pip install httpx")
    except Exception as e:
        print(f"⚠️  API test failed: {e}")

def run_all_tests():
    """Run all test functions"""
    print("🚀 Starting GeoMining System Tests")
    print("=" * 50)
    
    try:
        test_reward_calculation()
        print()
        
        test_distance_calculation()
        print()
        
        test_streak_calculation()
        print()
        
        test_token_config_loading()
        print()
        
        test_api_requests()
        print()
        
        print("=" * 50)
        print("🎉 All tests completed!")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"❌ Test assertion failed: {e}")
    except Exception as e:
        print(f"❌ Test error: {e}")

def demo_complete_flow():
    """Demonstrate a complete user flow"""
    print("🎮 Demonstrating complete user flow...")
    print("-" * 40)
    
    treasury = GeoMiningTreasury()
    user_id = "demo_user_123"
    wallet_address = "DemoWallet1111111111111111111111111111"
    
    # Simulate multiple check-ins
    locations = [
        (37.7749, -122.4194, "San Francisco Downtown"),
        (37.7849, -122.4294, "San Francisco North"),
        (37.7649, -122.4094, "San Francisco South"),
        (37.7749, -122.4394, "San Francisco West"),
    ]
    
    print(f"User: {user_id}")
    print(f"Wallet: {wallet_address}")
    print()
    
    total_earned = 0
    
    for i, (lat, lon, name) in enumerate(locations):
        print(f"Check-in #{i+1}: {name}")
        
        # Add time delay between check-ins
        base_time = datetime.utcnow() + timedelta(hours=i*2)
        location = Location(lat, lon, base_time)
        
        # Calculate reward
        amount, reason, valid = treasury.calculate_reward(user_id, location)
        
        if valid:
            # Simulate successful reward
            reward = UserReward(
                user_id=user_id,
                wallet_address=wallet_address,
                amount=amount,
                reward_type=reason,
                location=location,
                timestamp=base_time
            )
            
            if user_id not in treasury.user_history:
                treasury.user_history[user_id] = []
            treasury.user_history[user_id].append(reward)
            
            total_earned += amount
            print(f"  ✅ Earned: {amount} tokens ({reason})")
        else:
            print(f"  ❌ No reward: {reason}")
        
        print(f"  📍 Location: {lat:.4f}, {lon:.4f}")
        print()
    
    # Show final stats
    stats = treasury.get_user_stats(user_id)
    print("Final Statistics:")
    print(f"  Total Tokens: {stats['total_tokens']}")
    print(f"  Check-ins: {stats['check_ins']}")
    print(f"  Current Streak: {stats['current_streak']} days")
    print()
    print("🎉 Demo completed successfully!")

if __name__ == "__main__":
    print("GeoMining System Test Suite")
    print("Choose an option:")
    print("1. Run all tests")
    print("2. Demo complete flow")
    print("3. Run specific test")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        run_all_tests()
    elif choice == "2":
        demo_complete_flow()
    elif choice == "3":
        print("\nAvailable tests:")
        print("a. Reward calculation")
        print("b. Distance calculation")
        print("c. Streak calculation")
        print("d. Config loading")
        print("e. API endpoints")
        
        test_choice = input("Enter test letter: ").strip().lower()
        
        if test_choice == "a":
            test_reward_calculation()
        elif test_choice == "b":
            test_distance_calculation()
        elif test_choice == "c":
            test_streak_calculation()
        elif test_choice == "d":
            test_token_config_loading()
        elif test_choice == "e":
            test_api_requests()
        else:
            print("Invalid choice")
    else:
        print("Invalid choice")