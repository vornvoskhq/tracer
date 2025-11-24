"""
GeoMining Treasury Manager
Handles token distribution, rewards, and treasury operations
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from solana.rpc.api import Client
from solders.keypair import Keypair
from solders.pubkey import Pubkey as PublicKey

# Token Program ID (standard SPL Token program)
TOKEN_PROGRAM_ID = PublicKey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RewardConfig:
    """Configuration for different types of mining rewards"""
    base_reward: float = 10.0  # Base tokens per check-in
    distance_bonus: float = 0.1  # Extra tokens per km from last location
    time_bonus: float = 0.5  # Extra tokens per hour since last check-in
    streak_multiplier: float = 1.2  # Multiplier for consecutive days
    max_daily_rewards: int = 10  # Maximum check-ins per day
    min_distance_m: float = 100.0  # Minimum distance between check-ins (meters)
    min_time_hours: float = 1.0  # Minimum time between check-ins (hours)

@dataclass
class Location:
    """Geographic location data"""
    latitude: float
    longitude: float
    timestamp: datetime
    accuracy: float = 10.0  # GPS accuracy in meters

@dataclass
class UserReward:
    """User reward transaction record"""
    user_id: str
    wallet_address: str
    amount: float
    reward_type: str
    location: Location
    transaction_signature: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class GeoMiningTreasury:
    """Manages the GeoMining token treasury and reward distribution"""
    
    def __init__(self, config_path: str = "token_config.json", network: str = "devnet"):
        """
        Initialize treasury manager
        
        Args:
            config_path: Path to token configuration file
            network: Solana network
        """
        self.network = network
        self.config = self._load_config(config_path)
        self.reward_config = RewardConfig()
        
        # Initialize Solana client
        rpc_urls = {
            "devnet": "https://api.devnet.solana.com",
            "testnet": "https://api.testnet.solana.com",
            "mainnet-beta": "https://api.mainnet-beta.solana.com"
        }
        self.client = Client(rpc_urls[network])
        
        # Load treasury keypair
        self.treasury_keypair = self._load_treasury_keypair()
        
        # Token configuration (simplified for demo)
        self.token_mint = PublicKey.from_string(self.config["token_address"])
        
        # Treasury account for holding tokens
        self.treasury_account = None
        
        # In-memory storage for demo (use database in production)
        self.user_history: Dict[str, List[UserReward]] = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load token configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
    
    def _load_treasury_keypair(self, keypair_path: str = "mint_authority.json") -> Keypair:
        """Load treasury keypair (same as mint authority for simplicity)"""
        try:
            with open(keypair_path, 'r') as f:
                secret_key = json.load(f)
            return Keypair.from_secret_key(bytes(secret_key))
        except FileNotFoundError:
            logger.error(f"Treasury keypair not found: {keypair_path}")
            raise
    
    def set_treasury_account(self, account_address: str):
        """Set the treasury token account address"""
        self.treasury_account = PublicKey(account_address)
    
    def calculate_distance_km(self, loc1: Location, loc2: Location) -> float:
        """
        Calculate distance between two locations using Haversine formula
        
        Args:
            loc1: First location
            loc2: Second location
            
        Returns:
            Distance in kilometers
        """
        from math import radians, sin, cos, sqrt, atan2
        
        # Earth radius in kilometers
        R = 6371.0
        
        lat1, lon1 = radians(loc1.latitude), radians(loc1.longitude)
        lat2, lon2 = radians(loc2.latitude), radians(loc2.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def calculate_reward(self, user_id: str, location: Location) -> Tuple[float, str, bool]:
        """
        Calculate reward amount for a user's location check-in
        
        Args:
            user_id: User identifier
            location: Current location
            
        Returns:
            Tuple of (reward_amount, reason, is_valid)
        """
        user_rewards = self.user_history.get(user_id, [])
        
        # Check if user has previous check-ins
        if not user_rewards:
            return (
                self.reward_config.base_reward,
                "First check-in bonus",
                True
            )
        
        # Get last check-in
        last_reward = user_rewards[-1]
        time_since_last = (location.timestamp - last_reward.timestamp).total_seconds() / 3600
        
        # Validate minimum time between check-ins
        if time_since_last < self.reward_config.min_time_hours:
            return (
                0.0,
                f"Must wait {self.reward_config.min_time_hours} hours between check-ins",
                False
            )
        
        # Calculate distance from last location
        distance_km = self.calculate_distance_km(location, last_reward.location)
        distance_m = distance_km * 1000
        
        # Validate minimum distance
        if distance_m < self.reward_config.min_distance_m:
            return (
                0.0,
                f"Must move at least {self.reward_config.min_distance_m}m from last check-in",
                False
            )
        
        # Check daily limit
        today_rewards = [
            r for r in user_rewards 
            if r.timestamp.date() == location.timestamp.date()
        ]
        
        if len(today_rewards) >= self.reward_config.max_daily_rewards:
            return (
                0.0,
                f"Daily limit of {self.reward_config.max_daily_rewards} check-ins reached",
                False
            )
        
        # Calculate base reward
        reward = self.reward_config.base_reward
        
        # Distance bonus
        distance_bonus = distance_km * self.reward_config.distance_bonus
        reward += distance_bonus
        
        # Time bonus
        time_bonus = min(time_since_last * self.reward_config.time_bonus, 24)  # Cap at 24 hours
        reward += time_bonus
        
        # Streak bonus (consecutive days)
        streak_days = self._calculate_streak(user_rewards, location.timestamp)
        if streak_days > 1:
            streak_bonus = reward * (self.reward_config.streak_multiplier - 1) * min(streak_days / 7, 2)
            reward += streak_bonus
        
        reason = f"Base: {self.reward_config.base_reward}, Distance: +{distance_bonus:.1f}, Time: +{time_bonus:.1f}"
        if streak_days > 1:
            reason += f", Streak: {streak_days} days"
        
        return (round(reward, 2), reason, True)
    
    def _calculate_streak(self, user_rewards: List[UserReward], current_time: datetime) -> int:
        """Calculate consecutive day streak"""
        if not user_rewards:
            return 1
        
        dates = set()
        for reward in user_rewards[-30:]:  # Check last 30 rewards
            dates.add(reward.timestamp.date())
        
        streak = 1
        current_date = current_time.date()
        
        for i in range(1, 30):
            check_date = current_date - timedelta(days=i)
            if check_date in dates:
                streak += 1
            else:
                break
        
        return streak
    
    def create_user_token_account(self, user_wallet: PublicKey) -> Optional[PublicKey]:
        """
        Create a token account for a user's wallet (simulated for demo)
        
        Args:
            user_wallet: User's wallet public key
            
        Returns:
            Token account address or None if failed
        """
        try:
            # For demo purposes, we'll return the user's wallet address
            # In production, this would create an actual associated token account
            logger.info(f"Simulated token account creation for user {user_wallet}")
            return user_wallet
        except Exception as e:
            logger.error(f"Failed to create token account: {e}")
            return None
    
    def send_reward(self, user_id: str, wallet_address: str, location: Location) -> Optional[UserReward]:
        """
        Process and send reward to user (simulated for demo)
        
        Args:
            user_id: User identifier
            wallet_address: User's Solana wallet address
            location: Location for check-in
            
        Returns:
            UserReward object if successful, None if failed
        """
        try:
            # Calculate reward
            amount, reason, is_valid = self.calculate_reward(user_id, location)
            
            if not is_valid:
                logger.warning(f"Invalid reward for user {user_id}: {reason}")
                return None
            
            # Simulate token transfer (for demo purposes)
            # In production, this would perform actual blockchain transactions
            tx_sig = f"demo_tx_{user_id}_{len(self.user_history.get(user_id, []))}"
            
            # Create reward record
            reward = UserReward(
                user_id=user_id,
                wallet_address=wallet_address,
                amount=amount,
                reward_type=reason,
                location=location,
                transaction_signature=tx_sig
            )
            
            # Store in history
            if user_id not in self.user_history:
                self.user_history[user_id] = []
            self.user_history[user_id].append(reward)
            
            logger.info(f"✅ Simulated reward: {amount} tokens to {wallet_address}. TX: {tx_sig}")
            return reward
            
        except Exception as e:
            logger.error(f"Failed to process reward: {e}")
            return None
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get user's mining statistics"""
        rewards = self.user_history.get(user_id, [])
        
        if not rewards:
            return {
                "total_rewards": 0,
                "total_tokens": 0.0,
                "check_ins": 0,
                "current_streak": 0,
                "last_check_in": None
            }
        
        total_tokens = sum(r.amount for r in rewards)
        current_streak = self._calculate_streak(rewards, datetime.utcnow())
        
        return {
            "total_rewards": len(rewards),
            "total_tokens": total_tokens,
            "check_ins": len(rewards),
            "current_streak": current_streak,
            "last_check_in": rewards[-1].timestamp.isoformat() if rewards else None
        }
    
    def get_treasury_balance(self) -> Optional[float]:
        """Get current treasury token balance (simulated for demo)"""
        try:
            # For demo purposes, return a simulated balance
            # In production, this would query the actual token account balance
            total_distributed = sum(
                sum(reward.amount for reward in rewards)
                for rewards in self.user_history.values()
            )
            starting_balance = 1_000_000_000.0  # 1 billion tokens
            remaining_balance = starting_balance - total_distributed
            
            logger.info(f"Simulated treasury balance: {remaining_balance:,.2f} tokens")
            return remaining_balance
        except Exception as e:
            logger.error(f"Failed to get treasury balance: {e}")
            return 1_000_000_000.0  # Return starting balance on error


# Example usage
if __name__ == "__main__":
    # Initialize treasury
    treasury = GeoMiningTreasury()
    
    # Set treasury account (you'll get this from token creation)
    # treasury.set_treasury_account("YOUR_TREASURY_ACCOUNT_ADDRESS")
    
    # Example reward calculation
    user_id = "user123"
    location = Location(
        latitude=37.7749,
        longitude=-122.4194,
        timestamp=datetime.utcnow()
    )
    
    amount, reason, valid = treasury.calculate_reward(user_id, location)
    print(f"Reward calculation: {amount} tokens - {reason} (Valid: {valid})")
    
    # Get treasury balance
    balance = treasury.get_treasury_balance()
    if balance:
        print(f"Treasury balance: {balance:,.2f} tokens")