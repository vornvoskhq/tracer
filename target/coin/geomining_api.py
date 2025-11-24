"""
GeoMining API Backend
FastAPI backend for geolocation-based token mining with Solana wallet integration
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
from typing import Optional, List, Dict
from datetime import datetime
import json
import logging
from treasury_manager import GeoMiningTreasury, Location
from solders.pubkey import Pubkey as PublicKey

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Initialize FastAPI app
app = FastAPI(
    title="GeoMining API",
    description="Solana-based geolocation mining rewards system",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize treasury manager
treasury = GeoMiningTreasury()

# Pydantic models for API requests/responses
class WalletConnectionRequest(BaseModel):
    """Request to connect a wallet"""
    wallet_address: str
    signature: Optional[str] = None  # For wallet verification
    message: Optional[str] = None    # Message that was signed
    
    @validator('wallet_address')
    def validate_wallet_address(cls, v):
        try:
            PublicKey.from_string(v)
            return v
        except Exception:
            raise ValueError('Invalid Solana wallet address')

class LocationCheckIn(BaseModel):
    """Location check-in request"""
    user_id: str
    wallet_address: str
    latitude: float
    longitude: float
    accuracy: Optional[float] = 10.0
    timestamp: Optional[datetime] = None
    
    @validator('latitude')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v
    
    @validator('longitude')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v
    
    @validator('accuracy')
    def validate_accuracy(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Accuracy must be positive')
        return v

class RewardResponse(BaseModel):
    """Response for successful reward"""
    success: bool
    amount: float
    reason: str
    transaction_signature: Optional[str]
    new_balance: Optional[float]
    message: str

class UserStatsResponse(BaseModel):
    """User statistics response"""
    user_id: str
    total_rewards: int
    total_tokens: float
    check_ins: int
    current_streak: int
    last_check_in: Optional[str]
    treasury_balance: Optional[float]

class TokenInfoResponse(BaseModel):
    """Token information response"""
    token_address: str
    network: str
    decimals: int
    total_supply: Optional[float]
    treasury_balance: Optional[float]

# In-memory storage for connected wallets (use database in production)
connected_wallets: Dict[str, Dict] = {}

# API Endpoints

@app.get("/", tags=["General"])
async def root():
    """Health check endpoint"""
    return {
        "message": "GeoMining API is running",
        "network": treasury.network,
        "token_address": treasury.config.get("token_address"),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/wallet/connect", tags=["Wallet"], response_model=Dict)
async def connect_wallet(request: WalletConnectionRequest):
    """
    Connect a Solana wallet to the platform
    
    In production, you should verify the signature to ensure wallet ownership
    """
    try:
        wallet_address = request.wallet_address
        
        # Store wallet connection (in production, verify signature first)
        connected_wallets[wallet_address] = {
            "connected_at": datetime.utcnow().isoformat(),
            "signature": request.signature,
            "message": request.message
        }
        
        logger.info(f"Wallet connected: {wallet_address}")
        
        return {
            "success": True,
            "message": "Wallet connected successfully",
            "wallet_address": wallet_address,
            "connected_at": connected_wallets[wallet_address]["connected_at"]
        }
        
    except Exception as e:
        logger.error(f"Wallet connection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to connect wallet: {str(e)}"
        )

@app.get("/wallet/{wallet_address}/status", tags=["Wallet"])
async def get_wallet_status(wallet_address: str):
    """Check if wallet is connected and get basic info"""
    try:
        PublicKey.from_string(wallet_address)  # Validate address format
        
        is_connected = wallet_address in connected_wallets
        
        return {
            "wallet_address": wallet_address,
            "is_connected": is_connected,
            "connection_info": connected_wallets.get(wallet_address)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid wallet address"
        )

@app.post("/mining/checkin", tags=["Mining"], response_model=RewardResponse)
async def location_checkin(checkin: LocationCheckIn):
    """
    Process a location check-in and distribute rewards
    """
    try:
        # Verify wallet is connected
        if checkin.wallet_address not in connected_wallets:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Wallet not connected. Please connect your wallet first."
            )
        
        # Create location object
        location = Location(
            latitude=checkin.latitude,
            longitude=checkin.longitude,
            timestamp=checkin.timestamp or datetime.utcnow(),
            accuracy=checkin.accuracy or 10.0
        )
        
        # Process reward
        reward = treasury.send_reward(
            user_id=checkin.user_id,
            wallet_address=checkin.wallet_address,
            location=location
        )
        
        if reward:
            # Get updated treasury balance
            treasury_balance = treasury.get_treasury_balance()
            
            return RewardResponse(
                success=True,
                amount=reward.amount,
                reason=reward.reward_type,
                transaction_signature=reward.transaction_signature,
                new_balance=treasury_balance,
                message=f"Successfully earned {reward.amount} GeoTokens!"
            )
        else:
            # Check why reward failed
            amount, reason, valid = treasury.calculate_reward(checkin.user_id, location)
            
            return RewardResponse(
                success=False,
                amount=0.0,
                reason=reason,
                transaction_signature=None,
                new_balance=None,
                message=reason
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Check-in failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Check-in processing failed: {str(e)}"
        )

@app.get("/mining/preview", tags=["Mining"])
async def preview_reward(user_id: str, latitude: float, longitude: float):
    """
    Preview reward amount without actually processing the check-in
    """
    try:
        location = Location(
            latitude=latitude,
            longitude=longitude,
            timestamp=datetime.utcnow()
        )
        
        amount, reason, valid = treasury.calculate_reward(user_id, location)
        
        return {
            "potential_reward": amount,
            "reason": reason,
            "is_valid": valid,
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Reward preview failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to calculate reward preview: {str(e)}"
        )

@app.get("/user/{user_id}/stats", tags=["User"], response_model=UserStatsResponse)
async def get_user_stats(user_id: str):
    """Get user mining statistics"""
    try:
        stats = treasury.get_user_stats(user_id)
        treasury_balance = treasury.get_treasury_balance()
        
        return UserStatsResponse(
            user_id=user_id,
            total_rewards=stats["total_rewards"],
            total_tokens=stats["total_tokens"],
            check_ins=stats["check_ins"],
            current_streak=stats["current_streak"],
            last_check_in=stats["last_check_in"],
            treasury_balance=treasury_balance
        )
        
    except Exception as e:
        logger.error(f"Failed to get user stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user statistics: {str(e)}"
        )

@app.get("/user/{user_id}/history", tags=["User"])
async def get_user_history(user_id: str, limit: int = 50):
    """Get user's reward history"""
    try:
        user_rewards = treasury.user_history.get(user_id, [])
        
        # Convert to serializable format and limit results
        history = []
        for reward in user_rewards[-limit:]:
            history.append({
                "amount": reward.amount,
                "reward_type": reward.reward_type,
                "location": {
                    "latitude": reward.location.latitude,
                    "longitude": reward.location.longitude,
                    "accuracy": reward.location.accuracy
                },
                "timestamp": reward.timestamp.isoformat(),
                "transaction_signature": reward.transaction_signature
            })
        
        return {
            "user_id": user_id,
            "total_records": len(user_rewards),
            "returned_records": len(history),
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Failed to get user history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user history: {str(e)}"
        )

@app.get("/token/info", tags=["Token"], response_model=TokenInfoResponse)
async def get_token_info():
    """Get token information"""
    try:
        treasury_balance = treasury.get_treasury_balance()
        
        return TokenInfoResponse(
            token_address=treasury.config["token_address"],
            network=treasury.network,
            decimals=treasury.config.get("decimals", 9),
            total_supply=None,  # Would need to query this from blockchain
            treasury_balance=treasury_balance
        )
        
    except Exception as e:
        logger.error(f"Failed to get token info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve token information: {str(e)}"
        )

@app.get("/admin/treasury", tags=["Admin"])
async def get_treasury_info():
    """Get treasury information (admin endpoint)"""
    try:
        balance = treasury.get_treasury_balance()
        
        return {
            "treasury_account": str(treasury.treasury_account) if treasury.treasury_account else None,
            "balance": balance,
            "network": treasury.network,
            "total_users": len(treasury.user_history),
            "total_rewards_distributed": sum(
                len(rewards) for rewards in treasury.user_history.values()
            )
        }
        
    except Exception as e:
        logger.error(f"Failed to get treasury info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve treasury information: {str(e)}"
        )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting GeoMining API...")
    
    # Set treasury account if available
    try:
        # This would typically be loaded from configuration
        # treasury.set_treasury_account("YOUR_TREASURY_ACCOUNT_ADDRESS")
        logger.info("Treasury initialized successfully")
    except Exception as e:
        logger.warning(f"Treasury initialization failed: {e}")
    
    logger.info("GeoMining API started successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "geomining_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )