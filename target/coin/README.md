# 🌍 GeoMining Token System

A Solana-based geolocation mining system that rewards users with custom SPL tokens for visiting different locations. Perfect for location-based apps, tourism, exploration games, and community engagement.

## 🚀 Features

- **Custom SPL Token**: Create your own geomining token on Solana
- **Location-Based Rewards**: Users earn tokens by checking in at different locations
- **Smart Reward Algorithm**: Distance bonuses, time bonuses, and streak multipliers
- **Wallet Integration**: Support for Phantom, Solflare, and other Solana wallets
- **Anti-Gaming Measures**: Minimum distance and time requirements
- **RESTful API**: Complete backend for mobile/web integration
- **Treasury Management**: Automated token distribution system

## 📋 Quick Start

### Prerequisites

- Python 3.8+
- Node.js (for frontend development)
- Solana CLI tools

### Automated Setup

Run the setup script to automatically configure everything:

```bash
python setup_geomining.py
```

This will:
1. Install required dependencies
2. Configure Solana CLI
3. Create your custom token
4. Set up treasury management
5. Create a demo frontend

### Manual Setup

1. **Clone and Install Dependencies**
```bash
git clone <your-repo>
cd geomining-token
pip install -r requirements.txt
```

2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your settings
```

3. **Install Solana CLI**
```bash
# Unix/Linux/macOS
sh -c "$(curl -sSfL https://release.solana.com/stable/install)"

# Windows
# Download from https://github.com/solana-labs/solana/releases
```

4. **Create Your Token**
```bash
python token_creator.py
```

5. **Start the API Server**
```bash
python geomining_api.py
```

6. **Open the Demo Frontend**
```bash
# Open frontend_example.html in your browser
# Or serve it with:
python -m http.server 3000
```

## 🏗️ Architecture

### Core Components

1. **Token Creator** (`token_creator.py`)
   - Creates SPL tokens on Solana
   - Manages mint authority
   - Handles initial token distribution

2. **Treasury Manager** (`treasury_manager.py`)
   - Calculates location-based rewards
   - Manages token distribution
   - Tracks user statistics and history

3. **API Backend** (`geomining_api.py`)
   - RESTful API for wallet integration
   - Location check-in endpoints
   - User statistics and history

4. **Frontend Integration** (`frontend_example.html`)
   - Wallet connection interface
   - Location-based check-ins
   - Real-time statistics display

## 💰 Reward System

### Base Rewards
- **Check-in Reward**: 10 tokens per location
- **Distance Bonus**: +0.1 tokens per kilometer from last location
- **Time Bonus**: +0.5 tokens per hour since last check-in
- **Streak Bonus**: 20% multiplier for consecutive days

### Anti-Gaming Measures
- Minimum 100 meters between check-ins
- Minimum 1 hour between check-ins
- Maximum 10 check-ins per day
- GPS accuracy validation

### Example Reward Calculation
```
Base: 10 tokens
+ Distance: 5km × 0.1 = +0.5 tokens
+ Time: 3 hours × 0.5 = +1.5 tokens
+ Streak: 5 days = 20% bonus = +2.4 tokens
Total: 14.4 tokens
```

## 🔌 API Endpoints

### Wallet Management
```http
POST /wallet/connect
GET /wallet/{address}/status
```

### Mining Operations
```http
POST /mining/checkin     # Process location check-in
GET /mining/preview      # Preview reward amount
```

### User Data
```http
GET /user/{id}/stats     # User statistics
GET /user/{id}/history   # Reward history
```

### Token Information
```http
GET /token/info          # Token details
GET /admin/treasury      # Treasury status
```

### Example Check-in Request
```json
{
  "user_id": "user123",
  "wallet_address": "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU",
  "latitude": 37.7749,
  "longitude": -122.4194,
  "accuracy": 5.0
}
```

## 🔧 Configuration

### Token Settings
```json
{
  "name": "GeoToken",
  "symbol": "GEO",
  "decimals": 9,
  "initial_supply": 1000000000,
  "distribution": {
    "mining_rewards": "60%",
    "team": "20%",
    "marketing": "10%",
    "reserve": "10%"
  }
}
```

### Reward Configuration
```python
@dataclass
class RewardConfig:
    base_reward: float = 10.0
    distance_bonus: float = 0.1
    time_bonus: float = 0.5
    streak_multiplier: float = 1.2
    max_daily_rewards: int = 10
    min_distance_m: float = 100.0
    min_time_hours: float = 1.0
```

## 🔐 Security Features

### Wallet Verification
- Signature verification for wallet ownership
- Non-custodial wallet integration
- Secure transaction signing

### Location Validation
- GPS accuracy requirements
- Distance and time validation
- Rate limiting and abuse prevention

### Treasury Security
- Multi-signature support (configurable)
- Secure keypair management
- Transaction logging and monitoring

## 🌐 Frontend Integration

### Wallet Connection (JavaScript)
```javascript
// Connect Phantom wallet
const response = await window.solana.connect();
const walletAddress = response.publicKey.toString();

// Register with API
await fetch('/wallet/connect', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ wallet_address: walletAddress })
});
```

### Location Check-in
```javascript
// Get user location
navigator.geolocation.getCurrentPosition(async (position) => {
  const checkin = {
    user_id: userId,
    wallet_address: walletAddress,
    latitude: position.coords.latitude,
    longitude: position.coords.longitude,
    accuracy: position.coords.accuracy
  };
  
  const response = await fetch('/mining/checkin', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(checkin)
  });
  
  const result = await response.json();
  console.log(`Earned ${result.amount} tokens!`);
});
```

## 📱 Mobile Integration

### React Native Example
```javascript
import Geolocation from '@react-native-geolocation-service';

const checkIn = async () => {
  Geolocation.getCurrentPosition(
    async (position) => {
      const response = await fetch(`${API_BASE}/mining/checkin`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          wallet_address: walletAddress,
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          accuracy: position.coords.accuracy
        })
      });
      
      const result = await response.json();
      // Handle result
    },
    (error) => console.log(error),
    { enableHighAccuracy: true, timeout: 15000, maximumAge: 10000 }
  );
};
```

## 🚀 Deployment

### Development
```bash
# Start API server
uvicorn geomining_api:app --reload --host 0.0.0.0 --port 8000

# API documentation
# http://localhost:8000/docs
```

### Production
```bash
# Using Docker
docker build -t geomining-api .
docker run -p 8000:8000 geomining-api

# Using systemd
sudo systemctl enable geomining
sudo systemctl start geomining
```

### Environment Variables
```bash
export SOLANA_NETWORK=mainnet-beta
export TOKEN_ADDRESS=your_token_address
export TREASURY_ACCOUNT=your_treasury_account
```

## 📊 Monitoring and Analytics

### Treasury Monitoring
```python
# Check treasury balance
balance = treasury.get_treasury_balance()
print(f"Treasury: {balance:,.2f} tokens")

# User statistics
stats = treasury.get_user_stats(user_id)
print(f"User earned {stats['total_tokens']} tokens")
```

### API Monitoring
- Request logging and metrics
- Error tracking and alerting
- Performance monitoring
- User activity analytics

## 🔄 Upgrade and Migration

### Token Upgrades
- Implement token migration contracts
- Maintain backward compatibility
- Communicate changes to users

### Database Migration
```python
# Example migration script
def migrate_user_data():
    # Migrate from in-memory to database
    # Update reward calculation logic
    # Preserve user history
    pass
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup
```bash
git clone <your-fork>
cd geomining-token
pip install -r requirements-dev.txt
pre-commit install
```

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

### Common Issues

**Q: Token creation fails**
A: Ensure you have enough SOL for transaction fees and proper Solana CLI setup

**Q: Wallet connection not working**
A: Check that Phantom wallet is installed and on the correct network

**Q: Location not accurate**
A: Enable high-accuracy GPS and ensure location permissions are granted

**Q: Rewards not appearing**
A: Check minimum distance/time requirements and daily limits

### Getting Help

- 📧 Email: support@yourdomain.com
- 💬 Discord: [Your Discord Server]
- 📖 Documentation: [Your Docs URL]
- 🐛 Issues: [GitHub Issues]

## 🗺️ Roadmap

### Phase 1: Core Features ✅
- [x] SPL token creation
- [x] Basic reward system
- [x] Wallet integration
- [x] API backend

### Phase 2: Enhanced Features 🚧
- [ ] Mobile app
- [ ] Advanced analytics
- [ ] Social features
- [ ] NFT integration

### Phase 3: Ecosystem 🔮
- [ ] Marketplace integration
- [ ] Third-party API
- [ ] Enterprise features
- [ ] Cross-chain support

---

Built with ❤️ for the Solana ecosystem. Happy mining! 🌍⛏️