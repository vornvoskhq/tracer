# 🚀 Quick Start Guide - GeoMining Token

## Current Situation
Your setup detected that Solana CLI needs to be added to your PATH. Here's how to fix this and get your geomining token running:

## Step 1: Add Solana CLI to PATH

Run this command in your terminal:
```bash
export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"
```

To make this permanent, add it to your shell profile:
```bash
echo 'export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## Step 2: Verify Solana CLI

```bash
solana --version
# Should show: solana-cli 1.16.0 (or similar)
```

## Step 3: Configure Solana

```bash
# Set to devnet
solana config set --url https://api.devnet.solana.com

# Create a keypair (if you don't have one)
solana-keygen new --no-bip39-passphrase

# Get some devnet SOL for transaction fees
solana airdrop 2

# Check your balance
solana balance
```

## Step 4: Install Missing Python Package

```bash
pip install spl-token==0.2.0
```

## Step 5: Create Your Token

```bash
python create_token_only.py
```

This will:
- ✅ Create your custom GeoMining token
- ✅ Set up a treasury with 1 billion tokens
- ✅ Save configuration files

## Step 6: Start the API Server

```bash
python geomining_api.py
```

The API will start on http://localhost:8000

## Step 7: Test the Frontend

1. Open `frontend_example.html` in your browser
2. Install Phantom wallet extension if you haven't
3. Set Phantom to **Devnet** network
4. Connect your wallet
5. Allow location access
6. Start mining tokens!

## Alternative: One-Command Setup

If you want to try the automated setup again after fixing PATH:

```bash
python setup_geomining.py
```

## Troubleshooting

### "solana: command not found"
- Make sure you added Solana to PATH (Step 1)
- Restart your terminal
- Verify with `solana --version`

### "Insufficient funds"
- Request more devnet SOL: `solana airdrop 2`
- Check balance: `solana balance`

### "Token creation failed"
- Ensure you're on devnet: `solana config get`
- Check you have SOL for fees: `solana balance`
- Try airdrop again: `solana airdrop 2`

### Frontend wallet issues
- Set Phantom wallet to **Devnet** network
- Make sure you have the same wallet address as your Solana CLI

## What You'll Have

After successful setup:

1. **Custom SPL Token** on Solana devnet
2. **Treasury** with 1 billion tokens for rewards
3. **API Backend** for wallet integration and mining
4. **Web Frontend** for user interaction
5. **Complete reward system** with location-based bonuses

## Next Steps

Once everything is working:

1. **Customize rewards**: Edit `treasury_manager.py` to adjust reward amounts
2. **Deploy to mainnet**: Change network settings for production
3. **Build mobile app**: Use the API endpoints for React Native/Flutter
4. **Add features**: NFT rewards, social features, marketplace integration

## Need Help?

Check the full documentation in `README.md` or run the test suite:
```bash
python test_system.py
```

---

🎉 **You're building the future of location-based tokenomics!** 🌍