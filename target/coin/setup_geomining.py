"""
GeoMining Setup Script
Automated setup for the complete GeoMining token system
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path

def run_command(command, description, check=True):
    """Run a shell command with error handling"""
    print(f"\n📋 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def check_prerequisites():
    """Check if required tools are installed"""
    print("🔍 Checking prerequisites...")
    
    # Check Python
    try:
        import sys
        print(f"✅ Python {sys.version.split()[0]} found")
    except:
        print("❌ Python not found")
        sys.exit(1)
    
    # Check if Solana CLI is installed
    result = run_command("solana --version", "Checking Solana CLI", check=False)
    if result.returncode != 0:
        print("❌ Solana CLI not found. Installing...")
        install_solana_cli()
    else:
        print("✅ Solana CLI found")
    
    print("✅ Prerequisites check complete")

def install_solana_cli():
    """Install Solana CLI"""
    print("📦 Installing Solana CLI...")
    
    if os.name == 'nt':  # Windows
        command = 'cmd /c "curl https://release.solana.com/v1.16.0/solana-install-init-x86_64-pc-windows-msvc.exe --output solana-install-init.exe && solana-install-init.exe v1.16.0"'
    else:  # Unix-like
        command = 'sh -c "$(curl -sSfL https://release.solana.com/v1.16.0/install)"'
    
    run_command(command, "Installing Solana CLI")
    
    # Add to PATH (user needs to restart terminal)
    print("⚠️  Please restart your terminal after setup completes to use Solana CLI")

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # Install from requirements.txt
    run_command("pip install -r requirements.txt", "Installing Python packages")
    
    print("✅ Dependencies installed")

def setup_solana_config():
    """Configure Solana for development"""
    print("⚙️  Configuring Solana...")
    
    # Check if solana is in PATH, if not add it
    solana_path = os.path.expanduser("~/.local/share/solana/install/active_release/bin")
    if os.path.exists(solana_path):
        os.environ["PATH"] = f"{solana_path}:{os.environ.get('PATH', '')}"
        print(f"✅ Added Solana CLI to PATH: {solana_path}")
    
    # Try to set devnet URL
    result = run_command("solana config set --url https://api.devnet.solana.com", "Setting devnet URL", check=False)
    
    if result.returncode != 0:
        print("⚠️  Solana CLI not found in PATH. Manual setup required:")
        print("1. Add Solana to your PATH:")
        print(f"   export PATH=\"{solana_path}:$PATH\"")
        print("2. Restart your terminal")
        print("3. Run: solana config set --url https://api.devnet.solana.com")
        print("4. Run: solana-keygen new --no-bip39-passphrase")
        print("5. Run: solana airdrop 2")
        return False
    
    # Create a new keypair if none exists
    keypair_path = os.path.expanduser("~/.config/solana/id.json")
    if not os.path.exists(keypair_path):
        run_command("solana-keygen new --no-bip39-passphrase", "Creating new keypair")
    
    # Get some devnet SOL
    run_command("solana airdrop 2", "Requesting devnet SOL", check=False)
    
    print("✅ Solana configuration complete")
    return True

def create_token():
    """Create the GeoMining token"""
    print("🪙 Creating GeoMining token...")
    
    try:
        from token_creator import GeoMiningTokenCreator
        
        # Initialize creator
        creator = GeoMiningTokenCreator("devnet")
        
        # Create mint authority
        print("Creating mint authority...")
        mint_authority = creator.create_or_load_mint_authority()
        
        # Request SOL for transaction fees
        print("Requesting SOL for transaction fees...")
        creator.airdrop_sol(2.0)
        
        # Wait a bit for airdrop to process
        time.sleep(5)
        
        # Create the token
        print("Creating token...")
        token_address = creator.create_token(decimals=9)
        
        if token_address:
            print(f"✅ Token created: {token_address}")
            
            # Create treasury account
            print("Creating treasury account...")
            treasury_account = creator.create_token_account()
            
            if treasury_account:
                # Mint initial supply
                print("Minting initial supply...")
                initial_supply = 1_000_000_000 * (10 ** 9)  # 1B tokens
                success = creator.mint_initial_supply(initial_supply, treasury_account)
                
                if success:
                    print(f"✅ Minted 1,000,000,000 tokens to treasury")
                    
                    # Save configuration
                    creator.save_token_config()
                    
                    # Update config with treasury account
                    with open("token_config.json", "r") as f:
                        config = json.load(f)
                    config["treasury_account"] = str(treasury_account)
                    with open("token_config.json", "w") as f:
                        json.dump(config, f, indent=2)
                    
                    print("✅ Token setup complete!")
                    
                    return {
                        "token_address": str(token_address),
                        "treasury_account": str(treasury_account),
                        "mint_authority": str(mint_authority.public_key)
                    }
                else:
                    print("❌ Failed to mint initial supply")
            else:
                print("❌ Failed to create treasury account")
        else:
            print("❌ Failed to create token")
            
    except Exception as e:
        print(f"❌ Token creation failed: {e}")
        return None

def setup_treasury():
    """Setup treasury manager with the created token"""
    print("💰 Setting up treasury manager...")
    
    try:
        # Load token configuration
        with open("token_config.json", "r") as f:
            config = json.load(f)
        
        # Update the API with treasury account
        if "treasury_account" in config:
            print(f"Treasury account: {config['treasury_account']}")
            print("✅ Treasury configuration loaded")
            return True
        else:
            print("❌ Treasury account not found in configuration")
            return False
            
    except Exception as e:
        print(f"❌ Treasury setup failed: {e}")
        return False

def create_example_frontend():
    """Create a simple HTML frontend example"""
    print("🌐 Creating frontend example...")
    
    frontend_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GeoMining - Location-Based Token Mining</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .warning { background: #fff3cd; color: #856404; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        button:disabled { background: #6c757d; cursor: not-allowed; }
        .wallet-info { background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .stat-card { background: white; padding: 15px; border-radius: 5px; text-align: center; }
    </style>
</head>
<body>
    <h1>🌍 GeoMining Token System</h1>
    <p>Earn tokens by checking in at different locations!</p>
    
    <div class="container">
        <h2>Wallet Connection</h2>
        <div id="wallet-status" class="status warning">Please connect your Solana wallet</div>
        <button id="connect-wallet">Connect Phantom Wallet</button>
        <div id="wallet-info" class="wallet-info" style="display: none;">
            <strong>Connected Wallet:</strong> <span id="wallet-address"></span>
        </div>
    </div>
    
    <div class="container">
        <h2>Location Check-in</h2>
        <div id="location-status" class="status warning">Location access not granted</div>
        <button id="get-location">Get My Location</button>
        <button id="checkin-btn" disabled>Check In & Mine Tokens</button>
        <div id="location-info" style="display: none;">
            <p><strong>Latitude:</strong> <span id="latitude"></span></p>
            <p><strong>Longitude:</strong> <span id="longitude"></span></p>
            <p><strong>Accuracy:</strong> <span id="accuracy"></span> meters</p>
        </div>
    </div>
    
    <div class="container">
        <h2>Your Mining Stats</h2>
        <div class="stats">
            <div class="stat-card">
                <h3>Total Tokens</h3>
                <div id="total-tokens">0</div>
            </div>
            <div class="stat-card">
                <h3>Check-ins</h3>
                <div id="total-checkins">0</div>
            </div>
            <div class="stat-card">
                <h3>Current Streak</h3>
                <div id="current-streak">0 days</div>
            </div>
            <div class="stat-card">
                <h3>Last Reward</h3>
                <div id="last-reward">Never</div>
            </div>
        </div>
        <button id="refresh-stats">Refresh Stats</button>
    </div>
    
    <div id="messages"></div>

    <script>
        const API_BASE = 'http://localhost:8000';
        let currentWallet = null;
        let currentLocation = null;
        let userId = 'user_' + Math.random().toString(36).substr(2, 9);

        // Wallet connection
        document.getElementById('connect-wallet').addEventListener('click', async () => {
            try {
                if (window.solana && window.solana.isPhantom) {
                    const response = await window.solana.connect();
                    currentWallet = response.publicKey.toString();
                    
                    // Call API to register wallet
                    const apiResponse = await fetch(`${API_BASE}/wallet/connect`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ wallet_address: currentWallet })
                    });
                    
                    if (apiResponse.ok) {
                        document.getElementById('wallet-status').className = 'status success';
                        document.getElementById('wallet-status').textContent = 'Wallet connected successfully!';
                        document.getElementById('wallet-address').textContent = currentWallet;
                        document.getElementById('wallet-info').style.display = 'block';
                        document.getElementById('connect-wallet').textContent = 'Connected ✓';
                        document.getElementById('connect-wallet').disabled = true;
                        
                        refreshStats();
                    }
                } else {
                    showMessage('Please install Phantom wallet extension', 'error');
                }
            } catch (error) {
                console.error('Wallet connection failed:', error);
                showMessage('Failed to connect wallet', 'error');
            }
        });

        // Location access
        document.getElementById('get-location').addEventListener('click', () => {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(
                    (position) => {
                        currentLocation = {
                            latitude: position.coords.latitude,
                            longitude: position.coords.longitude,
                            accuracy: position.coords.accuracy
                        };
                        
                        document.getElementById('location-status').className = 'status success';
                        document.getElementById('location-status').textContent = 'Location acquired!';
                        document.getElementById('latitude').textContent = currentLocation.latitude.toFixed(6);
                        document.getElementById('longitude').textContent = currentLocation.longitude.toFixed(6);
                        document.getElementById('accuracy').textContent = currentLocation.accuracy.toFixed(0);
                        document.getElementById('location-info').style.display = 'block';
                        
                        if (currentWallet) {
                            document.getElementById('checkin-btn').disabled = false;
                        }
                    },
                    (error) => {
                        console.error('Location error:', error);
                        showMessage('Failed to get location. Please allow location access.', 'error');
                    }
                );
            } else {
                showMessage('Geolocation is not supported by this browser.', 'error');
            }
        });

        // Check-in and mining
        document.getElementById('checkin-btn').addEventListener('click', async () => {
            if (!currentWallet || !currentLocation) {
                showMessage('Please connect wallet and get location first', 'error');
                return;
            }

            try {
                document.getElementById('checkin-btn').disabled = true;
                document.getElementById('checkin-btn').textContent = 'Processing...';

                const response = await fetch(`${API_BASE}/mining/checkin`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: userId,
                        wallet_address: currentWallet,
                        latitude: currentLocation.latitude,
                        longitude: currentLocation.longitude,
                        accuracy: currentLocation.accuracy
                    })
                });

                const result = await response.json();

                if (result.success) {
                    showMessage(`🎉 Earned ${result.amount} tokens! ${result.reason}`, 'success');
                    refreshStats();
                } else {
                    showMessage(`❌ ${result.message}`, 'warning');
                }

            } catch (error) {
                console.error('Check-in failed:', error);
                showMessage('Check-in failed. Please try again.', 'error');
            } finally {
                document.getElementById('checkin-btn').disabled = false;
                document.getElementById('checkin-btn').textContent = 'Check In & Mine Tokens';
            }
        });

        // Refresh stats
        document.getElementById('refresh-stats').addEventListener('click', refreshStats);

        async function refreshStats() {
            if (!currentWallet) return;

            try {
                const response = await fetch(`${API_BASE}/user/${userId}/stats`);
                const stats = await response.json();

                document.getElementById('total-tokens').textContent = stats.total_tokens.toFixed(2);
                document.getElementById('total-checkins').textContent = stats.check_ins;
                document.getElementById('current-streak').textContent = `${stats.current_streak} days`;
                document.getElementById('last-reward').textContent = 
                    stats.last_check_in ? new Date(stats.last_check_in).toLocaleString() : 'Never';

            } catch (error) {
                console.error('Failed to refresh stats:', error);
            }
        }

        function showMessage(message, type = 'info') {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `status ${type}`;
            messageDiv.textContent = message;
            messagesDiv.appendChild(messageDiv);

            setTimeout(() => {
                messageDiv.remove();
            }, 5000);
        }

        // Auto-refresh stats every 30 seconds
        setInterval(() => {
            if (currentWallet) refreshStats();
        }, 30000);
    </script>
</body>
</html>"""
    
    with open("frontend_example.html", "w") as f:
        f.write(frontend_html)
    
    print("✅ Frontend example created: frontend_example.html")

def main():
    """Main setup function"""
    print("🚀 GeoMining Token System Setup")
    print("=" * 50)
    
    # Step 1: Check prerequisites
    check_prerequisites()
    
    # Step 2: Install dependencies
    install_dependencies()
    
    # Step 3: Setup Solana
    solana_ready = setup_solana_config()
    
    if not solana_ready:
        print("\n" + "⚠️ " * 20)
        print("SOLANA CLI SETUP REQUIRED")
        print("⚠️ " * 20)
        print("Please complete the Solana CLI setup manually, then run:")
        print("python create_token_only.py")
        print("python geomining_api.py")
        return
    
    # Step 4: Create token
    token_info = create_token()
    
    if token_info:
        # Step 5: Setup treasury
        setup_treasury()
        
        # Step 6: Create frontend example
        create_example_frontend()
        
        print("\n" + "=" * 50)
        print("🎉 SETUP COMPLETE!")
        print("=" * 50)
        print(f"Token Address: {token_info['token_address']}")
        print(f"Treasury Account: {token_info['treasury_account']}")
        print(f"Mint Authority: {token_info['mint_authority']}")
        print()
        print("Next steps:")
        print("1. Start the API server: python geomining_api.py")
        print("2. Open frontend_example.html in your browser")
        print("3. Install Phantom wallet extension")
        print("4. Connect wallet and start mining!")
        print()
        print("API Documentation: http://localhost:8000/docs")
        print("=" * 50)
    else:
        print("❌ Setup failed during token creation")
        sys.exit(1)

if __name__ == "__main__":
    main()