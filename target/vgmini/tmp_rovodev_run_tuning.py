#!/usr/bin/env python3
"""
Runner for the equal weight tuning process
"""

import sys
import os
import subprocess

def run_tuning_for_enhanced_model():
    """Run tuning on the enhanced mixed model"""
    
    print("🎯 Running Equal-Weight Parameter Tuning")
    print("=" * 50)
    
    # Parameters for tuning
    symbol = "SPY"  # Use SPY as representative symbol
    experiment = "equal_weight_dual_enhanced_mixed"
    lookback = 252  # 1 year of data for tuning
    
    print(f"📊 Tuning Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Experiment: {experiment}")
    print(f"   Lookback: {lookback} days")
    
    # Check if trained model exists
    model_path = f"results/{experiment}/models/trained_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ No trained model found at {model_path}")
        print(f"💡 Run: ./vg train --model {experiment}")
        return
        
    print(f"✅ Found trained model: {model_path}")
    
    # Build tuning command
    cmd = [
        sys.executable, "-m", "scripts.tune_equal_weight",
        "--symbol", symbol,
        "--experiment", experiment,
        "--lookback", str(lookback)
    ]
    
    print(f"\n🚀 Running tuning command:")
    print(f"   {' '.join(cmd)}")
    
    try:
        # Run tuning in the vgmini directory
        result = subprocess.run(cmd, cwd=".", capture_output=True, text=True)
        
        print(f"\n📋 Tuning Results:")
        print("=" * 30)
        
        if result.returncode == 0:
            print("✅ Tuning completed successfully!")
            print("\n📊 Output:")
            print(result.stdout)
            
            # Extract optimal parameters if they're in the output
            if "ew_activation_threshold:" in result.stdout:
                print("\n🎯 Optimal Parameters Found!")
                lines = result.stdout.split('\n')
                for line in lines:
                    if any(param in line for param in ['ew_activation_threshold:', 'ew_temperature:', 'hysteresis_margin:']):
                        print(f"   {line.strip()}")
            
        else:
            print("❌ Tuning failed!")
            print(f"Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error running tuning: {e}")

def compare_before_after_tuning():
    """Compare model performance before and after tuning"""
    
    print(f"\n🆚 Performance Comparison Recommendation:")
    print("=" * 40)
    print("After tuning, you should:")
    print("1. Update the experiment config with optimal parameters")
    print("2. Retrain the model with tuned parameters") 
    print("3. Run pipeline to see improved results")
    print("4. Compare trading performance metrics")

if __name__ == "__main__":
    run_tuning_for_enhanced_model()
    compare_before_after_tuning()
    
    print(f"\n💡 Next Steps:")
    print("1. First retrain: ./vg train --model equal_weight_dual_enhanced_mixed")
    print("2. Then run tuning: python tmp_rovodev_run_tuning.py")
    print("3. Update config with optimal parameters")
    print("4. Retrain with tuned parameters")
    print("5. Test pipeline: ./vg pipeline --model equal_weight_dual_enhanced_mixed")