#!/usr/bin/env python3
"""
Compare the original equal_weight_dual with the new equal_weight_dual_enhanced_mixed
to answer the key questions about what changed
"""

import sys
import os
sys.path.append('src')
import pickle

def compare_models():
    """Compare configurations and parameters of both models"""
    
    print("🔍 Model Comparison Analysis")
    print("=" * 60)
    
    # Model paths
    original_path = "results/equal_weight_dual/models/trained_model.pkl"
    new_path = "results/equal_weight_dual_enhanced_mixed/models/trained_model.pkl"
    
    models = {}
    
    # Load both models if they exist
    for name, path in [("Original equal_weight_dual", original_path), 
                       ("New equal_weight_dual_enhanced_mixed", new_path)]:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    model_data = pickle.load(f)
                models[name] = model_data
                print(f"✅ Loaded {name}")
            except Exception as e:
                print(f"❌ Failed to load {name}: {e}")
        else:
            print(f"❌ {name} not found at {path}")
    
    if len(models) == 0:
        print("No models found to compare!")
        return
        
    print(f"\n📊 Detailed Model Comparison")
    print("=" * 60)
    
    for model_name, model_data in models.items():
        print(f"\n🔍 {model_name}")
        print("-" * 40)
        
        # Basic info
        keys = list(model_data.keys())
        print(f"Data keys: {keys}")
        
        # Buy model analysis
        buy_model = model_data.get('buy_model')
        if buy_model:
            print(f"\n📈 Buy Model:")
            print(f"   Type: {type(buy_model).__name__}")
            print(f"   Mode: {getattr(buy_model, 'mode', 'N/A')}")
            print(f"   Correlation signs: {getattr(buy_model, 'correlation_signs', 'N/A')}")
            print(f"   Calibrate prior: {getattr(buy_model, 'calibrate_prior', 'N/A')}")
            print(f"   Activation threshold: {getattr(buy_model, 'activation_threshold', 'N/A')}")
            print(f"   Temperature: {getattr(buy_model, 'temperature', 'N/A')}")
            print(f"   Aggregation: {getattr(buy_model, 'aggregation', 'N/A')}")
            
            if hasattr(buy_model, 'signs_'):
                print(f"   Signs: {buy_model.signs_}")
                negative_count = sum(1 for s in buy_model.signs_ if s < 0)
                positive_count = sum(1 for s in buy_model.signs_ if s > 0)
                print(f"   Signs summary: {positive_count} positive, {negative_count} negative")
                
            if hasattr(buy_model, 'intercept_'):
                print(f"   Intercept: {buy_model.intercept_}")
        
        # Feature names
        feature_names = model_data.get('feature_names', [])
        if feature_names:
            print(f"\n📋 Features ({len(feature_names)}):")
            for i, name in enumerate(feature_names):
                sign = buy_model.signs_[i] if buy_model and hasattr(buy_model, 'signs_') else 'N/A'
                print(f"   {i+1:2d}. {name:<30} (sign: {sign:4.1f})")
        
        # Config if available
        config = model_data.get('config', {})
        if config:
            print(f"\n⚙️  Training Config:")
            for key, value in config.items():
                if key.startswith('ew_'):
                    print(f"   {key}: {value}")
    
    # Compare if we have both models
    if len(models) == 2:
        print(f"\n🆚 COMPARISON SUMMARY")
        print("=" * 40)
        
        original = models["Original equal_weight_dual"]
        new = models["New equal_weight_dual_enhanced_mixed"]
        
        orig_buy = original.get('buy_model')
        new_buy = new.get('buy_model')
        
        if orig_buy and new_buy:
            print(f"\n🔧 Configuration Differences:")
            
            configs = [
                ('correlation_signs', 'correlation_signs'),
                ('calibrate_prior', 'calibrate_prior'), 
                ('activation_threshold', 'activation_threshold'),
                ('temperature', 'temperature'),
                ('aggregation', 'aggregation')
            ]
            
            for attr_name, display_name in configs:
                orig_val = getattr(orig_buy, attr_name, 'N/A')
                new_val = getattr(new_buy, attr_name, 'N/A')
                
                if orig_val != new_val:
                    print(f"   {display_name:20}: {orig_val:10} → {new_val:10} ⚠️ CHANGED")
                else:
                    print(f"   {display_name:20}: {orig_val:10} → {new_val:10} ✓ same")
            
            # Compare signs
            if hasattr(orig_buy, 'signs_') and hasattr(new_buy, 'signs_'):
                orig_signs = orig_buy.signs_
                new_signs = new_buy.signs_
                
                signs_different = not all(abs(a - b) < 1e-6 for a, b in zip(orig_signs, new_signs))
                
                if signs_different:
                    print(f"\n⚠️  SIGNS CHANGED:")
                    orig_pos = sum(1 for s in orig_signs if s > 0)
                    new_pos = sum(1 for s in new_signs if s > 0)
                    print(f"   Original: {orig_pos}/{len(orig_signs)} positive")
                    print(f"   New:      {new_pos}/{len(new_signs)} positive")
                else:
                    print(f"\n✓ Signs are identical")
            
            # Compare intercepts
            orig_intercept = getattr(orig_buy, 'intercept_', 0)
            new_intercept = getattr(new_buy, 'intercept_', 0)
            
            if abs(orig_intercept - new_intercept) > 1e-6:
                print(f"\n⚠️  INTERCEPT CHANGED:")
                print(f"   Original: {orig_intercept:.6f}")
                print(f"   New:      {new_intercept:.6f}")
            else:
                print(f"\n✓ Intercepts are similar")

def answer_key_questions():
    """Answer the three key questions based on analysis"""
    
    print(f"\n🎯 ANSWERS TO KEY QUESTIONS")
    print("=" * 50)
    
    print(f"\n1️⃣ Was equal_weight_dual originally working as true equal-weight?")
    print("   Looking at equal_weight_dual.yaml:")
    print("   - ew_correlation_signs: false  ← TRUE equal-weight!")
    print("   - ew_calibrate_prior: false   ← No bias adjustment!")
    print("   - ew_temperature: 1.0         ← No smoothing!")
    print("   ✅ ANSWER: YES, it was true equal-weight (just count signals)")
    
    print(f"\n2️⃣ When did correlation signs get added?")
    print("   Looking at global.yaml:")
    print("   - ew_correlation_signs: true  ← Global default enables it!")
    print("   - ew_calibrate_prior: true    ← Global default enables it!")
    print("   ✅ ANSWER: Global defaults override experiment configs")
    print("      The global config corrupted the simple equal-weight model!")
    
    print(f"\n3️⃣ What value does this 'training' actually provide?")
    print("   Current training does:")
    print("   - Learns which signals are 'bad' vs 'good' historically")
    print("   - Biases probabilities based on past target frequency")
    print("   - Smooths differences between activation rates")
    print("   ❌ PROBLEM: This defeats the purpose of equal-weight!")
    print("   ✅ ANSWER: Current training provides NEGATIVE value")
    print("      It corrupts simple signal counting with historical bias")

if __name__ == "__main__":
    compare_models()
    answer_key_questions()
    
    print(f"\n💡 SOLUTION")
    print("=" * 30)
    print("To restore true equal-weight behavior:")
    print("1. Set ew_correlation_signs: false  (don't learn inversions)")
    print("2. Set ew_calibrate_prior: false    (don't add bias)")
    print("3. Set ew_temperature: 1.0          (no smoothing)")
    print("4. Use ew_activation_threshold: 0.0 (binary signals)")
    print("5. Keep ew_buy_aggregation: fraction (count signals)")
    print("")
    print("Result: probability = activated_signals / total_signals")
    print("Simple, interpretable, no training needed!")