#!/usr/bin/env python3
"""
Verification tests for equal-weight model to ensure it's working properly
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
from src.ml_models import EqualWeightSignalClassifier

def test_equal_weight_basic():
    """Test basic equal-weight functionality"""
    print("🧪 Equal-Weight Model Verification Tests")
    print("=" * 60)
    
    # Test 1: Basic functionality with simple data
    print("\n📊 Test 1: Basic Functionality")
    print("-" * 30)
    
    # Create simple test data where we know what should happen
    X_simple = np.array([
        [1, 1, 1, 0, 0, 0],  # 3/6 features active = 0.5
        [1, 1, 1, 1, 1, 1],  # 6/6 features active = 1.0
        [0, 0, 0, 0, 0, 0],  # 0/6 features active = 0.0
        [1, 0, 1, 0, 1, 0],  # 3/6 features active = 0.5
    ])
    
    y_simple = np.array([0, 1, 0, 1])  # Simple binary targets
    
    print(f"Test data shape: {X_simple.shape}")
    print(f"Feature activation rates:")
    for i, row in enumerate(X_simple):
        activation_rate = np.mean(row)
        print(f"  Sample {i}: {activation_rate:.1%} ({np.sum(row)}/6 features active)")
    
    # Create and train model
    model = EqualWeightSignalClassifier(
        mode='buy',
        correlation_signs=False,  # Disable correlation signs for predictable behavior
        activation_threshold=0.0,
        temperature=1.0,
        calibrate_prior=False,    # Disable calibration for predictable behavior
        aggregation='fraction'
    )
    
    model.fit(X_simple, y_simple)
    
    print(f"\nModel parameters:")
    print(f"  Signs: {model.signs_}")
    print(f"  Intercept: {model.intercept_}")
    
    # Test predictions
    probas = model.predict_proba(X_simple)
    
    print(f"\nPredictions:")
    for i, (row, proba) in enumerate(zip(X_simple, probas)):
        activation_rate = np.mean(row)
        buy_prob = proba[1]
        print(f"  Sample {i}: {activation_rate:.1%} activation → {buy_prob:.3f} probability")
    
    # Verify that different activation rates produce different probabilities
    unique_probas = np.unique(probas[:, 1])
    if len(unique_probas) > 1:
        print("✅ PASS: Model produces different probabilities for different inputs")
    else:
        print("❌ FAIL: Model produces identical probabilities for different inputs")
        print(f"   All probabilities: {unique_probas}")
        
    return len(unique_probas) > 1

def test_correlation_signs():
    """Test correlation signs feature"""
    print("\n📊 Test 2: Correlation Signs")
    print("-" * 30)
    
    # Create data where features are negatively correlated with targets
    X_corr = np.array([
        [1, 1, 1],  # High features → negative target
        [1, 1, 0],  
        [1, 0, 0], 
        [0, 0, 0],  # Low features → positive target
        [0, 0, 1],
        [0, 1, 1],
    ])
    
    y_corr = np.array([0, 0, 0, 1, 1, 1])  # Inverted relationship
    
    print("Data with negative correlation:")
    for i, (row, target) in enumerate(zip(X_corr, y_corr)):
        print(f"  Features: {row} → Target: {target}")
    
    # Test with correlation signs enabled
    model_corr = EqualWeightSignalClassifier(
        mode='buy',
        correlation_signs=True,   # Enable correlation signs
        activation_threshold=0.0,
        temperature=1.0,
        calibrate_prior=False,
        aggregation='fraction'
    )
    
    model_corr.fit(X_corr, y_corr)
    
    print(f"\nModel with correlation signs:")
    print(f"  Signs: {model_corr.signs_}")
    print(f"  Expected: negative values (features negatively correlated)")
    
    probas_corr = model_corr.predict_proba(X_corr)
    
    print(f"\nPredictions with correlation signs:")
    for i, (row, proba, target) in enumerate(zip(X_corr, probas_corr, y_corr)):
        buy_prob = proba[1]
        print(f"  Features: {row} → Prob: {buy_prob:.3f} (target: {target})")
    
    # Test without correlation signs
    model_no_corr = EqualWeightSignalClassifier(
        mode='buy',
        correlation_signs=False,  # Disable correlation signs
        activation_threshold=0.0,
        temperature=1.0,
        calibrate_prior=False,
        aggregation='fraction'
    )
    
    model_no_corr.fit(X_corr, y_corr)
    
    print(f"\nModel without correlation signs:")
    print(f"  Signs: {model_no_corr.signs_}")
    print(f"  Expected: all positive (default)")
    
    signs_different = not np.array_equal(model_corr.signs_, model_no_corr.signs_)
    if signs_different:
        print("✅ PASS: Correlation signs feature works")
    else:
        print("❌ FAIL: Correlation signs feature not working")
        
    return signs_different

def test_real_data_variance():
    """Test with real training data to see if variance exists"""
    print("\n📊 Test 3: Real Data Variance")
    print("-" * 30)
    
    try:
        # Load actual training data from mixed model
        import pickle
        model_path = "results/equal_weight_dual_enhanced_mixed/models/trained_model.pkl"
        
        if not os.path.exists(model_path):
            print("❌ No trained model found - run training first")
            return False
            
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        buy_model = model_data['buy_model']
        feature_names = model_data.get('feature_names', [])
        
        print(f"Loaded model with {len(feature_names)} features")
        print(f"Feature names: {feature_names}")
        print(f"Model signs: {buy_model.signs_}")
        print(f"Model intercept: {buy_model.intercept_}")
        
        # Test with some varied feature combinations
        n_features = len(buy_model.signs_)
        test_cases = [
            np.zeros(n_features),                           # All zeros
            np.ones(n_features),                           # All ones  
            np.array([1] * (n_features//2) + [0] * (n_features//2)),  # Half/half
            np.random.binomial(1, 0.3, n_features),       # 30% activation
            np.random.binomial(1, 0.7, n_features),       # 70% activation
        ]
        
        print(f"\nTesting model with varied inputs:")
        probas = []
        for i, test_case in enumerate(test_cases):
            test_case = test_case.reshape(1, -1)
            proba = buy_model.predict_proba(test_case)[0, 1]
            activation_rate = np.mean(test_case)
            probas.append(proba)
            print(f"  Test {i+1}: {activation_rate:.1%} activation → {proba:.6f} probability")
            
        # Check variance
        prob_variance = np.var(probas)
        unique_probas = len(np.unique(np.round(probas, 6)))
        
        print(f"\nVariance analysis:")
        print(f"  Probability variance: {prob_variance:.8f}")
        print(f"  Unique probabilities: {unique_probas}/5")
        
        if prob_variance > 1e-6 and unique_probas > 1:
            print("✅ PASS: Model shows variance in predictions")
            return True
        else:
            print("❌ FAIL: Model produces identical/near-identical predictions")
            print("  This suggests the saved model has issues")
            return False
            
    except Exception as e:
        print(f"❌ Error testing real data: {e}")
        return False

def test_temperature_effects():
    """Test temperature parameter effects"""
    print("\n📊 Test 4: Temperature Effects")
    print("-" * 30)
    
    X_temp = np.array([
        [0.8, 0.2, 0.6, 0.4],  # Mixed activation
        [0.9, 0.1, 0.7, 0.3],  # Slightly different
    ])
    
    y_temp = np.array([1, 0])
    
    temperatures = [0.1, 0.5, 1.0, 2.0]
    
    print("Testing different temperatures:")
    for temp in temperatures:
        model = EqualWeightSignalClassifier(
            mode='buy',
            correlation_signs=False,
            activation_threshold=0.0,
            temperature=temp,
            calibrate_prior=False,
            aggregation='fraction'
        )
        
        model.fit(X_temp, y_temp)
        probas = model.predict_proba(X_temp)
        
        prob_diff = abs(probas[0, 1] - probas[1, 1])
        print(f"  Temperature {temp:0.1f}: {probas[0, 1]:.4f} vs {probas[1, 1]:.4f} (diff: {prob_diff:.4f})")
    
    print("✅ Temperature test completed")
    return True

if __name__ == "__main__":
    print("🔍 Equal-Weight Model Verification Suite")
    print("========================================")
    
    results = []
    
    try:
        results.append(test_equal_weight_basic())
        results.append(test_correlation_signs()) 
        results.append(test_real_data_variance())
        results.append(test_temperature_effects())
        
        print(f"\n🎯 SUMMARY")
        print("=" * 30)
        passed = sum(results)
        total = len(results)
        
        print(f"Tests passed: {passed}/{total}")
        
        if passed == total:
            print("✅ Equal-weight model implementation appears correct")
            print("   Issue may be in training data or configuration")
        else:
            print("❌ Equal-weight model has implementation issues")
            print("   Need to fix basic functionality first")
            
    except Exception as e:
        print(f"❌ Error in verification suite: {e}")
        import traceback
        traceback.print_exc()