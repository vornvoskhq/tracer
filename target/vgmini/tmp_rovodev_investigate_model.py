#!/usr/bin/env python3
"""
Investigate the saved model to understand why ranking produces different results
"""

import sys
import os
sys.path.append('src')
import pickle
import numpy as np
from src.ml_models import EqualWeightSignalClassifier

def investigate_saved_model():
    """Compare the saved model vs fresh model"""
    
    print("🔍 VGMini Saved Model Investigation")
    print("=" * 60)
    
    model_path = "results/equal_weight_dual_enhanced/models/trained_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
        
    print(f"📂 Loading saved model: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Model data keys: {list(model_data.keys())}")
        
        # Extract models
        buy_model = model_data.get('buy_model')
        sell_model = model_data.get('sell_model')
        scaler = model_data.get('scaler')
        
        print(f"\n🎯 Buy Model Analysis:")
        if buy_model:
            print(f"   Type: {type(buy_model).__name__}")
            if hasattr(buy_model, 'mode'):
                print(f"   Mode: {buy_model.mode}")
            if hasattr(buy_model, 'aggregation'):
                print(f"   Aggregation: {buy_model.aggregation}")
            if hasattr(buy_model, 'correlation_signs'):
                print(f"   Correlation signs: {buy_model.correlation_signs}")
            if hasattr(buy_model, 'activation_threshold'):
                print(f"   Activation threshold: {buy_model.activation_threshold}")
            if hasattr(buy_model, 'temperature'):
                print(f"   Temperature: {buy_model.temperature}")
            if hasattr(buy_model, 'signs_'):
                print(f"   Signs shape: {buy_model.signs_.shape if buy_model.signs_ is not None else None}")
                print(f"   Signs values: {buy_model.signs_}")
            if hasattr(buy_model, 'intercept_'):
                print(f"   Intercept: {buy_model.intercept_}")
        else:
            print("   ❌ Buy model not found")
            
        print(f"\n🎯 Sell Model Analysis:")
        if sell_model:
            print(f"   Type: {type(sell_model).__name__}")
            if hasattr(sell_model, 'mode'):
                print(f"   Mode: {sell_model.mode}")
        else:
            print("   ❌ Sell model not found")
            
        print(f"\n🎯 Scaler Analysis:")
        if scaler:
            print(f"   Type: {type(scaler).__name__}")
        else:
            print("   ❌ Scaler not found")
            
        # Test the saved model on test data
        print(f"\n🧪 Testing Saved Model:")
        test_features = np.array([
            [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # AAPL pattern
            [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],  # QQQ pattern  
        ])
        
        if buy_model:
            try:
                probas = buy_model.predict_proba(test_features)
                print(f"   AAPL pattern probability: {probas[0, 1]:.3f}")
                print(f"   QQQ pattern probability: {probas[1, 1]:.3f}")
            except Exception as e:
                print(f"   ❌ Error testing model: {e}")
        
        # Compare with fresh model
        print(f"\n⚖️  Comparing with Fresh Model:")
        fresh_model = EqualWeightSignalClassifier(
            mode='buy',
            correlation_signs=True,
            activation_threshold=0.0,
            temperature=1.0,
            calibrate_prior=True,
            aggregation='fraction'
        )
        
        # Fit fresh model on same test data
        fresh_model.fit(test_features, None)
        fresh_probas = fresh_model.predict_proba(test_features)
        
        print(f"   Fresh model - AAPL: {fresh_probas[0, 1]:.3f}")
        print(f"   Fresh model - QQQ: {fresh_probas[1, 1]:.3f}")
        
        if buy_model:
            saved_probas = buy_model.predict_proba(test_features)
            print(f"   Saved model - AAPL: {saved_probas[0, 1]:.3f}")
            print(f"   Saved model - QQQ: {saved_probas[1, 1]:.3f}")
            
            print(f"\n🔍 Differences:")
            print(f"   AAPL diff: {abs(fresh_probas[0, 1] - saved_probas[0, 1]):.3f}")
            print(f"   QQQ diff: {abs(fresh_probas[1, 1] - saved_probas[1, 1]):.3f}")
            
            if abs(fresh_probas[0, 1] - saved_probas[0, 1]) > 0.01:
                print(f"❌ SIGNIFICANT DIFFERENCE in model predictions!")
                print(f"   This explains why ranking shows different results")
            else:
                print(f"✅ Models produce similar results")
                print(f"   Issue must be elsewhere in the pipeline")
        
    except Exception as e:
        print(f"❌ Error investigating model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    investigate_saved_model()