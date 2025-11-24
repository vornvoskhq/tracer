#!/usr/bin/env python3
"""
Debug script to trace exactly what config values are being passed to the model
during training to identify why the override isn't working
"""

import sys
import os
sys.path.append('src')

from src.yaml_config import load_merged_experiment_yaml
import json

def debug_config_merge():
    """Debug the config merge process step by step"""
    
    print("🔍 Config Override Debug Analysis")
    print("=" * 60)
    
    model_name = "equal_weight_dual_enhanced_mixed"
    
    print(f"\n📋 Step 1: Loading Raw Config Files")
    print("-" * 40)
    
    # Load global config
    global_path = "configs/global.yaml"
    exp_path = f"configs/experiments/{model_name}.yaml"
    
    import yaml
    
    try:
        with open(global_path, 'r') as f:
            global_config = yaml.safe_load(f)
        print(f"✅ Loaded global config from: {global_path}")
        
        global_model = global_config.get('model', {})
        print(f"📊 Global model config:")
        for key in ['ew_correlation_signs', 'ew_calibrate_prior', 'ew_temperature', 'ew_activation_threshold']:
            value = global_model.get(key, 'NOT_SET')
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ Error loading global config: {e}")
        return
        
    try:
        with open(exp_path, 'r') as f:
            exp_config = yaml.safe_load(f)
        print(f"\n✅ Loaded experiment config from: {exp_path}")
        
        exp_model = exp_config.get('model', {})
        print(f"📊 Experiment model config:")
        for key in ['ew_correlation_signs', 'ew_calibrate_prior', 'ew_temperature', 'ew_activation_threshold']:
            value = exp_model.get(key, 'NOT_SET')
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ Error loading experiment config: {e}")
        return
        
    print(f"\n📋 Step 2: Testing YAML Merge Function")
    print("-" * 40)
    
    try:
        merged_config = load_merged_experiment_yaml(model_name)
        print(f"✅ Merged config loaded successfully")
        
        print(f"📊 Merged equal-weight model parameters:")
        for key in ['ew_correlation_signs', 'ew_calibrate_prior', 'ew_temperature', 'ew_activation_threshold']:
            value = merged_config.get(key, 'NOT_SET')
            print(f"   {key}: {value}")
            
        # Check if model type is set correctly
        model_type = merged_config.get('model_type')
        print(f"\n🎯 Model type: {model_type}")
        
        print(f"\n🔍 Full merged config keys:")
        config_keys = sorted(merged_config.keys())
        for key in config_keys:
            if key.startswith('ew_') or key in ['model_type', 'name', 'description']:
                print(f"   {key}: {merged_config[key]}")
        
    except Exception as e:
        print(f"❌ Error merging configs: {e}")
        import traceback
        traceback.print_exc()
        return
        
    print(f"\n📋 Step 3: Test Model Instantiation")
    print("-" * 40)
    
    try:
        from src.ml_models import EqualWeightSignalClassifier
        
        # Extract equal-weight parameters from merged config
        ew_params = {
            'correlation_signs': merged_config.get('ew_correlation_signs', True),
            'calibrate_prior': merged_config.get('ew_calibrate_prior', True), 
            'activation_threshold': merged_config.get('ew_activation_threshold', 0.0),
            'temperature': merged_config.get('ew_temperature', 1.0),
            'aggregation': merged_config.get('ew_buy_aggregation', 'fraction')
        }
        
        print(f"📊 Parameters passed to EqualWeightSignalClassifier:")
        for key, value in ew_params.items():
            print(f"   {key}: {value}")
            
        # Create model with these parameters
        test_model = EqualWeightSignalClassifier(
            mode='buy',
            correlation_signs=ew_params['correlation_signs'],
            calibrate_prior=ew_params['calibrate_prior'],
            activation_threshold=ew_params['activation_threshold'], 
            temperature=ew_params['temperature'],
            aggregation=ew_params['aggregation']
        )
        
        print(f"\n✅ Model created successfully with parameters:")
        print(f"   correlation_signs: {test_model.correlation_signs}")
        print(f"   calibrate_prior: {test_model.calibrate_prior}")
        print(f"   activation_threshold: {test_model.activation_threshold}")
        print(f"   temperature: {test_model.temperature}")
        print(f"   aggregation: {test_model.aggregation}")
        
    except Exception as e:
        print(f"❌ Error creating test model: {e}")
        import traceback
        traceback.print_exc()

def trace_training_initialization():
    """Trace how the training framework loads and uses config"""
    
    print(f"\n📋 Step 4: Trace Training Framework Config Loading")
    print("-" * 50)
    
    try:
        from src.experiment_configs import ExperimentConfig
        
        # Load config same way training does
        model_name = "equal_weight_dual_enhanced_mixed"
        config_data = load_merged_experiment_yaml(model_name)
        
        print(f"✅ Loaded config data via load_merged_experiment_yaml")
        
        # Convert to ExperimentConfig
        config = ExperimentConfig.from_dict(config_data)
        
        print(f"✅ Created ExperimentConfig from dict")
        print(f"📊 ExperimentConfig equal-weight settings:")
        
        # Check what ExperimentConfig actually has
        ew_attrs = [attr for attr in dir(config) if attr.startswith('ew_')]
        for attr in ew_attrs:
            value = getattr(config, attr, 'NOT_SET')
            print(f"   {attr}: {value}")
            
        # Check specific problematic values
        print(f"\n🎯 Critical values:")
        print(f"   ew_correlation_signs: {getattr(config, 'ew_correlation_signs', 'NOT_SET')}")
        print(f"   ew_calibrate_prior: {getattr(config, 'ew_calibrate_prior', 'NOT_SET')}")
        print(f"   ew_temperature: {getattr(config, 'ew_temperature', 'NOT_SET')}")
        
    except Exception as e:
        print(f"❌ Error in training framework config trace: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_config_merge()
    trace_training_initialization()
    
    print(f"\n💡 EXPECTED vs ACTUAL")
    print("=" * 30)
    print("Based on equal_weight_dual_enhanced_mixed.yaml, we should see:")
    print("   ew_correlation_signs: False")
    print("   ew_calibrate_prior: False") 
    print("   ew_temperature: 1.0")
    print("")
    print("But training logs show:")
    print("   corr_signs=True, calibrate=True, temp=0.75")
    print("")
    print("If the merged config shows the correct values but training")
    print("uses wrong values, the issue is in how ExperimentConfig")
    print("passes parameters to the model creation.")