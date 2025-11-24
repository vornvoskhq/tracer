#!/usr/bin/env python3
"""
VGMini Trading Framework - Single Entry Point
Professional command-line interface for running trading experiments
"""

import argparse
import json
import os
import sys
import subprocess
import shutil
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

def check_virtual_environment():
    """Check if we're running in a virtual environment"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def get_venv_python():
    """Get the path to the virtual environment Python executable"""
    if os.name == 'nt':  # Windows
        return os.path.join('.venv', 'Scripts', 'python.exe')
    else:  # Unix/Linux/macOS
        return os.path.join('.venv', 'bin', 'python')

def setup_virtual_environment():
    """Create and setup virtual environment with comprehensive error handling"""
    print("🔧 Setting up virtual environment...")
    
    # Create venv if it doesn't exist
    if not os.path.exists('.venv'):
        print("📦 Creating virtual environment...")
        try:
            subprocess.run([sys.executable, '-m', 'venv', '.venv'], 
                         check=True, stderr=subprocess.PIPE, text=True)
            print("✅ Virtual environment created")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e.stderr}")
            print("💡 Try installing python3-venv: sudo apt-get install python3-venv")
            sys.exit(1)
    
    venv_python = get_venv_python()
    
    # Ensure pip is up to date first
    try:
        print("🔄 Upgrading pip and setuptools...")
        subprocess.run(
            [venv_python, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools'],
            check=True, stderr=subprocess.PIPE, text=True, stdout=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to upgrade pip: {e.stderr}")
    
    # Install dependencies from pyproject.toml if it exists
    if os.path.exists('pyproject.toml'):
        print("📦 Installing project dependencies...")
        try:
            result = subprocess.run(
                [venv_python, '-m', 'pip', 'install', '-e', '.'],
                check=True, capture_output=True, text=True
            )
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies. Error:\n{e.stderr}")
            print("\n💡 Attempting individual package installation...")
            
            # Fallback to installing core dependencies individually
            core_deps = [
                'pandas>=2.0.0', 'numpy>=1.24.0', 'scikit-learn>=1.3.0',
                'matplotlib>=3.7.0', 'seaborn>=0.12.0', 'ta>=0.10.0'
            ]
            try:
                print("📦 Installing core dependencies...")
                subprocess.run(
                    [venv_python, '-m', 'pip', 'install'] + core_deps,
                    check=True, stderr=subprocess.PIPE, text=True
                )
                print("✅ Core dependencies installed")
                
                # Try installing the package again
                print("📦 Installing package in development mode...")
                subprocess.run(
                    [venv_python, '-m', 'pip', 'install', '-e', '.'],
                    check=True, stderr=subprocess.PIPE, text=True
                )
                print("✅ Package installed in development mode")
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install core dependencies: {e.stderr}")
                print("\n💡 Please try installing the dependencies manually:")
                print("1. source .venv/bin/activate")
                print("2. pip install -e .")
                sys.exit(1)
    
    elif os.path.exists('requirements.txt'):
        print("📦 Installing dependencies from requirements.txt...")
        try:
            subprocess.run(
                [venv_python, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                check=True, stderr=subprocess.PIPE, text=True
            )
            print("✅ Dependencies installed from requirements.txt")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e.stderr}")
            sys.exit(1)
    else:
        print("⚠️  No pyproject.toml or requirements.txt found")
    
    return venv_python

def restart_with_venv():
    """Restart the script with the virtual environment Python"""
    if check_virtual_environment():
        # Already in venv, continue
        return False
    
    # Check if venv exists, create if not
    if not os.path.exists('.venv'):
        setup_virtual_environment()
    
    venv_python = get_venv_python()
    
    # Check if venv python exists
    if not os.path.exists(venv_python):
        setup_virtual_environment()
    
    # Restart with venv python, preserving module invocation if used
    print("🔄 Restarting with virtual environment...")
    argv0 = sys.argv[0]
    # If argv0 looks like a script path (ends with .py or exists on disk), run it as a script
    # Otherwise, assume it was invoked as a module name and preserve -m semantics
    if (argv0.endswith('.py') or os.path.exists(argv0)):
        new_argv = [venv_python, argv0] + sys.argv[1:]
    else:
        new_argv = [venv_python, '-m', argv0] + sys.argv[1:]
    os.execv(venv_python, new_argv)

# Validate we're in the right directory
if not os.path.exists('src') or not os.path.exists('vgmini.py'):
    print("❌ Please run vgmini.py from the project root directory")
    print("💡 Make sure you're in the directory containing src/ and vgmini.py")
    sys.exit(1)

# Check and restart with venv if needed
restart_with_venv()

# Add src to path
sys.path.append('src')

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from src.main_framework import TradingFramework
    from src.experiment_configs import ExperimentConfig
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("🔧 Try running: pip install -r requirements.txt")
    sys.exit(1)

def run_pipeline(model_name: str, top_count: int = 100):
    """Run the complete VGMini pipeline: ranking -> decorrelation -> trailstop -> limiter -> charts"""
    import os
    from pathlib import Path
    
    print(f"[pipeline] Running complete pipeline for model: {model_name}")
    
    # Step 1: Generate ranking
    print(f"[pipeline] Ranking top {top_count} symbols...")
    result = subprocess.run([sys.executable, __file__, 'rank', model_name, '--top', str(top_count), '--quiet'], 
                          capture_output=False)
    if result.returncode != 0:
        print(f"[pipeline] Ranking failed with return code {result.returncode}")
        return
    
    # Verify ranking file was created
    ranking_file = f"results/{model_name}_ranking.txt"
    if not os.path.exists(ranking_file):
        print(f"[pipeline] Error: Ranking file not found: {ranking_file}")
        return
    print(f"Wrote ranking to vgmini/{ranking_file}")
    
    # Step 2: Decorrelation  
    print(f"[pipeline] Decorrelating...")
    try:
        from src.decorrelate import run_decorrelate, load_config
        import tempfile
        import yaml
        import shutil
        
        # Load the decorrelate config and update the ranking file path and model name
        config = load_config('configs/decorrelate.yaml')
        config['input']['ranking_file'] = ranking_file
        
        # Replace model name placeholder in output prefix
        if 'output' in config and 'prefix' in config['output']:
            config['output']['prefix'] = config['output']['prefix'].replace('{{model_name}}', model_name)
        
        # Write temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_config:
            yaml.dump(config, temp_config)
            temp_config_path = temp_config.name
        
        try:
            out_txt, out_json = run_decorrelate(temp_config_path)
            print(f"De-correlated shortlist: {out_txt}")
            
            # Copy decorrelated results to where mithra.sh can read it
            vgmini_ranking_file = f"{model_name}_ranking.txt"  # For reports.py to find
            
            if os.path.exists(out_txt):
                shutil.copy2(out_txt, vgmini_ranking_file)  # Copy to vgmini root for reports.py
                print(f"Final ranking saved to: {vgmini_ranking_file}")
                ranking_file = vgmini_ranking_file  # Update for next steps
            
        finally:
            # Clean up temporary config file
            import os
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
                
    except Exception as e:
        print(f"[pipeline] Decorrelation failed: {e}")
        return
        
    # Step 3: Trail stops
    print(f"[pipeline] Computing trail stops...")
    try:
        from src.trailstop import run_trailstop
        trailstop_json = run_trailstop(out_txt, 'results/trailstop')
        print(f"Trailstop analysis: {trailstop_json}")
    except Exception as e:
        print(f"[pipeline] Trail stop analysis failed: {e}")
        return
    
    # Step 4: Limiter analysis  
    print(f"[pipeline] Analyzing swing highs/lows...")
    try:
        from src.limiter import run_limiter
        limiter_json = run_limiter(out_txt, 'results/limiter')
        print(f"Limiter analysis: {limiter_json}")
    except Exception as e:
        print(f"[pipeline] Limiter analysis failed: {e}")
        return
        
    # Step 5: Render charts
    print(f"[pipeline] Rendering charts...")
    try:
        from src.render_charts import render_charts_with_trailstop
        # Parse symbols from the final ranking file (decorrelated)
        symbols = []
        if os.path.exists(ranking_file):
            with open(ranking_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('De-correlated') and not line.startswith('Total'):
                        parts = line.split()
                        if len(parts) >= 2 and parts[0].isdigit():  # Must start with rank number
                            symbols.append(parts[1])  # Symbol is typically second column
        
        if symbols:
            render_charts_with_trailstop(symbols, 'results/reports', trailstop_json, ranking_file)
            print(f"[pipeline] Charts rendered for {len(symbols)} symbols")
            if len(symbols) <= 10:  # Only show symbols if 10 or fewer
                print(f"   Symbols: {', '.join(symbols)}")
            else:
                print(f"   Top 5: {', '.join(symbols[:5])}")
        else:
            print(f"[pipeline] No symbols found in ranking file: {ranking_file}")
    except Exception as e:
        print(f"[pipeline] Chart rendering failed: {e}")
        return
    
    # Step 6: Create model config reference for mithra.sh
    print(f"[pipeline] Preparing model configuration reference...")
    try:
        # Create a simple reference file pointing to the YAML config
        config_ref_file = f"results/{model_name}_config.yaml"
        yaml_config_path = f"configs/experiments/{model_name}.yaml"
        
        if os.path.exists(yaml_config_path):
            # Create symlink or copy to results for easier access by mithra.sh
            import shutil
            shutil.copy2(yaml_config_path, config_ref_file)
            print(f"Model config reference: {config_ref_file}")
            print(f"Source config: {yaml_config_path}")
        else:
            print(f"Warning: Source config not found: {yaml_config_path}")
            
    except Exception as e:
        print(f"Warning: Config reference creation failed: {e}")
    
    # Step 7: Summary
    print(f"[pipeline] Pipeline completed successfully!")
    print(f"📊 Final outputs for mithra.sh:")
    print(f"   📋 Ranking: {ranking_file} (decorrelated)")
    print(f"   📈 Charts: results/reports/")
    print(f"   ⚙️  Model config: configs/experiments/{model_name}.yaml (editable)")
    print(f"   🛡️  Trailstop: results/trailstop/trailstop.json")
    print(f"   ⚖️  Limiter: results/limiter/limiter.json")
    print(f"\n✅ Ready for mithra.sh integration!")


def list_configs():
    """List available experiments from YAML (preferred) and legacy JSON configs."""
    from src.yaml_config import list_yaml_experiments

    # YAML experiments
    yaml_exps = list_yaml_experiments()
    print("📋 Available Experiments (YAML):")
    print("=" * 50)
    if yaml_exps:
        for name in sorted(yaml_exps.keys()):
            meta = yaml_exps[name]
            print(f"\n🔧 {name}")
            print(f"   File: {os.path.relpath(meta['path'])}")
            print(f"   Description: {meta.get('description','No description')}")
    else:
        print("(none found under configs/experiments)")

    # Legacy JSON configs under results/configs for backward-compatibility
    config_dir = "results/configs"
    if os.path.exists(config_dir):
        configs = [f for f in os.listdir(config_dir) if f.endswith('.json')]
        if configs:
            print("\n📦 Legacy Configurations (JSON):")
            print("=" * 50)
            for config_file in sorted(configs):
                config_path = os.path.join(config_dir, config_file)
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    name = config_data.get('name', config_file.replace('.json', ''))
                    description = config_data.get('description', 'No description')
                    symbols = config_data.get('symbols', [])
                    print(f"\n🔧 {name}")
                    print(f"   File: {config_file}")
                    print(f"   Description: {description}")
                    if symbols:
                        print(f"   Symbols: {', '.join(symbols)}")
                except Exception as e:
                    print(f"\n❌ {config_file} - Error reading: {e}")
    else:
        print("\n(no legacy JSON config directory found)")

def run_experiment(config_name: str, verbose: bool = False):
    """Run a single experiment by name.
    Priority: YAML in configs/experiments/<name>.yaml, else legacy results/configs/<name>.json
    """
    from src.yaml_config import load_merged_experiment_yaml

    print(f"🔍 DEBUG: Starting run_experiment for '{config_name}'")
    
    # Attempt YAML first
    yaml_path = os.path.join("configs", "experiments", f"{config_name}.yaml")
    json_path = f"results/configs/{config_name}.json"
    
    print(f"🔍 DEBUG: Checking YAML path: {yaml_path}")
    print(f"🔍 DEBUG: YAML exists: {os.path.exists(yaml_path)}")
    print(f"🔍 DEBUG: JSON path: {json_path}")
    print(f"🔍 DEBUG: JSON exists: {os.path.exists(json_path)}")

    if not os.path.exists(yaml_path):
        print(f"❌ YAML config not found: {yaml_path}")
        print("💡 Use 'vgmini.py list' to see available experiments")
        return None
    
    try:
        config_data = load_merged_experiment_yaml(config_name)
        if not config_data:
            print(f"❌ Failed to load config for {config_name}")
            return None
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None
    
    if config_data is None:
        print(f"❌ DEBUG: Final config_data is None - this should not happen!")
        return None

    print(f"🚀 Running experiment: {config_name}")
    print("=" * 60)
    
    try:
        config = ExperimentConfig.from_dict(config_data)
        
        if verbose:
            print(f"📊 Configuration Details:")
            print(f"   Symbols: {', '.join(config.symbols)}")
            print(f"   Date Range: {config.start_date} to {config.end_date}")
            print(f"   Features: {len(config.enabled_features)} enabled")
            print(f"   Confidence Threshold: {config.confidence_threshold}")
        
        # Save YAML snapshot if available
        try:
            import yaml
            os.makedirs(f"results/{config.name}", exist_ok=True)
            with open(f"results/{config.name}/config_snapshot.yaml", "w") as cf:
                yaml.safe_dump(config_data, cf, sort_keys=False)
        except Exception as _e:
            if verbose:
                print(f"⚠️  Failed to write config snapshot: {_e}")
        
        # Initialize framework with configuration overrides
        config_overrides = {
            'model_config': {
                'buy_threshold': config.buy_threshold,
                'sell_threshold': config.sell_threshold,
                'forecast_horizon': config.forecast_horizon,
                'test_size': config.test_size,
                'class_weight': config.class_weight,
                'penalty': config.penalty,
                'solver': config.solver,
                'max_iter': config.max_iter,
                'tol': config.tol,
                'model_type': config.model_type,
                'target_strategy': getattr(config, 'target_strategy', 'dual'),
                'n_estimators': config.n_estimators,
                'max_depth': config.max_depth,
                'learning_rate': config.learning_rate,
                'subsample': config.subsample,
                'colsample_bytree': config.colsample_bytree,
                'ema_periods': config.ema_periods,
                'volume_window': config.volume_window,
                'base_models': getattr(config, 'base_models', None),
                'weights': getattr(config, 'weights', None),
                # Equal-weight knobs
                'ew_correlation_signs': getattr(config, 'ew_correlation_signs', False),
                'ew_calibrate_prior': getattr(config, 'ew_calibrate_prior', False),
                'ew_activation_threshold': getattr(config, 'ew_activation_threshold', 0.0),
                'ew_temperature': getattr(config, 'ew_temperature', 1.0),
                'ew_buy_aggregation': getattr(config, 'ew_buy_aggregation', 'fraction'),
                'ew_sell_aggregation': getattr(config, 'ew_sell_aggregation', 'fraction')
            },
            'backtest_config': {
                'initial_capital': config.initial_capital,
                'commission': config.commission,
                'slippage': getattr(config, 'slippage', None),
                'confidence_threshold': config.confidence_threshold,
                'max_position_size': config.max_position_size,
                'threshold_strategy': getattr(config, 'threshold_strategy', 'absolute'),
                'threshold_percentile': getattr(config, 'threshold_percentile', 0.7),
                'threshold_window': getattr(config, 'threshold_window', 60),
                'auto_threshold_fallback': getattr(config, 'auto_threshold_fallback', True),
                'fallback_percentile': getattr(config, 'fallback_percentile', 0.7),
                'fallback_window': getattr(config, 'fallback_window', 60),
                'buy_consecutive_days': getattr(config, 'buy_consecutive_days', 1),
                'sell_consecutive_days': getattr(config, 'sell_consecutive_days', 1),
                'hysteresis_margin': getattr(config, 'hysteresis_margin', 0.0),
                'trade_cooldown_days': getattr(config, 'trade_cooldown_days', 0),
                'min_holding_days': getattr(config, 'min_holding_days', 0)
            },
            'technical_indicators': config.technical_indicators,
            'enabled_features': config.enabled_features,
            'name': config.name,  # Pass experiment name for special handling
            'analysis': config.analysis,
            'visualization': config.visualization
        }
        
        framework = TradingFramework(config_overrides=config_overrides)
        
        # Run experiment
        print("⚙️  Starting experiment...")
        results = framework.run_experiment(
            symbols=config.symbols,
            start_date=config.start_date,
            end_date=config.end_date,
            experiment_name=config.name
        )
        
        # Close framework
        framework.close()
        
        # If the framework returned an error, report and exit early
        if results is None or ('error' in results):
            print("\n" + "=" * 60)
            print("❌ EXPERIMENT FAILED")
            print("=" * 60)
            if results and 'error' in results:
                print(f"   Error: {results['error']}")
            print(f"\n💾 Partial results saved to: results/{config_name}/ (if any)")
            return results
        
        # Print summary
        print("\n" + "=" * 60)
        print("✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"\n📈 Key Results:")
        
        # Extract backtest results from the experiment results structure
        if 'backtest_results' in results:
            backtest_results = results['backtest_results']
            
            total_trades = 0
            avg_return = 0
            avg_sharpe = 0
            worst_drawdown = 0
            avg_volatility = 0
            avg_win_rate = 0
            symbol_count = 0
            
            print(f"   Individual Symbol Results:")
            for symbol, bt_results in backtest_results.items():
                if 'error' not in bt_results:
                    total_return = bt_results.get('total_return', 0)
                    sharpe = bt_results.get('sharpe_ratio', 0)
                    drawdown = bt_results.get('max_drawdown', 0)
                    volatility = bt_results.get('volatility', 0)
                    win_rate = bt_results.get('win_rate', 0)
                    trades = bt_results.get('total_trades', 0)
                    
                    print(f"     {symbol}: {total_return:.2%} return, {sharpe:.2f} Sharpe, {trades} trades")
                    
                    total_trades += trades
                    avg_return += total_return
                    avg_sharpe += sharpe
                    avg_volatility += volatility
                    avg_win_rate += win_rate
                    worst_drawdown = min(worst_drawdown, drawdown)
                    symbol_count += 1
            
            if symbol_count > 0:
                avg_return /= symbol_count
                avg_sharpe /= symbol_count
                avg_volatility /= symbol_count
                avg_win_rate /= symbol_count
                
                print(f"   Average Return: {avg_return:.2%}")
                print(f"   Average Sharpe: {avg_sharpe:.2f}")
                print(f"   Average Volatility: {avg_volatility:.2%}")
                print(f"   Worst Drawdown: {worst_drawdown:.2%}")
                print(f"   Average Win Rate: {avg_win_rate:.2%}")
                print(f"   Total Trades: {total_trades}")
            else:
                print(f"   No valid backtest results found")
        else:
            print(f"   No backtest results found in experiment output")
        
        print(f"\n💾 Results saved to: results/{config_name}/")
        
        return results
        
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None

def compare_all_experiments(verbose: bool = False):
    """Compare all completed experiments in results/ folder"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("❌ No results directory found")
        return
    
    # Find all experiment directories with results
    experiment_dirs = []
    for item in os.listdir(results_dir):
        experiment_path = os.path.join(results_dir, item)
        if os.path.isdir(experiment_path) and item != "configs":
            # Check if it has experiment results
            results_file = os.path.join(experiment_path, "experiment_results.json")
            if os.path.exists(results_file):
                experiment_dirs.append(item)
    
    if len(experiment_dirs) < 2:
        print(f"❌ Found only {len(experiment_dirs)} completed experiments. Need at least 2 to compare.")
        print("💡 Run some experiments first:")
        print("   python vgmini.py dual_signal")
        print("   python vgmini.py dual_signal_v2")
        return
    
    print(f"🔬 Comparing {len(experiment_dirs)} completed experiments:")
    for exp in sorted(experiment_dirs):
        print(f"   • {exp}")
    print("=" * 60)
    
    # Load results from each experiment
    results = {}
    for exp_name in experiment_dirs:
        results_file = os.path.join(results_dir, exp_name, "experiment_results.json")
        try:
            with open(results_file, 'r') as f:
                results[exp_name] = json.load(f)
            print(f"✅ Loaded results for {exp_name}")
        except Exception as e:
            print(f"❌ Failed to load results for {exp_name}: {e}")
            continue
    
    if len(results) < 2:
        print("❌ Failed to load enough experiment results for comparison")
        return
    
    # Perform comparison
    compare_loaded_results(results, verbose)

def compare_experiments(config_names: List[str], verbose: bool = False):
    """Compare multiple experiments by running them"""
    if len(config_names) < 2:
        print("❌ Need at least 2 experiments to compare")
        return
    
    print(f"🔬 Comparing {len(config_names)} experiments:")
    for name in config_names:
        print(f"   • {name}")
    print("=" * 60)
    
    # Run all experiments
    results = {}
    for config_name in config_names:
        print(f"\n🏃 Running {config_name}...")
        result = run_experiment(config_name, verbose=False)
        if result is None:
            print(f"❌ Failed to run {config_name}")
            return
        results[config_name] = result
    
    # Perform comparison
    compare_loaded_results(results, verbose)
    
    return results

def compare_loaded_results(results: Dict[str, Any], verbose: bool = False):
    """Compare loaded experiment results"""
    config_names = list(results.keys())
    
    # Compare results
    print(f"\n{'='*80}")
    print("📊 EXPERIMENT COMPARISON")
    print(f"{'='*80}")
    
    # Metrics to compare
    metrics = [
        ('total_return', 'Total Return', '%'),
        ('sharpe_ratio', 'Sharpe Ratio', ''),
        ('max_drawdown', 'Max Drawdown', '%'),
        ('volatility', 'Volatility', '%'),
        ('total_trades', 'Total Trades', ''),
        ('win_rate', 'Win Rate', '%')
    ]
    
    # Create comparison table
    print(f"\n{'Metric':<20}", end='')
    for name in config_names:
        print(f"{name:<15}", end='')
    print("Best")
    print("-" * (20 + 15 * len(config_names) + 10))
    
    comparison_data = []
    
    for metric, display_name, unit in metrics:
        print(f"{display_name:<20}", end='')
        
        values = []
        for name in config_names:
            # Extract metrics from backtest_results structure
            value = 0
            if 'backtest_results' in results[name]:
                backtest_results = results[name]['backtest_results']
                total_value = 0
                count = 0
                
                for symbol, bt_results in backtest_results.items():
                    symbol_value = bt_results.get(metric, 0)
                    
                    if metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility', 'win_rate']:
                        total_value += symbol_value
                        count += 1
                    elif metric == 'total_trades':
                        # Use the total_trades field directly
                        total_value += bt_results.get('total_trades', 0)
                        count = 1  # Don't average trades, sum them
                
                if count > 0:
                    if metric == 'total_trades':
                        value = total_value  # Sum of trades
                    else:
                        value = total_value / count  # Average of other metrics
            
            values.append(value)
            
            if unit == '%':
                print(f"{value:.2%}".ljust(15), end='')
            elif metric == 'sharpe_ratio':
                print(f"{value:.3f}".ljust(15), end='')
            else:
                print(f"{value:.0f}".ljust(15), end='')
        
        # Determine best (lowest for drawdown and volatility, highest for others)
        if metric in ['max_drawdown', 'volatility']:
            # For risk metrics, lower absolute values are better
            abs_values = [abs(v) for v in values]
            best_idx = abs_values.index(min(abs_values)) if any(v != 0 for v in abs_values) else 0
        else:
            best_idx = values.index(max(values)) if any(v != 0 for v in values) else 0
        
        print(f"{config_names[best_idx]}")
        
        comparison_data.append({
            'metric': metric,
            'values': dict(zip(config_names, values)),
            'best': config_names[best_idx]
        })
    
    # Feature analysis
    print(f"\n{'='*80}")
    print("🔍 FEATURE ANALYSIS")
    print(f"{'='*80}")
    
    for config_name in config_names:
        # Try to get feature importance from model_results or shap_analysis
        buy_features = None
        # Label equal-weight model if detected
        model_type_label = results[config_name].get('config', {}).get('model_config', {}).get('model_type', '')
        if str(model_type_label).lower() == 'equal_weight':
            print(f"\nℹ️  {config_name} uses Equal-Weight model (uniform coefficients by design; SHAP/LIME may vary).")
        
        if 'model_results' in results[config_name] and 'buy_model' in results[config_name]['model_results']:
            buy_features = results[config_name]['model_results']['buy_model'].get('feature_importance', {})
        elif 'shap_analysis' in results[config_name] and 'comprehensive_analysis' in results[config_name]['shap_analysis']:
            comp_analysis = results[config_name]['shap_analysis']['comprehensive_analysis']
            if 'buy_model' in comp_analysis and 'linear_coefficients' in comp_analysis['buy_model']:
                buy_features = comp_analysis['buy_model']['linear_coefficients']
        
        if buy_features:
            print(f"\n📋 {config_name} - Top 5 Buy Model Features:")
            # Filter out zero values and RSI features for dual_signal_v2
            filtered_features = {}
            for feature, importance in buy_features.items():
                if abs(importance) > 1e-6:  # Filter out essentially zero values
                    # For dual_signal_v2, exclude RSI features
                    if config_name == 'dual_signal_v2' and 'rsi' in feature.lower():
                        continue
                    filtered_features[feature] = importance
            
            buy_sorted = sorted(filtered_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            
            for i, (feature, importance) in enumerate(buy_sorted):
                direction = "📈" if importance > 0 else "📉"
                print(f"   {i+1}. {feature}: {importance:.4f} {direction}")
        else:
            print(f"\n📋 {config_name} - No feature importance data available")
    
    # Create visualization
    create_comparison_visualization(results, config_names, comparison_data)
    
    # Summary
    print(f"\n{'='*80}")
    print("🎯 SUMMARY")
    print(f"{'='*80}")
    
    # Count wins for each experiment
    wins = {name: 0 for name in config_names}
    for data in comparison_data:
        wins[data['best']] += 1
    
    print(f"\n🏆 Performance Ranking:")
    sorted_wins = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    for i, (name, win_count) in enumerate(sorted_wins):
        print(f"   {i+1}. {name}: {win_count}/{len(metrics)} metrics won")
    
    # Write comparison summary to results/summary/ (CSV + HTML)
    import os, csv
    summary_dir = os.path.join('results', 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    
    # CSV with metrics table
    csv_path = os.path.join(summary_dir, 'comparison_summary.csv')
    fieldnames = ['metric'] + config_names + ['best']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparison_data:
            out = {'metric': row['metric']}
            out.update(row['values'])
            out['best'] = row['best']
            writer.writerow(out)
    
    # Simple HTML table
    html_path = os.path.join(summary_dir, 'comparison_summary.html')
    try:
        rows = []
        rows.append('<html><head><meta charset="utf-8"><style>table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:6px}</style></head><body>')
        rows.append(f'<h2>Experiment Comparison</h2>')
        rows.append('<table>')
        header = '<tr>' + ''.join([f'<th>{h}</th>' for h in fieldnames]) + '</tr>'
        rows.append(header)
        for row in comparison_data:
            tds = [f'<td>{row["metric"]}</td>']
            for name in config_names:
                val = row['values'][name]
                cell = f"{val:.4f}" if isinstance(val, (int, float)) else f"{val}"
                tds.append(f'<td>{cell}</td>')
            tds.append(f'<td>{row["best"]}</td>')
            rows.append('<tr>' + ''.join(tds) + '</tr>')
        rows.append('</table>')
        rows.append('</body></html>')
        with open(html_path, 'w') as f:
            f.write('\n'.join(rows))
        print(f"\n📑 Comparison summaries saved to: {summary_dir}")
    except Exception as e:
        print(f"⚠️  Failed to write HTML summary: {e}")

def create_comparison_visualization(results: Dict[str, Any], config_names: List[str], comparison_data: List[Dict]):
    """Create comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Experiment Comparison: {" vs ".join(config_names)}', fontsize=16, fontweight='bold')
    
    # 1. Performance metrics
    ax1 = axes[0, 0]
    metrics = ['total_return', 'sharpe_ratio', 'win_rate']
    x = np.arange(len(metrics))
    width = 0.8 / len(config_names)
    
    for i, name in enumerate(config_names):
        values = [next(d['values'][name] for d in comparison_data if d['metric'] == m) for m in metrics]
        ax1.bar(x + i * width, values, width, label=name, alpha=0.8)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Value')
    ax1.set_title('Key Performance Metrics')
    ax1.set_xticks(x + width * (len(config_names) - 1) / 2)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Portfolio comparison (if available)
    ax2 = axes[0, 1]
    portfolio_plotted = False
    for name in config_names:
        if 'portfolio_history' in results[name]:
            portfolio = results[name]['portfolio_history']
            ax2.plot(portfolio['date'], portfolio['total_value'], label=name, linewidth=2)
            portfolio_plotted = True
    
    if portfolio_plotted:
        ax2.set_title('Portfolio Value Over Time')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Portfolio history not available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Portfolio Value Over Time')
    
    # 3. Risk metrics
    ax3 = axes[1, 0]
    risk_metrics = ['max_drawdown', 'volatility']
    x_risk = np.arange(len(risk_metrics))
    
    for i, name in enumerate(config_names):
        values = [abs(next(d['values'][name] for d in comparison_data if d['metric'] == m)) for m in risk_metrics]
        ax3.bar(x_risk + i * width, values, width, label=name, alpha=0.8)
    
    ax3.set_xlabel('Risk Metrics')
    ax3.set_ylabel('Value (absolute)')
    ax3.set_title('Risk Metrics Comparison')
    ax3.set_xticks(x_risk + width * (len(config_names) - 1) / 2)
    ax3.set_xticklabels(risk_metrics)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Trade statistics
    ax4 = axes[1, 1]
    trade_values = [next(d['values'][name] for d in comparison_data if d['metric'] == 'total_trades') for name in config_names]
    
    ax4.bar(config_names, trade_values, alpha=0.8)
    ax4.set_xlabel('Experiments')
    ax4.set_ylabel('Number of Trades')
    ax4.set_title('Trading Activity Comparison')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot (overwrite previous)
    comparison_path = "results/latest_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to: {comparison_path}")

def train_model(model_name: str, verbose: bool = False):
    """Train a model using the specified experiment configuration"""
    print(f"🏋️  Training model: {model_name}")
    print("=" * 60)
    
    # Load YAML config only
    yaml_config_path = f"configs/experiments/{model_name}.yaml"
    if not os.path.exists(yaml_config_path):
        print(f"❌ YAML config not found: {yaml_config_path}")
        print("💡 Use 'vgmini.py list' to see available experiments")
        return
    
    try:
        from src.yaml_config import load_merged_experiment_yaml
        from src.experiment_configs import ExperimentConfig
        from src.main_framework import TradingFramework
        
        config_data = load_merged_experiment_yaml(model_name)
        if not config_data:
            print(f"❌ Failed to load config for {model_name}")
            return
            
        config = ExperimentConfig.from_dict(config_data)
        
        if verbose:
            print(f"📊 Training Configuration:")
            print(f"   Model Type: {config.model_type}")
            print(f"   Symbols: {len(config.symbols)} symbols")
            print(f"   Date Range: {config.start_date} to {config.end_date}")
            print(f"   Features: {len(config.enabled_features)} enabled")
        
        # Initialize framework for training only
        config_overrides = {
            'model_config': {
                'buy_threshold': config.buy_threshold,
                'sell_threshold': config.sell_threshold,
                'forecast_horizon': config.forecast_horizon,
                'model_type': config.model_type,
                # Add other necessary config overrides...
            },
            'enabled_features': config.enabled_features,
            'name': config.name,
        }
        
        framework = TradingFramework(config_overrides=config_overrides)
        
        print("⚙️  Training and testing model...")
        # Run full experiment (training + testing)
        results = framework.run_experiment(
            symbols=config.symbols,
            start_date=config.start_date,
            end_date=config.end_date,
            experiment_name=config.name
        )
        
        framework.close()
        
        if results and 'error' not in results:
            print("✅ Model training completed successfully!")
            print(f"💾 Model saved to: results/{model_name}/models/trained_model.pkl")
        else:
            print("❌ Model training failed")
            
    except Exception as e:
        print(f"❌ Error training model: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


def rank_symbols(model_name: str, top_n: int = 10, verbose: bool = False, threshold: float = None, sort_by: str = 'er', downside_k: float = 0.7, quiet: bool = False):
    """Rank all database symbols by buy potential using a previously trained model.
    
    Expects a trained model at results/{model_name.lower()}/models/trained_model.pkl
    and uses the same feature configuration as the experiment to compute features
    up to the latest available date per symbol.
    """
    print(f"🎯 Ranking symbols using {model_name} model")
    print("=" * 60)

    import json
    from datetime import timedelta

    # Resolve paths and config
    exp_name = model_name.lower()
    model_path = f"results/{exp_name}/models/trained_model.pkl"
    config_path = f"results/configs/{exp_name}.json"

    if not os.path.exists(model_path):
        print(f"❌ Trained model not found: {model_path}")
        print(f"💡 Run the experiment first: python vgmini.py {exp_name}")
        return

    # Load experiment config if available (to match training-time feature settings)
    exp_config = None
    
    # Load YAML experiment config only
    yaml_config_path = f"configs/experiments/{exp_name}.yaml"
    if os.path.exists(yaml_config_path):
        try:
            from src.yaml_config import load_merged_experiment_yaml
            exp_config = load_merged_experiment_yaml(exp_name)
        except Exception as e:
            if verbose:
                print(f"⚠️  Failed to load YAML experiment config ({yaml_config_path}): {e}")
    else:
        print(f"❌ YAML config not found: {yaml_config_path}")
        print("💡 Use 'vgmini.py list' to see available experiments")
        return

    try:
        import numpy as np
        import pandas as pd
        from src.data_loader import StockDataLoader
        from src.feature_engineering import FeatureEngineer
        from src.ml_models import StockSignalPredictor
        from src.config import model_config as global_model_config
        import logging, time

        # Reduce noisy INFO logs from data loader during ranking
        logging.getLogger('src.data_loader').setLevel(logging.WARNING)
        logging.getLogger('sqlalchemy').setLevel(logging.WARNING)

        # Sync key model config parameters with experiment for feature calc (EMA, volume window, horizons)
        if exp_config:
            if 'ema_periods' in exp_config:
                global_model_config.ema_periods = exp_config['ema_periods']
            if 'volume_window' in exp_config:
                global_model_config.volume_window = exp_config['volume_window']
            if 'forecast_horizon' in exp_config:
                global_model_config.forecast_horizon = exp_config['forecast_horizon']
            if 'buy_threshold' in exp_config:
                global_model_config.buy_threshold = exp_config['buy_threshold']
            if 'sell_threshold' in exp_config:
                global_model_config.sell_threshold = exp_config['sell_threshold']

        # Load model
        predictor = StockSignalPredictor()
        predictor.load_models(model_path)
        print(f"✅ Loaded model and scaler from {model_path}")

        # Determine threshold to use
        if threshold is None:
            if isinstance(exp_config, dict) and 'confidence_threshold' in exp_config:
                threshold = float(exp_config['confidence_threshold'])
            else:
                print(f"❌ No confidence_threshold found in experiment config for {model_name}")
                print(f"💡 Add 'confidence_threshold' to the model config or pass --threshold explicitly")
                print(f"💡 For equal_weight models, try values like 0.3-0.4")
                print(f"💡 For logistic models, try values like 0.5-0.7")
                return

        # Load universe of symbols from DB
        loader = StockDataLoader()
        symbols_df = loader.get_symbols()
        symbols_to_rank = symbols_df['symbol'].tolist()
        if not symbols_to_rank:
            print("❌ No symbols found in the database")
            return

        # Determine a reasonable lookback window for feature calc
        # Use max EMA + 2x horizon + buffer
        max_ema = max(getattr(global_model_config, 'ema_periods', [9]))
        horizon = int(getattr(global_model_config, 'forecast_horizon', 5))
        lookback_days = int(max(260, max_ema + 2 * horizon + 50))

        # Determine end date from DB summary for daily interval only
        summary = loader.get_data_summary(interval='1d')
        end_ts = summary.get('date_range', {}).get('end_date', None)
        if end_ts is None:
            print("❌ Could not determine latest date from database")
            return
        end_date = pd.to_datetime(end_ts).normalize()
        start_date = (end_date - pd.Timedelta(days=lookback_days)).date().isoformat()
        end_date_str = end_date.date().isoformat()

        # Feature engineer with experiment settings (for RSI/MACD/etc)
        # Pass the raw experiment dict so _get_config_value picks overrides
        fe = FeatureEngineer(config=exp_config if isinstance(exp_config, dict) else None)

        symbol_scores = []
        processed = 0
        successful_predictions = 0

        total = len(symbols_to_rank)
        print(f"📊 Starting analysis for {total} symbols (lookback ~{lookback_days} days ending {end_date_str})...")
        start_time = time.time()

        def print_progress():
            import os, sys
            plain = os.getenv('VGM_PLAIN_PROGRESS', '') == '1' or not sys.stdout.isatty()
            elapsed = time.time() - start_time
            eta = (elapsed / processed * (total - processed)) if processed else 0
            if plain:
                # Throttle plain prints to avoid spam
                step = max(1, total // 50)
                if (processed % step == 0) or (processed == total):
                    print(f"[{processed}/{total}] ok {successful_predictions} | ETA {int(eta)}s")
            else:
                bar_len = 30
                done = int(bar_len * processed / total) if total else bar_len
                bar = '#' * done + '.' * (bar_len - done)
                print(f"\r⏳ [{bar}] {processed}/{total} | ok {successful_predictions} | ETA {int(eta)}s", end='', flush=True)

        for symbol in symbols_to_rank:
            try:
                df = loader.get_symbol_data(symbol, start_date=start_date, end_date=end_date_str)
                if df is None or df.empty:
                    if verbose:
                        print(f"⚠️  {symbol}: No data in range {start_date}..{end_date_str}")
                    continue

                # Ensure chronological order
                df = df.sort_values('datetime').reset_index(drop=True)

                # Compute features WITHOUT creating targets or dropping the last horizon rows
                fdf = df.copy()
                # Core indicators
                fdf = fe.calculate_ema(fdf, global_model_config.ema_periods)
                fdf = fe.calculate_macd(fdf)
                fdf = fe.calculate_heikin_ashi(fdf)
                fdf = fe.calculate_volume_indicators(fdf)
                fdf = fe.calculate_additional_indicators(fdf)
                fdf = fe.create_crossover_signals(fdf)

                # Take the last row as the current state
                last_row = fdf.iloc[[-1]]  # keep DataFrame shape

                # Build feature vector in the EXACT order used by the model
                feature_names = predictor.feature_names
                missing = [c for c in feature_names if c not in last_row.columns]
                if missing:
                    # If some features are missing (config mismatch), fill with 0s
                    for c in missing:
                        last_row[c] = 0.0
                    # Reorder after adding
                    last_row = last_row

                X = last_row[feature_names].fillna(0).values

                # Predict signals
                signals = predictor.predict_signals(X, confidence_threshold=threshold)
                buy_proba = float(signals['buy_probability'][0])
                sell_proba = float(signals['sell_probability'][0])
                buy_signal = int(signals['buy_signals'][0])
                margin = buy_proba - float(threshold)

                # Expected growth proxy: p*target - (1-p)*k*downside_sigma
                # Use 60-bar downside stdev of daily returns as proxy
                recent = df.tail(60).copy() if len(df) >= 2 else df.copy()
                recent['ret'] = recent['close'].pct_change()
                downside = recent['ret'].clip(upper=0).abs()
                downside_sigma = float(downside.std(skipna=True) or 0.0)
                target_gain = float(getattr(global_model_config, 'buy_threshold', 0.01))
                er = buy_proba * target_gain - (1.0 - buy_proba) * downside_k * downside_sigma

                symbol_scores.append({
                    'symbol': symbol,
                    'buy_score': buy_proba,
                    'sell_score': sell_proba,
                    'expected_return': er,
                    'downside_sigma': downside_sigma,
                    'margin': margin,
                    'buy_signal': buy_signal,
                    'latest_price': float(last_row['close'].iloc[0]),
                    'latest_date': str(last_row['datetime'].iloc[0].date()) if 'datetime' in last_row else end_date_str
                })
                successful_predictions += 1

                if verbose:
                    icon = "🚀" if buy_signal else "⏸️"
                    print(f"{icon} {symbol}: score={buy_proba:.3f} price=${float(last_row['close'].iloc[0]):.2f}")

            except Exception as e:
                if verbose:
                    print(f"❌ {symbol}: {e}")
                continue
            finally:
                processed += 1
                print_progress()

        # Finish progress line
        print()

        if not symbol_scores:
            print("❌ No symbols could be analyzed successfully")
            return

        # Sort and display top N
        if sort_by.lower() in ['er', 'expected_return']:
            symbol_scores.sort(key=lambda x: x.get('expected_return', 0.0), reverse=True)
        else:
            symbol_scores.sort(key=lambda x: x['buy_score'], reverse=True)
        top = min(top_n, len(symbol_scores))

        # Prepare a consolidated text output for file
        lines = []
        sort_desc = 'Expected Return' if sort_by.lower() in ['er','expected_return'] else 'Buy Score'
        header_main = f"\n🏆 TOP {top} BUY OPPORTUNITIES (threshold={threshold:.2f}, sorted by {sort_desc})"
        lines.append(header_main)
        lines.append("=" * 90)
        lines.append(f"{'Rank':<4} {'Symbol':<8} {'Score':<8} {'Margin':<8} {'Signal':<8} {'Price':<10} {'Date':<12}")
        lines.append("-" * 90)

        if not quiet:
            print(header_main)
            print("=" * 90)
            print(f"{'Rank':<4} {'Symbol':<8} {'Score':<8} {'Margin':<8} {'Signal':<8} {'Price':<10} {'Date':<12}")
            print("-" * 90)

            for i, data in enumerate(symbol_scores[:top]):
                signal_text = "BUY" if data['buy_signal'] else "HOLD"
                signal_icon = "🚀" if data['buy_signal'] else "⏸️"
                line = f"{i+1:<4} {data['symbol']:<8} {data['buy_score']:.3f}    {data['margin']:.3f}   {signal_icon} {signal_text:<6} ${data['latest_price']:<8.2f} {data['latest_date']}"
                print(line)
                lines.append(line)

            # Additional table: only symbols priced > $20
            filtered_scores = [s for s in symbol_scores if s['latest_price'] > 20]
            if filtered_scores:
                # Already sorted by buy_score above, but ensure ordering after filtering
                filtered_scores.sort(key=lambda x: x['buy_score'], reverse=True)
                topf = min(top_n, len(filtered_scores))
                header_f = f"\n🏆 TOP {topf} BUY OPPORTUNITIES (Price > $20, threshold={threshold:.2f})"
                print(header_f)
                print("=" * 90)
                print(f"{'Rank':<4} {'Symbol':<8} {'Score':<8} {'Margin':<8} {'Signal':<8} {'Price':<10} {'Date':<12}")
                print("-" * 90)
                lines.append(header_f)
                lines.append("=" * 90)
                lines.append(f"{'Rank':<4} {'Symbol':<8} {'Score':<8} {'Margin':<8} {'Signal':<8} {'Price':<10} {'Date':<12}")
                lines.append("-" * 90)
                for i, data in enumerate(filtered_scores[:topf]):
                    signal_text = "BUY" if data['buy_signal'] else "HOLD"
                    signal_icon = "🚀" if data['buy_signal'] else "⏸️"
                    line = f"{i+1:<4} {data['symbol']:<8} {data['buy_score']:.3f}    {data['margin']:.3f}   {signal_icon} {signal_text:<6} ${data['latest_price']:<8.2f} {data['latest_date']}"
                    print(line)
                    lines.append(line)
        else:
            # In quiet mode, just add the data to lines for file output
            for i, data in enumerate(symbol_scores[:top]):
                signal_text = "BUY" if data['buy_signal'] else "HOLD"
                signal_icon = "🚀" if data['buy_signal'] else "⏸️"
                line = f"{i+1:<4} {data['symbol']:<8} {data['buy_score']:.3f}    {data['margin']:.3f}   {signal_icon} {signal_text:<6} ${data['latest_price']:<8.2f} {data['latest_date']}"
                lines.append(line)

        # Add full ranked list of ALL symbols to the file output (not printed to console)
        lines.append("")
        sort_desc = 'Expected Return' if sort_by.lower() in ['er','expected_return'] else 'Buy Score'
        lines.append(f"📋 ALL SYMBOLS (sorted by {sort_desc}, threshold={threshold:.2f})")
        lines.append("=" * 90)
        lines.append(f"{'Rank':<4} {'Symbol':<8} {'Score':<8} {'Margin':<8} {'Signal':<8} {'Price':<10} {'Date':<12}")
        lines.append("-" * 90)
        for i, data in enumerate(symbol_scores):
            signal_text = "BUY" if data['buy_signal'] else "HOLD"
            signal_icon = "🚀" if data['buy_signal'] else "⏸️"
            line = f"{i+1:<4} {data['symbol']:<8} {data['buy_score']:.3f}    {data['margin']:.3f}   {signal_icon} {signal_text:<6} ${data['latest_price']:<8.2f} {data['latest_date']}"
            lines.append(line)

        print("\n📊 Summary:")
        print(f"   Symbols in DB: {len(symbols_to_rank)}")
        print(f"   Processed: {processed}")
        print(f"   Successful predictions: {successful_predictions}")
        print(f"   Buy signals flagged: {sum(1 for s in symbol_scores if s['buy_signal'])}")
        print(f"   Model: {model_name}")

        # Save consolidated output to file in project root
        # Write to results directory to keep vgmini root clean
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, f"{exp_name}_ranking.txt")
        try:
            with open(out_path, 'w') as f:
                f.write("\n".join(lines))
                f.write("\n\n📊 Summary:\n")
                f.write(f"   Symbols in DB: {len(symbols_to_rank)}\n")
                f.write(f"   Processed: {processed}\n")
                f.write(f"   Successful predictions: {successful_predictions}\n")
                f.write(f"   Buy signals flagged: {sum(1 for s in symbol_scores if s['buy_signal'])}\n")
                f.write(f"   Model: {model_name}\n")
                f.write(f"   Threshold: {threshold:.2f}\n")
            print(f"\n💾 Ranking saved to: {out_path}")
        except Exception as fe:
            print(f"⚠️  Failed to write ranking file: {fe}")

    except Exception as e:
        print(f"❌ Error during ranking: {e}")
        if verbose:
            import traceback
            traceback.print_exc()

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="VGMini Trading Framework - Professional Trading Strategy Backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vgmini.py list                              # List available configurations
  vgmini.py train --model equal_weight_dual   # Train and test model
  vgmini.py rank equal_weight_dual            # Rank symbols using trained model
  vgmini.py pipeline --model equal_weight_dual # Run complete pipeline
  vgmini.py equal_weight_dual                 # Run full experiment (train+test)
  vgmini.py compare                           # Compare all completed experiments
        """
    )
    
    parser.add_argument('command', nargs='?', help='Command: list, train, rank, pipeline, compare, or experiment name')
    parser.add_argument('model_or_arg', nargs='?', help='Model name or additional argument')
    parser.add_argument('--model', type=str, help='Model name for train/test/pipeline commands')
    parser.add_argument('--top', type=int, default=10, help='Number of top symbols (for rank/pipeline)')
    parser.add_argument('--threshold', type=float, help='Override confidence threshold for ranking')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress verbose ranking output (for pipeline use)')
    
    args = parser.parse_args()
    
    print("🎯 VGMini Trading Framework")
    print("=" * 40)
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'list':
        list_configs()
    
    elif args.command == 'train':
        model_name = args.model or args.model_or_arg
        if not model_name:
            print("❌ Model name required for training")
            print("💡 Example: vgmini.py train --model equal_weight_dual")
            return
        train_model(model_name, verbose=args.verbose)
    
    elif args.command == 'rank':
        model_name = args.model_or_arg
        if not model_name:
            print("❌ Model name required for ranking")
            print("💡 Example: vgmini.py rank equal_weight_dual")
            return
        rank_symbols(model_name, top_n=args.top, verbose=args.verbose, threshold=args.threshold, quiet=args.quiet)
    
    elif args.command == 'pipeline':
        model_name = args.model
        if not model_name:
            print("❌ Model name required for pipeline")
            print("💡 Example: vgmini.py pipeline --model equal_weight_dual")
            return
        run_pipeline(model_name)
    
    elif args.command == 'compare':
        compare_all_experiments(verbose=args.verbose)
    
    else:
        # Treat as experiment name (run full experiment: train + test)
        run_experiment(args.command, verbose=args.verbose)

if __name__ == "__main__":
    main()