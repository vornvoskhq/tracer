# VGMini Trading Framework

A professional trading framework for backtesting and deploying algorithmic trading strategies.

## Features

- Backtesting engine with multiple metrics
- Machine learning model integration
- Performance visualization
- Risk management tools

## Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Usage

```bash
# List YAML experiments
python vgmini.py list

# Run Sprint 1 baseline (equal-weight dual)
python vgmini.py equal_weight_dual

# Run Sprint 2 examples (buy-only)
python vgmini.py logistic_buy_only
python vgmini.py xgboost_buy_only

# Compare completed experiments (from results/*/experiment_results.json)
python vgmini.py compare
```
