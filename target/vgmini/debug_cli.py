#!/usr/bin/env python3
"""
Debug CLI wrapper for stock-chart2 integration.
Redirects calls to the main infer_cli.py functionality, filtering out unsupported arguments.
"""

import sys
import subprocess
import os

def filter_args(args):
    """Filter out arguments that infer_cli.py doesn't support"""
    filtered = []
    skip_next = False
    
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
            
        # Skip unsupported arguments from stock-chart2
        if arg in ['--buy-threshold', '--sell-threshold']:
            skip_next = True  # Skip the argument and its value
            continue
        
        # Keep supported arguments: --experiment, --symbol, --timeframe, --threshold
        filtered.append(arg)
    
    return filtered

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    infer_cli_path = os.path.join(script_dir, "infer_cli.py")
    
    # Filter arguments to only pass supported ones
    filtered_args = filter_args(sys.argv[1:])
    
    # Pass filtered arguments to infer_cli.py
    try:
        result = subprocess.run([sys.executable, infer_cli_path] + filtered_args, 
                              capture_output=False, text=True)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"Error calling infer_cli.py: {e}", file=sys.stderr)
        sys.exit(1)