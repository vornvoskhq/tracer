# VGMini Call Graph Generator

This script analyzes the VGMini codebase and generates a call graph starting from the entry point (`./vg` → `vgmini.py`) and traces all function calls with their filenames.

## Features

- **Entry Point Analysis**: Starts from `target/vgmini/vgmini.py` (the main entry point)
- **Function Discovery**: Finds all function definitions across the codebase
- **Call Tracking**: Traces function calls within and between modules
- **Import Resolution**: Resolves internal project imports (src.* modules)
- **Multiple Output Formats**: Text, JSON, and Graphviz DOT formats

## Usage

### Basic Usage (Text Output)
```bash
python call_graph_generator.py
```

### JSON Output
```bash
python call_graph_generator.py --format json --output call_graph.json
```

### Graphviz DOT Output (for visualization)
```bash
python call_graph_generator.py --format dot --output call_graph.dot
```

### Custom Root Directory
```bash
python call_graph_generator.py --root /path/to/project
```

## Output Formats

### Text Format
Human-readable format showing:
- Summary statistics
- Entry point analysis
- File-by-file breakdown of functions
- Function call relationships
- Project imports

### JSON Format
Machine-readable format containing:
- Metadata (files analyzed, function counts)
- Complete file structure with functions and imports
- Call graph relationships

### DOT Format
Graphviz format for creating visual call graphs. Convert to image with:
```bash
dot -Tpng call_graph.dot -o call_graph.png
```

## How It Works

1. **Entry Point**: Starts analysis from `target/vgmini/vgmini.py`
2. **AST Parsing**: Uses Python's `ast` module to parse source files
3. **Import Resolution**: Follows internal imports (src.* modules) to discover more files
4. **Function Extraction**: Identifies all function definitions and their locations
5. **Call Analysis**: Tracks function calls within each function scope
6. **Graph Building**: Builds a complete call graph representation

## Key Components Analyzed

The script analyzes all Python files in the VGMini project, including:

- **Entry Point**: `vgmini.py` - Main CLI interface
- **Core Framework**: `src/main_framework.py` - Trading framework
- **Data Components**: 
  - `src/data_loader.py` - Data loading and DB access
  - `src/feature_engineering.py` - Feature calculation
- **ML Components**:
  - `src/ml_models.py` - Machine learning models
  - `src/model_interpretability.py` - SHAP/LIME analysis
- **Trading Components**:
  - `src/backtesting.py` - Backtesting engine
  - `src/visualization.py` - Chart generation
- **Analysis Tools**:
  - `src/decorrelate.py` - Symbol decorrelation
  - `src/trailstop.py` - Trail stop analysis
  - `src/limiter.py` - Swing high/low analysis
- **Configuration**: `src/config.py` - System configuration

## Example Output

```
================================================================================
VGMini Call Graph Analysis
================================================================================

📊 Summary:
   Files analyzed: 15
   Total functions: 127
   Total function calls: 423

🚀 Entry Point: target/vgmini/vgmini.py
   Main functions:
     • main() [line 1344]
     • run_experiment() [line 347]
     • run_pipeline() [line 162]

📁 Files and Functions:
--------------------------------------------------

📄 target/vgmini/vgmini.py
   Functions:
     • main() [line 1344]
       Calls:
         → argparse.ArgumentParser
         → check_virtual_environment
     • run_experiment() [line 347]
       Calls:
         → TradingFramework
         → ExperimentConfig.from_dict
```

## Limitations

- **External Libraries**: Only tracks internal project calls, not external library calls
- **Dynamic Calls**: Cannot track dynamically generated function calls (eval, getattr, etc.)
- **String-based Imports**: May miss some import patterns using string manipulation

## Requirements

- Python 3.6+
- Only uses standard library modules (ast, pathlib, collections, etc.)
- No external dependencies required