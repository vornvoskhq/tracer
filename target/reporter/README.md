# Stock Analysis Reporter

A CLI tool that analyzes stocks from VGMini rankings using Large Language Models (LLMs) to provide qualitative investment analysis.

## Features

- **LLM-Powered Analysis**: Uses OpenRouter API to access various LLM models for stock analysis
- **VGMini Integration**: Automatically loads top-ranked stocks from VGMini outputs  
- **Configurable Prompts**: Multiple analysis templates (technical, sector, fundamental focus)
- **Cost Tracking**: Monitors LLM usage and estimated costs
- **Flexible Output**: Console display and file saving options
- **Batch Processing**: Analyze multiple top-ranked stocks at once

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API key**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenRouter API key
   ```

3. **Run analysis**:
   ```bash
   # Analyze a specific stock
   python reporter_cli.py AAPL
   
   # Analyze top 3 stocks from VGMini
   python reporter_cli.py --top 3
   
   # Use technical analysis prompt
   python reporter_cli.py TSLA --prompt technical_focus.txt
   ```

## Usage Examples

```bash
# List available prompts
python reporter_cli.py --list-prompts

# List top stocks from VGMini rankings  
python reporter_cli.py --list-stocks

# Analyze with specific model
python reporter_cli.py AAPL --model "openai/gpt-4o"

# Console only (no file saving)
python reporter_cli.py AAPL --no-save

# Batch analyze top 5 stocks  
python reporter_cli.py --top 5
```

## Configuration

Edit `config.yaml` to customize:

- **LLM Models**: Available models with short names, pricing, and recommendations
- **Length Templates**: Response length options with custom instructions  
- **Default Settings**: Model, prompt, length template, theme
- **Console Theme**: Normal or dystopian (post-soviet/slum aesthetic)
- **Paths**: VGMini integration paths, output directories

**Key Features from Tracer Integration:**
- ✅ **Short model names**: Use `claude-sonnet` instead of full API names
- ✅ **Cost tracking**: Real-time cost estimation for each analysis
- ✅ **Model selection**: Pre-configured low-cost options with recommendations
- ✅ **No CLI overrides**: All configuration managed through config.yaml

## Available Prompts

- `default_analysis.txt`: Dystopian financial analyst reflecting economic collapse
- `concise_brutal.txt`: Ultra-brief surveillance state format (brutal facts only)  
- `detailed_collapse.txt`: Comprehensive collapse documentation
- `sector_analysis.txt`: Sector-focused analysis  
- `technical_focus.txt`: Technical analysis emphasis

**Dystopian Theme**: All prompts emphasize using ONLY provided data (no fabrication) and reflect themes of economic collapse, mental health crisis, and post-soviet decay.

Create custom prompts in the `prompts/` directory using `{stock_data}` placeholder.

## Output

Reports are saved to `reports/SYMBOL.txt` with:
- Company overview and sector analysis
- Technical and fundamental insights  
- Investment outlook and risk assessment
- Timestamp and model information

## Integration Notes

Currently integrates with VGMini ranking data. Full technical/fundamental data integration is planned for future releases.

## Requirements

- Python 3.8+
- OpenRouter API key
- VGMini ranking outputs (optional for custom analysis)