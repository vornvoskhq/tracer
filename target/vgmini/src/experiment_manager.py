"""
Experiment management utilities for organizing and comparing multiple experiments
"""
import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
import shutil

class ExperimentManager:
    """Manages multiple experiments and provides comparison utilities"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        
    def list_experiments(self) -> pd.DataFrame:
        """List all experiments with key metrics"""
        experiments = []
        
        for exp_dir in self.results_dir.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
                exp_data = self._load_experiment_summary(exp_dir)
                if exp_data:
                    experiments.append(exp_data)
        
        if experiments:
            df = pd.DataFrame(experiments)
            return df.sort_values('run_date', ascending=False)
        else:
            return pd.DataFrame()
    
    def _load_experiment_summary(self, exp_dir: Path) -> Dict[str, Any]:
        """Load experiment summary from results file"""
        results_file = exp_dir / "experiment_results.json"
        
        if not results_file.exists():
            return None
            
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            # Extract key metrics
            summary = {
                'experiment_id': exp_dir.name,
                'experiment_name': data.get('experiment_name', 'unknown'),
                'run_date': data.get('timestamp', ''),
                'symbols': ', '.join(data.get('symbols', [])),
                'date_range': f"{data.get('start_date', '')} to {data.get('end_date', '')}",
            }
            
            # Extract performance metrics if available
            if 'backtest_results' in data and 'aggregate' in data['backtest_results']:
                agg = data['backtest_results']['aggregate']
                summary.update({
                    'total_return': agg.get('avg_total_return', 0),
                    'sharpe_ratio': agg.get('avg_sharpe_ratio', 0),
                    'max_drawdown': agg.get('avg_max_drawdown', 0),
                    'total_trades': agg.get('total_trades', 0),
                })
            
            # Extract model metrics if available
            if 'model_results' in data:
                model_res = data['model_results']
                summary.update({
                    'buy_model_auc': model_res.get('buy_model', {}).get('test_auc', 0),
                    'sell_model_auc': model_res.get('sell_model', {}).get('test_auc', 0),
                })
            
            return summary
            
        except Exception as e:
            print(f"Error loading experiment {exp_dir.name}: {e}")
            return None
    
    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiments side by side"""
        comparisons = []
        
        for exp_id in experiment_ids:
            exp_dir = self.results_dir / exp_id
            if exp_dir.exists():
                summary = self._load_experiment_summary(exp_dir)
                if summary:
                    comparisons.append(summary)
        
        if comparisons:
            df = pd.DataFrame(comparisons)
            # Reorder columns for better comparison
            key_cols = ['experiment_name', 'total_return', 'sharpe_ratio', 'max_drawdown', 
                       'total_trades', 'buy_model_auc', 'sell_model_auc']
            available_cols = [col for col in key_cols if col in df.columns]
            other_cols = [col for col in df.columns if col not in available_cols]
            return df[available_cols + other_cols]
        else:
            return pd.DataFrame()
    
    def get_best_experiment(self, metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """Get the best performing experiment by specified metric"""
        experiments_df = self.list_experiments()
        
        if experiments_df.empty:
            return None
        
        if metric not in experiments_df.columns:
            print(f"Metric '{metric}' not found. Available metrics: {list(experiments_df.columns)}")
            return None
        
        best_idx = experiments_df[metric].idxmax()
        return experiments_df.loc[best_idx].to_dict()
    
    def archive_experiment(self, experiment_id: str, archive_dir: str = "archived_experiments"):
        """Archive an experiment to free up space"""
        exp_dir = self.results_dir / experiment_id
        archive_path = self.results_dir / archive_dir
        
        if not exp_dir.exists():
            print(f"Experiment {experiment_id} not found")
            return False
        
        archive_path.mkdir(exist_ok=True)
        destination = archive_path / experiment_id
        
        shutil.move(str(exp_dir), str(destination))
        print(f"Experiment {experiment_id} archived to {destination}")
        return True
    
    def delete_experiment(self, experiment_id: str, confirm: bool = False):
        """Delete an experiment permanently"""
        exp_dir = self.results_dir / experiment_id
        
        if not exp_dir.exists():
            print(f"Experiment {experiment_id} not found")
            return False
        
        if not confirm:
            print(f"Are you sure you want to delete experiment {experiment_id}?")
            print("This action cannot be undone. Use confirm=True to proceed.")
            return False
        
        shutil.rmtree(exp_dir)
        print(f"Experiment {experiment_id} deleted permanently")
        return True
    
    def export_experiment_summary(self, output_file: str = "experiment_summary.csv"):
        """Export all experiments to CSV for analysis"""
        experiments_df = self.list_experiments()
        
        if experiments_df.empty:
            print("No experiments found")
            return
        
        output_path = self.results_dir / output_file
        experiments_df.to_csv(output_path, index=False)
        print(f"Experiment summary exported to {output_path}")
    
    def create_experiment_report(self, experiment_id: str) -> str:
        """Generate a detailed report for a specific experiment"""
        exp_dir = self.results_dir / experiment_id
        results_file = exp_dir / "experiment_results.json"
        
        if not results_file.exists():
            return f"Experiment {experiment_id} not found"
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        report = []
        report.append(f"# Experiment Report: {experiment_id}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Basic info
        report.append("## Experiment Configuration")
        report.append(f"- **Name**: {data.get('experiment_name', 'N/A')}")
        report.append(f"- **Symbols**: {', '.join(data.get('symbols', []))}")
        report.append(f"- **Date Range**: {data.get('start_date')} to {data.get('end_date')}")
        report.append(f"- **Timestamp**: {data.get('timestamp', 'N/A')}")
        report.append("")
        
        # Performance metrics
        if 'backtest_results' in data and 'aggregate' in data['backtest_results']:
            agg = data['backtest_results']['aggregate']
            report.append("## Performance Metrics")
            report.append(f"- **Total Return**: {agg.get('avg_total_return', 0):.2%}")
            report.append(f"- **Sharpe Ratio**: {agg.get('avg_sharpe_ratio', 0):.2f}")
            report.append(f"- **Max Drawdown**: {agg.get('avg_max_drawdown', 0):.2%}")
            report.append(f"- **Total Trades**: {agg.get('total_trades', 0)}")
            report.append("")
        
        # Model performance
        if 'model_results' in data:
            model_res = data['model_results']
            report.append("## Model Performance")
            
            if 'buy_model' in model_res:
                buy_model = model_res['buy_model']
                report.append(f"### Buy Model")
                report.append(f"- **Train AUC**: {buy_model.get('train_auc', 0):.3f}")
                report.append(f"- **Test AUC**: {buy_model.get('test_auc', 0):.3f}")
                
                # Top features
                if 'feature_importance' in buy_model:
                    importance = buy_model['feature_importance']
                    sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                    report.append(f"- **Top Features**:")
                    for feature, weight in sorted_features:
                        report.append(f"  - {feature}: {weight:.3f}")
                report.append("")
            
            if 'sell_model' in model_res:
                sell_model = model_res['sell_model']
                report.append(f"### Sell Model")
                report.append(f"- **Train AUC**: {sell_model.get('train_auc', 0):.3f}")
                report.append(f"- **Test AUC**: {sell_model.get('test_auc', 0):.3f}")
                report.append("")
        
        # Files generated
        report.append("## Generated Files")
        files = []
        if (exp_dir / "models").exists():
            files.append("- Models: `models/trained_model.pkl`")
        if (exp_dir / "visualizations").exists():
            viz_files = list((exp_dir / "visualizations").glob("*"))
            files.append(f"- Visualizations: {len(viz_files)} files")
        if (exp_dir / "analysis").exists():
            analysis_files = list((exp_dir / "analysis").glob("*"))
            files.append(f"- Analysis: {len(analysis_files)} files")
        
        report.extend(files)
        
        return "\n".join(report)
    
    def cleanup_failed_experiments(self):
        """Remove incomplete experiment directories"""
        cleaned = 0
        
        for exp_dir in self.results_dir.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
                results_file = exp_dir / "experiment_results.json"
                
                # If no results file, consider it failed
                if not results_file.exists():
                    print(f"Removing incomplete experiment: {exp_dir.name}")
                    shutil.rmtree(exp_dir)
                    cleaned += 1
        
        print(f"Cleaned up {cleaned} incomplete experiments")
        return cleaned

def main():
    """CLI interface for experiment management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage trading experiments")
    parser.add_argument('command', choices=['list', 'compare', 'best', 'report', 'cleanup', 'export'])
    parser.add_argument('--experiments', nargs='+', help='Experiment IDs for comparison')
    parser.add_argument('--metric', default='sharpe_ratio', help='Metric for best experiment')
    parser.add_argument('--output', help='Output file for export')
    
    args = parser.parse_args()
    
    manager = ExperimentManager()
    
    if args.command == 'list':
        df = manager.list_experiments()
        print(df.to_string(index=False))
    
    elif args.command == 'compare':
        if not args.experiments:
            print("Please specify experiment IDs with --experiments")
            return
        df = manager.compare_experiments(args.experiments)
        print(df.to_string(index=False))
    
    elif args.command == 'best':
        best = manager.get_best_experiment(args.metric)
        if best:
            print(f"Best experiment by {args.metric}:")
            for key, value in best.items():
                print(f"  {key}: {value}")
    
    elif args.command == 'report':
        if not args.experiments or len(args.experiments) != 1:
            print("Please specify exactly one experiment ID")
            return
        report = manager.create_experiment_report(args.experiments[0])
        print(report)
    
    elif args.command == 'cleanup':
        manager.cleanup_failed_experiments()
    
    elif args.command == 'export':
        output_file = args.output or "experiment_summary.csv"
        manager.export_experiment_summary(output_file)

if __name__ == "__main__":
    main()