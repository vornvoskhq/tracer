#!/usr/bin/env python3
"""
Generate a comprehensive summary report for manual trading decisions
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

def generate_summary_report(
    shortlist_path: str,
    trailstop_json_path: str,
    orderer_json_path: str,
    reports_dir: str,
    decorrelate_json_path: str = None,
    ranking_txt_path: str = None
) -> str:
    """
    Generate a comprehensive text summary for trading decisions
    
    Args:
        shortlist_path: Path to decorrelated shortlist
        trailstop_json_path: Path to trailstop analysis JSON
        orderer_json_path: Path to orderer format JSON
        reports_dir: Directory where report will be saved
        decorrelate_json_path: Path to decorrelate JSON (optional)
        ranking_txt_path: Path to ranking text file for console data (optional)
    
    Returns:
        Path to generated summary report
    """
    
    # Load data
    symbols = []
    ranking_data = {}
    if Path(shortlist_path).exists():
        with open(shortlist_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].isdigit():
                    rank = int(parts[0])
                    symbol = parts[1]
                    symbols.append(symbol)
                    # Try to parse additional ranking data
                    if len(parts) >= 3:
                        try:
                            score = float(parts[2])
                            ranking_data[symbol] = {'rank': rank, 'score': score}
                        except:
                            ranking_data[symbol] = {'rank': rank}
                    else:
                        ranking_data[symbol] = {'rank': rank}
    
    trailstop_data = []
    if Path(trailstop_json_path).exists():
        with open(trailstop_json_path, 'r') as f:
            trailstop_data = json.load(f)
    
    orderer_data = {}
    if Path(orderer_json_path).exists():
        with open(orderer_json_path, 'r') as f:
            orderer_data = json.load(f)
    
    # Load decorrelate data if available
    decorrelate_data = {}
    if decorrelate_json_path and Path(decorrelate_json_path).exists():
        with open(decorrelate_json_path, 'r') as f:
            decorrelate_data = json.load(f)
    
    # Capture ranking console data
    ranking_data_console = {}
    if ranking_txt_path:
        ranking_data_console = capture_ranking_data(ranking_txt_path)
    
    # Capture decorrelate console data
    decorrelate_console = capture_decorrelate_console_data(shortlist_path)
    
    # Create lookup dictionaries
    trailstop_by_symbol = {item['symbol']: item for item in trailstop_data if 'symbol' in item}
    orderer_orders = {order['symbol']: order for order in orderer_data.get('orders', [])}
    
    # Generate report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "=" * 80,
        "VGMini Trading Analysis Summary",
        "=" * 80,
        f"Generated: {timestamp}",
        f"Symbols Analyzed: {len(symbols)}",
        f"Orders Ready: {len(orderer_orders)}",
        "",
    ]
    
    # Add pipeline analysis from console data
    report_lines.extend([
        "PIPELINE ANALYSIS",
        "=" * 80,
        "",
    ])
    
    # Model performance from ranking console output
    if ranking_data_console:
        report_lines.extend([
            "Model Performance:",
            f"  • Model Type: {ranking_data_console.get('model_type', 'N/A')}",
            f"  • Symbols in Database: {ranking_data_console.get('total_symbols', 'N/A')}",
            f"  • Symbols Processed: {ranking_data_console.get('processed', 'N/A')}",
            f"  • Successful Predictions: {ranking_data_console.get('successful_predictions', 'N/A')}",
            f"  • Buy Signals Flagged: {ranking_data_console.get('buy_signals', 'N/A')}",
            "",
        ])
        
        # Calculate success rate
        if ranking_data_console.get('processed') and ranking_data_console.get('successful_predictions'):
            success_rate = ranking_data_console['successful_predictions'] / ranking_data_console['processed']
            report_lines.append(f"  • Prediction Success Rate: {success_rate:.1%}")
            report_lines.append("")
    
    # Decorrelation analysis from console output
    if decorrelate_console or ranking_data_console:
        orig_count = ranking_data_console.get('buy_signals', 'N/A')
        final_count = decorrelate_console.get('final_count', len(symbols))
        
        report_lines.extend([
            "Decorrelation Analysis:",
            f"  • Original Buy Signals: {orig_count}",
            f"  • After Decorrelation: {final_count}",
        ])
        
        if orig_count != 'N/A' and isinstance(orig_count, int) and orig_count > 0:
            reduction = (orig_count - final_count) / orig_count
            report_lines.append(f"  • Symbols Filtered: {reduction:.1%}")
        
        if decorrelate_console.get('has_cluster_metrics'):
            report_lines.append("  • Cluster Metrics Available: ✓ (ER, CM, Stab columns)")
        
        report_lines.append("")
    
    # Additional cluster analysis from JSON if available
    if decorrelate_data:
        if 'cluster_metrics' in decorrelate_data:
            cm = decorrelate_data['cluster_metrics']
            report_lines.extend([
                "Advanced Cluster Analysis:",
                f"  • Momentum Score: {cm.get('momentum_score', 'N/A')}",
                f"  • Stability Score: {cm.get('stability_score', 'N/A')}",
                f"  • Expected Return: {cm.get('expected_return', 'N/A')}",
                f"  • Risk Score: {cm.get('risk_score', 'N/A')}",
                "",
            ])
    
    report_lines.extend([
        "TRADING CANDIDATES",
        "=" * 80,
        "",
        f"{'Rank':<4} {'Symbol':<8} {'Score':<8} {'Entry':<8} {'Stop':<8} {'Target':<8} {'Risk':<6} {'R:R':<6} {'ATR':<6} {'Trend':<6} {'ER':<6} {'CM':<6} {'Stab':<6}",
        "-" * 100,
    ])
    
    # Parse cluster metrics from shortlist if available
    cluster_metrics_by_symbol = {}
    if Path(shortlist_path).exists():
        with open(shortlist_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 7 and parts[0].isdigit():  # Has cluster data
                    try:
                        rank = int(parts[0])
                        symbol = parts[1]
                        cluster = parts[2] if len(parts) > 2 else "N/A"
                        er = float(parts[3]) if len(parts) > 3 else 0
                        cm = float(parts[4]) if len(parts) > 4 else 0  
                        stab = float(parts[5]) if len(parts) > 5 else 0
                        cluster_metrics_by_symbol[symbol] = {
                            'cluster': cluster,
                            'er': er,
                            'cm': cm,
                            'stab': stab
                        }
                    except:
                        pass

    for i, symbol in enumerate(symbols, 1):
        ts_data = trailstop_by_symbol.get(symbol, {})
        order_data = orderer_orders.get(symbol, {})
        rank_data = ranking_data.get(symbol, {})
        cluster_data = cluster_metrics_by_symbol.get(symbol, {})
        
        # Extract data with fallbacks
        rank = rank_data.get('rank', i)
        score = rank_data.get('score', 0)
        entry = ts_data.get('entry_price', 0)
        stop = ts_data.get('stop_loss_price', 0)
        target = ts_data.get('take_profit_price', 0)
        atr = ts_data.get('atr', 0)
        trend = "✓" if ts_data.get('trend_positive', False) else "✗"
        
        # Cluster metrics
        er = cluster_data.get('er', 0)
        cm = cluster_data.get('cm', 0)
        stab = cluster_data.get('stab', 0)
        
        # Calculate risk and reward
        risk = entry - stop if entry and stop else 0
        reward = target - entry if target and entry else 0
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Format values
        score_str = f"{score:.3f}" if score else "N/A"
        entry_str = f"${entry:.2f}" if entry else "N/A"
        stop_str = f"${stop:.2f}" if stop else "N/A"
        target_str = f"${target:.2f}" if target else "N/A"
        risk_str = f"${risk:.2f}" if risk else "N/A"
        rr_str = f"1:{rr_ratio:.1f}" if rr_ratio > 0 else "N/A"
        atr_str = f"${atr:.2f}" if atr else "N/A"
        er_str = f"{er:.3f}" if er else "N/A"
        cm_str = f"{cm:.3f}" if cm else "N/A"
        stab_str = f"{stab:.3f}" if stab else "N/A"
        
        # Mark executable orders
        executable = "🟢" if symbol in orderer_orders else "🔴"
        
        report_lines.append(
            f"{rank:<4} {symbol:<8} {score_str:<8} {entry_str:<8} {stop_str:<8} {target_str:<8} {risk_str:<6} {rr_str:<6} {atr_str:<6} {trend:<6} {er_str:<6} {cm_str:<6} {stab_str:<6}"
        )
    
    report_lines.extend([
        "",
        "RISK ANALYSIS",
        "=" * 80,
        "",
    ])
    
    # Risk analysis
    total_risk = 0
    best_rr = 0
    worst_rr = float('inf')
    trend_positive = 0
    
    for symbol in symbols:
        ts_data = trailstop_by_symbol.get(symbol, {})
        if symbol in orderer_orders:
            order = orderer_orders[symbol]
            risk = order.get('metadata', {}).get('risk_per_share', 0) * order.get('quantity', 1)
            total_risk += risk
        
        entry = ts_data.get('entry_price', 0)
        stop = ts_data.get('stop_loss_price', 0)
        target = ts_data.get('take_profit_price', 0)
        
        if entry and stop and target:
            risk_amt = entry - stop
            reward_amt = target - entry
            if risk_amt > 0:
                rr = reward_amt / risk_amt
                best_rr = max(best_rr, rr)
                worst_rr = min(worst_rr, rr)
        
        if ts_data.get('trend_positive', False):
            trend_positive += 1
    
    report_lines.extend([
        f"Total Portfolio Risk (if all executed): ${total_risk:.2f}",
        f"Symbols with Positive Trend: {trend_positive}/{len(symbols)} ({trend_positive/len(symbols)*100:.1f}%)",
        f"Best R:R Ratio: 1:{best_rr:.1f}" if best_rr > 0 else "Best R:R Ratio: N/A",
        f"Worst R:R Ratio: 1:{worst_rr:.1f}" if worst_rr < float('inf') else "Worst R:R Ratio: N/A",
        "",
        "EXECUTION NOTES",
        "=" * 80,
        "",
        "⚠️  SAFETY REMINDER:",
        "• Review individual charts before executing ANY orders",
        "• Consider executing only 1-3 best setups, not all at once",
        "• Verify current market conditions and news",
        "• Ensure proper position sizing for your account",
        "",
        "📁 FILES GENERATED:",
        f"• Charts: {reports_dir}/*_FULL.png (with price level overlays)",
        f"• Charts: {reports_dir}/*_CROSS.png (crossover analysis)",
        f"• Summary: {reports_dir}/TRADING_SUMMARY.txt (this file)",
        f"• Orderer Data: {orderer_json_path}",
        f"• Full Analysis: {trailstop_json_path}",
        f"• Decorrelate Data: {decorrelate_json_path}" if decorrelate_json_path else "",
        "",
        "🚀 TO EXECUTE ORDERS:",
        "1. Review individual charts and analysis above",
        "2. Select 1-3 best setups (DO NOT execute all)",
        "3. Run orderer in dry-run mode first:",
        "   cd orderer",
        f"   python orderer_vgmini.py {orderer_json_path}",
        "4. If satisfied, execute selected orders:",
        f"   python orderer_vgmini.py {orderer_json_path} --live --max-position 1000",
        "",
        "=" * 80,
    ])
    
    # Save report
    report_content = "\n".join(report_lines)
    report_path = Path(reports_dir) / "TRADING_SUMMARY.txt"
    
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"📊 Trading summary report: {report_path}")
    return str(report_path)


def capture_ranking_data(ranking_txt_path: str) -> Dict[str, Any]:
    """
    Parse the ranking text file to extract model performance and buy signals
    """
    if not Path(ranking_txt_path).exists():
        return {}
    
    data = {}
    with open(ranking_txt_path, 'r') as f:
        content = f.read()
    
    # Extract summary stats
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if 'Symbols in DB:' in line:
            try:
                data['total_symbols'] = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Processed:' in line:
            try:
                data['processed'] = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Successful predictions:' in line:
            try:
                data['successful_predictions'] = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Buy signals flagged:' in line:
            try:
                data['buy_signals'] = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Model:' in line:
            try:
                data['model_type'] = line.split(':')[1].strip()
            except:
                pass
    
    return data


def capture_decorrelate_console_data(shortlist_txt_path: str) -> Dict[str, Any]:
    """
    Extract decorrelate metrics from the shortlist text file header
    """
    if not Path(shortlist_txt_path).exists():
        return {}
    
    data = {}
    with open(shortlist_txt_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if 'Total selected:' in line:
            try:
                data['final_count'] = int(line.split(':')[1].strip())
            except:
                pass
        elif line.startswith('#') and 'Symbol' in line:
            # Parse header to understand columns
            if 'ER' in line and 'CM' in line and 'Stab' in line:
                data['has_cluster_metrics'] = True
            break
    
    return data