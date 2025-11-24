import os
import json
import subprocess
from pathlib import Path
from typing import List, Tuple

import yaml

from .decorrelate import load_config, _resolve_existing


def parse_shortlist(txt_path: str) -> List[str]:
    syms: List[str] = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                # lines like " 1  SPY   ..."
                sym = parts[1]
                if sym.isalpha():
                    syms.append(sym)
    return syms


def parse_shortlist_with_scores(txt_path: str) -> List[Tuple[str, float]]:
    """Parse shortlist and return (symbol, score) tuples ordered by confidence score"""
    syms_with_scores: List[Tuple[str, float]] = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7 and parts[0].isdigit():
                # lines like " 1  CLNE         57    0.272    1.000    0.698    0.955"
                # parts: ['1', 'CLNE', '57', '0.272', '1.000', '0.698', '0.955']
                sym = parts[1]
                try:
                    score = float(parts[6])  # The final "Score" column
                    if sym.isalpha():
                        syms_with_scores.append((sym, score))
                except (ValueError, IndexError):
                    # Fallback: just add symbol without score
                    if sym.isalpha():
                        syms_with_scores.append((sym, 0.0))
    
    # Sort by score descending (highest confidence first)
    syms_with_scores.sort(key=lambda x: x[1], reverse=True)
    return syms_with_scores


from concurrent.futures import ThreadPoolExecutor, as_completed

def _render_one(sym: str, reports_dir: str, cross_only: bool, trailstop_data: dict = None) -> Tuple[str, str]:
    exe = ["cargo", "run", "--quiet", "--bin", "render_png", "--"]
    out_name = f"{sym}_{'CROSS' if cross_only else 'FULL'}.png"
    out_path = str(Path(reports_dir) / out_name)
    cmd = exe + ["--symbol", sym, "--out", out_path]
    
    # Add trailstop price levels if available
    if trailstop_data and not cross_only:  # Only add to FULL charts
        if 'entry_price' in trailstop_data:
            cmd += ["--entry-price", str(trailstop_data['entry_price'])]
        if 'stop_loss_price' in trailstop_data:
            cmd += ["--trail-stop-price", str(trailstop_data['stop_loss_price'])]
            cmd += ["--hard-stop-price", str(trailstop_data['stop_loss_price'])]  # Same for now
        if 'take_profit_price' in trailstop_data:
            cmd += ["--take-profit-price", str(trailstop_data['take_profit_price'])]
        if 'atr' in trailstop_data:
            cmd += ["--atr", str(trailstop_data['atr'])]
        if 'risk_per_share' in trailstop_data:
            # Calculate risk per share as entry - stop
            risk_per_share = trailstop_data['entry_price'] - trailstop_data['stop_loss_price']
            cmd += ["--risk-per-share", str(risk_per_share)]
    
    if cross_only:
        cmd += ["--cross-only"]
    print("Rendering:", sym, "->", out_path)
    if trailstop_data and not cross_only:
        print(f"  With levels: Entry=${trailstop_data.get('entry_price', 'N/A')} Stop=${trailstop_data.get('stop_loss_price', 'N/A')}")
    subprocess.run(cmd, cwd=_resolve_existing("stock-chart2"), check=True)
    return sym, out_path


def clear_reports_directory(reports_dir: str):
    """Clear all PNG files from the reports directory"""
    reports_path = Path(reports_dir)
    if reports_path.exists():
        for png_file in reports_path.glob("*.png"):
            try:
                png_file.unlink()
                print(f"Removed: {png_file.name}")
            except Exception as e:
                print(f"Warning: Could not remove {png_file.name}: {e}")


def render_for_symbols(symbols: List[str], reports_dir: str, cross_only: bool = False, max_workers: int = 4, trailstop_data: dict = None, preserve_order: bool = False) -> List[Tuple[str, str]]:
    out_paths: List[Tuple[str, str]] = []
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    
    # Create symbol lookup for trailstop data
    levels_by_symbol = {}
    if trailstop_data:
        for item in trailstop_data:
            if 'symbol' in item:
                levels_by_symbol[item['symbol']] = item
    
    if preserve_order and cross_only:
        # For _CROSS charts, render sequentially to preserve confidence ordering
        print(f"Rendering {len(symbols)} _CROSS charts sequentially (preserving confidence order)...")
        for sym in symbols:
            levels = levels_by_symbol.get(sym)
            sym_result, out_path = _render_one(sym, reports_dir, cross_only, levels)
            out_paths.append((sym_result, out_path))
    else:
        # Use parallel rendering for _FULL charts
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = []
            for sym in symbols:
                levels = levels_by_symbol.get(sym)
                futs.append(ex.submit(_render_one, sym, reports_dir, cross_only, levels))
            
            for fut in as_completed(futs):
                sym, out_path = fut.result()
                out_paths.append((sym, out_path))
    return out_paths


def render_charts_with_trailstop(symbols: List[str], reports_dir: str, trailstop_json_file: str, shortlist_path: str = None) -> List[Tuple[str, str]]:
    """
    Render charts with trailstop levels overlaid
    
    Args:
        symbols: List of symbols to render
        reports_dir: Output directory for charts
        trailstop_json_file: Path to trailstop JSON file from vg trailstop
        shortlist_path: Path to shortlist file (for ordering _CROSS charts by confidence)
    
    Returns:
        List of (symbol, output_path) tuples
    """
    import json
    
    # Clear previous reports
    print(f"Clearing previous reports from {reports_dir}...")
    clear_reports_directory(reports_dir)
    
    # Load trailstop data
    trailstop_data = []
    if Path(trailstop_json_file).exists():
        with open(trailstop_json_file, 'r') as f:
            trailstop_data = json.load(f)
        print(f"Loaded trailstop data for {len(trailstop_data)} symbols")
    else:
        print(f"Warning: Trailstop file not found: {trailstop_json_file}")
    
    # Render full charts with levels (use original symbol order)
    full_charts = render_for_symbols(symbols, reports_dir, cross_only=False, trailstop_data=trailstop_data)
    
    # For cross charts, use confidence-based ordering if shortlist_path is provided
    cross_symbols = symbols
    if shortlist_path and Path(shortlist_path).exists():
        try:
            symbols_with_scores = parse_shortlist_with_scores(shortlist_path)
            cross_symbols = [sym for sym, score in symbols_with_scores]
            print(f"Ordering _CROSS charts by confidence score (highest first):")
            for i, (sym, score) in enumerate(symbols_with_scores[:5]):  # Show top 5
                print(f"  {i+1}. {sym} (Score: {score:.3f})")
            if len(symbols_with_scores) > 5:
                print(f"  ... and {len(symbols_with_scores)-5} more")
        except Exception as e:
            print(f"Warning: Could not parse scores from shortlist, using original order: {e}")
            cross_symbols = symbols
    
    # Render cross-only charts (no levels, ordered by confidence, preserve order)
    cross_charts = render_for_symbols(cross_symbols, reports_dir, cross_only=True, preserve_order=True)
    
    return full_charts + cross_charts


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Render PNG charts for de-correlated shortlist")
    ap.add_argument("--config", default="vgmini/configs/decorrelate.yaml")
    ap.add_argument("--shortlist", default="vgmini/results/decorrelated/equal_weight_dual.txt")
    ap.add_argument("--reports-dir", default="vgmini/results/reports")
    args = ap.parse_args()

    shortlist_path = _resolve_existing(args.shortlist)
    reports_dir = _resolve_existing(args.reports_dir)

    symbols = parse_shortlist(shortlist_path)
    if not symbols:
        raise SystemExit(f"No symbols parsed from {shortlist_path}")

    # Render full charts
    render_for_symbols(symbols, reports_dir, cross_only=False)
    # Render CROSS-only charts
    render_for_symbols(symbols, reports_dir, cross_only=True)

    print(f"Done. Charts in {reports_dir}")


if __name__ == "__main__":
    main()
