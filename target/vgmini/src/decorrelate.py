import os
import re
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yaml


@dataclass
class SymbolAnnotation:
    symbol: str
    expected_return: float  # ER-momentum from vgmini ranking
    rating: str             # BUY/HOLD/SELL or similar
    cluster_id: Optional[int] = None
    cluster_momentum: Optional[float] = None
    retention_roll: Optional[float] = None
    turnover_roll: Optional[float] = None
    stability_score: Optional[float] = None
    composite_score: Optional[float] = None
    reasons: Optional[List[str]] = None


DEFAULT_REDUNDANCY_GROUPS = {
    'broad_market': ['SPY', 'VOO', 'IVV', 'VTI', 'ITOT', 'SPLG'],
    'nasdaq': ['QQQ', 'QQQM', 'TQQQ', 'ONEQ'],
    'bonds': ['TLT', 'IEF', 'BND', 'AGG', 'GOVT'],
    'gold': ['GLD', 'IAU', 'GLDM'],
    'international': ['EFA', 'VEA', 'IEFA'],
    'emerging': ['EEM', 'VWO', 'IEMG'],
    'small_cap': ['IWM', 'VB', 'IJR'],
}


def _repo_root() -> str:
    # module file is vgmini/src/decorrelate.py -> root is two levels up
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _resolve_existing(path: str) -> str:
    """If path is not absolute and doesn't exist relative to CWD, try repo root."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(_repo_root(), path)
    return candidate


def load_config(path: str) -> Dict:
    path = _resolve_existing(path)
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _normalize(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    if s.empty:
        return s
    mn, mx = s.min(), s.max()
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn == 0:
        # fallback: zeros
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


def parse_vgmini_ranking(path: str, rating_filter: str = 'BUY') -> pd.DataFrame:
    """
    Parse vgmini rank text file; extract rows and the file-level date.
    Expected line structure includes a trailing date column per row and/or a header date line.
    Returns DataFrame with columns: symbol, expected_return, rating, date (str)
    """
    lines = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            lines.append(line.rstrip('\n'))
    # Try to capture a header date if present
    file_date = None
    header_date_re = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
    for ln in lines[:10]:
        m = header_date_re.search(ln)
        if m:
            file_date = m.group(1)
            break
    rows = []
    # SYMBOL  prob  exp  LABEL  $price  YYYY-MM-DD at end
    row_re = re.compile(r"\b([A-Z]{1,6})\b\s+([0-9]\.[0-9]{3})\s+([\-]?[0-9]\.[0-9]{3})\s+.*?\b(BUY|HOLD|SELL)\b.*?\b(20\d{2}-\d{2}-\d{2})\b")
    fallback_re = re.compile(r"\b([A-Z]{1,6})\b\s+([0-9]\.[0-9]{3})\s+([\-]?[0-9]\.[0-9]{3})\s+.*?\b(BUY|HOLD|SELL)\b")
    for ln in lines:
        m = row_re.search(ln)
        if not m:
            m = fallback_re.search(ln)
        if not m:
            continue
        sym = m.group(1)
        try:
            exp_ret = float(m.group(3))
        except Exception:
            continue
        rating = m.group(4)
        date_str = None
        if m.re is row_re:
            try:
                date_str = m.group(5)
            except Exception:
                date_str = None
        if not date_str:
            date_str = file_date
        rows.append((sym, exp_ret, rating, date_str))
    df = pd.DataFrame(rows, columns=['symbol', 'expected_return', 'rating', 'date']).drop_duplicates('symbol')
    if rating_filter and rating_filter.upper() in {'BUY', 'HOLD'}:
        df = df[df['rating'] == rating_filter.upper()].copy()
    return df


def load_latest_cluster_data(db_path: str, prefer_date: Optional[str] = None) -> Tuple[int, Dict[str, int], Dict[int, Dict[str, float]], Optional[str]]:
    """
    Returns: (run_id, symbol->cluster_id, cluster_id->metrics, date_end)
    If prefer_date is provided and exists in cluster_run.date_end, use that run; else use latest.
    metrics include: momentum_score, retention_roll, turnover_roll
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    row = None
    if prefer_date:
        cur.execute("SELECT id, date_end FROM cluster_run WHERE date_end=? ORDER BY id DESC LIMIT 1", (prefer_date,))
        row = cur.fetchone()
    if not row:
        cur.execute("SELECT id, date_end FROM cluster_run ORDER BY date_end DESC LIMIT 1")
        row = cur.fetchone()
    if not row:
        raise RuntimeError("No cluster2 runs found in SQLite. Please run cluster2 first.")
    run_id = int(row[0])
    date_end = row[1]

    cur.execute("SELECT symbol, cluster_id FROM cluster_member WHERE run_id=?", (run_id,))
    sym_rows = cur.fetchall()
    symbol_to_cluster = {s: int(cid) for s, cid in sym_rows}

    cur.execute("SELECT cluster_id, momentum_score, retention_roll, turnover_roll FROM cluster_momentum WHERE run_id=?", (run_id,))
    mom_rows = cur.fetchall()
    cluster_metrics = {
        int(cid): {
            'momentum_score': (ms if ms is not None else np.nan),
            'retention_roll': (rr if rr is not None else np.nan),
            'turnover_roll': (tr if tr is not None else np.nan),
        }
        for cid, ms, rr, tr in mom_rows
    }
    conn.close()
    return run_id, symbol_to_cluster, cluster_metrics, date_end


def compute_scores(df: pd.DataFrame, cluster_metrics: Dict[int, Dict[str, float]], weights: Dict[str, float], use_retention_instead_of_turnover: bool) -> pd.DataFrame:
    # Map cluster metrics to symbols
    df = df.copy()
    df['cluster_momentum'] = df['cluster_id'].map(lambda c: cluster_metrics.get(c, {}).get('momentum_score', np.nan))
    df['retention_roll'] = df['cluster_id'].map(lambda c: cluster_metrics.get(c, {}).get('retention_roll', np.nan))
    df['turnover_roll'] = df['cluster_id'].map(lambda c: cluster_metrics.get(c, {}).get('turnover_roll', np.nan))

    # Normalize ER and CM across candidates
    df['E_norm'] = _normalize(df['expected_return'])
    df['C_norm'] = _normalize(df['cluster_momentum'].fillna(df['cluster_momentum'].median()))

    if use_retention_instead_of_turnover:
        stab = df['retention_roll']
    else:
        stab = 1.0 - df['turnover_roll']
    df['S'] = stab
    df['S'] = df['S'].where(np.isfinite(df['S']), df['S'].median())
    df['S'] = df['S'].clip(lower=0.0, upper=1.0)

    w_er = float(weights.get('expected_return', 0.5))
    w_cm = float(weights.get('cluster_momentum', 0.35))
    w_st = float(weights.get('stability', 0.15))

    df['composite_score'] = w_er * df['E_norm'] + w_cm * df['C_norm'] + w_st * df['S']
    return df


def dedupe_and_select(df: pd.DataFrame, selection_cfg: Dict, redundancy_groups: Dict[str, List[str]]) -> Tuple[List[Dict], Dict]:
    target_size = int(selection_cfg.get('target_size', 20))
    per_cluster_cap_frac = float(selection_cfg.get('per_cluster_cap', 0.3))
    per_cluster_cap = max(1, int(math.ceil(target_size * per_cluster_cap_frac)))

    apply_same_cluster = bool(selection_cfg.get('dedupe', {}).get('same_cluster', True))
    apply_groups = bool(selection_cfg.get('dedupe', {}).get('redundancy_groups', True))

    groups = redundancy_groups or DEFAULT_REDUNDANCY_GROUPS

    selected = []
    removed = []
    reasons_map = {}
    cluster_counts: Dict[int, int] = {}
    group_counts: Dict[str, int] = {g: 0 for g in groups}

    # Sort by composite score desc
    df_sorted = df.sort_values('composite_score', ascending=False)

    for _, row in df_sorted.iterrows():
        sym = row['symbol']
        cid = int(row['cluster_id']) if pd.notna(row['cluster_id']) else None
        if len(selected) >= target_size:
            removed.append((sym, 'overflow'))
            continue

        # per-cluster cap
        if cid is not None and cluster_counts.get(cid, 0) >= per_cluster_cap:
            removed.append((sym, f'cluster_cap_{cid}'))
            continue

        # same-cluster de-dup: allow first, reject later ones in same cluster
        if apply_same_cluster and cid is not None and any(s['cluster_id'] == cid for s in selected):
            removed.append((sym, f'same_cluster_{cid}'))
            continue

        # redundancy groups: allow one per group
        if apply_groups:
            rejected = False
            for gname, gsyms in groups.items():
                if sym in gsyms and group_counts.get(gname, 0) >= 1:
                    removed.append((sym, f'redundancy_group_{gname}'))
                    rejected = True
                    break
            if rejected:
                continue

        # accept
        selected.append({
            'symbol': sym,
            'cluster_id': cid,
            'expected_return': float(row['expected_return']) if pd.notna(row['expected_return']) else None,
            'cluster_momentum': float(row['cluster_momentum']) if pd.notna(row['cluster_momentum']) else None,
            'stability': float(row['S']) if pd.notna(row['S']) else None,
            'composite_score': float(row['composite_score']) if pd.notna(row['composite_score']) else None,
        })
        if cid is not None:
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
        if apply_groups:
            for gname, gsyms in groups.items():
                if sym in gsyms:
                    group_counts[gname] = group_counts.get(gname, 0) + 1

    summary = {
        'total_candidates': int(len(df_sorted)),
        'selected_count': int(len(selected)),
        'removed_count': int(len(removed)),
        'removed_reasons': {r: sum(1 for _, rr in removed if rr == r) for r in sorted(set(rr for _, rr in removed))},
        'cluster_distribution': {str(k): int(v) for k, v in sorted(cluster_counts.items(), key=lambda kv: kv[0])},
        'group_distribution': {k: int(v) for k, v in group_counts.items() if v > 0},
    }
    return selected, summary


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _refresh_cluster2(cluster2_cfg: Dict) -> None:
    """Optionally refresh cluster2 latest run and momentum before decorrelation."""
    refresh = bool(cluster2_cfg.get('refresh_before_run', False))
    if not refresh:
        return
    # Prefer running from cluster2 module dir to avoid import issues
    try:
        import subprocess
        cmd = [sys.executable, '-m', 'src.cli', 'run']
        subprocess.run(cmd, cwd=os.path.join(os.getcwd(), 'cluster2'), check=True)
    except Exception as e:
        print(f"[vgmini.decorrelate] Warning: failed to refresh cluster2: {e}")


def run_decorrelate(cfg_path: str) -> Tuple[str, str]:
    cfg = load_config(cfg_path)

    input_cfg = cfg.get('input', {})
    output_cfg = cfg.get('output', {})
    selection_cfg = cfg.get('selection', {})
    scoring_cfg = cfg.get('scoring', {})
    cluster2_cfg = cfg.get('cluster2', {})
    redundancy_cfg = cfg.get('redundancy', {})

    # Optional cluster2 refresh
    _refresh_cluster2(cluster2_cfg)

    ranking_file = _resolve_existing(input_cfg.get('ranking_file', 'vgmini/equal_weight_dual_ranking.txt'))
    out_dir = _resolve_existing(output_cfg.get('dir', 'vgmini/results/decorrelated'))
    prefix = output_cfg.get('prefix', 'equal_weight_dual')

    filter_rating = selection_cfg.get('filter', 'BUY').upper()

    # Parse ranking
    rank_df = parse_vgmini_ranking(ranking_file, rating_filter=filter_rating)
    if rank_df.empty:
        raise RuntimeError(f"No candidates found in ranking file {ranking_file} with filter={filter_rating}")

    # Load cluster2 data
    db_path = _resolve_existing(cluster2_cfg.get('db_path', 'cluster2/data/cluster2_state.db'))
    prefer_date = None
    if cluster2_cfg.get('run_date_mode', 'latest') == 'match_rank_date':
        # Choose the most recent date present in the ranking file
        rank_dates_all = [d for d in rank_df['date'].dropna().unique().tolist() if isinstance(d, str)]
        if rank_dates_all:
            prefer_date = sorted(rank_dates_all)[-1]
    run_id, sym_to_cluster, cluster_metrics, cluster_date = load_latest_cluster_data(db_path, prefer_date=prefer_date)

    # Optional same-date enforcement between vgmini rank file and cluster2
    if bool(input_cfg.get('enforce_same_date', False)):
        rank_dates = sorted({d for d in rank_df['date'].dropna().unique().tolist() if isinstance(d, str)})
        if not rank_dates:
            raise RuntimeError("Ranking file has no detectable date; cannot enforce same date.")
        if cluster_date and cluster_date not in rank_dates:
            raise RuntimeError(f"Date mismatch: ranking dates {rank_dates} vs cluster2 {cluster_date}. Set input.enforce_same_date=false to ignore.")

    # Annotate with cluster_id
    rank_df['cluster_id'] = rank_df['symbol'].map(sym_to_cluster)

    # Drop symbols with no cluster assignment
    before = len(rank_df)
    rank_df = rank_df.dropna(subset=['cluster_id']).copy()
    rank_df['cluster_id'] = rank_df['cluster_id'].astype(int)
    dropped_na = before - len(rank_df)

    # Compute scores
    weights = scoring_cfg.get('weights', {})
    use_retention = bool(scoring_cfg.get('use_retention_instead_of_turnover', False))
    scored = compute_scores(rank_df, cluster_metrics, weights, use_retention)

    # Apply min cluster momentum filter if set
    min_cm = scoring_cfg.get('min_cluster_momentum', None)
    if min_cm is not None:
        scored = scored[scored['cluster_momentum'].fillna(-1e9) >= float(min_cm)].copy()

    # Dedupe and select
    redundancy_groups = redundancy_cfg.get('groups', DEFAULT_REDUNDANCY_GROUPS)
    selected, summary = dedupe_and_select(scored, selection_cfg, redundancy_groups)

    # Compose outputs
    ensure_dir(out_dir)
    # Date suffix: try cluster2 run date for reproducibility
    # Decide output filenames
    include_date = bool(output_cfg.get('include_date_suffix', True))
    date_suffix = None
    if include_date:
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT date_end FROM cluster_run WHERE id=?", (run_id,))
            row = cur.fetchone()
            if row and row[0]:
                date_suffix = row[0]
            conn.close()
        except Exception:
            pass
        if not date_suffix:
            from datetime import datetime
            date_suffix = datetime.now().date().isoformat()
        out_txt = os.path.join(out_dir, f"{prefix}_{date_suffix}.txt")
        out_json = os.path.join(out_dir, f"{prefix}_{date_suffix}.json")
    else:
        out_txt = os.path.join(out_dir, f"{prefix}.txt")
        out_json = os.path.join(out_dir, f"{prefix}.json")

    # Write text output
    header = f"{'#':>2}  {'Symbol':<6}  {'Cluster':>7}  {'ER':>7}  {'CM':>7}  {'Stab':>7}  {'Score':>7}"
    lines_out = []
    for i, row in enumerate(selected, 1):
        er = f"{row['expected_return']:.3f}" if row['expected_return'] is not None else "  n/a"
        cm = f"{row['cluster_momentum']:.3f}" if row['cluster_momentum'] is not None else "  n/a"
        st = f"{row['stability']:.3f}" if row['stability'] is not None else "  n/a"
        sc = f"{row['composite_score']:.3f}" if row['composite_score'] is not None else "  n/a"
        lines_out.append(f"{i:2d}  {row['symbol']:<6}  {str(row['cluster_id']):>7}  {er:>7}  {cm:>7}  {st:>7}  {sc:>7}")

    # Write text output
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(f"De-correlated shortlist (filter={filter_rating})\n")
        f.write(f"Total selected: {len(selected)}\n\n")
        f.write(header + "\n")
        for ln in lines_out:
            f.write(ln + "\n")

    # Also print to console
    print("\nDe-correlated shortlist (filter={})".format(filter_rating))
    print("Total selected: {}".format(len(selected)))
    print("\n" + header)
    for ln in lines_out[:max(10, len(lines_out))]:
        print(ln)

    # Write JSON summary
    payload = {
        'config': cfg,
        'summary': summary,
        'selected': selected,
        'dropped_no_cluster_assignment': int(dropped_na),
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    return out_txt, out_json


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='vgmini/configs/decorrelate.yaml')
    args = ap.parse_args()
    out_txt, out_json = run_decorrelate(args.config)
    print(f"Wrote: {out_txt}\nWrote: {out_json}")
