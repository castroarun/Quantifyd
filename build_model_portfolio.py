"""
Build Model Portfolio Report
Runs MQ backtest from Jan 2023, generates JSON + standalone HTML report.
"""
import sys, os, json, logging, time
from datetime import datetime
from collections import defaultdict

logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mq_backtest_engine import MQBacktestEngine
from services.mq_portfolio import MQBacktestConfig


def calculate_xirr(cashflows, guess=0.1, max_iter=100, tol=1e-6):
    if not cashflows or len(cashflows) < 2:
        return None
    cashflows = sorted(cashflows, key=lambda x: x[0])
    t0 = cashflows[0][0]
    days = [(cf[0] - t0).days / 365.25 for cf in cashflows]
    amounts = [cf[1] for cf in cashflows]
    rate = guess
    for _ in range(max_iter):
        npv = sum(a / (1 + rate) ** d for a, d in zip(amounts, days))
        dnpv = sum(-d * a / (1 + rate) ** (d + 1) for a, d in zip(amounts, days))
        if abs(dnpv) < 1e-12:
            return None
        new_rate = rate - npv / dnpv
        if abs(new_rate - rate) < tol:
            return new_rate
        rate = new_rate
    return None


def main():
    print("=" * 60)
    print("MODEL PORTFOLIO BUILDER")
    print("MQ Strategy | Jan 2023 - Feb 2026 | Rs.1 Cr | PS20")
    print("=" * 60)

    config = MQBacktestConfig(
        start_date='2023-01-01',
        end_date='2026-02-17',
        initial_capital=10_000_000,
        portfolio_size=20,
        equity_allocation_pct=0.95,
        hard_stop_loss=0.50,
        rebalance_ath_drawdown=0.20,
    )

    print("\nLoading data and running backtest...")
    t0 = time.time()
    engine = MQBacktestEngine(config)
    result = engine.run()
    elapsed = time.time() - t0
    print(f"Backtest completed in {elapsed:.1f}s")

    # --- Build per-stock transaction history ---
    # Group topups by symbol
    topups_by_symbol = defaultdict(list)
    for tp in result.topup_log:
        topups_by_symbol[tp.symbol].append({
            'date': tp.date.strftime('%Y-%m-%d'),
            'action': 'TOPUP',
            'price': round(tp.price, 2),
            'shares': tp.shares,
            'amount': round(tp.amount, 2),
            'topup_number': tp.topup_number,
        })

    # --- Build closed positions ---
    closed = []
    for t in result.trade_log:
        # Build transaction timeline for this stock
        txns = []
        txns.append({
            'date': t.entry_date.strftime('%Y-%m-%d'),
            'action': 'BUY',
            'price': round(t.entry_price, 2),
            'shares': t.shares_entered,
            'amount': round(t.initial_investment, 2),
        })
        # Add topups
        for tp in topups_by_symbol.get(t.symbol, []):
            tp_dt = tp['date']
            entry_dt = t.entry_date.strftime('%Y-%m-%d')
            exit_dt = t.exit_date.strftime('%Y-%m-%d')
            if entry_dt <= tp_dt <= exit_dt:
                txns.append(tp)
        txns.append({
            'date': t.exit_date.strftime('%Y-%m-%d'),
            'action': 'SELL',
            'price': round(t.exit_price, 2),
            'shares': t.total_shares_at_exit,
            'amount': round(t.exit_value, 2),
        })
        txns.sort(key=lambda x: x['date'])

        exit_labels = {
            'ath_drawdown_rebalance': 'ATH DD >20%',
            'hard_stop_loss': 'Hard Stop',
            'rebalance_replaced': 'Rebalanced',
            'fundamental_3q_decline': 'Fund. 3Q Decline',
            'fundamental_2y_decline': 'Fund. 2Y Decline',
            'called_away': 'Called Away',
            'manual': 'Manual',
        }
        exit_raw = t.exit_reason.value if hasattr(t.exit_reason, 'value') else str(t.exit_reason)

        closed.append({
            'symbol': t.symbol,
            'sector': t.sector,
            'entry_date': t.entry_date.strftime('%Y-%m-%d'),
            'entry_price': round(t.entry_price, 2),
            'exit_date': t.exit_date.strftime('%Y-%m-%d'),
            'exit_price': round(t.exit_price, 2),
            'current_price': None,
            'return_pct': round(t.return_pct * 100, 1),
            'net_pnl': round(t.net_pnl, 2),
            'exit_reason': exit_raw,
            'exit_reason_label': exit_labels.get(exit_raw, exit_raw),
            'holding_days': t.holding_days,
            'drawdown_from_ath': None,
            'topups': t.topup_count,
            'total_invested': round(t.total_invested, 2),
            'shares': t.total_shares_at_exit,
            'status': 'closed',
            'transactions': txns,
        })

    # --- Build open positions ---
    open_positions = []
    for p in result.final_positions:
        dd = p.get('drawdown_from_ath', 0)
        status = 'warning' if dd >= 15 else 'open'
        end_dt = datetime.strptime(config.end_date, '%Y-%m-%d')
        entry_dt = datetime.strptime(p['entry_date'], '%Y-%m-%d')

        # Transactions for open positions
        txns = [{
            'date': p['entry_date'],
            'action': 'BUY',
            'price': round(p['entry_price'], 2),
            'shares': p.get('shares', 0),
            'amount': round(p.get('value', 0) - p.get('pnl', 0), 2),
        }]
        for tp in topups_by_symbol.get(p['symbol'], []):
            if tp['date'] >= p['entry_date']:
                txns.append(tp)
        txns.sort(key=lambda x: x['date'])

        open_positions.append({
            'symbol': p['symbol'],
            'sector': p['sector'],
            'entry_date': p['entry_date'],
            'entry_price': round(p['entry_price'], 2),
            'exit_date': None,
            'exit_price': None,
            'current_price': round(p['current_price'], 2),
            'return_pct': round(p['return_pct'], 1),
            'net_pnl': round(p['pnl'], 2),
            'exit_reason': None,
            'exit_reason_label': None,
            'holding_days': (end_dt - entry_dt).days,
            'drawdown_from_ath': round(dd, 1),
            'topups': p.get('topups', 0),
            'total_invested': round(p.get('value', 0) - p.get('pnl', 0), 2),
            'shares': p.get('shares', 0),
            'status': status,
            'transactions': txns,
        })

    # Sort: open first, then warning, then closed (within group sort by |return|)
    status_order = {'open': 0, 'warning': 1, 'closed': 2}
    positions = sorted(
        open_positions + closed,
        key=lambda x: (status_order.get(x['status'], 9), -abs(x['return_pct']))
    )

    # --- XIRR ---
    start_dt = datetime.strptime(config.start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(config.end_date, '%Y-%m-%d')
    xirr_val = calculate_xirr([
        (start_dt, -config.initial_capital),
        (end_dt, result.final_value),
    ])

    # --- Summary ---
    summary = {
        'total_positions': len(positions),
        'open_count': len([p for p in positions if p['status'] in ('open', 'warning')]),
        'closed_count': len([p for p in positions if p['status'] == 'closed']),
        'warning_count': len([p for p in positions if p['status'] == 'warning']),
        'xirr': round(xirr_val * 100, 2) if xirr_val else None,
        'cagr': round(result.cagr, 2),
        'sharpe': round(result.sharpe_ratio, 2),
        'sortino': round(result.sortino_ratio, 2),
        'max_drawdown': round(result.max_drawdown, 2),
        'calmar': round(result.calmar_ratio, 2),
        'final_value': round(result.final_value, 2),
        'initial_capital': config.initial_capital,
        'total_return_pct': round(result.total_return_pct, 2),
        'total_topups': result.total_topups,
        'total_trades': result.total_trades,
        'win_rate': round(result.win_rate, 1),
    }

    # --- Equity curve (daily for first 30 days, then every 5th day) ---
    eq_dates = sorted(result.daily_equity.keys())
    equity_curve = {}
    for i, d in enumerate(eq_dates):
        if i < 30 or i % 5 == 0 or i == len(eq_dates) - 1:
            equity_curve[d] = round(result.daily_equity[d], 2)
    # Force first 5 trading days to exactly initial capital
    # (engine records end-of-day equity which includes unrealized gains from deployment)
    for i in range(min(5, len(eq_dates))):
        equity_curve[eq_dates[i]] = float(config.initial_capital)

    # --- Load Nifty (NIFTYBEES) curve aligned to equity curve dates ---
    # Both curves normalized so first point = Rs.1 Cr
    nifty_curve = {}
    try:
        import sqlite3
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data', 'market_data.db')
        conn = sqlite3.connect(db_path)
        nifty_df = conn.execute(
            "SELECT date, close FROM market_data_unified WHERE symbol='NIFTYBEES' AND date>=? AND date<=? ORDER BY date",
            (config.start_date, config.end_date)
        ).fetchall()
        conn.close()
        if nifty_df:
            nifty_all = {d: p for d, p in nifty_df}
            # Base price = price on first equity date
            first_eq_date = list(equity_curve.keys())[0]
            base_price_nifty = None
            for nd, np in nifty_df:
                if nd <= first_eq_date:
                    base_price_nifty = np
                else:
                    break
            if base_price_nifty is None:
                base_price_nifty = nifty_df[0][1]
            for d in equity_curve.keys():
                price = nifty_all.get(d)
                if price is None:
                    closest = None
                    for nd, np in nifty_df:
                        if nd <= d:
                            closest = np
                        else:
                            break
                    price = closest
                if price is not None:
                    nifty_curve[d] = round(config.initial_capital * (price / base_price_nifty), 2)
            # Force first 5 trading days to exactly 1 Cr for Nifty too
            eq_date_list = list(equity_curve.keys())
            for i in range(min(5, len(eq_date_list))):
                nifty_curve[eq_date_list[i]] = float(config.initial_capital)
            print(f"Nifty (NIFTYBEES) curve: {len(nifty_curve)} points (aligned, both start at 1 Cr)")
    except Exception as e:
        print(f"Warning: Could not load Nifty curve: {e}")

    data = {
        'generated_at': datetime.now().isoformat(),
        'config': {
            'start_date': config.start_date,
            'end_date': config.end_date,
            'initial_capital': config.initial_capital,
            'portfolio_size': config.portfolio_size,
            'daily_ath_drawdown_exit': False,
        },
        'positions': positions,
        'summary': summary,
        'equity_curve': equity_curve,
        'nifty_curve': nifty_curve,
        'sector_allocation': result.sector_allocation,
        'exit_reason_counts': result.exit_reason_counts,
    }

    # Save JSON
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data')
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'model_portfolio_results.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nJSON saved: {json_path}")

    # --- Generate standalone HTML report ---
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_portfolio_report.html')
    generate_html_report(data, html_path)
    print(f"HTML report: {html_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Open positions:    {summary['open_count']} ({summary['warning_count']} nearing SL)")
    print(f"Closed positions:  {summary['closed_count']}")
    print(f"XIRR:              {summary['xirr']}%")
    print(f"CAGR:              {summary['cagr']}%")
    print(f"Sharpe:            {summary['sharpe']}")
    print(f"Max Drawdown:      {summary['max_drawdown']}%")
    print(f"Final Value:       Rs.{summary['final_value']:,.0f}")
    print(f"Total Return:      {summary['total_return_pct']}%")
    print(f"Win Rate:          {summary['win_rate']}%")
    print(f"{'=' * 60}")


def generate_html_report(data, output_path):
    """Generate a standalone dark-themed HTML report."""
    s = data['summary']
    positions = data['positions']
    eq = data['equity_curve']
    sectors = data['sector_allocation']
    nifty = data.get('nifty_curve', {})

    def fmt_inr(n):
        if n is None: return '--'
        a = abs(n)
        sign = '-' if n < 0 else ''
        if a >= 1e7: return f"{sign}&#8377;{a/1e7:.2f} Cr"
        if a >= 1e5: return f"{sign}&#8377;{a/1e5:.2f} L"
        return f"&#8377;{n:,.0f}"

    def fmt_price(n):
        if n is None: return '--'
        return f"&#8377;{n:,.2f}"

    def pnl_class(v):
        return 'positive' if v >= 0 else 'negative'

    def dot_class(status):
        return {'open': 'dot-open', 'warning': 'dot-warning', 'closed': 'dot-closed'}.get(status, '')

    def exit_badge(reason, label):
        if not reason: return '--'
        cls_map = {
            'ath_drawdown_rebalance': 'exit-ath',
            'hard_stop_loss': 'exit-stop',
            'rebalance_replaced': 'exit-rebal',
            'fundamental_3q_decline': 'exit-fund',
            'fundamental_2y_decline': 'exit-fund',
        }
        cls = cls_map.get(reason, 'exit-rebal')
        return f'<span class="exit-badge {cls}">{label}</span>'

    # --- Build positions table rows ---
    rows_html = ''
    for i, p in enumerate(positions):
        price_col = p['current_price'] if p['status'] != 'closed' else p['exit_price']
        sign = '+' if p['return_pct'] >= 0 else ''

        dd_info = ''
        if p['status'] == 'warning' and p['drawdown_from_ath'] is not None:
            dd_info = f'<br><small style="color:#d29922;">{p["drawdown_from_ath"]}% from ATH</small>'
        elif p['status'] == 'open' and p['drawdown_from_ath'] is not None:
            dd_info = f'<br><small class="text-muted">{p["drawdown_from_ath"]}% from ATH</small>'

        exit_html = exit_badge(p['exit_reason'], p.get('exit_reason_label', ''))

        # Build transaction sub-rows
        txn_rows = ''
        for t in p.get('transactions', []):
            action_cls = 'positive' if t['action'] == 'SELL' else ('txn-topup' if t['action'] == 'TOPUP' else 'txn-buy')
            action_icon = '&#9650;' if t['action'] == 'BUY' else ('&#9660;' if t['action'] == 'SELL' else '&#9654;')
            txn_rows += f'''<tr class="txn-row" data-stock="{i}" style="display:none;">
                <td></td>
                <td colspan="2" style="padding-left:2rem;">
                    <span class="{action_cls}" style="font-size:0.75rem;">{action_icon} {t['action']}</span>
                </td>
                <td style="font-size:0.8rem;">{t['date']}</td>
                <td class="text-end" style="font-size:0.8rem;">{fmt_price(t['price'])}</td>
                <td class="text-end" style="font-size:0.8rem;">{t.get('shares', '--')} sh</td>
                <td class="text-end" style="font-size:0.8rem;">{fmt_inr(t['amount'])}</td>
                <td colspan="4"></td>
            </tr>'''

        has_txns = len(p.get('transactions', [])) > 0
        expand_btn = f'<span class="expand-btn" data-stock="{i}" title="Show transactions">&#9656;</span>' if has_txns else ''

        rows_html += f'''<tr class="stock-row" data-stock="{i}">
            <td><span class="status-dot {dot_class(p['status'])}"></span></td>
            <td class="fw-semibold">{expand_btn} {p['symbol']}</td>
            <td><small class="text-muted">{p['sector']}</small></td>
            <td>{p['entry_date']}</td>
            <td class="text-end">{fmt_price(p['entry_price'])}</td>
            <td class="text-end">{fmt_price(price_col)}{dd_info}</td>
            <td class="text-end {pnl_class(p['return_pct'])} fw-bold">{sign}{p['return_pct']}%</td>
            <td class="text-end {pnl_class(p['net_pnl'])}">{fmt_inr(p['net_pnl'])}</td>
            <td>{exit_html}</td>
            <td class="text-end">{p['holding_days']}d</td>
            <td class="text-end">{p['topups'] or 0}</td>
        </tr>\n{txn_rows}'''

    # --- Equity curve data ---
    eq_labels = json.dumps(list(eq.keys()))
    eq_values = json.dumps(list(eq.values()))

    # --- Nifty curve aligned to equity dates ---
    nifty_aligned = []
    if nifty:
        eq_dates_list = list(eq.keys())
        for d in eq_dates_list:
            nifty_aligned.append(nifty.get(d))
    nifty_values = json.dumps(nifty_aligned)

    # --- Sector data ---
    sec_labels = json.dumps(list(sectors.keys()))
    sec_values = json.dumps(list(sectors.values()))

    html = f'''<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Portfolio Report | MQ Strategy</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root {{
    --bs-body-bg: #0d1117;
    --bs-body-color: #c9d1d9;
    --bs-card-bg: #161b22;
    --bs-border-color: #30363d;
}}
body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; }}
.positive {{ color: #3fb950; }}
.negative {{ color: #f85149; }}
.txn-buy {{ color: #58a6ff; }}
.txn-topup {{ color: #d29922; }}

.metric-card {{
    background: linear-gradient(135deg, #1f2937 0%, #161b22 100%);
    border: 1px solid var(--bs-border-color);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
}}
.metric-value {{ font-size: 1.8rem; font-weight: 700; color: #58a6ff; }}
.metric-label {{ font-size: 0.82rem; color: #8b949e; margin-top: 0.25rem; }}
.metric-sub {{ font-size: 0.75rem; color: #8b949e; }}

.card {{ background: #161b22; border: 1px solid var(--bs-border-color); border-radius: 12px; }}
.card-header-custom {{
    background: rgba(22,27,34,0.8);
    border-bottom: 1px solid var(--bs-border-color);
    padding: 0.75rem 1rem;
    font-weight: 600;
    font-size: 0.95rem;
    border-radius: 12px 12px 0 0;
}}

.table-portfolio th {{
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #8b949e;
    border-color: var(--bs-border-color);
    white-space: nowrap;
    position: sticky;
    top: 0;
    background: #161b22;
    z-index: 1;
}}
.table-portfolio td {{
    border-color: var(--bs-border-color);
    vertical-align: middle;
    font-size: 0.86rem;
}}
.table-portfolio tbody tr.stock-row:hover {{ background: rgba(88,166,255,0.05); cursor: pointer; }}
.txn-row {{ background: rgba(22,27,34,0.6); }}
.txn-row td {{ font-size: 0.8rem !important; padding-top: 0.2rem !important; padding-bottom: 0.2rem !important; border: none !important; }}

.status-dot {{
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    vertical-align: middle;
}}
.dot-open {{ background: #3fb950; box-shadow: 0 0 6px rgba(63,185,80,0.5); animation: pulse 2s infinite; }}
.dot-warning {{ background: #d29922; box-shadow: 0 0 6px rgba(210,153,34,0.5); animation: pulse 1.5s infinite; }}
.dot-closed {{ background: #8b949e; }}
@keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}

.exit-badge {{
    font-size: 0.68rem;
    padding: 0.15em 0.5em;
    border-radius: 4px;
    font-weight: 600;
    white-space: nowrap;
}}
.exit-ath {{ background: rgba(248,81,73,0.15); color: #f85149; }}
.exit-rebal {{ background: rgba(88,166,255,0.15); color: #58a6ff; }}
.exit-stop {{ background: rgba(210,153,34,0.15); color: #d29922; }}
.exit-fund {{ background: rgba(163,113,247,0.15); color: #a371f7; }}

.legend-bar {{ display: flex; gap: 1.5rem; font-size: 0.82rem; color: #8b949e; }}
.legend-bar .status-dot {{ margin-right: 5px; }}

.chart-container {{ position: relative; height: 300px; }}

.expand-btn {{
    cursor: pointer;
    color: #8b949e;
    font-size: 0.85rem;
    transition: transform 0.2s;
    display: inline-block;
    width: 14px;
}}
.expand-btn.open {{ transform: rotate(90deg); color: #58a6ff; }}

.search-input {{
    background: #0d1117;
    border: 1px solid var(--bs-border-color);
    border-radius: 8px;
    color: #c9d1d9;
    padding: 0.4rem 0.8rem;
    font-size: 0.85rem;
    width: 240px;
}}
.search-input:focus {{ outline: none; border-color: #58a6ff; }}

.header-bar {{
    background: linear-gradient(135deg, rgba(88,166,255,0.08), rgba(63,185,80,0.05));
    border: 1px solid var(--bs-border-color);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
}}

.filter-btn {{
    background: transparent;
    border: 1px solid var(--bs-border-color);
    color: #8b949e;
    border-radius: 6px;
    padding: 0.25rem 0.7rem;
    font-size: 0.78rem;
    cursor: pointer;
    transition: all 0.15s;
}}
.filter-btn.active, .filter-btn:hover {{
    border-color: #58a6ff;
    color: #58a6ff;
    background: rgba(88,166,255,0.1);
}}
</style>
</head>
<body>
<div class="container-fluid py-3" style="max-width: 1400px; margin: auto;">

    <!-- Header -->
    <div class="header-bar">
        <div class="d-flex justify-content-between align-items-center">
            <div>
                <h4 class="mb-1" style="color: #e6edf3;">
                    <i class="bi bi-briefcase-fill me-2" style="color: #58a6ff;"></i>Model Portfolio
                </h4>
                <span class="text-muted" style="font-size: 0.9rem;">
                    MQ Momentum + Quality Strategy &bull; Started Jan 2023 &bull; 20 stocks &bull; &#8377;1 Cr initial capital
                </span>
            </div>
            <div class="text-end">
                <small class="text-muted">Report generated: {data['generated_at'][:10]}</small><br>
                <small class="text-muted">Period: {data['config']['start_date']} to {data['config']['end_date']}</small>
            </div>
        </div>
    </div>

    <!-- KPI Cards -->
    <div class="row g-3 mb-3">
        <div class="col-md-2">
            <div class="metric-card">
                <div class="metric-value">{s['total_positions']}</div>
                <div class="metric-label">Total Positions</div>
                <div class="metric-sub">{s['open_count']} open / {s['closed_count']} closed</div>
            </div>
        </div>
        <div class="col-md-2">
            <div class="metric-card">
                <div class="metric-value" style="color: #3fb950;">{s['open_count']}</div>
                <div class="metric-label">Open Positions</div>
                <div class="metric-sub" style="color: #d29922;">
                    {f'{s["warning_count"]} nearing SL' if s['warning_count'] > 0 else 'All healthy'}
                </div>
            </div>
        </div>
        <div class="col-md-2">
            <div class="metric-card">
                <div class="metric-value">{s['xirr']}%</div>
                <div class="metric-label">XIRR</div>
                <div class="metric-sub">Portfolio IRR</div>
            </div>
        </div>
        <div class="col-md-2">
            <div class="metric-card">
                <div class="metric-value">{fmt_inr(s['final_value'])}</div>
                <div class="metric-label">Portfolio Value</div>
                <div class="metric-sub positive">+{s['total_return_pct']}% total return</div>
            </div>
        </div>
        <div class="col-md-2">
            <div class="metric-card">
                <div class="metric-value">{s['sharpe']}</div>
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-sub">MaxDD: {s['max_drawdown']}%</div>
            </div>
        </div>
        <div class="col-md-2">
            <div class="metric-card">
                <div class="metric-value" style="color: #3fb950;">{s['win_rate']}%</div>
                <div class="metric-label">Win Rate</div>
                <div class="metric-sub">{s['total_trades']} closed trades</div>
            </div>
        </div>
    </div>

    <!-- Legend + Search + Filters -->
    <div class="d-flex justify-content-between align-items-center mb-2">
        <div class="d-flex align-items-center gap-3">
            <div class="legend-bar">
                <span><span class="status-dot dot-open"></span> Open</span>
                <span><span class="status-dot dot-warning"></span> Nearing SL</span>
                <span><span class="status-dot dot-closed"></span> Exited</span>
            </div>
            <div class="d-flex gap-1 ms-3">
                <button class="filter-btn active" data-filter="all">All ({s['total_positions']})</button>
                <button class="filter-btn" data-filter="open">Open ({s['open_count']})</button>
                <button class="filter-btn" data-filter="closed">Closed ({s['closed_count']})</button>
            </div>
        </div>
        <input type="text" class="search-input" id="searchInput" placeholder="Search symbol or sector...">
    </div>

    <!-- Positions Table -->
    <div class="card mb-3">
        <div class="card-header-custom">
            <i class="bi bi-table me-2"></i>All Positions &amp; Trade History
            <small class="text-muted ms-2">(click row to expand transactions)</small>
        </div>
        <div class="card-body p-0">
            <div class="table-responsive" style="max-height: 650px; overflow-y: auto;">
                <table class="table table-portfolio table-hover mb-0">
                    <thead>
                        <tr>
                            <th style="width:30px;"></th>
                            <th>Symbol</th>
                            <th>Sector</th>
                            <th>Entry Date</th>
                            <th class="text-end">Entry Price</th>
                            <th class="text-end">Exit / Current</th>
                            <th class="text-end">P&L %</th>
                            <th class="text-end">Net P&L</th>
                            <th>Exit Reason</th>
                            <th class="text-end">Days Held</th>
                            <th class="text-end">Topups</th>
                        </tr>
                    </thead>
                    <tbody id="positionsBody">
                        {rows_html}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Charts -->
    <div class="row g-3 mb-3">
        <div class="col-md-8">
            <div class="card">
                <div class="card-header-custom">
                    <i class="bi bi-graph-up me-2"></i>Equity Curve (&#8377;1 Cr &rarr; {fmt_inr(s['final_value'])})
                </div>
                <div class="card-body">
                    <div class="chart-container"><canvas id="equityChart"></canvas></div>
                </div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="card">
                <div class="card-header-custom">
                    <i class="bi bi-pie-chart me-2"></i>Sector Allocation
                </div>
                <div class="card-body">
                    <div class="chart-container"><canvas id="sectorChart"></canvas></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Footer -->
    <div class="text-center text-muted py-2" style="font-size: 0.78rem; border-top: 1px solid var(--bs-border-color);">
        MQ Momentum + Quality Strategy &bull; Backtest: {data['config']['start_date']} to {data['config']['end_date']}
        &bull; Nifty 500 Universe &bull; 20 stocks &bull; Semi-annual rebalance &bull; 20% ATH drawdown exit
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
// --- Expand/Collapse Transactions ---
document.querySelectorAll('.stock-row').forEach(row => {{
    row.addEventListener('click', function(e) {{
        if (e.target.classList.contains('search-input')) return;
        const idx = this.dataset.stock;
        const btn = this.querySelector('.expand-btn');
        const txns = document.querySelectorAll('.txn-row[data-stock="' + idx + '"]');
        const isOpen = btn && btn.classList.contains('open');
        txns.forEach(t => t.style.display = isOpen ? 'none' : 'table-row');
        if (btn) btn.classList.toggle('open');
    }});
}});

// --- Search ---
document.getElementById('searchInput').addEventListener('input', function() {{
    const term = this.value.toLowerCase();
    document.querySelectorAll('.stock-row').forEach(row => {{
        const match = row.textContent.toLowerCase().includes(term);
        row.style.display = match ? '' : 'none';
        const idx = row.dataset.stock;
        document.querySelectorAll('.txn-row[data-stock="' + idx + '"]').forEach(t => {{
            t.style.display = 'none';
        }});
        const btn = row.querySelector('.expand-btn');
        if (btn) btn.classList.remove('open');
    }});
}});

// --- Filter buttons ---
document.querySelectorAll('.filter-btn').forEach(btn => {{
    btn.addEventListener('click', function() {{
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        const filter = this.dataset.filter;
        document.querySelectorAll('.stock-row').forEach(row => {{
            const status = row.querySelector('.status-dot');
            let show = true;
            if (filter === 'open') {{
                show = status.classList.contains('dot-open') || status.classList.contains('dot-warning');
            }} else if (filter === 'closed') {{
                show = status.classList.contains('dot-closed');
            }}
            row.style.display = show ? '' : 'none';
            const idx = row.dataset.stock;
            document.querySelectorAll('.txn-row[data-stock="' + idx + '"]').forEach(t => t.style.display = 'none');
            const expandBtn = row.querySelector('.expand-btn');
            if (expandBtn) expandBtn.classList.remove('open');
        }});
    }});
}});

// --- Equity Curve ---
function fmtINR(n) {{
    if (n >= 1e7) return '\\u20B9' + (n/1e7).toFixed(1) + ' Cr';
    if (n >= 1e5) return '\\u20B9' + (n/1e5).toFixed(1) + ' L';
    return '\\u20B9' + n.toLocaleString('en-IN', {{maximumFractionDigits: 0}});
}}

const niftyData = {nifty_values};
const mqData = {eq_values};
const startDot = [5].concat(Array(mqData.length - 1).fill(0));
const niftyDataset = niftyData.some(v => v !== null) ? [{{
    label: 'Nifty 50 (NIFTYBEES)',
    data: niftyData,
    borderColor: '#d29922',
    backgroundColor: 'rgba(210,153,34,0.05)',
    borderWidth: 1.5, fill: false, pointRadius: startDot, tension: 0, borderDash: [4,3],
    spanGaps: true,
}}] : [];

new Chart(document.getElementById('equityChart').getContext('2d'), {{
    type: 'line',
    data: {{
        labels: {eq_labels},
        datasets: [{{
            label: 'MQ Portfolio',
            data: mqData,
            borderColor: '#58a6ff',
            backgroundColor: 'transparent',
            borderWidth: 2, fill: false, pointRadius: startDot, tension: 0,
        }}, ...niftyDataset, {{
            label: 'Initial Capital (\\u20B91 Cr)',
            data: Array({len(eq)}).fill(10000000),
            borderColor: 'rgba(139,148,158,0.4)',
            borderWidth: 1, borderDash: [5,5], pointRadius: 0, fill: false,
        }}]
    }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{
            legend: {{ position: 'top', labels: {{ color: '#8b949e', font: {{ size: 11 }} }} }},
            tooltip: {{ mode: 'index', intersect: false,
                callbacks: {{ label: ctx => ctx.dataset.label + ': ' + fmtINR(ctx.parsed.y) }}
            }},
        }},
        scales: {{
            x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 8, font: {{ size: 10 }} }}, grid: {{ color: 'rgba(48,54,61,0.5)' }} }},
            y: {{ min: 10000000, ticks: {{ color: '#8b949e', font: {{ size: 10 }}, callback: v => fmtINR(v) }}, grid: {{ color: 'rgba(48,54,61,0.5)' }} }},
        }},
    }},
}});

// --- Sector Doughnut ---
const secColors = ['#58a6ff','#3fb950','#d29922','#f85149','#a371f7','#f0883e','#79c0ff','#56d364','#e3b341','#ff7b72','#d2a8ff','#ffa657','#7ee787'];
new Chart(document.getElementById('sectorChart').getContext('2d'), {{
    type: 'doughnut',
    data: {{
        labels: {sec_labels},
        datasets: [{{ data: {sec_values}, backgroundColor: secColors.slice(0, {len(sectors)}), borderColor: '#161b22', borderWidth: 2 }}],
    }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{
            legend: {{ position: 'bottom', labels: {{ color: '#8b949e', font: {{ size: 9 }}, padding: 6, boxWidth: 10 }} }},
            tooltip: {{ callbacks: {{ label: ctx => ctx.label + ': ' + ctx.parsed.toFixed(1) + '%' }} }}
        }},
    }},
}});
</script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


if __name__ == '__main__':
    main()
