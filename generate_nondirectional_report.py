"""
Generate HTML Report for Non-Directional Options Strategy
=========================================================
Produces a comprehensive dark-theme report with:
- Equity curves (Chart.js)
- Trade-level analysis
- Monthly P&L heatmap
- Strike distance & signal comparisons
"""

import os, sys, json
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nondirectional_report.html')
SUMMARY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nondirectional_v3_summary.csv')
TRADES_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nondirectional_v3_trades.csv')


def main():
    df_summary = pd.read_csv(SUMMARY_CSV)
    df_trades = pd.read_csv(TRADES_CSV)

    # =========================================================================
    # Prepare data for charts
    # =========================================================================

    # Best configs for equity curve display
    best_configs = [
        'BANKNIFTY_short_strangle_biweekly_monthly_BB_only_SD1.5_TF2.0_L5',
        'BANKNIFTY_short_strangle_biweekly_monthly_BB_only_SD1.5_TF1.5_L5',
        'BANKNIFTY_short_strangle_biweekly_monthly_BB_only_SD1.5_noTF_L5',
        'BANKNIFTY_short_strangle_weekly_monthly_BB_only_SD1.5_noTF_L5',
        'BANKNIFTY_short_strangle_biweekly_monthly_BB_only_SD2.0_TF2.0_L5',
        'NIFTY50_short_strangle_weekly_monthly_BB_only_SD1.5_noTF_L5',
    ]

    equity_data = {}
    for config in best_configs:
        trades = df_trades[df_trades['config'] == config].sort_values('entry_date')
        if trades.empty:
            continue
        capital = 10_00_000
        curve = [{'date': trades.iloc[0]['entry_date'][:10], 'capital': capital}]
        for _, t in trades.iterrows():
            capital += t['pnl_rs']
            curve.append({'date': t['exit_date'][:10], 'capital': round(capital, 2)})
        equity_data[config] = curve

    # Monthly P&L for best config
    best_label = 'BANKNIFTY_short_strangle_biweekly_monthly_BB_only_SD1.5_TF2.0_L5'
    best_trades = df_trades[df_trades['config'] == best_label].copy()
    if not best_trades.empty:
        best_trades['month'] = best_trades['entry_date'].str[:7]
        monthly_pnl = best_trades.groupby('month')['pnl_rs'].sum().to_dict()
    else:
        monthly_pnl = {}

    # Trade outcomes for best config
    if not best_trades.empty:
        outcomes = best_trades['outcome'].value_counts().to_dict()
    else:
        outcomes = {}

    # Viable strategies table
    viable = df_summary[
        (df_summary['win_rate'] > 50) & (df_summary['profit_factor'] >= 1)
    ].sort_values('cagr_pct', ascending=False).head(25)

    # =========================================================================
    # Generate HTML
    # =========================================================================

    # Colors for equity curves
    colors = ['#00e676', '#ff9800', '#29b6f6', '#e040fb', '#ffd740', '#ef5350']

    equity_datasets = []
    for i, (config, curve) in enumerate(equity_data.items()):
        short_name = config.replace('BANKNIFTY_', 'BN_').replace('NIFTY50_', 'N50_')
        short_name = short_name.replace('short_strangle_', 'SS_').replace('iron_condor_', 'IC_')
        equity_datasets.append({
            'label': short_name,
            'data': [{'x': p['date'], 'y': p['capital']} for p in curve],
            'borderColor': colors[i % len(colors)],
            'backgroundColor': 'transparent',
            'borderWidth': 2,
            'pointRadius': 0,
            'tension': 0.3,
        })

    # Monthly P&L bar chart data
    monthly_labels = sorted(monthly_pnl.keys())
    monthly_values = [monthly_pnl[m] for m in monthly_labels]
    monthly_colors = ['#00e676' if v >= 0 else '#ef5350' for v in monthly_values]

    # Trade detail rows for best config
    trade_rows_html = ''
    for _, t in best_trades.iterrows():
        outcome_class = 'win' if t['pnl_rs'] > 0 else 'loss'
        zone_status = '✓' if t['zone_held'] else '✗'
        trade_rows_html += f"""
        <tr class="{outcome_class}">
            <td>{t['entry_date'][:10]}</td>
            <td>{t['exit_date'][:10]}</td>
            <td>{t['entry_price']:,.0f}</td>
            <td>{t['call_strike']:,.0f}</td>
            <td>{t['put_strike']:,.0f}</td>
            <td>{t['zone_width_pct']:.1f}%</td>
            <td>{t['premium']:.1f}</td>
            <td>{t['max_price']:,.0f}</td>
            <td>{t['min_price']:,.0f}</td>
            <td>{t['max_breach_pct']:.2f}%</td>
            <td class="{'positive' if t['pnl_rs'] > 0 else 'negative'}">{t['pnl_rs']:+,.0f}</td>
            <td>{zone_status}</td>
            <td>{t['outcome']}</td>
            <td>{t['signals']}</td>
        </tr>"""

    # Viable strategies rows
    viable_rows_html = ''
    for _, r in viable.iterrows():
        pf = f"{r['profit_factor']:.1f}" if r['profit_factor'] < 100 else '&#8734;'
        tf = r.get('trend_filter', 'N/A')
        viable_rows_html += f"""
        <tr>
            <td class="label-cell">{r['label']}</td>
            <td>{tf}</td>
            <td>{r['lots']}</td>
            <td>{r['total_trades']}</td>
            <td>{r['win_rate']:.1f}%</td>
            <td>{pf}</td>
            <td class="positive">{r['cagr_pct']:.1f}%</td>
            <td class="{'positive' if r['total_return_pct'] > 0 else 'negative'}">{r['total_return_pct']:+.1f}%</td>
            <td>{r['max_drawdown_pct']:.1f}%</td>
            <td>{r['avg_pnl_per_trade']:+,.0f}</td>
            <td>Rs {r['final_capital']:,.0f}</td>
        </tr>"""

    # Best config summary
    best_row = df_summary[df_summary['label'] == best_label].iloc[0] if best_label in df_summary['label'].values else None

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quantifyd - Non-Directional Strategy Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #0a0a0a; color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif; padding: 20px; }}

    .container {{ max-width: 1400px; margin: 0 auto; }}

    h1 {{ color: #00e676; font-size: 28px; margin-bottom: 5px; }}
    h2 {{ color: #29b6f6; font-size: 20px; margin: 30px 0 15px; border-bottom: 1px solid #333; padding-bottom: 8px; }}
    h3 {{ color: #ff9800; font-size: 16px; margin: 20px 0 10px; }}
    .subtitle {{ color: #888; font-size: 14px; margin-bottom: 30px; }}

    /* Summary cards */
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
    .card {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 15px; text-align: center; }}
    .card .value {{ font-size: 28px; font-weight: 700; color: #00e676; }}
    .card .value.negative {{ color: #ef5350; }}
    .card .label {{ font-size: 12px; color: #888; margin-top: 5px; text-transform: uppercase; letter-spacing: 1px; }}

    /* Charts */
    .chart-container {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 20px; margin: 20px 0; }}
    .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    @media (max-width: 900px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}

    /* Tables */
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 10px 0; }}
    th {{ background: #1a1a1a; color: #29b6f6; padding: 10px 8px; text-align: left; border-bottom: 2px solid #333;
          position: sticky; top: 0; }}
    td {{ padding: 8px; border-bottom: 1px solid #222; }}
    tr:hover {{ background: #1a1a2a; }}
    tr.win {{ border-left: 3px solid #00e676; }}
    tr.loss {{ border-left: 3px solid #ef5350; }}
    .positive {{ color: #00e676; }}
    .negative {{ color: #ef5350; }}
    .label-cell {{ font-size: 11px; font-family: monospace; max-width: 350px; word-break: break-all; }}

    /* Findings */
    .findings {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 20px; margin: 20px 0; }}
    .findings ul {{ list-style: none; padding: 0; }}
    .findings li {{ padding: 8px 0; border-bottom: 1px solid #222; }}
    .findings li:before {{ content: '▸ '; color: #00e676; }}
    .highlight {{ background: #1a2a1a; border-left: 3px solid #00e676; padding: 10px 15px; margin: 10px 0; border-radius: 4px; }}

    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
    .badge-green {{ background: #1a3a1a; color: #00e676; }}
    .badge-red {{ background: #3a1a1a; color: #ef5350; }}
    .badge-orange {{ background: #3a2a1a; color: #ff9800; }}

    .pine-section {{ background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 20px 0; }}
    .pine-section code {{ color: #c9d1d9; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">

<h1>Non-Directional Options Strategy Report</h1>
<p class="subtitle">NIFTY & BANKNIFTY Range Detection | Backtest: Mar 2023 - Mar 2026 (36 months) | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<!-- ================================================================== -->
<h2>Best Strategy: BankNifty Short Strangle Bi-Weekly + Trend Filter</h2>
<!-- ================================================================== -->

<div class="cards">
    <div class="card">
        <div class="value">{best_row['total_return_pct'] if best_row is not None else 0:+.1f}%</div>
        <div class="label">Total Return</div>
    </div>
    <div class="card">
        <div class="value">{best_row['win_rate'] if best_row is not None else 0:.0f}%</div>
        <div class="label">Win Rate</div>
    </div>
    <div class="card">
        <div class="value">{best_row['profit_factor'] if best_row is not None else 0:.1f}</div>
        <div class="label">Profit Factor</div>
    </div>
    <div class="card">
        <div class="value negative">{best_row['max_drawdown_pct'] if best_row is not None else 0:.1f}%</div>
        <div class="label">Max Drawdown</div>
    </div>
    <div class="card">
        <div class="value">{best_row['total_trades'] if best_row is not None else 0}</div>
        <div class="label">Total Trades</div>
    </div>
    <div class="card">
        <div class="value">₹{best_row['avg_pnl_per_trade'] if best_row is not None else 0:+,.0f}</div>
        <div class="label">Avg P&L / Trade</div>
    </div>
    <div class="card">
        <div class="value">{best_row['cagr_pct'] if best_row is not None else 0:.1f}%</div>
        <div class="label">CAGR</div>
    </div>
    <div class="card">
        <div class="value">₹{best_row['final_capital'] if best_row is not None else 0:,.0f}</div>
        <div class="label">Final Capital</div>
    </div>
</div>

<div class="highlight">
    <strong>Config:</strong> BB Squeeze + Trend Filter (2.0 ATR) | 1.5x ATR strike distance | Bi-weekly hold (10 bars) | Monthly options (30 DTE) | 5 lots per trade on ₹10L capital
</div>

<!-- ================================================================== -->
<h2>Equity Curves</h2>
<!-- ================================================================== -->

<div class="chart-container">
    <canvas id="equityChart" height="300"></canvas>
</div>

<!-- ================================================================== -->
<h2>Monthly P&L — Best Config</h2>
<!-- ================================================================== -->

<div class="chart-container">
    <canvas id="monthlyChart" height="200"></canvas>
</div>

<!-- ================================================================== -->
<h2>Outcome Distribution</h2>
<!-- ================================================================== -->

<div class="chart-row">
    <div class="chart-container">
        <canvas id="outcomeChart" height="250"></canvas>
    </div>
    <div class="chart-container">
        <h3>Key Findings</h3>
        <div class="findings">
            <ul>
                <li><strong>Trend Filter is the game-changer</strong> — TF2.0 boosts CAGR from 18.4% to 22.2%, WR from 74% to 86%, PF from 7.8 to 15.9. Blocks bad signals during crashes.</li>
                <li><strong>Monthly options (30 DTE) &gt; Weekly (7 DTE)</strong> — BankNifty weeklies discontinued. Monthly options collect more premium with same hold period.</li>
                <li><strong>TF2.0 is the sweet spot</strong> — TF1.5 is too strict (loses 2 trades, 93% WR but lower CAGR). TF2.5 too loose (lets some trend trades through).</li>
                <li><strong>BankNifty dominates Nifty</strong> — All top 25 configs are BankNifty. Higher premiums from wider ATR.</li>
                <li><strong>Short Strangle &gt; Iron Condor</strong> — IC best: 3.5% CAGR vs SS: 22.2% CAGR. Wing cost eats too much premium.</li>
                <li><strong>Bi-weekly hold with monthly options optimal</strong> — 22.2% CAGR, 29 trades, 1.9% DD. Weekly hold: 20.9% CAGR but 3.4% DD.</li>
                <li><strong>Max drawdown ultra-low</strong> — Best config DD only 1.9% (₹19K on ₹10L). Profit factor 15.9. Win rate 86%.</li>
            </ul>
        </div>
    </div>
</div>

<!-- ================================================================== -->
<h2>All Viable Strategies (WR > 50%, PF > 1.0)</h2>
<!-- ================================================================== -->

<div style="overflow-x: auto;">
<table>
    <thead>
        <tr>
            <th>Configuration</th>
            <th>TF</th>
            <th>Lots</th>
            <th>Trades</th>
            <th>Win Rate</th>
            <th>PF</th>
            <th>CAGR</th>
            <th>Return</th>
            <th>Max DD</th>
            <th>Avg P&L</th>
            <th>Final Cap</th>
        </tr>
    </thead>
    <tbody>
        {viable_rows_html}
    </tbody>
</table>
</div>

<!-- ================================================================== -->
<h2>Trade Log — Best Config</h2>
<!-- ================================================================== -->

<div style="overflow-x: auto; max-height: 600px;">
<table>
    <thead>
        <tr>
            <th>Entry</th>
            <th>Exit</th>
            <th>Price</th>
            <th>CE Strike</th>
            <th>PE Strike</th>
            <th>Width</th>
            <th>Premium</th>
            <th>Max High</th>
            <th>Min Low</th>
            <th>Breach</th>
            <th>P&L (₹)</th>
            <th>Zone</th>
            <th>Outcome</th>
            <th>Signals</th>
        </tr>
    </thead>
    <tbody>
        {trade_rows_html}
    </tbody>
</table>
</div>

<!-- ================================================================== -->
<h2>Pine Script — TradingView Setup</h2>
<!-- ================================================================== -->

<div class="pine-section">
    <h3>How to Use</h3>
    <ol style="padding-left: 20px; line-height: 2;">
        <li>Open TradingView → Pine Editor → New Indicator</li>
        <li>Paste the code from <code>pinescripts/nondirectional_range_detector.pine</code></li>
        <li>Apply to BANKNIFTY daily chart</li>
        <li>When you see the <span class="badge badge-green">green diamond</span> signal:
            <ul style="margin-top: 5px;">
                <li>Check the info table (top right) for exact CE and PE strikes</li>
                <li>The <span class="badge badge-orange">orange lines</span> are your short strikes (SELL)</li>
                <li>The <span style="color: #9c27b0;">purple dotted lines</span> are protection wings (BUY for iron condor)</li>
                <li>The <span style="color: #26a69a;">green shaded zone</span> is the expected range</li>
            </ul>
        </li>
        <li>Set up alerts: right-click indicator → "Add Alert" → select "Range Signal"</li>
    </ol>

    <h3 style="margin-top: 20px;">Recommended Settings</h3>
    <table style="max-width: 500px;">
        <tr><th>Parameter</th><th>BankNifty</th><th>Nifty</th></tr>
        <tr><td>BB Squeeze Period</td><td>10</td><td>10</td></tr>
        <tr><td>ATR Period</td><td>14</td><td>14</td></tr>
        <tr><td>Signal Mode</td><td>BB Only</td><td>BB Only</td></tr>
        <tr><td>Trend Filter</td><td>ON (2.0 ATR)</td><td>ON (2.0 ATR)</td></tr>
        <tr><td>Strike Distance</td><td>1.5x ATR</td><td>1.5x ATR</td></tr>
        <tr><td>Wing Distance</td><td>3.0x ATR</td><td>3.0x ATR</td></tr>
        <tr><td>Hold Period</td><td>10 bars</td><td>5 bars</td></tr>
        <tr><td>Strike Rounding</td><td>100</td><td>50</td></tr>
        <tr><td>Min Bars Gap</td><td>10</td><td>5</td></tr>
    </table>
</div>

<!-- ================================================================== -->
<h2>Strategy Execution Playbook</h2>
<!-- ================================================================== -->

<div class="findings">
    <h3>Entry Rules</h3>
    <ul>
        <li>Apply indicator on BANKNIFTY daily chart</li>
        <li>Wait for green diamond signal (BB Squeeze active)</li>
        <li>Enter bi-weekly short strangle: Sell CE + PE at strikes shown in table</li>
        <li>Entry only on signal day — no chasing next day</li>
        <li>Minimum 10 bars gap between trades (no overlapping positions)</li>
    </ul>

    <h3 style="margin-top: 15px;">Position Sizing</h3>
    <ul>
        <li>5 lots per ₹10L capital (best risk-adjusted: 22.2% CAGR, 1.9% DD)</li>
        <li>3 lots for conservative sizing (14.0% CAGR, 2.2% DD)</li>
        <li>BankNifty lot = 15 units (5 lots = 75 units)</li>
        <li>Monthly expiry options (30 DTE) — BankNifty weeklies no longer available</li>
    </ul>

    <h3 style="margin-top: 15px;">Exit Rules</h3>
    <ul>
        <li><strong>Primary:</strong> Hold to expiry (let theta do the work)</li>
        <li><strong>Stop loss:</strong> Exit if either leg goes 2x premium collected</li>
        <li><strong>Early exit:</strong> Close at 80% of max profit (premium captured)</li>
        <li><strong>Adjustment:</strong> If one strike is breached, consider rolling the untested side closer</li>
    </ul>

    <h3 style="margin-top: 15px;">Risk Management</h3>
    <ul>
        <li>Never sell strangles on event days (RBI policy, Budget, Elections, earnings)</li>
        <li>Avoid first 2 days and last day of weekly expiry</li>
        <li>If VIX > 20, skip the trade or go with iron condor instead</li>
        <li>Maximum 1 open position at a time</li>
    </ul>
</div>

</div>

<!-- ================================================================== -->
<!-- CHARTS JS -->
<!-- ================================================================== -->

<script>
// Equity Curve
const equityCtx = document.getElementById('equityChart').getContext('2d');
new Chart(equityCtx, {{
    type: 'line',
    data: {{
        datasets: {json.dumps(equity_datasets)}
    }},
    options: {{
        responsive: true,
        plugins: {{
            title: {{ display: true, text: 'Equity Curves — Top Strategies', color: '#e0e0e0' }},
            legend: {{ labels: {{ color: '#e0e0e0', font: {{ size: 11 }} }} }},
        }},
        scales: {{
            x: {{
                type: 'time',
                time: {{ unit: 'month' }},
                grid: {{ color: '#222' }},
                ticks: {{ color: '#888' }}
            }},
            y: {{
                grid: {{ color: '#222' }},
                ticks: {{
                    color: '#888',
                    callback: function(v) {{ return '₹' + (v/100000).toFixed(1) + 'L'; }}
                }}
            }}
        }}
    }}
}});

// Monthly P&L
const monthlyCtx = document.getElementById('monthlyChart').getContext('2d');
new Chart(monthlyCtx, {{
    type: 'bar',
    data: {{
        labels: {json.dumps(monthly_labels)},
        datasets: [{{
            label: 'Monthly P&L (₹)',
            data: {json.dumps(monthly_values)},
            backgroundColor: {json.dumps(monthly_colors)},
            borderColor: {json.dumps(monthly_colors)},
            borderWidth: 1,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{
            title: {{ display: true, text: 'Monthly P&L — BN Short Strangle Bi-Weekly', color: '#e0e0e0' }},
            legend: {{ display: false }},
        }},
        scales: {{
            x: {{ grid: {{ color: '#222' }}, ticks: {{ color: '#888' }} }},
            y: {{
                grid: {{ color: '#222' }},
                ticks: {{
                    color: '#888',
                    callback: function(v) {{ return '₹' + (v/1000).toFixed(0) + 'K'; }}
                }}
            }}
        }}
    }}
}});

// Outcome Pie
const outcomeCtx = document.getElementById('outcomeChart').getContext('2d');
new Chart(outcomeCtx, {{
    type: 'doughnut',
    data: {{
        labels: {json.dumps(list(outcomes.keys()))},
        datasets: [{{
            data: {json.dumps(list(outcomes.values()))},
            backgroundColor: ['#00e676', '#ffd740', '#29b6f6', '#ef5350', '#e040fb'],
            borderColor: '#0a0a0a',
            borderWidth: 2,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{
            title: {{ display: true, text: 'Trade Outcomes', color: '#e0e0e0' }},
            legend: {{ labels: {{ color: '#e0e0e0' }} }},
        }},
    }}
}});
</script>

</body>
</html>"""

    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Report generated: {OUTPUT_HTML}")


if __name__ == '__main__':
    main()
