"""Inject updated Model Portfolio section into tactical_capital_pool.html"""
import json, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, 'backtest_data', 'model_portfolio_results.json')
HTML_PATH = os.path.join(BASE_DIR, 'tactical_capital_pool.html')

with open(JSON_PATH) as f:
    data = json.load(f)

s = data['summary']
positions = data['positions']
eq = data['equity_curve']
nifty = data.get('nifty_curve', {})
sectors = data['sector_allocation']

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

# Status dot - inline styles to guarantee rendering
def dot_html(status):
    base = 'display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle;'
    if status == 'open':
        return f'<span style="{base}background:#10b981;box-shadow:0 0 6px rgba(16,185,129,0.5);animation:mp-pulse 2s infinite;"></span>'
    elif status == 'warning':
        return f'<span style="{base}background:#f59e0b;box-shadow:0 0 6px rgba(245,158,11,0.5);animation:mp-pulse 1.5s infinite;"></span>'
    else:
        return f'<span style="{base}background:#64748b;"></span>'

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
    return f'<span class="mp-exit-badge {cls}">{label}</span>'

# Build table rows
rows = ''
for p in positions:
    price_col = p['current_price'] if p['status'] != 'closed' else p['exit_price']
    sign = '+' if p['return_pct'] >= 0 else ''
    pcls = 'color:var(--green)' if p['return_pct'] >= 0 else 'color:var(--red)'
    ncls = 'color:var(--green)' if p['net_pnl'] >= 0 else 'color:var(--red)'

    dd_info = ''
    if p['status'] == 'warning' and p['drawdown_from_ath'] is not None:
        dd_info = f'<br><small style="color:var(--accent);">{p["drawdown_from_ath"]}% from ATH</small>'
    elif p['status'] == 'open' and p['drawdown_from_ath'] is not None:
        dd_info = f'<br><small style="color:#64748b;">{p["drawdown_from_ath"]}% from ATH</small>'

    ex = exit_badge(p['exit_reason'], p.get('exit_reason_label', ''))

    rows += f'''                        <tr>
                            <td>{dot_html(p['status'])}</td>
                            <td style="font-weight:600;">{p['symbol']}</td>
                            <td><small style="color:#64748b;">{p['sector']}</small></td>
                            <td>{p['entry_date']}</td>
                            <td style="text-align:right;">{fmt_price(p['entry_price'])}</td>
                            <td style="text-align:right;">{fmt_price(price_col)}{dd_info}</td>
                            <td style="text-align:right;font-weight:700;{pcls}">{sign}{p['return_pct']}%</td>
                            <td style="text-align:right;{ncls}">{fmt_inr(p['net_pnl'])}</td>
                            <td>{ex}</td>
                            <td style="text-align:right;">{p['holding_days']}d</td>
                        </tr>
'''

# Equity curve JS data
eq_labels = json.dumps(list(eq.keys()))
eq_values = json.dumps(list(eq.values()))

# Nifty aligned
nifty_aligned = []
if nifty:
    for d in eq.keys():
        nifty_aligned.append(nifty.get(d))
nifty_js = json.dumps(nifty_aligned)

# Sector data
sec_labels = json.dumps(list(sectors.keys()))
sec_values = json.dumps([round(v, 1) for v in sectors.values()])
sec_count = len(sectors)

# Build the section HTML
section = f'''<!-- ================================================================ -->
<!-- MODEL PORTFOLIO (MQ Core 60% Allocation)                        -->
<!-- ================================================================ -->
<section id="model-portfolio" class="section-mid">
    <div class="container">
        <div class="section-tag">Chapter 8</div>
        <h2>MQ Core &mdash; <span class="live-blink">Live</span> Holdings</h2>
        <p class="subtitle">The <strong>MQ (Momentum + Quality)</strong> component of the Tactical Capital Pool &mdash; the 60% core equity allocation. This is what the portfolio looks like if started in January 2023 with &#8377;1 Crore. Does not include KC6, IPO, or debt fund allocations.</p>

        <div class="info-box">
            <strong>MQ Core Only.</strong> This section shows the 60% long-term equity allocation (Momentum + Quality strategy). The tactical 40% pool (KC6 Mean Reversion + IPO Scalper + IPO Swing + Liquid Debt) is tracked separately above.
        </div>

        <!-- KPI Metrics -->
        <div class="metrics-grid" style="grid-template-columns: repeat(5, 1fr);">
            <div class="metric-card">
                <div class="metric-value">{s['total_positions']}</div>
                <div class="metric-label">Positions</div>
                <div class="metric-detail"><span style="color:#10b981;">{s['open_count']} open</span> &bull; <span style="color:#f59e0b;">{s['warning_count']} nearing SL</span> &bull; <span style="color:#64748b;">{s['closed_count']} exited</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{s['xirr']}%</div>
                <div class="metric-label">XIRR</div>
                <div class="metric-detail">Portfolio IRR</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{fmt_inr(s['final_value'])}</div>
                <div class="metric-label">Portfolio Value</div>
                <div class="metric-detail" style="color:var(--green);">+{s['total_return_pct']}% total return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{s['sharpe']}</div>
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-detail">MaxDD: {s['max_drawdown']}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color:var(--green);">{s['win_rate']}%</div>
                <div class="metric-label">Win Rate</div>
                <div class="metric-detail">{s['total_trades']} closed trades</div>
            </div>
        </div>

        <!-- Status Legend -->
        <div style="display:flex; gap:1.5rem; margin:1rem 0 0.5rem; font-size:13px; color:#94a3b8;">
            <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#10b981;box-shadow:0 0 6px rgba(16,185,129,0.5);margin-right:5px;vertical-align:middle;"></span> Open</span>
            <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#f59e0b;box-shadow:0 0 6px rgba(245,158,11,0.5);margin-right:5px;vertical-align:middle;"></span> Nearing SL (&ge;15% ATH DD)</span>
            <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#64748b;margin-right:5px;vertical-align:middle;"></span> Exited</span>
        </div>

        <!-- Positions Table -->
        <div style="overflow-x:auto; max-height:600px; overflow-y:auto; border-radius:8px; border:1px solid var(--border);">
            <table class="comparison-table" style="width:100%; margin:0;">
                <thead>
                    <tr>
                        <th style="width:30px;"></th>
                        <th>Symbol</th>
                        <th>Sector</th>
                        <th>Entry Date</th>
                        <th style="text-align:right;">Entry Price</th>
                        <th style="text-align:right;">Exit / Current</th>
                        <th style="text-align:right;">P&amp;L %</th>
                        <th style="text-align:right;">Net P&amp;L</th>
                        <th>Exit Reason</th>
                        <th style="text-align:right;">Days Held</th>
                    </tr>
                </thead>
                <tbody>
{rows}                </tbody>
            </table>
        </div>

        <!-- Charts -->
        <div style="display:grid; grid-template-columns:2fr 1fr; gap:1.5rem; margin-top:1.5rem;">
            <div class="chart-container" style="border-radius:12px; border:1px solid var(--border); padding:1rem; background:var(--bg-dark);">
                <h4 style="font-size:14px; margin-bottom:0.75rem; color:#e2e8f0;">
                    Equity Curve (&#8377;1 Cr &rarr; {fmt_inr(s['final_value'])})
                </h4>
                <div style="position:relative; height:300px;">
                    <canvas id="mpEquityChart"></canvas>
                </div>
            </div>
            <div class="chart-container" style="border-radius:12px; border:1px solid var(--border); padding:1rem; background:var(--bg-dark);">
                <h4 style="font-size:14px; margin-bottom:0.75rem; color:#e2e8f0;">Sector Allocation</h4>
                <div style="position:relative; height:300px;">
                    <canvas id="mpSectorChart"></canvas>
                </div>
            </div>
        </div>

        <div style="text-align:center; font-size:11px; color:#64748b; margin-top:1.5rem; padding-top:0.75rem; border-top:1px solid var(--border);">
            MQ Momentum + Quality Strategy &bull; Backtest: {data['config']['start_date']} to {data['config']['end_date']}
            &bull; Nifty 500 Universe &bull; {data['config']['portfolio_size']} stocks &bull; Semi-annual rebalance &bull; 20% ATH drawdown exit
        </div>
    </div>
</section>
'''

# Chart JS for model portfolio (injected inside the existing try/catch block)
chart_js = f'''
    // --- Model Portfolio Charts ---
    try {{
        var mpNiftyData = {nifty_js};
        var mpMqData = {eq_values};
        var mpLen = mpMqData.length;
        var mpStartDot = [5].concat(Array(mpLen - 1).fill(0));
        var mpNiftyDataset = mpNiftyData.some(function(v) {{ return v !== null; }}) ? [{{
            label: 'Nifty 50 (NIFTYBEES)',
            data: mpNiftyData,
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245,158,11,0.05)',
            borderWidth: 1.5, fill: false, pointRadius: mpStartDot, tension: 0, borderDash: [4,3],
            spanGaps: true,
        }}] : [];

        var mpDs = [{{
            label: 'MQ Portfolio',
            data: mpMqData,
            borderColor: '#3b82f6',
            backgroundColor: 'transparent',
            borderWidth: 2, fill: false, pointRadius: mpStartDot, tension: 0,
        }}].concat(mpNiftyDataset).concat([{{
            label: 'Initial Capital (\\u20B91 Cr)',
            data: Array({len(eq)}).fill(10000000),
            borderColor: 'rgba(148,163,184,0.4)',
            borderWidth: 1, borderDash: [5,5], pointRadius: 0, fill: false,
        }}]);

        new Chart(document.getElementById('mpEquityChart').getContext('2d'), {{
            type: 'line',
            data: {{
                labels: {eq_labels},
                datasets: mpDs
            }},
            options: {{
                responsive: true, maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'top', labels: {{ color: '#94a3b8', font: {{ size: 11, family: 'Inter' }} }} }},
                    tooltip: {{ mode: 'index', intersect: false,
                        callbacks: {{ label: function(ctx) {{ return ctx.dataset.label + ': \\u20B9' + (ctx.parsed.y/1e7).toFixed(2) + ' Cr'; }} }}
                    }},
                }},
                scales: {{
                    x: {{ ticks: {{ color: '#94a3b8', maxTicksLimit: 8, font: {{ size: 10 }} }}, grid: {{ color: 'rgba(148,163,184,0.1)' }} }},
                    y: {{ min: 10000000, ticks: {{ color: '#94a3b8', font: {{ size: 10 }}, callback: function(v) {{ return '\\u20B9' + (v/1e7).toFixed(1) + 'Cr'; }} }}, grid: {{ color: 'rgba(148,163,184,0.1)' }} }},
                }},
            }},
        }});

        var mpSecColors = ['#10b981','#3b82f6','#f59e0b','#ef4444','#8b5cf6','#f97316','#06b6d4','#84cc16','#eab308','#fb7185','#a78bfa','#fb923c','#22d3ee'];
        new Chart(document.getElementById('mpSectorChart').getContext('2d'), {{
            type: 'doughnut',
            data: {{
                labels: {sec_labels},
                datasets: [{{ data: {sec_values}, backgroundColor: mpSecColors.slice(0, {sec_count}), borderColor: '#1e293b', borderWidth: 2, hoverOffset: 8 }}],
            }},
            options: {{
                responsive: true, maintainAspectRatio: false, cutout: '55%',
                plugins: {{
                    legend: {{ position: 'bottom', labels: {{ color: '#94a3b8', font: {{ size: 10, family: 'Inter' }}, padding: 8, boxWidth: 10 }} }},
                    tooltip: {{ callbacks: {{ label: function(ctx) {{ return ctx.label + ': ' + ctx.parsed.toFixed(1) + '%'; }} }} }}
                }},
            }},
        }});
    }} catch(e) {{ console.warn('MP charts error:', e.message); }}
'''

# Read the existing HTML
with open(HTML_PATH, 'r', encoding='utf-8') as f:
    html = f.read()

# Find and replace the model portfolio section
# The section starts at <!-- MODEL PORTFOLIO --> and goes until the closing </section>
import re

# Replace section content
start_marker = '<!-- ================================================================ -->\n<!-- MODEL PORTFOLIO (MQ Core 60% Allocation)                        -->\n<!-- ================================================================ -->'
# Find from the marker to the end of the section (</section> followed by newlines)
section_pattern = re.compile(
    r'<!-- ={64} -->\n<!-- MODEL PORTFOLIO \(MQ Core 60% Allocation\)\s+-->\n<!-- ={64} -->\n<section id="model-portfolio".*?</section>',
    re.DOTALL
)
match = section_pattern.search(html)
if match:
    html = html[:match.start()] + section + html[match.end():]
    print(f"Replaced model portfolio section (lines {html[:match.start()].count(chr(10))+1}-...)")
else:
    print("ERROR: Could not find model portfolio section marker!")
    exit(1)

# Replace the chart JS (between the MP charts try/catch)
chart_pattern = re.compile(
    r'// --- Model Portfolio Charts ---\n\s*try \{.*?\} catch\(e\) \{ console\.warn\(\'MP charts error:\'.*?\}',
    re.DOTALL
)
chart_match = chart_pattern.search(html)
if chart_match:
    html = html[:chart_match.start()] + chart_js.strip() + html[chart_match.end():]
    print(f"Replaced model portfolio chart JS")
else:
    # Insert before the closing catch of the main try block
    # Look for "} catch(e) { console.warn('Chart.js not loaded"
    fallback_marker = "    } catch(e) { console.warn('Chart.js not loaded (expected on file:// protocol):', e.message); }"
    idx = html.find(fallback_marker)
    if idx >= 0:
        html = html[:idx] + '\n' + chart_js + '\n' + html[idx:]
        print(f"Inserted model portfolio chart JS before main catch")
    else:
        print("WARNING: Could not find chart JS insertion point!")

with open(HTML_PATH, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\nDone! Updated: {HTML_PATH}")
print(f"Positions: {s['total_positions']} ({s['open_count']} open, {s['closed_count']} closed, {s['warning_count']} warning)")
print(f"CAGR: {s['cagr']}% | Sharpe: {s['sharpe']} | MaxDD: {s['max_drawdown']}%")
print(f"Final Value: Rs.{s['final_value']:,.0f}")
