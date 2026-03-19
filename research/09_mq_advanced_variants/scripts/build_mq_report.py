"""
Build the MQ Investment Report HTML from report_data.json.
Generates a self-contained HTML file with embedded data and charts.
"""
import json, os

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report_data.json')
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mq_investment_report.html')

data = json.load(open(DATA_FILE))

# Extract chart arrays
bm_dates = sorted(data['benchmark']['equity_monthly'].keys())
chart_labels = [d[:7] for d in bm_dates]  # "2010-01", "2010-02", ...
chart_bm = [data['benchmark']['equity_monthly'][d] for d in bm_dates]

# Equity comparison chart: all systems normalized to Rs 1 Cr at start
eq_labels = chart_labels
eq_bm = chart_bm  # Already normalized to Rs 1 Cr by generate_report_data.py
eq_basic = [data['mq_basic']['equity_monthly'].get(d, None) for d in bm_dates]
eq_full = [data['mq_full']['equity_monthly'].get(d, None) for d in bm_dates]

# Full arrays for other uses
chart_basic = [data['mq_basic']['equity_monthly'].get(d, None) for d in bm_dates]
chart_full = [data['mq_full']['equity_monthly'].get(d, None) for d in bm_dates]

# Drawdown arrays
chart_dd_bm = [data['benchmark']['drawdown_monthly'].get(d, 0) for d in bm_dates]
chart_dd_basic = [data['mq_basic']['drawdown_monthly'].get(d, 0) for d in bm_dates]
chart_dd_full = [data['mq_full']['drawdown_monthly'].get(d, 0) for d in bm_dates]

# Yearly returns
yr_keys = sorted(data['benchmark']['yearly_returns'].keys())
yr_bm = [data['benchmark']['yearly_returns'][y] for y in yr_keys]
yr_basic = [data['mq_basic']['yearly_returns'].get(y, 0) for y in yr_keys]
yr_full = [data['mq_full']['yearly_returns'].get(y, 0) for y in yr_keys]

# Key metrics
bm = data['benchmark']
mb = data['mq_basic']
mf = data['mq_full']
period_years = data['period']['years']

def fmt_cr(val):
    """Format as crores"""
    cr = val / 10000000
    if cr >= 100:
        return f"{cr:,.0f}"
    return f"{cr:,.1f}"

# Build yearly returns table rows (3 columns: Nifty, Basic MQ, Full System)
yr_rows = ""
for y in yr_keys:
    bm_ret = data['benchmark']['yearly_returns'].get(y, '-')
    basic_ret = data['mq_basic']['yearly_returns'].get(y, '-')
    full_ret = data['mq_full']['yearly_returns'].get(y, '-')

    bm_class = 'positive' if isinstance(bm_ret, (int, float)) and bm_ret > 0 else 'negative'
    basic_class = 'positive' if isinstance(basic_ret, (int, float)) and basic_ret > 0 else 'negative'
    full_class = 'positive' if isinstance(full_ret, (int, float)) and full_ret > 0 else 'negative'

    yr_rows += f"""<tr>
        <td>{y}</td>
        <td class="{bm_class}">{bm_ret}%</td>
        <td class="{basic_class}">{basic_ret}%</td>
        <td class="{full_class}">{full_ret}%</td>
    </tr>\n"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>The MQ System &mdash; Momentum + Quant | {period_years}-Year Backtest Report by Castronix</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

        :root {{
            --bg-hero: #0f172a;
            --bg-dark: #0f172a;
            --bg-mid: #1e293b;
            --bg-light: #f8fafc;
            --bg-card: rgba(30, 41, 59, 0.9);
            --accent: #f59e0b;
            --accent-hover: #d97706;
            --green: #10b981;
            --green-light: #34d399;
            --red: #ef4444;
            --blue: #3b82f6;
            --purple: #8b5cf6;
            --text-light: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-dark: #1e293b;
            --text-muted: #64748b;
            --border: rgba(148, 163, 184, 0.15);
            --max-w: 1200px;
            --radius: 16px;
            --radius-sm: 8px;
        }}

        html {{ scroll-behavior: smooth; }}
        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: var(--bg-dark);
            color: var(--text-light);
            line-height: 1.7;
            font-size: 16px;
            -webkit-font-smoothing: antialiased;
        }}

        /* Navigation */
        nav {{
            position: fixed; top: 0; left: 0; right: 0; z-index: 100;
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            padding: 0 24px;
        }}
        nav .nav-inner {{
            max-width: var(--max-w);
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
            height: 60px;
        }}
        nav .logo {{
            font-weight: 800;
            font-size: 18px;
            color: var(--accent);
            letter-spacing: -0.5px;
        }}
        nav .nav-links {{ display: flex; gap: 24px; }}
        nav .nav-links a {{
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 13px;
            font-weight: 500;
            transition: color 0.2s;
        }}
        nav .nav-links a:hover {{ color: var(--accent); }}

        /* Sections */
        section {{ padding: 100px 24px; }}
        .container {{ max-width: var(--max-w); margin: 0 auto; }}
        .section-light {{ background: var(--bg-light); color: var(--text-dark); }}
        .section-mid {{ background: var(--bg-mid); }}

        /* Typography */
        .section-tag {{
            display: inline-block;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--accent);
            margin-bottom: 16px;
        }}
        .section-light .section-tag {{ color: var(--accent-hover); }}
        h2 {{
            font-size: clamp(28px, 4vw, 42px);
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 20px;
            letter-spacing: -1px;
        }}
        .section-light h2 {{ color: var(--text-dark); }}
        .subtitle {{
            font-size: 18px;
            color: var(--text-secondary);
            max-width: 680px;
            margin-bottom: 48px;
        }}
        .section-light .subtitle {{ color: var(--text-muted); }}

        /* Hero */
        #hero {{
            min-height: 100vh;
            display: flex;
            align-items: center;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
            position: relative;
            overflow: hidden;
            padding-top: 60px;
        }}
        #hero::before {{
            content: '';
            position: absolute;
            top: -50%; left: -50%;
            width: 200%; height: 200%;
            background: radial-gradient(ellipse at 30% 50%, rgba(245, 158, 11, 0.06) 0%, transparent 60%),
                        radial-gradient(ellipse at 70% 80%, rgba(16, 185, 129, 0.04) 0%, transparent 50%);
        }}
        .hero-content {{ position: relative; z-index: 1; }}
        .hero-title {{
            font-size: clamp(36px, 6vw, 64px);
            font-weight: 900;
            line-height: 1.1;
            letter-spacing: -2px;
            margin-bottom: 24px;
        }}
        .hero-title .highlight {{ color: var(--accent); }}
        .hero-subtitle {{
            font-size: clamp(18px, 2.5vw, 22px);
            color: var(--text-secondary);
            max-width: 600px;
            margin-bottom: 48px;
            line-height: 1.6;
        }}
        .hero-stats {{
            display: flex;
            gap: 48px;
            flex-wrap: wrap;
        }}
        .hero-stat {{
            text-align: left;
        }}
        .hero-stat .number {{
            font-size: 36px;
            font-weight: 800;
            color: var(--accent);
            letter-spacing: -1px;
        }}
        .hero-stat .label {{
            font-size: 13px;
            color: var(--text-secondary);
            font-weight: 500;
            margin-top: 4px;
        }}

        /* Metric Cards */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin: 40px 0;
        }}
        .metric-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 28px;
            text-align: center;
        }}
        .section-light .metric-card {{
            background: white;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .metric-card .metric-value {{
            font-size: 32px;
            font-weight: 800;
            letter-spacing: -1px;
            margin-bottom: 6px;
        }}
        .metric-card .metric-label {{
            font-size: 13px;
            color: var(--text-secondary);
            font-weight: 500;
        }}
        .section-light .metric-card .metric-label {{ color: var(--text-muted); }}
        .metric-card.accent .metric-value {{ color: var(--accent); }}
        .metric-card.green .metric-value {{ color: var(--green); }}
        .metric-card.red .metric-value {{ color: var(--red); }}
        .metric-card.blue .metric-value {{ color: var(--blue); }}

        /* Comparison Table */
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 32px 0;
            font-size: 14px;
        }}
        .comparison-table th, .comparison-table td {{
            padding: 14px 16px;
            text-align: center;
            border-bottom: 1px solid var(--border);
        }}
        .comparison-table th {{
            font-weight: 700;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
        }}
        .section-light .comparison-table th {{ color: var(--text-muted); }}
        .section-light .comparison-table td {{ border-color: #e2e8f0; }}
        .comparison-table td.positive {{ color: var(--green); font-weight: 600; }}
        .comparison-table td.negative {{ color: var(--red); font-weight: 600; }}
        .comparison-table tr:hover {{ background: rgba(245, 158, 11, 0.05); }}

        /* Chart containers */
        .chart-container {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 32px;
            margin: 32px 0;
        }}
        .section-light .chart-container {{
            background: white;
            border: 1px solid #e2e8f0;
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        .chart-subtitle {{
            font-size: 13px;
            color: var(--text-secondary);
            margin-bottom: 24px;
        }}
        .section-light .chart-subtitle {{ color: var(--text-muted); }}

        /* Steps */
        .steps {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 24px;
            margin: 48px 0;
        }}
        .step {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 32px 24px;
            text-align: center;
            position: relative;
        }}
        .step-number {{
            width: 40px; height: 40px;
            background: var(--accent);
            color: var(--bg-dark);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 16px;
            margin: 0 auto 16px;
        }}
        .step h4 {{
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 8px;
            color: #f1f5f9;
        }}
        .step p {{
            font-size: 14px;
            color: #e2e8f0;
            line-height: 1.5;
        }}

        /* Big comparison */
        .big-compare {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 32px;
            margin: 48px 0;
        }}
        .compare-card {{
            border-radius: var(--radius);
            padding: 40px 32px;
            text-align: center;
        }}
        .compare-card.nifty {{
            background: linear-gradient(135deg, #1e293b, #334155);
            border: 1px solid var(--border);
        }}
        .compare-card.mq-basic {{
            background: linear-gradient(135deg, #78350f, #92400e);
            border: 2px solid var(--accent);
        }}
        .compare-card .cc-label {{
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 16px;
            opacity: 0.8;
        }}
        .compare-card .cc-amount {{
            font-size: clamp(32px, 5vw, 48px);
            font-weight: 900;
            letter-spacing: -2px;
            margin-bottom: 8px;
        }}
        .compare-card .cc-detail {{
            font-size: 14px;
            opacity: 0.7;
        }}

        /* Tool cards */
        .tool-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
            margin: 48px 0;
        }}
        .tool-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 32px;
        }}
        .tool-card .tool-icon {{
            font-size: 32px;
            margin-bottom: 16px;
        }}
        .tool-card h4 {{
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        .tool-card p {{
            font-size: 14px;
            color: var(--text-secondary);
            line-height: 1.6;
        }}

        /* Warning box */
        .warning-box {{
            background: rgba(239, 68, 68, 0.08);
            border: 1px solid rgba(239, 68, 68, 0.2);
            border-radius: var(--radius);
            padding: 24px 28px;
            margin: 24px 0;
        }}
        .warning-box h4 {{
            color: var(--red);
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        .warning-box p, .warning-box li {{
            font-size: 14px;
            color: var(--text-secondary);
            line-height: 1.6;
        }}
        .warning-box ul {{ margin-left: 20px; margin-top: 8px; }}

        /* Info box */
        .info-box {{
            background: rgba(59, 130, 246, 0.08);
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: var(--radius);
            padding: 24px 28px;
            margin: 24px 0;
        }}
        .info-box h4 {{
            color: var(--blue);
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        .info-box p {{
            font-size: 14px;
            color: var(--text-secondary);
            line-height: 1.6;
        }}

        /* Footer */
        footer {{
            padding: 48px 24px;
            text-align: center;
            border-top: 1px solid var(--border);
        }}
        footer p {{
            font-size: 13px;
            color: var(--text-secondary);
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            section {{ padding: 60px 16px; }}
            .hero-stats {{ gap: 32px; }}
            .hero-stat .number {{ font-size: 28px; }}
            nav .nav-links {{ display: none; }}
            .big-compare {{ grid-template-columns: 1fr; }}
            .steps {{ grid-template-columns: 1fr; }}
        }}

        /* Table scrollable on mobile */
        .table-wrap {{
            overflow-x: auto;
            margin: 32px 0;
            border-radius: var(--radius);
        }}

        /* Callout */
        .callout {{
            background: rgba(245, 158, 11, 0.08);
            border-left: 4px solid var(--accent);
            padding: 20px 24px;
            border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
            margin: 32px 0;
            font-size: 17px;
            font-weight: 500;
            color: var(--text-light);
            line-height: 1.6;
        }}
        .section-light .callout {{ color: var(--text-dark); }}

        /* Two-col layout */
        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 48px;
            align-items: start;
        }}
        @media (max-width: 768px) {{
            .two-col {{ grid-template-columns: 1fr; gap: 24px; }}
        }}

        .text-accent {{ color: var(--accent); }}
        .text-green {{ color: var(--green); }}
        .text-red {{ color: var(--red); }}
        .fw-800 {{ font-weight: 800; }}
    </style>
</head>
<body>

<!-- Navigation -->
<nav>
    <div class="nav-inner">
        <div class="logo"><span style="color:#22D3EE; font-size:22px; margin-right:6px;">C</span>MQ+ System</div>
        <div class="nav-links">
            <a href="#market">The Market</a>
            <a href="#compounding">Compounding</a>
            <a href="#system">The System</a>
            <a href="#results">Results</a>
            <a href="#cash-trap">Capital Recycling</a>
            <a href="#live-portfolio">Live Portfolio</a>
            <a href="#tools">Get Started</a>
        </div>
    </div>
</nav>

<!-- Hero Section -->
<section id="hero">
    <div class="container hero-content">
        <p class="section-tag">{period_years}-Year Backtest Report &bull; Castronix Research</p>
        <h1 class="hero-title">
            The <span class="highlight">MQ System</span><br>
            Momentum + Quant
        </h1>
        <p class="hero-subtitle" style="font-size:20px; color:var(--accent); font-weight:600; margin-bottom:12px; letter-spacing:0.5px;">
            Quant-powered stock picking. Zero guesswork.
        </p>
        <p class="hero-subtitle">
            A systematic, data-driven approach to Indian equities &mdash;
            ranking stocks by quantitative momentum signals and letting the numbers decide.
            Here's what {period_years} years of data shows.
        </p>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="number" data-count="{period_years}" data-decimals="1">0</div>
                <div class="label">Years of Data</div>
            </div>
            <div class="hero-stat">
                <div class="number" data-count="395" data-decimals="0">0</div>
                <div class="label">Stocks Tested</div>
            </div>
            <div class="hero-stat">
                <div class="number" data-count="{mf['cagr']}" data-suffix="%" data-decimals="1">0</div>
                <div class="label">CAGR (Full System)</div>
            </div>
            <div class="hero-stat">
                <div class="number" data-count="{mf['max_drawdown']}" data-suffix="%" data-decimals="0">0</div>
                <div class="label">Max Drawdown</div>
            </div>
        </div>
    </div>
</section>

<!-- Chapter 1: The Market Reality -->
<section id="market" class="section-light">
    <div class="container">
        <p class="section-tag">Chapter 1</p>
        <h2>The Indian Stock Market:<br>A {period_years}-Year Story</h2>
        <p class="subtitle">
            The Nifty 50 index has been one of the world's best performing markets.
            But the journey hasn't been smooth.
        </p>

        <div class="metrics-grid">
            <div class="metric-card green">
                <div class="metric-value">{bm['cagr']}%</div>
                <div class="metric-label">Average Annual Return (CAGR)</div>
            </div>
            <div class="metric-card accent">
                <div class="metric-value">{int(bm['total_return_pct']):,}%</div>
                <div class="metric-label">Total Return ({period_years} years)</div>
            </div>
            <div class="metric-card green">
                <div class="metric-value">+{bm['best_year']['return']}%</div>
                <div class="metric-label">Best Year ({bm['best_year']['year']})</div>
            </div>
            <div class="metric-card red">
                <div class="metric-value">{bm['worst_year']['return']}%</div>
                <div class="metric-label">Worst Year ({bm['worst_year']['year']})</div>
            </div>
        </div>

        <div class="callout">
            If you invested <strong>Rs 1 Crore</strong> in the Nifty 50 in January {data['period']['start'][:4]},
            it would be worth <strong class="text-green">Rs {fmt_cr(bm['final_value'])} Crores</strong> today.
            That's {bm['cagr']}% compounding every single year. Not bad &mdash; but can we do better?
        </div>

        <div class="chart-container">
            <div class="chart-title">Nifty 50: The {period_years}-Year Journey</div>
            <div class="chart-subtitle">Rs 1 Crore invested in January {data['period']['start'][:4]}. Note the stomach-churning drops along the way.</div>
            <canvas id="chartBenchmark"></canvas>
        </div>

        <div class="warning-box">
            <h4>The Painful Truth About Market Crashes</h4>
            <p>The market lost <strong>38%</strong> during COVID 2020.
            Most investors panic and sell at the bottom, missing the recovery. The Nifty's max drawdown
            over this period was <strong>{abs(bm['max_drawdown'])}%</strong> &mdash; meaning at one point,
            more than a third of your money was gone (on paper).</p>
        </div>
    </div>
</section>

<!-- Chapter 2: The Compounding Edge -->
<section id="compounding" class="section-mid">
    <div class="container">
        <p class="section-tag">Chapter 2</p>
        <h2>The Magic of a <span class="text-accent">Few Extra Percent</span></h2>
        <p class="subtitle">
            The difference between 11% and 20% annual returns sounds small.
            Over 20 years, it's life-changing.
        </p>

        <div class="chart-container">
            <div class="chart-title">Rs 1 Crore Growing at Different Rates</div>
            <div class="chart-subtitle">The gap between returns widens dramatically over time. This is the power of compounding.</div>
            <canvas id="chartCompounding"></canvas>
        </div>

        <div class="table-wrap">
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Annual Return</th>
                        <th>After 5 Years</th>
                        <th>After 10 Years</th>
                        <th>After 15 Years</th>
                        <th>After 20 Years</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>7% (FD-like)</td>
                        <td>Rs 1.4 Cr</td>
                        <td>Rs 2.0 Cr</td>
                        <td>Rs 2.8 Cr</td>
                        <td>Rs 3.9 Cr</td>
                    </tr>
                    <tr>
                        <td><strong>~11% (Nifty 50)</strong></td>
                        <td>Rs 1.7 Cr</td>
                        <td>Rs 2.8 Cr</td>
                        <td>Rs 4.8 Cr</td>
                        <td><strong>Rs 8.1 Cr</strong></td>
                    </tr>
                    <tr>
                        <td><strong class="text-accent">~17% (Basic MQ)</strong></td>
                        <td>Rs 2.2 Cr</td>
                        <td>Rs 4.8 Cr</td>
                        <td>Rs 10.5 Cr</td>
                        <td><strong class="text-accent">Rs 23.1 Cr</strong></td>
                    </tr>
                    <tr>
                        <td><strong class="text-green">~20% (MQ + Recycling)</strong></td>
                        <td>Rs 2.5 Cr</td>
                        <td>Rs 6.2 Cr</td>
                        <td>Rs 15.4 Cr</td>
                        <td><strong class="text-green">Rs 38.3 Cr</strong></td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="callout">
            At ~11% (Nifty), Rs 1 Crore becomes Rs 8 Crores in 20 years.
            The full MQ+ system at ~{mf['cagr']:.0f}% turns it into <strong class="text-green">Rs {mf['final_value']/10000000:.0f} Crores</strong>.
            That's <strong>nearly {mf['final_value']/bm['final_value']:.0f}x more wealth</strong> &mdash; from a disciplined, rules-based approach.
        </div>
    </div>
</section>

<!-- Chapter 3: The MQ System -->
<section id="system" class="section-light">
    <div class="container">
        <p class="section-tag">Chapter 3</p>
        <h2>The MQ System: <br>Quantitative Rules, Powerful Results</h2>
        <p class="subtitle">
            No predictions. No market timing. No stock tips.
            Just a clear set of rules that anyone can follow.
        </p>

        <div class="steps">
            <div class="step">
                <div class="step-number">1</div>
                <h4>Start with the Universe</h4>
                <p>Begin with the Nifty 500 universe &mdash; India's top 500 companies by market cap. No penny stocks, no obscure names.</p>
            </div>
            <div class="step">
                <div class="step-number">2</div>
                <h4>Rank by Momentum</h4>
                <p>Sort all stocks by how close they are to their all-time high. Stocks near their ATH are showing strength &mdash; the market is voting for them.</p>
            </div>
            <div class="step">
                <div class="step-number">3</div>
                <h4>Pick the Top 30</h4>
                <p>Select the top 30 stocks by momentum ranking. Equal weight &mdash; no guessing which stock will be the winner.</p>
            </div>
            <div class="step">
                <div class="step-number">4</div>
                <h4>Equal Weight, Natural Balance</h4>
                <p>Every stock gets the same allocation (~3.3% each). No stock picking bias, no overweight bets. With 30 stocks across sectors, the portfolio is diversified by construction &mdash; no single stock or sector can sink the ship.</p>
            </div>
            <div class="step">
                <div class="step-number">5</div>
                <h4>Manage Risk Daily</h4>
                <p>Every single day, the system checks: has any stock dropped 20% from its peak since entry? If yes, sell immediately. Why 20%? Because a 20% drop from the all-time high is the textbook definition of entering bear territory &mdash; once a stock crosses that threshold, momentum has broken.</p>
            </div>
            <div class="step">
                <div class="step-number">6</div>
                <h4>Rebalance</h4>
                <p>Every 6 months (January &amp; July), re-screen the entire universe and update the portfolio. Rinse and repeat.</p>
            </div>
        </div>

        <div class="info-box">
            <h4>Why Momentum Works</h4>
            <p>Momentum is one of the most robust factors in finance, documented across 200+ years of data in every major market.
            Stocks that are going up tend to keep going up (and vice versa). The academic evidence is overwhelming &mdash;
            yet most investors fight the trend instead of riding it.</p>
        </div>
    </div>
</section>

<!-- Chapter 4: The Results -->
<section id="results">
    <div class="container">
        <p class="section-tag">Chapter 4</p>
        <h2>{period_years} Years of Results:<br>The Numbers Don't Lie</h2>
        <p class="subtitle">
            Here's what happened when we ran the MQ System on {period_years} years of real market data,
            with real transaction costs, through market crashes.
        </p>

        <!-- Big comparison cards -->
        <div class="big-compare">
            <div class="compare-card nifty">
                <div class="cc-label">Nifty 50 Index</div>
                <div class="cc-amount">Rs {fmt_cr(bm['final_value'])} Cr</div>
                <div class="cc-detail">{bm['cagr']}% CAGR &bull; Max loss: {abs(bm['max_drawdown'])}%</div>
            </div>
            <div class="compare-card mq-basic">
                <div class="cc-label">Basic MQ 30-Stock</div>
                <div class="cc-amount">Rs {fmt_cr(mb['final_value'])} Cr</div>
                <div class="cc-detail">{mb['cagr']}% CAGR &bull; {mb['win_rate']}% win rate</div>
            </div>
            <div class="compare-card" style="background: linear-gradient(135deg, #064e3b, #065f46); border: 2px solid var(--green); border-radius: var(--radius);">
                <div class="cc-label">MQ + Capital Recycling</div>
                <div class="cc-amount" style="color:var(--green);">Rs {fmt_cr(mf['final_value'])} Cr</div>
                <div class="cc-detail">{mf['cagr']}% CAGR &bull; {mf['max_drawdown']}% Max Drawdown</div>
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-title">Growth of Rs 1 Crore: Nifty vs. Basic MQ vs. Full System ({data['period']['start'][:4]}&ndash;{data['period']['end'][:4]})</div>
            <div class="chart-subtitle">All three portfolios start with Rs 1 Cr. Logarithmic scale shows the real compounding difference.</div>
            <canvas id="chartEquity"></canvas>
        </div>

        <div class="chart-container" style="margin-top:40px;">
            <div class="chart-title">Drawdown from Peak: How Deep Do the Dips Go?</div>
            <div class="chart-subtitle">Shows the percentage decline from the highest point reached. Shallower dips = better risk management.</div>
            <canvas id="chartDrawdown"></canvas>
        </div>

        <!-- COVID Crash Deep Dive -->
        <div style="background: linear-gradient(135deg, rgba(239,68,68,0.06), rgba(245,158,11,0.06)); border: 1px solid rgba(239,68,68,0.15); border-radius: var(--radius); padding: 40px 36px; margin: 48px 0;">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:24px;">
                <div style="font-size:28px;">&#129466;</div>
                <div>
                    <h3 style="color:var(--red); font-size:22px; margin-bottom:2px;">Stress Test: The COVID Crash (March 2020)</h3>
                    <p style="color:var(--text-secondary); font-size:14px;">The ultimate test for any system &mdash; a 36% market crash in 30 days</p>
                </div>
            </div>
            <p style="color:var(--text-secondary); line-height:1.8; margin-bottom:24px;">
                In March 2020, the Nifty 50 crashed <strong style="color:var(--red);">-28.8%</strong> from its January high
                in a single month &mdash; the fastest crash in Indian market history. Trading halts, circuit breakers,
                panic everywhere. <strong style="color:var(--text-light);">This is exactly the scenario that breaks most investors.</strong>
            </p>

            <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:20px; margin-bottom:28px;">
                <div style="background:rgba(15,23,42,0.5); border-radius:var(--radius-sm); padding:24px; text-align:center;">
                    <div style="color:var(--text-secondary); font-size:12px; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">Nifty 50</div>
                    <div style="font-size:28px; font-weight:800; color:var(--red); margin-bottom:4px;">-27.7%</div>
                    <div style="color:var(--text-secondary); font-size:13px;">Jan &rarr; Mar 2020</div>
                    <div style="margin-top:12px; border-top:1px solid var(--border); padding-top:12px;">
                        <div style="color:var(--text-secondary); font-size:12px;">Recovery</div>
                        <div style="font-size:16px; font-weight:700; color:var(--text-light);">10 months</div>
                        <div style="color:var(--text-secondary); font-size:12px;">Nov 2020</div>
                    </div>
                </div>
                <div style="background:rgba(15,23,42,0.5); border-radius:var(--radius-sm); padding:24px; text-align:center;">
                    <div style="color:var(--accent); font-size:12px; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">Basic MQ</div>
                    <div style="font-size:28px; font-weight:800; color:var(--accent); margin-bottom:4px;">-19.4%</div>
                    <div style="color:var(--text-secondary); font-size:13px;">Jan &rarr; Mar 2020</div>
                    <div style="margin-top:12px; border-top:1px solid var(--border); padding-top:12px;">
                        <div style="color:var(--text-secondary); font-size:12px;">Recovery</div>
                        <div style="font-size:16px; font-weight:700; color:var(--accent);">8 months</div>
                        <div style="color:var(--text-secondary); font-size:12px;">Sep 2020</div>
                    </div>
                </div>
                <div style="background:rgba(15,23,42,0.5); border-radius:var(--radius-sm); padding:24px; text-align:center;">
                    <div style="color:var(--green); font-size:12px; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">MQ+</div>
                    <div style="font-size:28px; font-weight:800; color:var(--green); margin-bottom:4px;">-18.4%</div>
                    <div style="color:var(--text-secondary); font-size:13px;">Jan &rarr; Mar 2020</div>
                    <div style="margin-top:12px; border-top:1px solid var(--border); padding-top:12px;">
                        <div style="color:var(--text-secondary); font-size:12px;">Recovery</div>
                        <div style="font-size:16px; font-weight:700; color:var(--green);">7 months</div>
                        <div style="color:var(--text-secondary); font-size:12px;">Aug 2020</div>
                    </div>
                </div>
            </div>

            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px; margin-bottom:20px;">
                <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2); border-radius:var(--radius-sm); padding:20px;">
                    <div style="font-weight:700; color:var(--green); margin-bottom:8px;">What the System Did Right</div>
                    <ul style="color:var(--text-secondary); font-size:14px; line-height:1.7; margin-left:16px;">
                        <li>The 20% ATH drawdown exit triggered automatically &mdash; selling weak stocks before the full crash unfolded</li>
                        <li>MQ+ immediately parked freed cash into NIFTYBEES and debt, catching the recovery rally</li>
                        <li>By August 2020 (7 months), MQ+ was back at pre-crash levels. Nifty took 10 months.</li>
                    </ul>
                </div>
                <div style="background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.2); border-radius:var(--radius-sm); padding:20px;">
                    <div style="font-weight:700; color:var(--accent); margin-bottom:8px;">Full Year 2020 Returns</div>
                    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-top:12px;">
                        <div style="text-align:center;">
                            <div style="color:var(--text-secondary); font-size:12px;">Nifty</div>
                            <div style="font-size:20px; font-weight:800; color:var(--green);">+15.4%</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="color:var(--text-secondary); font-size:12px;">Basic MQ</div>
                            <div style="font-size:20px; font-weight:800; color:var(--green);">+22.2%</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="color:var(--text-secondary); font-size:12px;">MQ+</div>
                            <div style="font-size:20px; font-weight:800; color:var(--green);">+28.3%</div>
                        </div>
                    </div>
                    <p style="color:var(--text-secondary); font-size:13px; margin-top:12px; text-align:center;">
                        All three systems finished 2020 <strong style="color:var(--green);">positive</strong> &mdash; despite the worst crash since 2008.
                    </p>
                </div>
            </div>

            <div class="callout" style="margin:0;">
                <strong>Key takeaway:</strong> The MQ system didn't predict COVID. Nobody did.
                But the <strong>systematic 20% drawdown exit</strong> limited the damage, and capital recycling
                ensured the portfolio was <strong>fully invested when the recovery came</strong>.
                That's not luck &mdash; that's rules-based risk management.
            </div>
        </div>

        <!-- Metrics comparison: 3 columns -->
        <div class="table-wrap">
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Nifty 50</th>
                        <th>Basic MQ</th>
                        <th style="color:var(--green);">MQ + Recycling</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="text-align:left; font-weight:600;">Annual Return (CAGR)</td>
                        <td>{bm['cagr']}%</td>
                        <td class="positive">{mb['cagr']}%</td>
                        <td class="positive" style="font-weight:700;">{mf['cagr']}%</td>
                    </tr>
                    <tr>
                        <td style="text-align:left; font-weight:600;">Max Drawdown (Biggest Loss)</td>
                        <td class="negative">-{abs(bm['max_drawdown'])}%</td>
                        <td class="positive">-{mb['max_drawdown']}%</td>
                        <td>-{mf['max_drawdown']}%</td>
                    </tr>
                    <tr>
                        <td style="text-align:left; font-weight:600;">Rs 1 Cr Becomes</td>
                        <td>Rs {fmt_cr(bm['final_value'])} Cr</td>
                        <td class="positive">Rs {fmt_cr(mb['final_value'])} Cr</td>
                        <td class="positive" style="font-weight:700;">Rs {fmt_cr(mf['final_value'])} Cr</td>
                    </tr>
                    <tr>
                        <td style="text-align:left; font-weight:600;">Sharpe Ratio</td>
                        <td>&mdash;</td>
                        <td>{mb['sharpe']}</td>
                        <td class="positive" style="font-weight:700;">{mf['sharpe']}</td>
                    </tr>
                    <tr>
                        <td style="text-align:left; font-weight:600;">Total Trades</td>
                        <td>Buy &amp; Hold</td>
                        <td>{mb['total_trades']}</td>
                        <td>{mf['total_trades']}</td>
                    </tr>
                    <tr>
                        <td style="text-align:left; font-weight:600;">Win Rate</td>
                        <td>&mdash;</td>
                        <td>{mb['win_rate']}%</td>
                        <td>{mf['win_rate']}%</td>
                    </tr>
                    <tr>
                        <td style="text-align:left; font-weight:600;">Avg Winning Trade</td>
                        <td>&mdash;</td>
                        <td class="positive">+{mb['avg_win_pct']}%</td>
                        <td class="positive">+{mf['avg_win_pct']}%</td>
                    </tr>
                    <tr>
                        <td style="text-align:left; font-weight:600;">Avg Losing Trade</td>
                        <td>&mdash;</td>
                        <td class="negative">{mb['avg_loss_pct']}%</td>
                        <td class="negative">{mf['avg_loss_pct']}%</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Year-by-year returns -->
        <div class="chart-container">
            <div class="chart-title">Year-by-Year Returns</div>
            <div class="chart-subtitle">How each system performed in every year. Both MQ systems vs Nifty 50.</div>
            <canvas id="chartYearly"></canvas>
        </div>

        <details style="margin-top:24px;">
            <summary style="cursor:pointer; font-weight:600; color:var(--accent); font-size:15px;">
                View full year-by-year data table
            </summary>
            <div class="table-wrap" style="margin-top:16px;">
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Year</th>
                            <th>Nifty 50</th>
                            <th>Basic MQ</th>
                            <th style="color:var(--green);">MQ + Recycling</th>
                        </tr>
                    </thead>
                    <tbody>
                        {yr_rows}
                    </tbody>
                </table>
            </div>
        </details>
    </div>
</section>

<!-- Cash Trap Discovery & Fix -->
<section id="cash-trap" class="section-mid">
    <div class="container">
        <p class="section-tag">Capital Efficiency</p>
        <h2>Zero Idle Cash: <br><span class="text-accent">Every Rupee Works, Every Day</span></h2>
        <p class="subtitle">
            Most momentum systems have a hidden flaw &mdash; after crash exits, the portfolio sits in 100% cash
            for months, missing the recovery. The MQ+ system eliminates dead money entirely.
        </p>

        <!-- The Problem -->
        <div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.25); border-radius:var(--radius); padding:32px; margin:32px 0;">
            <h3 style="color:var(--red); margin-bottom:16px; font-size:20px;">The Problem With Naive Momentum</h3>
            <p style="color:var(--text-secondary); line-height:1.8;">
                In a basic momentum system, when stocks are sold (20% drawdown from peak), the cash earns <strong style="color:var(--text-light);">0%</strong>.
                During crashes, every stock triggers an exit &mdash; the portfolio goes to 100% cash and stays there until the next rebalance, sometimes months later.
            </p>
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:24px; margin-top:20px;">
                <div style="background:rgba(15,23,42,0.5); padding:20px; border-radius:var(--radius-sm);">
                    <div style="color:var(--red); font-weight:700; font-size:14px; text-transform:uppercase; letter-spacing:1px;">COVID 2020</div>
                    <div style="color:var(--text-light); font-size:24px; font-weight:700; margin:8px 0;">Months at 0%</div>
                    <div style="color:var(--text-secondary); font-size:14px;">A naive system held ZERO stocks from Mar to Jul 2020. Missed the massive recovery rally.</div>
                </div>
                <div style="background:rgba(15,23,42,0.5); padding:20px; border-radius:var(--radius-sm);">
                    <div style="color:var(--red); font-weight:700; font-size:14px; text-transform:uppercase; letter-spacing:1px;">Every Correction</div>
                    <div style="color:var(--text-light); font-size:24px; font-weight:700; margin:8px 0;">Dead money</div>
                    <div style="color:var(--text-secondary); font-size:14px;">2011, 2015, 2018, 2020 &mdash; each correction meant idle cash earning nothing while markets recovered.</div>
                </div>
            </div>
        </div>

        <!-- The MQ Solution: 3 Parts -->
        <h3 style="color:var(--text-light); margin:40px 0 20px; font-size:22px;">How MQ+ Solves This</h3>
        <p style="color:var(--text-secondary); margin-bottom:24px; line-height:1.7;">
            The system has a built-in capital recycling engine with three layers. No cash ever sits idle.
        </p>

        <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:20px; margin-bottom:40px;">
            <div style="background:var(--bg-card); border:1px solid var(--border); border-top:3px solid var(--accent); border-radius:var(--radius); padding:28px 24px;">
                <div style="color:var(--accent); font-weight:700; font-size:13px; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:12px;">Layer 1</div>
                <div style="font-weight:700; font-size:18px; margin-bottom:10px;">Cut &amp; Replace</div>
                <div style="color:var(--text-secondary); font-size:14px; line-height:1.7;">
                    When a stock enters bear territory (20% from its peak &mdash; the classic definition of a bear market),
                    sell it <em>that same day</em>. Immediately screen the universe and enter the next best MQ-ranked stock.
                    Capital is recycled within hours.
                </div>
            </div>
            <div style="background:var(--bg-card); border:1px solid var(--border); border-top:3px solid var(--green); border-radius:var(--radius); padding:28px 24px;">
                <div style="color:var(--green); font-weight:700; font-size:13px; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:12px;">Layer 2</div>
                <div style="font-weight:700; font-size:18px; margin-bottom:10px;">Smart Parking</div>
                <div style="color:var(--text-secondary); font-size:14px; line-height:1.7;">
                    If no replacement stock qualifies (common during crashes),
                    the leftover cash moves into a <strong style="color:var(--text-light);">NIFTYBEES ETF</strong> position
                    when Nifty is below its 200-day SMA &mdash; catching the recovery.
                </div>
            </div>
            <div style="background:var(--bg-card); border:1px solid var(--border); border-top:3px solid #22D3EE; border-radius:var(--radius); padding:28px 24px;">
                <div style="color:#22D3EE; font-weight:700; font-size:13px; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:12px;">Layer 3</div>
                <div style="font-weight:700; font-size:18px; margin-bottom:10px;">Debt Cushion</div>
                <div style="color:var(--text-secondary); font-size:14px; line-height:1.7;">
                    When Nifty is above 200 SMA (no bargain), idle cash earns
                    <strong style="color:var(--text-light);">6.5% p.a.</strong> in a liquid debt fund.
                    Auto-redeemed the moment a new stock opportunity appears.
                </div>
            </div>
        </div>

        <!-- Before vs After Summary -->
        <h3 style="color:var(--text-light); margin:48px 0 20px; font-size:22px;">The Impact: Basic MQ vs. Full System</h3>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:24px; margin-bottom:40px;">
            <div style="background:rgba(239,68,68,0.05); border:1px solid rgba(239,68,68,0.15); border-radius:var(--radius); padding:28px; text-align:center;">
                <div style="color:var(--text-muted); font-size:13px; text-transform:uppercase; letter-spacing:1px; margin-bottom:16px;">Without Capital Recycling</div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
                    <div>
                        <div style="color:var(--text-secondary); font-size:13px;">CAGR</div>
                        <div style="font-size:28px; font-weight:800; color:var(--text-light);">{mb['cagr']}%</div>
                    </div>
                    <div>
                        <div style="color:var(--text-secondary); font-size:13px;">Max Drawdown</div>
                        <div style="font-size:28px; font-weight:800; color:var(--red);">{mb['max_drawdown']}%</div>
                    </div>
                    <div>
                        <div style="color:var(--text-secondary); font-size:13px;">Rs 1 Cr &rarr;</div>
                        <div style="font-size:22px; font-weight:700; color:var(--text-light);">Rs {fmt_cr(mb['final_value'])} Cr</div>
                    </div>
                    <div>
                        <div style="color:var(--text-secondary); font-size:13px;">Sharpe Ratio</div>
                        <div style="font-size:22px; font-weight:700; color:var(--text-light);">{mb['sharpe']}</div>
                    </div>
                </div>
            </div>
            <div style="background:rgba(34,211,238,0.06); border:1px solid rgba(34,211,238,0.2); border-radius:var(--radius); padding:28px; text-align:center;">
                <div style="color:#22D3EE; font-size:13px; text-transform:uppercase; letter-spacing:1px; margin-bottom:16px;">With Capital Recycling</div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
                    <div>
                        <div style="color:var(--text-secondary); font-size:13px;">CAGR</div>
                        <div style="font-size:28px; font-weight:800; color:var(--green);">{mf['cagr']}%</div>
                    </div>
                    <div>
                        <div style="color:var(--text-secondary); font-size:13px;">Max Drawdown</div>
                        <div style="font-size:28px; font-weight:800; color:var(--green);">{mf['max_drawdown']}%</div>
                    </div>
                    <div>
                        <div style="color:var(--text-secondary); font-size:13px;">Rs 1 Cr &rarr;</div>
                        <div style="font-size:22px; font-weight:700; color:#22D3EE;">Rs {fmt_cr(mf['final_value'])} Cr</div>
                    </div>
                    <div>
                        <div style="color:var(--text-secondary); font-size:13px;">Sharpe Ratio</div>
                        <div style="font-size:22px; font-weight:700; color:var(--green);">{mf['sharpe']}</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="callout" style="margin-bottom:40px;">
            <strong>Key Insight:</strong> Capital recycling adds ~{round(mf['cagr'] - mb['cagr'], 1)}% extra CAGR per year
            by ensuring no rupee sits idle. Over {period_years} years, the full system generated
            <strong class="text-green">Rs {fmt_cr(mf['final_value'] - mb['final_value'])} Cr more</strong>
            than the basic version &mdash; purely from smarter cash management.
        </div>
    </div>
</section>

"""

# ═══════════════════════════════════════════════════════════
# BUILD LIVE PORTFOLIO SECTION (dynamic, needs Python loops)
# ═══════════════════════════════════════════════════════════
lp = data.get('live_portfolio', {})
lp_holdings = lp.get('live_holdings', [])
lp_closed = lp.get('closed_trades', [])

html += f"""<!-- Chapter 5: Live Portfolio Snapshot -->
<section id="live-portfolio" class="section-mid">
    <div class="container">
        <p class="section-tag">Chapter 5</p>
        <h2>Live Portfolio Snapshot: <br><span class="text-accent">What It Looks Like Today</span></h2>
        <p class="subtitle">
            If you had started the Basic MQ system on <strong>Jan 1, 2025</strong> with Rs 1 Crore,
            here's exactly what your portfolio would hold as of <strong>{lp.get('period_end', 'N/A')}</strong>.
        </p>
"""

if lp_holdings:
    num_h = lp.get('num_holdings', len(lp_holdings))
    num_w = lp.get('num_winners', 0)
    num_l = lp.get('num_losers', 0)
    fv = lp.get('final_value', 0)
    ret_pct = lp.get('portfolio_return_pct', 0)
    ret_color = 'var(--green)' if ret_pct > 0 else 'var(--red)'

    html += f"""
        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:20px; margin:32px 0;">
            <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); padding:24px; text-align:center;">
                <div style="color:var(--text-secondary); font-size:13px; text-transform:uppercase; letter-spacing:1px;">Stocks Held</div>
                <div style="font-size:32px; font-weight:800; color:var(--text-light); margin:8px 0;">{num_h}</div>
            </div>
            <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); padding:24px; text-align:center;">
                <div style="color:var(--text-secondary); font-size:13px; text-transform:uppercase; letter-spacing:1px;">Portfolio Value</div>
                <div style="font-size:32px; font-weight:800; color:var(--green); margin:8px 0;">Rs {fv/10000000:.2f} Cr</div>
            </div>
            <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); padding:24px; text-align:center;">
                <div style="color:var(--text-secondary); font-size:13px; text-transform:uppercase; letter-spacing:1px;">Winners / Losers</div>
                <div style="font-size:32px; font-weight:800; margin:8px 0;">
                    <span style="color:var(--green);">{num_w}</span>
                    <span style="color:var(--text-secondary); font-size:20px;">/</span>
                    <span style="color:var(--red);">{num_l}</span>
                </div>
            </div>
            <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); padding:24px; text-align:center;">
                <div style="color:var(--text-secondary); font-size:13px; text-transform:uppercase; letter-spacing:1px;">Portfolio Return</div>
                <div style="font-size:32px; font-weight:800; color:{ret_color}; margin:8px 0;">{ret_pct:+.1f}%</div>
            </div>
        </div>

        <div class="table-wrap">
            <table class="comparison-table" style="font-size:14px;">
                <thead>
                    <tr>
                        <th style="text-align:left;">#</th>
                        <th style="text-align:left;">Stock</th>
                        <th style="text-align:left;">Sector</th>
                        <th>Entry Date</th>
                        <th>Buy Price</th>
                        <th>Current Price</th>
                        <th>Return</th>
                        <th>P/L (Rs)</th>
                    </tr>
                </thead>
                <tbody>
"""
    for i, h in enumerate(lp_holdings, 1):
        rc = 'positive' if h['return_pct'] > 0 else ('negative' if h['return_pct'] < 0 else '')
        pnl_str = f"Rs {{abs(h['pnl']):,.0f}}" if h.get('pnl', 0) >= 0 else f"-Rs {{abs(h['pnl']):,.0f}}"
        # Use format() instead of f-string for the pnl to handle commas
        pnl_val = h.get('pnl', 0)
        if pnl_val >= 0:
            pnl_str = f"Rs {abs(pnl_val):,.0f}"
        else:
            pnl_str = f"-Rs {abs(pnl_val):,.0f}"

        html += f"""
                    <tr>
                        <td style="text-align:left; color:var(--text-secondary);">{i}</td>
                        <td style="text-align:left; font-weight:600;">{h['symbol']}</td>
                        <td style="text-align:left; color:var(--text-secondary); font-size:12px;">{h['sector']}</td>
                        <td>{h['entry_date']}</td>
                        <td>Rs {h['entry_price']:,.1f}</td>
                        <td>Rs {h['current_price']:,.1f}</td>
                        <td class="{rc}" style="font-weight:700;">{h['return_pct']:+.1f}%</td>
                        <td class="{rc}">{pnl_str}</td>
                    </tr>"""

    html += """
                </tbody>
            </table>
        </div>"""
else:
    html += f"""
        <div class="info-box" style="margin:32px 0;">
            <h4>No Open Positions</h4>
            <p>
                All stocks from the Jan 2025 entry have been exited via the 20% ATH drawdown rule.
                At the July 2025 rebalance, the system would re-enter fresh stocks based on updated rankings.
            </p>
        </div>"""

# Closed trades during snapshot period
if lp_closed:
    html += f"""
        <div style="margin-top:40px;">
            <h3 style="color:var(--text-light); font-size:18px; margin-bottom:16px;">
                Exited During Period ({len(lp_closed)} trades)
            </h3>
            <p style="color:var(--text-secondary); margin-bottom:16px; font-size:14px;">
                These stocks were sold because they hit the 20% ATH drawdown exit trigger.
            </p>
            <div class="table-wrap">
                <table class="comparison-table" style="font-size:13px;">
                    <thead>
                        <tr>
                            <th style="text-align:left;">Stock</th>
                            <th>Entry</th>
                            <th>Exit Reason</th>
                            <th>Return</th>
                            <th>Days Held</th>
                        </tr>
                    </thead>
                    <tbody>"""
    for t in lp_closed:
        trc = 'positive' if t['return_pct'] > 0 else ('negative' if t['return_pct'] < 0 else '')
        reason_display = t.get('exit_reason', '').replace('_', ' ').title()
        html += f"""
                        <tr>
                            <td style="text-align:left; font-weight:600;">{t['symbol']}</td>
                            <td>{t['entry_date']}</td>
                            <td style="color:var(--text-secondary);">{reason_display}</td>
                            <td class="{trc}" style="font-weight:700;">{t['return_pct']:+.1f}%</td>
                            <td>{t['holding_days']}d</td>
                        </tr>"""
    html += """
                    </tbody>
                </table>
            </div>
        </div>"""

html += """
    </div>
</section>
"""

# ═══════════════════════════════════════════════════════════
# CONTINUE WITH REMAINING SECTIONS
# ═══════════════════════════════════════════════════════════
html += f"""<!-- Chapter 6: Getting Started -->
<section id="tools" class="section-light">
    <div class="container">
        <p class="section-tag">Chapter 6</p>
        <h2>Getting Started: <br>Tools You Need</h2>
        <p class="subtitle">
            You don't need expensive software or fancy algorithms.
            Here are the free and low-cost tools to run the MQ system yourself.
        </p>

        <div class="tool-cards">
            <div class="tool-card">
                <div class="tool-icon">&#128270;</div>
                <h4>Screener.in</h4>
                <p>
                    Free stock screener for Indian markets. Filter Nifty 500 stocks by
                    "Close to 52-week high" to find momentum candidates.
                    Sort by distance from ATH &mdash; top stocks are your MQ picks.
                </p>
            </div>
            <div class="tool-card">
                <div class="tool-icon">&#128200;</div>
                <h4>TradingView</h4>
                <p>
                    Free charting platform. Set alerts for when stocks drop 20% from their peak
                    (ATH drawdown exit). Use weekly charts to confirm momentum trends before entering.
                </p>
            </div>
            <div class="tool-card">
                <div class="tool-icon">&#9889;</div>
                <h4>Zerodha Kite</h4>
                <p>
                    India's largest broker. Use Kite for execution. Their API (KiteConnect)
                    enables automated trading if you want to fully automate the MQ system.
                    GTT orders handle stop-losses automatically.
                </p>
            </div>
            <div class="tool-card">
                <div class="tool-icon">&#128202;</div>
                <h4>Google Sheets</h4>
                <p>
                    Track your MQ portfolio in a simple spreadsheet. Log entry dates,
                    prices, ATH values, and current drawdown per stock.
                    Set conditional formatting to flag stocks crossing the 20% drawdown threshold.
                </p>
            </div>
        </div>

        <div class="info-box">
            <h4>Layer 1: Basic MQ &mdash; Manual Mode (5 Minutes, Twice a Year)</h4>
            <p>
                1. Go to <strong>Screener.in</strong> &rarr; Screens &rarr; Create New<br>
                2. Filter: Market Cap > 5,000 Cr (roughly Nifty 500)<br>
                3. Filter: Stock must be within <strong>15% of its all-time high</strong> (Current Price / ATH &ge; 0.85)<br>
                4. Sort by: Distance from ATH ascending (closest to ATH = strongest momentum)<br>
                5. Top 30 qualifying stocks = your MQ portfolio<br>
                6. Equal-weight each position (~3.3% of capital per stock)<br>
                7. Repeat every <strong>January and July</strong> &mdash; sell what dropped out, buy what came in
            </p>
            <p style="margin-top: 0.75rem; color: var(--text-secondary); font-size: 0.9rem;">
                This is the simplest version. Two trades a year, <strong>{mb['cagr']:.2f}% CAGR</strong>
                with {mb['max_drawdown']:.1f}% max drawdown over {period_years} years.
                No daily monitoring needed &mdash; set it and check back in 6 months.
            </p>
        </div>

        <div class="info-box" style="border-left-color: var(--accent-green);">
            <h4>Layer 2: MQ + Capital Recycling &mdash; Best Done Automated</h4>
            <p>
                Everything from Layer 1, <strong>plus</strong> three daily enhancements:<br><br>
                <strong>1. Daily ATH Drawdown Exit:</strong> Monitor every holding daily.
                If any stock drops <strong>20% from its post-entry peak</strong>, sell it that day.<br>
                <strong>2. Immediate Replacement:</strong> Same day, buy the next highest-ranked stock
                from the MQ list that you don't already hold.<br>
                <strong>3. Smart Cash Parking:</strong> Any idle cash (between exits and replacements)
                goes into <strong>NIFTYBEES</strong> (Nifty 50 ETF) + a liquid debt fund earning ~6.5% p.a.
            </p>
            <p style="margin-top: 0.75rem; padding: 0.6rem 0.8rem; background: rgba(255,183,77,0.08); border-radius: 6px; font-size: 0.9rem;">
                &#9888;&#65039; <strong>Practical note:</strong> This layer requires daily price monitoring,
                same-day execution of exits &amp; replacements, and splitting idle cash across instruments.
                While possible manually, it is <strong>significantly easier with automation</strong>
                (e.g., a Python script connected to your broker's API).
                The added ~{mf['cagr'] - mb['cagr']:.1f}% CAGR (total <strong>{mf['cagr']:.2f}% CAGR</strong>,
                {mf['max_drawdown']:.1f}% max drawdown) comes from disciplined daily execution
                that is hard to replicate by hand consistently over {period_years} years.
            </p>
        </div>
    </div>
</section>

<!-- Chapter 7: The Fine Print -->
<section id="fine-print" class="section-light">
    <div class="container">
        <p class="section-tag">Chapter 7</p>
        <h2>The Honest Truth: <br>Risks & Caveats</h2>
        <p class="subtitle">
            No system is perfect. Here's what you need to know before putting real money to work.
        </p>

        <div class="warning-box">
            <h4>Survivorship Bias Note</h4>
            <p>
                This backtest uses today's Nifty 500 constituents applied backwards to {data['period']['start'][:4]}.
                Stocks that went bankrupt, were delisted, or merged before today are excluded.
                This means the returns may be slightly <strong>overstated</strong> because we're
                only looking at stocks that survived. We use {data['period']['start'][:4]}&ndash;{data['period']['end'][:4]}
                ({period_years} years) to minimize this effect.
            </p>
        </div>

        <div class="two-col">
            <div>
                <h3 style="margin-bottom:16px; color:var(--text-dark);">What Could Go Wrong</h3>
                <div class="warning-box">
                    <ul>
                        <li><strong>Crashes hurt:</strong> Both MQ systems' max drawdown was around {mb['max_drawdown']}&ndash;{mf['max_drawdown']}%.
                            While better than Nifty's {abs(bm['max_drawdown'])}%, it's still significant capital loss at peak.</li>
                        <li><strong>Momentum crashes:</strong> When trends reverse suddenly, momentum stocks
                            get hit especially hard before the system can exit.</li>
                        <li><strong>Discipline is everything:</strong> The system only works if you follow it mechanically.
                            Skipping a rebalance or holding a loser "because it might come back" breaks the system.</li>
                        <li><strong>Taxes:</strong> Semi-annual rebalancing generates short-term capital gains (15% STCG).
                            This reduces net returns by ~1-2% annually.</li>
                        <li><strong>Slippage:</strong> In real trading, you won't get the exact prices from the backtest.
                            Expect 0.5-1% slippage per trade in mid/small caps.</li>
                    </ul>
                </div>
            </div>
            <div>
                <h3 style="margin-bottom:16px; color:var(--text-dark);">What Works in Your Favor</h3>
                <div class="info-box">
                    <ul style="margin-left:20px;">
                        <li><strong>Transaction costs included:</strong> The backtest includes STT, brokerage, GST,
                            and stamp duty. The returns are after costs.</li>
                        <li><strong>Conservative parameters:</strong> 20% ATH drawdown exit and 50% hard stop are
                            conservative. No aggressive leverage or options.</li>
                        <li><strong>Long-only:</strong> No short selling, no derivatives. Simple cash equity positions.</li>
                        <li><strong>Lower drawdown than Nifty:</strong> Both MQ systems had max drawdown of ~{mb['max_drawdown']}%
                            vs Nifty's {abs(bm['max_drawdown'])}%. The systematic exits protect capital.</li>
                        <li><strong>Evidence-based:</strong> Momentum is one of the most researched factors in finance,
                            documented by academics globally since the 1990s.</li>
                    </ul>
                </div>
                <div class="callout" style="margin-top:24px;">
                    <strong>The bottom line:</strong> Past performance does not guarantee future results.
                    This is a research tool, not investment advice. Always do your own due diligence.
                </div>
            </div>
        </div>
    </div>
</section>

<!-- Footer -->
<footer>
    <div style="margin-bottom:16px;">
        <span style="font-size:24px; font-weight:800; color:#22D3EE; letter-spacing:-0.5px;">C</span>
        <span style="font-size:16px; font-weight:700; color:var(--text-light); letter-spacing:-0.3px;">astronix</span>
        <span style="font-size:12px; color:var(--text-secondary); margin-left:8px;">Research &amp; Quantitative Systems</span>
    </div>
    <p>
        Built with Python &bull; Backtested on {period_years} years of NSE data ({data['period']['start'][:4]}&ndash;{data['period']['end'][:4]}) &bull;
        395 stocks from Nifty 500 universe &bull; Transaction costs included
    </p>
    <p style="margin-top:8px; font-size:13px; color:var(--text-secondary);">
        <a href="mailto:arun.castromin@gmail.com" style="color:#22D3EE; text-decoration:none;">arun.castromin@gmail.com</a>
        &bull; <a href="https://github.com/castroarun" style="color:var(--text-secondary); text-decoration:none;">GitHub</a>
        &bull; <a href="https://www.linkedin.com/in/aruncastromin" style="color:var(--text-secondary); text-decoration:none;">LinkedIn</a>
        &bull; <a href="https://x.com/MetricsTrader" style="color:var(--text-secondary); text-decoration:none;">@MetricsTrader</a>
    </p>
    <p style="margin-top:8px; font-size:12px; color:var(--text-muted);">
        Generated on {data['generated_at']} &bull; For educational purposes only. Not investment advice.
    </p>
</footer>

<!-- Chart.js Rendering -->
<script>
const LABELS = {json.dumps(chart_labels)};
const BM_EQ = {json.dumps(chart_bm)};

// Equity comparison (all start at Rs 1 Cr)
const EQ_LABELS = {json.dumps(eq_labels)};
const EQ_BM = {json.dumps(eq_bm)};
const EQ_BASIC = {json.dumps(eq_basic)};
const EQ_FULL = {json.dumps(eq_full)};

const DD_BM = {json.dumps(chart_dd_bm)};
const DD_BASIC = {json.dumps(chart_dd_basic)};
const DD_FULL = {json.dumps(chart_dd_full)};

const YR_LABELS = {json.dumps(yr_keys)};
const YR_BM = {json.dumps(yr_bm)};
const YR_BASIC = {json.dumps(yr_basic)};
const YR_FULL = {json.dumps(yr_full)};

// Format INR
function fmtCr(val) {{
    return '\\u20B9' + (val / 10000000).toFixed(1) + ' Cr';
}}

Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.color = '#94a3b8';

// 1. Benchmark chart
const ctxBm = document.getElementById('chartBenchmark').getContext('2d');
const gradBm = ctxBm.createLinearGradient(0, 0, 0, 250);
gradBm.addColorStop(0, 'rgba(59, 130, 246, 0.3)');
gradBm.addColorStop(1, 'rgba(59, 130, 246, 0.0)');

new Chart(ctxBm, {{
    type: 'line',
    data: {{
        labels: LABELS,
        datasets: [{{
            label: 'Nifty 50',
            data: BM_EQ,
            borderColor: '#3b82f6',
            backgroundColor: gradBm,
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 2
        }}]
    }},
    options: {{
        responsive: true,
        aspectRatio: 3,
        plugins: {{
            legend: {{ display: false }},
            tooltip: {{
                callbacks: {{
                    label: (ctx) => fmtCr(ctx.parsed.y)
                }}
            }}
        }},
        scales: {{
            x: {{
                grid: {{ display: false }},
                ticks: {{ maxTicksLimit: 10, color: '#64748b' }}
            }},
            y: {{
                grid: {{ color: 'rgba(148,163,184,0.1)' }},
                ticks: {{
                    callback: (v) => fmtCr(v),
                    color: '#64748b'
                }}
            }}
        }}
    }}
}});

// 2. Compounding chart with endpoint labels
const ctxComp = document.getElementById('chartCompounding').getContext('2d');
function compound(rate, years) {{
    const data = [10000000];
    for (let i = 1; i <= years; i++) data.push(Math.round(10000000 * Math.pow(1 + rate/100, i)));
    return data;
}}
const compLabels = Array.from({{length: 21}}, (_, i) => 'Year ' + i);

// Endpoint labels plugin
const endpointLabelsPlugin = {{
    id: 'endpointLabels',
    afterDatasetsDraw(chart) {{
        const {{ ctx }} = chart;
        chart.data.datasets.forEach((dataset, i) => {{
            const meta = chart.getDatasetMeta(i);
            const lastPoint = meta.data[meta.data.length - 1];
            if (!lastPoint) return;
            const val = dataset.data[dataset.data.length - 1];
            const label = (val / 10000000).toFixed(0);
            const x = lastPoint.x;
            const y = lastPoint.y;
            // Circle
            ctx.save();
            ctx.beginPath();
            ctx.arc(x, y, 18, 0, Math.PI * 2);
            ctx.fillStyle = dataset.borderColor;
            ctx.fill();
            // Text
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 10px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(label + 'Cr', x, y);
            ctx.restore();
        }});
    }}
}};

new Chart(ctxComp, {{
    type: 'line',
    data: {{
        labels: compLabels,
        datasets: [
            {{ label: '7% (FD)', data: compound(7, 20), borderColor: '#64748b', borderWidth: 1.5, borderDash: [5,5], pointRadius: 0, tension: 0.3 }},
            {{ label: '11% (Nifty)', data: compound(11, 20), borderColor: '#3b82f6', borderWidth: 2, pointRadius: 0, tension: 0.3 }},
            {{ label: '17% (Basic MQ)', data: compound(17, 20), borderColor: '#f59e0b', borderWidth: 2.5, pointRadius: 0, tension: 0.3 }},
            {{ label: '20% (MQ + Recycling)', data: compound(20, 20), borderColor: '#10b981', borderWidth: 2.5, pointRadius: 0, tension: 0.3 }}
        ]
    }},
    plugins: [endpointLabelsPlugin],
    options: {{
        responsive: true,
        aspectRatio: 3,
        layout: {{ padding: {{ right: 30 }} }},
        plugins: {{
            legend: {{ position: 'top', labels: {{ usePointStyle: true, padding: 20 }} }},
            tooltip: {{ callbacks: {{ label: (ctx) => ctx.dataset.label + ': ' + fmtCr(ctx.parsed.y) }} }}
        }},
        scales: {{
            x: {{ grid: {{ display: false }} }},
            y: {{
                type: 'logarithmic',
                grid: {{ color: 'rgba(148,163,184,0.1)' }},
                ticks: {{ callback: (v) => fmtCr(v) }}
            }}
        }}
    }}
}});

// 3. Equity curve comparison (3 lines: Nifty, Basic MQ, Full System)
const ctxEq = document.getElementById('chartEquity').getContext('2d');
const gradFull = ctxEq.createLinearGradient(0, 0, 0, 250);
gradFull.addColorStop(0, 'rgba(16, 185, 129, 0.15)');
gradFull.addColorStop(1, 'rgba(16, 185, 129, 0.0)');

new Chart(ctxEq, {{
    type: 'line',
    data: {{
        labels: EQ_LABELS,
        datasets: [
            {{
                label: 'Nifty 50',
                data: EQ_BM,
                borderColor: '#64748b',
                borderWidth: 1.5,
                pointRadius: 0,
                tension: 0.3
            }},
            {{
                label: 'Basic MQ',
                data: EQ_BASIC,
                borderColor: '#f59e0b',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3
            }},
            {{
                label: 'MQ + Recycling',
                data: EQ_FULL,
                borderColor: '#10b981',
                backgroundColor: gradFull,
                fill: true,
                borderWidth: 2.5,
                pointRadius: 0,
                tension: 0.3
            }}
        ]
    }},
    options: {{
        responsive: true,
        aspectRatio: 2.5,
        plugins: {{
            legend: {{ position: 'top', labels: {{ usePointStyle: true, padding: 20 }} }},
            tooltip: {{
                mode: 'index',
                intersect: false,
                callbacks: {{ label: (ctx) => ctx.dataset.label + ': ' + fmtCr(ctx.parsed.y) }}
            }}
        }},
        interaction: {{ mode: 'nearest', axis: 'x', intersect: false }},
        scales: {{
            x: {{ grid: {{ display: false }}, ticks: {{ maxTicksLimit: 10 }} }},
            y: {{
                type: 'logarithmic',
                grid: {{ color: 'rgba(148,163,184,0.15)' }},
                ticks: {{ callback: (v) => fmtCr(v) }}
            }}
        }}
    }}
}});

// 4. Drawdown chart (3 lines - stacked area for clarity)
const ctxDD = document.getElementById('chartDrawdown').getContext('2d');

// Nifty fill (light red)
const gradNiftyDD = ctxDD.createLinearGradient(0, 0, 0, 300);
gradNiftyDD.addColorStop(0, 'rgba(100, 116, 139, 0.0)');
gradNiftyDD.addColorStop(1, 'rgba(100, 116, 139, 0.2)');

// MQ fill (light orange)
const gradBasicDD = ctxDD.createLinearGradient(0, 0, 0, 300);
gradBasicDD.addColorStop(0, 'rgba(245, 158, 11, 0.0)');
gradBasicDD.addColorStop(1, 'rgba(245, 158, 11, 0.15)');

new Chart(ctxDD, {{
    type: 'line',
    data: {{
        labels: EQ_LABELS,
        datasets: [
            {{
                label: 'Nifty 50',
                data: DD_BM,
                borderColor: 'rgba(100, 116, 139, 0.8)',
                backgroundColor: gradNiftyDD,
                fill: true,
                borderWidth: 1,
                borderDash: [4, 3],
                pointRadius: 0,
                tension: 0.3,
                order: 3
            }},
            {{
                label: 'Basic MQ',
                data: DD_BASIC,
                borderColor: '#f59e0b',
                backgroundColor: gradBasicDD,
                fill: true,
                borderWidth: 2.5,
                pointRadius: 0,
                tension: 0.3,
                order: 2
            }},
            {{
                label: 'MQ+',
                data: DD_FULL,
                borderColor: '#10b981',
                borderWidth: 3,
                pointRadius: 0,
                tension: 0.3,
                order: 1
            }}
        ]
    }},
    options: {{
        responsive: true,
        aspectRatio: 2.5,
        plugins: {{
            legend: {{ position: 'top', labels: {{ usePointStyle: true, padding: 20 }} }},
            tooltip: {{
                mode: 'index',
                intersect: false,
                callbacks: {{ label: (ctx) => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + '%' }}
            }}
        }},
        interaction: {{ mode: 'nearest', axis: 'x', intersect: false }},
        scales: {{
            x: {{ grid: {{ display: false }}, ticks: {{ maxTicksLimit: 12 }} }},
            y: {{
                reverse: false,
                grid: {{ color: 'rgba(148,163,184,0.1)' }},
                ticks: {{ callback: (v) => v.toFixed(0) + '%' }},
                title: {{ display: true, text: 'Drawdown from Peak', color: '#94a3b8' }}
            }}
        }}
    }}
}});

// 5. Yearly returns bar chart (3 systems)
const ctxYr = document.getElementById('chartYearly').getContext('2d');

new Chart(ctxYr, {{
    type: 'bar',
    data: {{
        labels: YR_LABELS,
        datasets: [
            {{ label: 'Nifty 50', data: YR_BM, backgroundColor: 'rgba(100, 116, 139, 0.6)', borderRadius: 3 }},
            {{ label: 'Basic MQ', data: YR_BASIC, backgroundColor: 'rgba(245, 158, 11, 0.7)', borderRadius: 3 }},
            {{ label: 'MQ + Recycling', data: YR_FULL, backgroundColor: 'rgba(16, 185, 129, 0.7)', borderRadius: 3 }}
        ]
    }},
    options: {{
        responsive: true,
        aspectRatio: 3,
        plugins: {{
            legend: {{ position: 'top', labels: {{ usePointStyle: true, padding: 20 }} }},
            tooltip: {{ callbacks: {{ label: (ctx) => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + '%' }} }}
        }},
        scales: {{
            x: {{ grid: {{ display: false }} }},
            y: {{
                grid: {{ color: 'rgba(148,163,184,0.15)' }},
                ticks: {{ callback: (v) => v + '%' }}
            }}
        }}
    }}
}});

// ── Count-up animation for hero stats ──
function animateCountUp(el) {{
    const target = parseFloat(el.dataset.count);
    const decimals = parseInt(el.dataset.decimals || '0');
    const suffix = el.dataset.suffix || '';
    const duration = 1800;
    const start = performance.now();

    function tick(now) {{
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        // Ease-out cubic for smooth deceleration
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = eased * target;
        el.textContent = current.toFixed(decimals) + suffix;
        if (progress < 1) requestAnimationFrame(tick);
    }}
    requestAnimationFrame(tick);
}}

// Trigger when hero stats scroll into view (or on load if already visible)
const heroObserver = new IntersectionObserver((entries) => {{
    entries.forEach(entry => {{
        if (entry.isIntersecting) {{
            document.querySelectorAll('.hero-stat .number[data-count]').forEach(el => {{
                animateCountUp(el);
            }});
            heroObserver.disconnect();
        }}
    }});
}}, {{ threshold: 0.3 }});

const heroSection = document.getElementById('hero');
if (heroSection) heroObserver.observe(heroSection);
</script>

</body>
</html>"""

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Report generated: {OUTPUT_FILE}")
print(f"File size: {len(html):,} bytes ({len(html)//1024} KB)")
