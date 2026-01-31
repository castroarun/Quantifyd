"""
MQ Reporting Agent
===================

Generates standalone HTML reports from agent outputs.
Reports are self-contained with embedded Chart.js and Bootstrap.

Report types:
- Daily brief (from MonitoringAgent)
- Weekly digest (aggregated signals)
- Monthly screening (from ScreeningAgent)
- Rebalance (from RebalanceAgent)
- Backtest (from BacktestAgent)
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

from .mq_agent_db import get_agent_db, MQAgentDB
from .mq_agent_reports import (
    ScreeningReport, MonitoringReport, RebalanceReport, BacktestReportData
)

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / 'reports'
REPORTS_DIR.mkdir(exist_ok=True)

# HTML template constants
CDN_BOOTSTRAP = 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css'
CDN_CHARTJS = 'https://cdn.jsdelivr.net/npm/chart.js'
CDN_ICONS = 'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css'


class ReportingAgent:
    """Generates HTML reports from agent outputs."""

    def __init__(self, db: MQAgentDB = None):
        self.db = db or get_agent_db()

    def generate_daily_brief(self, report: MonitoringReport) -> str:
        """Generate daily monitoring brief. Returns file path."""
        signals_html = self._render_signals_table(report.signals)

        sections = f"""
        <div class="row mb-4">
            <div class="col-md-4">
                <div class="card bg-dark text-light">
                    <div class="card-body text-center">
                        <h6 class="text-muted">Portfolio Value</h6>
                        <h3>{self._fmt_currency(report.portfolio_value)}</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card bg-dark text-light">
                    <div class="card-body text-center">
                        <h6 class="text-muted">Debt Fund</h6>
                        <h3>{self._fmt_currency(report.debt_fund_balance)}</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card bg-dark text-light">
                    <div class="card-body text-center">
                        <h6 class="text-muted">Signals</h6>
                        <h3>{len(report.signals)}</h3>
                        <small>{report.consolidation_count} consolidating, {report.breakout_count} breakouts</small>
                    </div>
                </div>
            </div>
        </div>

        <h5>Active Signals</h5>
        {signals_html}
        """

        html = self._render_html(
            f"Daily Brief - {report.run_date.strftime('%b %d, %Y')}",
            sections
        )
        return self._save_report(html, 'daily_brief', report.run_date)

    def generate_weekly_digest(self, from_date: datetime, to_date: datetime) -> str:
        """Generate weekly digest of signals."""
        run_id = self.db.start_agent_run('reporting')

        # Aggregate signals from the past week
        all_signals = self.db.get_active_signals(limit=100)
        week_signals = [s for s in all_signals]  # Already filtered by status

        runs = self.db.get_recent_runs(limit=20)
        week_runs = [r for r in runs if r.get('run_date', '') >= from_date.isoformat()]

        sections = f"""
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card bg-dark text-light">
                    <div class="card-body text-center">
                        <h6 class="text-muted">Agent Runs This Week</h6>
                        <h3>{len(week_runs)}</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card bg-dark text-light">
                    <div class="card-body text-center">
                        <h6 class="text-muted">Active Signals</h6>
                        <h3>{len(week_signals)}</h3>
                    </div>
                </div>
            </div>
        </div>

        <h5>Signals</h5>
        <table class="table table-dark table-sm">
            <thead><tr><th>Type</th><th>Symbol</th><th>Priority</th><th>Date</th></tr></thead>
            <tbody>
            {''.join(f'<tr><td><span class="badge bg-{self._priority_color(s.get("priority",""))}">{s.get("signal_type","")}</span></td><td>{s.get("symbol","")}</td><td>{s.get("priority","")}</td><td>{s.get("created_date","")[:10]}</td></tr>' for s in week_signals)}
            </tbody>
        </table>

        <h5>Agent Activity</h5>
        <table class="table table-dark table-sm">
            <thead><tr><th>Agent</th><th>Date</th><th>Status</th><th>Summary</th></tr></thead>
            <tbody>
            {''.join(f'<tr><td>{r.get("agent_type","")}</td><td>{r.get("run_date","")[:16]}</td><td><span class="badge bg-{"success" if r.get("status")=="completed" else "danger"}">{r.get("status","")}</span></td><td>{r.get("summary","")}</td></tr>' for r in week_runs)}
            </tbody>
        </table>
        """

        html = self._render_html(
            f"Weekly Digest - {from_date.strftime('%b %d')} to {to_date.strftime('%b %d, %Y')}",
            sections
        )
        path = self._save_report(html, 'weekly_digest', to_date)

        self.db.complete_agent_run(
            run_id, report_path=path, report_type='weekly_digest',
            summary=f"Weekly: {len(week_runs)} runs, {len(week_signals)} signals"
        )
        return path

    def generate_monthly_screening(self, report: ScreeningReport) -> str:
        """Generate monthly screening report with funnel."""
        funnel = report.funnel

        # Top 30 table
        top30_rows = ''
        for stock in report.top_ranked[:30]:
            top30_rows += f"""
            <tr>
                <td>{stock.get('rank', '')}</td>
                <td><strong>{stock.get('symbol', '')}</strong></td>
                <td>{stock.get('sector', '')}</td>
                <td>{stock.get('distance_from_ath', '')}%</td>
                <td>{stock.get('composite_score', '')}</td>
                <td>{'Yes' if stock.get('passes_all') else 'No'}</td>
            </tr>"""

        sections = f"""
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card bg-dark text-light">
                    <div class="card-body text-center">
                        <h6 class="text-muted">Regime</h6>
                        <h3><span class="badge bg-{'success' if report.regime.regime == 'BULLISH' else 'warning' if report.regime.regime == 'HIGH_VOLATILITY' else 'danger'}">{report.regime.regime}</span></h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-dark text-light">
                    <div class="card-body text-center">
                        <h6 class="text-muted">Universe</h6>
                        <h3>{report.total_scanned}</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-dark text-light">
                    <div class="card-body text-center">
                        <h6 class="text-muted">Momentum</h6>
                        <h3>{report.momentum_passed}</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-dark text-light">
                    <div class="card-body text-center">
                        <h6 class="text-muted">Quality</h6>
                        <h3>{report.quality_passed}</h3>
                    </div>
                </div>
            </div>
        </div>

        <h5>Screening Funnel</h5>
        <canvas id="funnelChart" height="80"></canvas>

        <h5 class="mt-4">Top 30 Ranked Stocks</h5>
        <table class="table table-dark table-striped table-sm">
            <thead><tr><th>#</th><th>Symbol</th><th>Sector</th><th>ATH Distance</th><th>Quality Score</th><th>Passes All</th></tr></thead>
            <tbody>{top30_rows}</tbody>
        </table>

        <script>
        new Chart(document.getElementById('funnelChart'), {{
            type: 'bar',
            data: {{
                labels: ['Universe', 'Momentum Filter', 'Quality Filter', 'Top 30'],
                datasets: [{{
                    data: [{funnel.get('universe', 0)}, {funnel.get('momentum', 0)}, {funnel.get('quality', 0)}, {funnel.get('top_30', 0)}],
                    backgroundColor: ['#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd'],
                    borderRadius: 4,
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{ x: {{ grid: {{ color: '#334155' }} }}, y: {{ grid: {{ display: false }} }} }}
            }}
        }});
        </script>
        """

        html = self._render_html(
            f"Monthly Screening - {report.run_date.strftime('%b %Y')}",
            sections
        )
        return self._save_report(html, 'monthly_screening', report.run_date)

    def generate_rebalance_report(self, report: RebalanceReport) -> str:
        """Generate rebalance report with exits and entries."""
        exit_rows = ''.join(
            f"<tr><td>{e.get('symbol','')}</td><td class='text-danger'>{e.get('drawdown_pct','')}%</td><td>{e.get('reason','')}</td></tr>"
            for e in report.exits
        )
        entry_rows = ''.join(
            f"<tr><td>{e.get('symbol','')}</td><td>{e.get('sector','')}</td><td>{e.get('composite_score','')}</td></tr>"
            for e in report.entries
        )

        sections = f"""
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card bg-dark text-light border-danger">
                    <div class="card-body text-center">
                        <h6 class="text-danger">Exits</h6>
                        <h3>{len(report.exits)}</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card bg-dark text-light border-success">
                    <div class="card-body text-center">
                        <h6 class="text-success">Entries</h6>
                        <h3>{len(report.entries)}</h3>
                    </div>
                </div>
            </div>
        </div>

        <h5>Exits (ATH Drawdown > 20%)</h5>
        <table class="table table-dark table-sm">
            <thead><tr><th>Symbol</th><th>Drawdown</th><th>Reason</th></tr></thead>
            <tbody>{exit_rows if exit_rows else '<tr><td colspan="3" class="text-muted">No exits</td></tr>'}</tbody>
        </table>

        <h5>New Entries</h5>
        <table class="table table-dark table-sm">
            <thead><tr><th>Symbol</th><th>Sector</th><th>Quality Score</th></tr></thead>
            <tbody>{entry_rows if entry_rows else '<tr><td colspan="3" class="text-muted">No new entries</td></tr>'}</tbody>
        </table>
        """

        html = self._render_html(
            f"Rebalance Report - {report.run_date.strftime('%b %Y')}",
            sections
        )
        return self._save_report(html, 'rebalance', report.run_date)

    def generate_backtest_report(self, report: BacktestReportData) -> str:
        """Generate backtest report with equity curve and metrics."""
        m = report.metrics

        # Prepare chart data
        equity_dates = list(report.equity_curve.keys())
        equity_values = list(report.equity_curve.values())

        # Normalize strategy to base 100
        base = equity_values[0] if equity_values else 1
        strategy_norm = [round((v / base) * 100, 2) for v in equity_values]

        benchmark_datasets = ''
        colors = {'Nifty 50': '#fbbf24', 'Nifty 500': '#22d3ee'}
        for bm_name, bm_data in report.benchmark_curves.items():
            bm_values = [bm_data.get(d, None) for d in equity_dates]
            # Normalize benchmarks to base 100 (same scale as strategy)
            bm_base = next((v for v in bm_values if v is not None), 1)
            bm_norm = [round((v / bm_base) * 100, 2) if v is not None else None for v in bm_values]
            color = colors.get(bm_name, '#94a3b8')
            benchmark_datasets += f"""{{
                label: '{bm_name}',
                data: {json.dumps(bm_norm)},
                borderColor: '{color}',
                borderWidth: 1.5,
                pointRadius: 0,
                fill: false,
                spanGaps: true,
            }},"""

        sections = f"""
        <div class="row mb-4">
            <div class="col-md-2">
                <div class="card bg-dark text-light"><div class="card-body text-center">
                    <h6 class="text-muted">CAGR</h6><h3 class="text-success">{m.get('cagr', 0)}%</h3>
                </div></div>
            </div>
            <div class="col-md-2">
                <div class="card bg-dark text-light"><div class="card-body text-center">
                    <h6 class="text-muted">Sharpe</h6><h3>{m.get('sharpe_ratio', 0)}</h3>
                </div></div>
            </div>
            <div class="col-md-2">
                <div class="card bg-dark text-light"><div class="card-body text-center">
                    <h6 class="text-muted">Sortino</h6><h3>{m.get('sortino_ratio', 0)}</h3>
                </div></div>
            </div>
            <div class="col-md-2">
                <div class="card bg-dark text-light"><div class="card-body text-center">
                    <h6 class="text-muted">Max DD</h6><h3 class="text-danger">{m.get('max_drawdown', 0)}%</h3>
                </div></div>
            </div>
            <div class="col-md-2">
                <div class="card bg-dark text-light"><div class="card-body text-center">
                    <h6 class="text-muted">Win Rate</h6><h3>{m.get('win_rate', 0)}%</h3>
                </div></div>
            </div>
            <div class="col-md-2">
                <div class="card bg-dark text-light"><div class="card-body text-center">
                    <h6 class="text-muted">Trades</h6><h3>{m.get('total_trades', 0)}</h3>
                </div></div>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card bg-dark text-light"><div class="card-body text-center">
                    <h6 class="text-muted">Initial Capital</h6>
                    <h3>{self._fmt_currency(m.get('initial_capital', 0))}</h3>
                </div></div>
            </div>
            <div class="col-md-6">
                <div class="card bg-dark text-light"><div class="card-body text-center">
                    <h6 class="text-muted">Final Value</h6>
                    <h3 class="text-success">{self._fmt_currency(m.get('final_value', 0))}</h3>
                </div></div>
            </div>
        </div>

        <h5>Equity Curve (Normalized to Base 100)</h5>
        <canvas id="equityChart" height="120"></canvas>

        <script>
        new Chart(document.getElementById('equityChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(equity_dates)},
                datasets: [
                    {{
                        label: 'MQ Strategy',
                        data: {json.dumps(strategy_norm)},
                        borderColor: '#8b5cf6',
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false,
                    }},
                    {benchmark_datasets}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ position: 'top', labels: {{ color: '#e2e8f0' }} }},
                }},
                scales: {{
                    x: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8', maxTicksLimit: 12 }} }},
                    y: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }}
                }}
            }}
        }});
        </script>

        {self._render_trade_analysis(m)}
        """

        html = self._render_html(
            f"Backtest Report - {report.config.get('start_date', '')} to {report.config.get('end_date', '')}",
            sections
        )
        return self._save_report(html, 'backtest', report.run_date)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _render_trade_analysis(self, metrics: dict) -> str:
        """Render exit reason breakdown with P&L and trade analysis section."""
        exit_counts = metrics.get('exit_reason_counts', {})
        exit_pnl = metrics.get('exit_reason_pnl', {})
        total_trades = metrics.get('total_trades', 0)
        winning = metrics.get('winning_trades', 0)
        losing = metrics.get('losing_trades', 0)
        avg_win = metrics.get('avg_win_pct', 0)
        avg_loss = metrics.get('avg_loss_pct', 0)

        # Exit reason display names
        reason_labels = {
            'hard_stop_loss': ('30% Hard Stop', '#f85149'),
            'ath_drawdown_rebalance': ('ATH Drawdown >20%', '#d29922'),
            'fundamental_3q_decline': ('Fundamental 3Q Fail', '#bc8cff'),
            'rebalance_replaced': ('Rebalance Replace', '#58a6ff'),
            'fundamental_2y_decline': ('Fundamental 2Y Decline', '#a371f7'),
            'manual': ('Manual Exit', '#8b949e'),
        }

        # Build detailed exit reason rows
        detail_rows = ''
        chart_labels = []
        chart_values = []
        chart_colors = []

        for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
            label, color = reason_labels.get(reason, (reason, '#8b949e'))
            pct = round(count / total_trades * 100, 1) if total_trades > 0 else 0
            pnl_data = exit_pnl.get(reason, {})
            total_pnl = pnl_data.get('total_pnl', 0)
            avg_ret = pnl_data.get('avg_return_pct', 0)
            avg_hold = int(pnl_data.get('avg_holding_days', 0))
            reason_wr = pnl_data.get('win_rate', 0)
            worst = pnl_data.get('worst_return_pct', 0)
            best = pnl_data.get('best_return_pct', 0)

            pnl_color = '#3fb950' if total_pnl >= 0 else '#f85149'
            avg_color = '#3fb950' if avg_ret >= 0 else '#f85149'

            detail_rows += f"""
            <tr>
                <td><span style="color: {color}; font-weight: 600;">{label}</span></td>
                <td class="text-end">{count}</td>
                <td class="text-end">{pct}%</td>
                <td class="text-end" style="color: {pnl_color}; font-weight: 600;">{self._fmt_currency(total_pnl)}</td>
                <td class="text-end" style="color: {avg_color};">{avg_ret:+.1f}%</td>
                <td class="text-end">{reason_wr:.0f}%</td>
                <td class="text-end">{avg_hold}d</td>
                <td class="text-end"><span style="color: #f85149;">{worst:+.1f}%</span> / <span style="color: #3fb950;">{best:+.1f}%</span></td>
            </tr>"""
            chart_labels.append(label)
            chart_values.append(total_pnl)
            chart_colors.append(color)

        if not detail_rows:
            detail_rows = '<tr><td colspan="8" class="text-muted">No exit data available</td></tr>'

        # Grand total P&L
        grand_pnl = sum(d.get('total_pnl', 0) for d in exit_pnl.values())
        grand_color = '#3fb950' if grand_pnl >= 0 else '#f85149'

        # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        win_rate = metrics.get('win_rate', 0)
        loss_rate = 100 - win_rate
        expectancy = round((win_rate / 100 * avg_win) - (loss_rate / 100 * abs(avg_loss)), 2) if total_trades > 0 else 0

        return f"""
        <div class="row mb-4 mt-4">
            <div class="col-12">
                <h5>Exit Reason Breakdown & P/L</h5>
                <div class="table-responsive">
                <table class="table table-dark table-sm">
                    <thead><tr>
                        <th>Exit Reason</th>
                        <th class="text-end">Count</th>
                        <th class="text-end">%</th>
                        <th class="text-end">Total P&L</th>
                        <th class="text-end">Avg Return</th>
                        <th class="text-end">Win Rate</th>
                        <th class="text-end">Avg Hold</th>
                        <th class="text-end">Worst / Best</th>
                    </tr></thead>
                    <tbody>{detail_rows}</tbody>
                    <tfoot><tr class="border-top">
                        <td class="fw-bold">Total</td>
                        <td class="text-end fw-bold">{total_trades}</td>
                        <td></td>
                        <td class="text-end fw-bold" style="color: {grand_color};">{self._fmt_currency(grand_pnl)}</td>
                        <td colspan="4"></td>
                    </tr></tfoot>
                </table>
                </div>

                <canvas id="exitPnlChart" height="60" class="mt-3"></canvas>
                <script>
                new Chart(document.getElementById('exitPnlChart'), {{
                    type: 'bar',
                    data: {{
                        labels: {json.dumps(chart_labels)},
                        datasets: [{{
                            label: 'Total P&L',
                            data: {json.dumps(chart_values)},
                            backgroundColor: {json.dumps(chart_values)}.map(v => v >= 0 ? 'rgba(63,185,80,0.7)' : 'rgba(248,81,73,0.7)'),
                            borderColor: {json.dumps(chart_values)}.map(v => v >= 0 ? '#3fb950' : '#f85149'),
                            borderWidth: 1,
                            borderRadius: 4,
                        }}]
                    }},
                    options: {{
                        indexAxis: 'y',
                        responsive: true,
                        plugins: {{
                            legend: {{ display: false }},
                            tooltip: {{
                                callbacks: {{
                                    label: function(ctx) {{ return '\u20B9' + ctx.raw.toLocaleString('en-IN'); }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }},
                            y: {{ grid: {{ display: false }}, ticks: {{ color: '#e2e8f0' }} }}
                        }}
                    }}
                }});
                </script>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-6">
                <h5>Trade Analysis</h5>
                <div class="card bg-dark text-light">
                    <div class="card-body">
                        <div class="d-flex justify-content-between mb-3">
                            <div class="text-center">
                                <small class="text-muted d-block">Winners</small>
                                <span class="fs-4 fw-bold" style="color: #3fb950;">{winning}</span>
                            </div>
                            <div class="text-center">
                                <small class="text-muted d-block">Losers</small>
                                <span class="fs-4 fw-bold" style="color: #f85149;">{losing}</span>
                            </div>
                            <div class="text-center">
                                <small class="text-muted d-block">Win Rate</small>
                                <span class="fs-4 fw-bold">{win_rate}%</span>
                            </div>
                            <div class="text-center">
                                <small class="text-muted d-block">Topups</small>
                                <span class="fs-4 fw-bold" style="color: #58a6ff;">{metrics.get('total_topups', 0)}</span>
                            </div>
                        </div>
                        <hr style="border-color: #334155;">
                        <div class="d-flex justify-content-between mb-2">
                            <span class="text-muted">Avg Win</span>
                            <span style="color: #3fb950; font-weight: 600;">+{abs(avg_win):.1f}%</span>
                        </div>
                        <div class="d-flex justify-content-between mb-2">
                            <span class="text-muted">Avg Loss</span>
                            <span style="color: #f85149; font-weight: 600;">-{abs(avg_loss):.1f}%</span>
                        </div>
                        <hr style="border-color: #334155;">
                        <div class="d-flex justify-content-between">
                            <span class="text-muted">Expectancy</span>
                            <span class="fw-bold" style="color: {'#3fb950' if expectancy >= 0 else '#f85149'};">{'+' if expectancy >= 0 else ''}{expectancy}%</span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <h5>Key Thresholds</h5>
                <div class="card bg-dark text-light">
                    <div class="card-body">
                        <table class="table table-dark table-sm mb-0">
                            <thead><tr><th>Parameter</th><th class="text-end">Current</th><th>Impact</th></tr></thead>
                            <tbody>
                                <tr>
                                    <td>Hard Stop Loss</td>
                                    <td class="text-end fw-bold" style="color: #f85149;">30%</td>
                                    <td><small class="text-muted">{exit_pnl.get('hard_stop_loss', {{}}).get('count', 0)} exits, {self._fmt_currency(exit_pnl.get('hard_stop_loss', {{}}).get('total_pnl', 0))} P&L</small></td>
                                </tr>
                                <tr>
                                    <td>ATH Drawdown Exit</td>
                                    <td class="text-end fw-bold" style="color: #d29922;">20%</td>
                                    <td><small class="text-muted">{exit_pnl.get('ath_drawdown_rebalance', {{}}).get('count', 0)} exits, {self._fmt_currency(exit_pnl.get('ath_drawdown_rebalance', {{}}).get('total_pnl', 0))} P&L</small></td>
                                </tr>
                                <tr>
                                    <td>Fundamental 3Q</td>
                                    <td class="text-end fw-bold" style="color: #bc8cff;">3 Qtrs</td>
                                    <td><small class="text-muted">{exit_pnl.get('fundamental_3q_decline', {{}}).get('count', 0)} exits, {self._fmt_currency(exit_pnl.get('fundamental_3q_decline', {{}}).get('total_pnl', 0))} P&L</small></td>
                                </tr>
                                <tr>
                                    <td>Rebalance Replace</td>
                                    <td class="text-end fw-bold" style="color: #58a6ff;">Semi-Annual</td>
                                    <td><small class="text-muted">{exit_pnl.get('rebalance_replaced', {{}}).get('count', 0)} exits, {self._fmt_currency(exit_pnl.get('rebalance_replaced', {{}}).get('total_pnl', 0))} P&L</small></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        """

    def _render_signals_table(self, signals) -> str:
        if not signals:
            return '<p class="text-muted">No signals</p>'

        rows = ''
        for s in signals:
            color = self._priority_color(s.priority)
            type_color = {'EXIT': 'danger', 'TOPUP': 'warning', 'WATCH': 'info', 'WARNING': 'secondary'}.get(s.signal_type, 'light')
            rows += f"""
            <tr>
                <td><span class="badge bg-{type_color}">{s.signal_type}</span></td>
                <td><strong>{s.symbol}</strong></td>
                <td><span class="badge bg-{color}">{s.priority}</span></td>
                <td>{s.message}</td>
            </tr>"""

        return f"""
        <table class="table table-dark table-sm">
            <thead><tr><th>Type</th><th>Symbol</th><th>Priority</th><th>Details</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>"""

    def _priority_color(self, priority: str) -> str:
        return {'HIGH': 'danger', 'MEDIUM': 'warning', 'LOW': 'info'}.get(priority, 'secondary')

    def _fmt_currency(self, value: float) -> str:
        """Format as Indian currency."""
        if value >= 10_000_000:
            return f"₹{value/10_000_000:.1f}Cr"
        elif value >= 100_000:
            return f"₹{value/100_000:.1f}L"
        else:
            return f"₹{value:,.0f}"

    def _render_html(self, title: str, body_content: str) -> str:
        """Render standalone HTML report."""
        return f"""<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} | MQ Agent</title>
    <link href="{CDN_BOOTSTRAP}" rel="stylesheet">
    <link href="{CDN_ICONS}" rel="stylesheet">
    <script src="{CDN_CHARTJS}"></script>
    <style>
        body {{ background: #0f172a; color: #e2e8f0; }}
        .card {{ border: 1px solid #334155; }}
        .text-muted {{ color: #94a3b8 !important; }}
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h4><i class="bi bi-robot me-2"></i>{title}</h4>
            <small class="text-muted">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</small>
        </div>
        {body_content}
    </div>
</body>
</html>"""

    def _save_report(self, html: str, report_type: str, date: datetime) -> str:
        """Save report HTML and return relative path."""
        filename = f"{report_type}_{date.strftime('%Y%m%d_%H%M%S')}.html"
        path = REPORTS_DIR / filename
        path.write_text(html, encoding='utf-8')
        relative = f"reports/{filename}"
        logger.info(f"Report saved: {relative}")
        return relative
