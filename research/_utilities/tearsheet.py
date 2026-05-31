"""Reusable client-facing TEARSHEET generator (hedge-fund factsheet style).

generate_tearsheet(strat_nav, bench_nav, name, meta) -> writes a one-page PNG
factsheet (+ self-contained HTML wrapper). Visual-first: KPI strip, equity curve
vs benchmark (log), underwater drawdown, yearly bars vs index, monthly heatmap,
rolling 12m return, return distribution, and compact stat tables.

Inputs are NAV (price) series indexed by date (any frequency; monthly+ recommended
for the heatmap). Benchmark is rebased to the strategy's start. rf is annual.
"""
from __future__ import annotations
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

# palette (clean dark, gold accent — matches Quantifyd design language)
BG, PANEL, INK, MUT = "#0e1116", "#161b22", "#e6edf3", "#8b949e"
GOLD, GREEN, RED, BLUE = "#e3b341", "#3fb950", "#f85149", "#58a6ff"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUT,
    "ytick.color": MUT, "axes.edgecolor": "#30363d", "font.size": 9,
})


def _ann(idx):
    d = np.median(np.diff(idx.values).astype("timedelta64[D]").astype(int))
    return 12 if d > 20 else 252


def _metrics(nav, rf):
    nav = nav.dropna()
    r = nav.pct_change().dropna()
    ann = _ann(nav.index)
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(ann)
    rf_p = (1 + rf) ** (1 / ann) - 1
    ex = r - rf_p
    sharpe = ex.mean() / r.std() * np.sqrt(ann) if r.std() > 0 else np.nan
    dn = ex[ex < 0].std()
    sortino = ex.mean() / dn * np.sqrt(ann) if dn and dn > 0 else np.nan
    dd = nav / nav.cummax() - 1
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    return dict(cagr=cagr, total=nav.iloc[-1] / nav.iloc[0], vol=vol, sharpe=sharpe,
                sortino=sortino, maxdd=maxdd, calmar=calmar, dd=dd, r=r, ann=ann, yrs=yrs)


def _yearly(nav):
    ye = nav.groupby(nav.index.year).last()
    out = ye.pct_change()
    out.iloc[0] = ye.iloc[0] / nav.iloc[0] - 1
    return out


def generate_tearsheet(strat_nav, bench_nav, name, meta=None, out_dir=".", rf=0.065):
    meta = meta or {}
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    s = strat_nav.dropna().copy()
    s = s / s.iloc[0]
    b = bench_nav.reindex(s.index).ffill().bfill()
    b = b / b.iloc[0]
    sm, bm = _metrics(s, rf), _metrics(b, rf)
    sy, by = _yearly(s), _yearly(b)
    yrs = sorted(set(sy.index) & set(by.index))
    excess = sm["cagr"] - bm["cagr"]
    beat = float(np.mean([sy[y] > by[y] for y in yrs])) if yrs else np.nan
    cov = np.cov(sm["r"], bm["r"].reindex(sm["r"].index).fillna(0))[0, 1]
    beta = cov / np.var(bm["r"]) if np.var(bm["r"]) > 0 else np.nan
    corr = sm["r"].corr(bm["r"].reindex(sm["r"].index))

    fig = plt.figure(figsize=(12, 15.5))
    gs = gridspec.GridSpec(6, 4, height_ratios=[0.5, 0.7, 1.5, 1.15, 1.15, 1.0],
                           hspace=0.5, wspace=0.32,
                           left=0.06, right=0.965, top=0.955, bottom=0.045)

    # ---- header ----
    axh = fig.add_subplot(gs[0, :]); axh.axis("off")
    axh.text(0, 0.6, name, fontsize=20, fontweight="bold", color=INK)
    sub = f"{s.index[0]:%b %Y} – {s.index[-1]:%b %Y}  ·  {sm['yrs']:.1f}y  ·  vs {meta.get('bench','Benchmark')}"
    axh.text(0, 0.05, sub, fontsize=10, color=MUT)
    axh.text(1, 0.6, meta.get("tag", "BACKTEST"), fontsize=10, color=GOLD,
             ha="right", fontweight="bold")

    # ---- KPI strip ----
    axk = fig.add_subplot(gs[1, :]); axk.axis("off")
    kpis = [
        ("CAGR", f"{sm['cagr']*100:.1f}%", GREEN if sm['cagr'] > 0 else RED),
        ("Total Return", f"{sm['total']:.1f}x", INK),
        ("Excess vs Index", f"{excess*100:+.1f}%/yr", GREEN if excess > 0 else RED),
        ("Sharpe", f"{sm['sharpe']:.2f}", GOLD),
        ("Sortino", f"{sm['sortino']:.2f}", GOLD),
        ("Max Drawdown", f"{sm['maxdd']*100:.1f}%", RED),
        ("Calmar", f"{sm['calmar']:.2f}", GOLD),
        ("Yrs Beating Index", f"{beat*100:.0f}%", GREEN if beat and beat > 0.5 else RED),
    ]
    n = len(kpis)
    for i, (lab, val, c) in enumerate(kpis):
        x = i / n + 0.5 / n
        axk.text(x, 0.78, val, fontsize=16, fontweight="bold", color=c, ha="center")
        axk.text(x, 0.18, lab.upper(), fontsize=7.5, color=MUT, ha="center")
    for i in range(1, n):
        axk.axvline(i / n, color="#30363d", lw=0.6)

    # ---- hero: growth of 1 (log) ----
    ax1 = fig.add_subplot(gs[2, :])
    ax1.plot(s.index, s.values, color=GOLD, lw=2.0, label=f"{name} ({sm['cagr']*100:.0f}% CAGR)")
    ax1.plot(b.index, b.values, color=MUT, lw=1.4, ls="--",
             label=f"{meta.get('bench','Index')} ({bm['cagr']*100:.0f}% CAGR)")
    ax1.set_yscale("log"); ax1.set_title("Growth of ₹1 (log scale)", color=INK, loc="left", fontsize=11)
    ax1.legend(loc="upper left", facecolor=PANEL, edgecolor="#30363d", labelcolor=INK)
    ax1.grid(True, which="both", alpha=0.15)

    # ---- underwater drawdown: strategy vs index ----
    ax2 = fig.add_subplot(gs[3, :2])
    ax2.fill_between(sm["dd"].index, sm["dd"].values * 100, 0, color=GOLD, alpha=0.28,
                     label=f"Strategy (max {sm['maxdd']*100:.1f}%)")
    ax2.plot(sm["dd"].index, sm["dd"].values * 100, color=GOLD, lw=1.0)
    ax2.plot(bm["dd"].index, bm["dd"].values * 100, color=MUT, lw=1.1, ls="--",
             label=f"{meta.get('bench','Index')} (max {bm['maxdd']*100:.1f}%)")
    ax2.set_title("Drawdown vs index (%)", color=INK, loc="left", fontsize=11)
    ax2.legend(loc="lower left", facecolor=PANEL, edgecolor="#30363d", labelcolor=INK, fontsize=8)
    ax2.grid(True, alpha=0.15)

    # ---- yearly bars vs index (value-labelled; partial final year flagged) ----
    ax3 = fig.add_subplot(gs[3, 2:])
    x = np.arange(len(yrs)); w = 0.4
    sv = [sy[y] * 100 for y in yrs]; bv = [by[y] * 100 for y in yrs]
    ax3.bar(x - w/2, sv, w, color=GOLD, label=name)
    ax3.bar(x + w/2, bv, w, color=MUT, label=meta.get("bench", "Index"))
    for i, v in enumerate(sv):                       # label strategy bars so tiny ones (e.g. partial yr) read
        ax3.annotate(f"{v:.0f}", (x[i] - w/2, v), ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=5.5, color=GOLD)
    labels = [f"{y}*" if y == max(yrs) else str(y) for y in yrs]
    ax3.set_xticks(x); ax3.set_xticklabels(labels, rotation=90, fontsize=7)
    ax3.axhline(0, color="#30363d", lw=0.8)
    ax3.set_title("Annual return: strategy vs index (%)   * = partial year (YTD)",
                  color=INK, loc="left", fontsize=10)
    ax3.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor=INK, fontsize=7.5)
    ax3.grid(True, axis="y", alpha=0.15)

    # ---- monthly heatmap ----
    ax4 = fig.add_subplot(gs[4, :2])
    mr = s.pct_change()
    piv = mr.groupby([mr.index.year, mr.index.month]).sum().unstack() * 100
    piv = piv.reindex(columns=range(1, 13))
    im = ax4.imshow(piv.values, cmap="RdYlGn", aspect="auto", vmin=-12, vmax=12)
    ax4.set_xticks(range(12)); ax4.set_xticklabels(list("JFMAMJJASOND"), fontsize=7)
    ax4.set_yticks(range(len(piv.index))); ax4.set_yticklabels(piv.index, fontsize=6)
    ax4.set_title("Monthly returns (%)", color=INK, loc="left", fontsize=11)
    fig.colorbar(im, ax=ax4, fraction=0.025, pad=0.02)

    # ---- rolling 12m ----
    ax5 = fig.add_subplot(gs[4, 2:])
    win = sm["ann"]
    rs = s.pct_change().rolling(win).apply(lambda x: np.prod(1 + x) - 1) * 100
    rb = b.pct_change().rolling(win).apply(lambda x: np.prod(1 + x) - 1) * 100
    ax5.plot(rs.index, rs.values, color=GOLD, lw=1.3, label=name)
    ax5.plot(rb.index, rb.values, color=MUT, lw=1.0, ls="--", label=meta.get("bench", "Index"))
    ax5.axhline(0, color="#30363d", lw=0.8)
    ax5.set_title("Rolling 12-month return (%)", color=INK, loc="left", fontsize=11)
    ax5.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor=INK, fontsize=8)
    ax5.grid(True, alpha=0.15)

    # ---- stat tables (text) ----
    ax6 = fig.add_subplot(gs[5, :2]); ax6.axis("off")
    rows = [("Metric", "Strategy", meta.get("bench", "Index")),
            ("CAGR", f"{sm['cagr']*100:.1f}%", f"{bm['cagr']*100:.1f}%"),
            ("Volatility", f"{sm['vol']*100:.1f}%", f"{bm['vol']*100:.1f}%"),
            ("Sharpe", f"{sm['sharpe']:.2f}", f"{bm['sharpe']:.2f}"),
            ("Sortino", f"{sm['sortino']:.2f}", f"{bm['sortino']:.2f}"),
            ("Max Drawdown", f"{sm['maxdd']*100:.1f}%", f"{bm['maxdd']*100:.1f}%"),
            ("Calmar", f"{sm['calmar']:.2f}", f"{bm['calmar']:.2f}"),
            ("Best year", f"{max(sy.values)*100:.0f}%", f"{max(by.values)*100:.0f}%"),
            ("Worst year", f"{min(sy.values)*100:.0f}%", f"{min(by.values)*100:.0f}%")]
    _texttable(ax6, rows, "Performance & risk")

    ax7 = fig.add_subplot(gs[5, 2:]); ax7.axis("off")
    rows2 = [("Vs benchmark", ""),
             ("Excess CAGR", f"{excess*100:+.1f}%/yr"),
             ("Beta", f"{beta:.2f}"),
             ("Correlation", f"{corr:.2f}"),
             ("Years beating index", f"{beat*100:.0f}%"),
             ("Total return", f"{sm['total']:.1f}x  vs  {bm['total']:.1f}x")]
    _texttable(ax7, rows2, "Relative to index", twocol=False)

    foot = meta.get("disclosures",
                    "Backtest, net of modelled costs. Past performance is not indicative of "
                    "future results.")
    fig.text(0.06, 0.012, foot, fontsize=7, color=MUT)

    png = out_dir / "tearsheet.png"
    fig.savefig(png, dpi=120); plt.close(fig)
    _html(out_dir / "tearsheet.html", png, name, sub, foot)
    return png


def _texttable(ax, rows, title, twocol=True):
    ax.text(0, 1.02, title, fontsize=11, color=INK, fontweight="bold", transform=ax.transAxes)
    y = 0.88
    for i, row in enumerate(rows):
        c = GOLD if i == 0 else INK
        ax.text(0.0, y, str(row[0]), fontsize=8.5, color=(MUT if i else GOLD), transform=ax.transAxes)
        if len(row) >= 2:
            ax.text(0.62, y, str(row[1]), fontsize=8.5, color=c, transform=ax.transAxes, ha="right")
        if twocol and len(row) >= 3:
            ax.text(0.98, y, str(row[2]), fontsize=8.5, color=(MUT if i == 0 else MUT),
                    transform=ax.transAxes, ha="right")
        y -= 0.105


def _html(path, png, name, sub, foot):
    b64 = base64.b64encode(png.read_bytes()).decode()
    path.write_text(f"""<!doctype html><html><head><meta charset='utf-8'>
<title>{name} — Factsheet</title>
<style>body{{background:{BG};margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif}}
.wrap{{max-width:1100px;margin:0 auto;padding:18px}}
img{{width:100%;border-radius:10px;box-shadow:0 8px 40px #0008}}
.note{{color:{MUT};font-size:12px;margin-top:10px}}</style></head>
<body><div class='wrap'><img src='data:image/png;base64,{b64}'/>
<div class='note'>{foot}</div></div></body></html>""", encoding="utf-8")
