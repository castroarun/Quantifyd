"""research/111 deliverable 2 — NAS (live records) vs best-config CSL (replay), per index.
Basis: 6 lots per side (NIFTY NAS = 2 lots x 3 books; CSL scaled to 6 lots). Per binding
rules: live-first (NAS = actual booked P&L), n-days + ranges reported per source, visuals
(equity + drawdown) per index -> static/app/{nifty,sensex}_csl_vs_nas.png + summary JSON."""
import json, sqlite3, datetime, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

Q = "/home/arun/quantifyd"
LOTS_BY_MONTH = {"2026-03": 1, "2026-04": 5, "2026-05": 1, "2026-06": 1, "2026-07": 2, "2026-08": 3}
cfg = json.load(open(Q + "/static/app/straddles/csl_best_configs.json"))

def csl_book(sym, scale):
    daily = {}
    for k, b in cfg["best"][sym].items():
        for d, v in b.get("series", []):
            daily[d] = daily.get(d, 0) + v * scale
    return daily

def agg(f):
    c = pk = dd = 0
    for v in f: c += v; pk = max(pk, c); dd = min(dd, c - pk)
    n = len(f) or 1
    return dict(total=round(sum(f)), mean=round(sum(f) / n), win=round(100 * sum(1 for v in f if v > 0) / n),
                maxdd=round(dd), n=n, ratio=(round(sum(f) / abs(dd), 1) if dd < 0 else 99.0))

def nifty_nas_perlot():
    out = {}
    for name in ("nas_916_atm", "nas_916_atm2", "nas_916_atm4"):
        c = sqlite3.connect(Q + "/backtest_data/%s_trading.db" % name); c.row_factory = sqlite3.Row
        cols = [d[1] for d in c.execute("PRAGMA table_info(nas_atm_trades)")]
        pcol = next(x for x in ("pnl", "net_pnl") if x in cols)
        dcol = next(x for x in ("entry_time", "created_at") if x in cols)
        m = {}
        for r in c.execute("SELECT %s d,%s p FROM nas_atm_trades WHERE %s IS NOT NULL" % (dcol, pcol, pcol)):
            d = str(r["d"])[:10]
            m[d] = m.get(d, 0) + r["p"] / LOTS_BY_MONTH.get(d[:7], 1)
        out[name] = m
    return out

def sensex_nas_perlot():
    c = sqlite3.connect(Q + "/backtest_data/sensex_atm2_trading.db"); c.row_factory = sqlite3.Row
    m = {}
    for r in c.execute("SELECT trade_date d, net_pnl p, lots FROM nas_atm_trades WHERE net_pnl IS NOT NULL"):
        d = str(r["d"])[:10]; L = r["lots"] or 1
        m[d] = m.get(d, 0) + r["p"] / L
    return m

def eq(f):
    c = 0; o = []
    for v in f: c += v; o.append(c)
    return o

def ddc(e):
    p = 0; o = []
    for v in e: p = max(p, v); o.append(v - p)
    return o

summary = {}
for SYM, nas_streams, csl_scale in (
        ("NIFTY", None, 0.6),        # CSL 10-lot series -> 6 lots
        ("SENSEX", None, 1.2)):      # CSL 5-lot series -> 6 lots
    csl = csl_book(SYM, csl_scale)
    if SYM == "NIFTY":
        books = nifty_nas_perlot()
        nas = {}
        for m in books.values():
            for d, v in m.items(): nas[d] = nas.get(d, 0) + v * 2   # 2 lots x 3 books = 6
        nas_lbl = "NAS-916 x3 @2 lots each (live)"
        nas_rng = (min(nas), max(nas)) if nas else (None, None)
    else:
        m = sensex_nas_perlot()
        nas = {d: v * 6 for d, v in m.items()}                       # 6 lots
        nas_lbl = "SENSEX NAS atm2 @6 lots (live)"
        nas_rng = (min(nas), max(nas)) if nas else (None, None)
    days = sorted(set(csl) | set(nas))
    common = sorted(set(csl) & set(nas))
    cslf = [csl.get(d, 0) for d in days]
    nasf = [nas.get(d, 0) for d in days if d in nas]
    stack = [(csl.get(d, 0) + nas.get(d, 0)) / 2 for d in days]     # 50/50 at 6-lot each -> 6-lot book
    corr = None
    if len(common) > 5:
        try: corr = round(st.correlation([csl[d] for d in common], [nas[d] for d in common]), 2)
        except Exception: pass
    A_csl, A_nas, A_stk = agg(cslf), agg(nasf), agg(stack)
    summary[SYM] = {"csl": A_csl, "nas": A_nas, "stack": A_stk, "corr": corr,
                    "csl_days": [len(csl), min(csl), max(csl)],
                    "nas_days": [len(nas), nas_rng[0], nas_rng[1]], "common": len(common)}
    print("== %s (all @6 lots) ==" % SYM)
    print(" CSL best-config book (replay): n=%d %s..%s  tot %+d mean %+d win %d%% dd %+d ratio %s" % (
        len(csl), min(csl), max(csl), A_csl["total"], A_csl["mean"], A_csl["win"], A_csl["maxdd"], A_csl["ratio"]))
    print(" %s: n=%d %s..%s  tot %+d mean %+d win %d%% dd %+d ratio %s" % (
        nas_lbl, len(nas), nas_rng[0], nas_rng[1], A_nas["total"], A_nas["mean"], A_nas["win"], A_nas["maxdd"], A_nas["ratio"]))
    print(" 50/50 stack: tot %+d mean %+d dd %+d ratio %s | corr(CSL,NAS common %d d) = %s" % (
        A_stk["total"], A_stk["mean"], A_stk["maxdd"], A_stk["ratio"], len(common), corr))
    # chart
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    X = range(len(days))
    for f, lbl, cl, lw in ((cslf, "CSL best-config book @6 lots (replay)", "#2ca02c", 2.2),
                            ([nas.get(d, 0) for d in days], nas_lbl, "#d62728", 1.7),
                            (stack, "50/50 stack (6 lots)", "#1f77b4", 1.5)):
        e = eq(f); ax[0].plot(X, e, color=cl, lw=lw, label="{} (Rs{:+,})".format(lbl, round(e[-1])))
        ax[1].plot(X, ddc(e), color=cl, lw=1.4); ax[1].fill_between(X, ddc(e), 0, color=cl, alpha=0.08)
    ax[0].set_title("%s — best-config CSL vs NAS live book — all at 6 lots — %d days %s..%s (NAS traded %d days)" % (
        SYM, len(days), days[0], days[-1], len(nas)), fontsize=11)
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.25); ax[0].axhline(0, color="#888", lw=0.6)
    ax[1].set_title("Drawdown (Rs)", fontsize=10); ax[1].grid(alpha=0.25)
    tick = [0, len(days) // 2, len(days) - 1]
    ax[1].set_xticks(tick); ax[1].set_xticklabels([days[i] for i in tick])
    plt.tight_layout()
    for p in ("static/app/%s_csl_vs_nas.png" % SYM.lower(), "frontend/public/%s_csl_vs_nas.png" % SYM.lower()):
        fig.savefig(Q + "/" + p, dpi=110)
    plt.close(fig)
json.dump(summary, open(Q + "/research/111_sensex_manual_mgmt/results/deliverable2_summary.json", "w"))
print("DONE charts + summary")
