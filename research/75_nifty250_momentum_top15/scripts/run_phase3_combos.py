"""Research 75 phase 3 — can per-stock risk controls REPLACE the index gate?

User ask: test research/75 + quality filter / ATH-proximity / per-stock exit / retention
buffer, and find the best COMBINATIONS *without* the NIFTYBEES-100EMA gate. This ports the
research/41 "SMOOTHEST" machinery onto the research/75 engine and isolates each control's
marginal effect on CAGR vs drawdown, with the gate OFF (and a couple gate-ON references).

Controls:
  quality  : hold only names with >= 6 of the last 12 non-overlapping 21d blocks positive (causal)
  ath      : hold only names within 10% of their point-in-time all-time high (close >= 0.90*ATH)
  sma100   : per-stock event-driven exit — drop a held name the day it closes below its 100-SMA
  donch15  : per-stock event-driven exit — drop a held name below its prior-15-day low
  buffer   : keep a held name while it stays in the top N*mult by momentum (cuts churn)
All daily-marked, net 0.3% RT, midcap/large-mid, rs120 (Aurum family) & ret252, N=15, 2006-2026.
"""
import importlib.util, sys, time, csv, logging
from pathlib import Path
import numpy as np, pandas as pd
logging.disable(logging.WARNING)

HERE = Path("/home/arun/quantifyd/research/75_nifty250_momentum_top15")
RES = HERE / "results"


def imp(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m


p1 = imp("p1", str(HERE / "scripts" / "run_nifty250_momentum.py"))
p2 = imp("p2", str(HERE / "scripts" / "run_variants_phase2.py"))
rs2 = p1.rs2; BENCH = p1.BENCH; EXCLUDE = p1.EXCLUDE; RT = p1.RT; START = p1.START
rs2.BANDS.setdefault("n250", (0, 250))


def quality_ok(r21, s, d):
    h = r21[s].loc[:d].dropna()
    if len(h) < 232: return False
    samples = [h.iloc[-1 - 21 * k] for k in range(12)]
    return sum(1 for v in samples if v > 0) >= 6


def run(close, tv, ema, nbx_ema100, ind, band="mid", score="rs120", N=15,
        use_gate=False, use_quality=False, use_ath=False, buffer_mult=1.0,
        exit_mode="none", rt=RT, stcg=0.0):
    ath, sma100, low15, r21 = ind["ath"], ind["sma100"], ind["low15"], ind["r21"]
    idx = close.index[close.index >= START]
    _ss = pd.Series(idx, index=idx); me = set(_ss.groupby([idx.year, idx.month]).max().values)
    nbx = close[BENCH].ffill()
    cash, held = 1.0, {}; nav = []; prev = None
    st = dict(tax=0.0, cost=0.0, fills=0, turn_sum=0.0, gate_off=0, stop_exits=0)
    buf_n = int(N * buffer_mult)

    def E(): return sum(v[0] for v in held.values())

    def realize(stt, d):
        g = stt[0] - stt[1]
        if stcg and g > 0 and (d - stt[2]).days < 365:
            t = g * stcg; st['tax'] += t; return t
        return 0.0

    def liquidate(d):
        nonlocal cash
        for s in list(held):
            t = realize(held[s], d); cash += held.pop(s)[0] - t

    def gate_on(d):
        if not use_gate: return True
        v = nbx_ema100.loc[d]; return True if pd.isna(v) else (nbx.loc[d] > v)

    def do_fill(d, px):
        nonlocal cash, held
        sc = p2.momentum(close, d, score)
        if sc is None or len(sc) == 0: return
        uni = rs2.pit_universe(tv, close, d, band)
        cand = [s for s in sc.index if s in uni and s not in EXCLUDE and s != BENCH]
        if use_ath:
            cand = [s for s in cand if pd.notna(ath[s].loc[d]) and ath[s].loc[d] > 0
                    and pd.notna(px.get(s, np.nan)) and px[s] >= 0.90 * ath[s].loc[d]]
        if use_quality:
            cand = [s for s in cand if quality_ok(r21, s, d)]
        cand = [s for s in cand if pd.notna(px.get(s, np.nan))]
        if not cand:
            liquidate(d); return
        ranked = list(sc[cand].sort_values(ascending=False).index)
        buf = set(ranked[:buf_n]); top = ranked[:N]
        keep = [s for s in held if s in buf]
        add = [s for s in top if s not in keep]
        target = (keep + add)[:N]
        tot = E() + cash; w = tot / len(target)
        turn = len(set(held).symmetric_difference(set(target))) / max(1, 2 * len(target))
        nh = {}
        for s in target:
            nh[s] = held[s] if s in held else [w, w, d]; nh[s][0] = w
        for s in list(held):
            if s not in nh: realize(held[s], d)
        held = nh
        c = tot * rt * turn * 2; cash = tot - w * len(target) - c
        st['cost'] += c; st['fills'] += 1; st['turn_sum'] += turn

    for d in idx:
        px = close.loc[d]
        if held and prev is not None:
            for s, stt in held.items():
                p1v = px.get(s, np.nan); p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1v) and pd.notna(p0) and p0 > 0:
                    stt[0] += stt[0] * (p1v / p0 - 1.0)
        if cash > 0: cash *= p1.DAY_CASH
        # per-stock event-driven exits (the gate substitute)
        if exit_mode != "none" and held:
            for s in list(held):
                p1v = px.get(s, np.nan)
                lvl = (sma100[s].loc[d] if exit_mode == "sma100" else low15[s].loc[d])
                if pd.notna(p1v) and pd.notna(lvl) and p1v < lvl:
                    t = realize(held[s], d); cash += held.pop(s)[0] - t; st['stop_exits'] += 1
        if use_gate and d in me and not gate_on(d) and held:
            liquidate(d); st['gate_off'] += 1
        if d in me:
            if gate_on(d): do_fill(d, px)
            elif held: liquidate(d)
        nav.append((d, E() + cash)); prev = d
    return p1._stats(nav, st, START)


FIELDS = ["config", "band", "score", "gate", "quality", "ath", "buffer", "exit",
          "net_cagr", "post_tax20", "maxdd", "sharpe", "calmar", "mult",
          "fills", "avg_turn", "stop_exits", "mod_cagr", "mod_maxdd", "mod_calmar"]


def main():
    t0 = time.time()
    print("Loading ...", flush=True)
    close, tv = p1.load()
    ema = {p: close.ewm(span=p, adjust=False, min_periods=p).mean() for p in (50, 100, 200)}
    nbx_ema100 = close[BENCH].ffill().ewm(span=100, adjust=False, min_periods=100).mean()
    print("Precomputing indicators (ATH/SMA100/Donch15/r21) ...", flush=True)
    ind = dict(ath=close.cummax(),
               sma100=close.rolling(100, min_periods=100).mean(),
               low15=close.rolling(15, min_periods=15).min().shift(1),
               r21=close.pct_change(21))

    out = RES / "ranking_v3.csv"
    with open(out, "w", newline="") as f: csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    # (label, band, score, gate, quality, ath, buffer_mult, exit_mode)
    cfgs = [
        ("mid_rs120_GATE_ref",         "mid", "rs120", True,  False, False, 1.0, "none"),
        ("mid_rs120_PURE_nogate",      "mid", "rs120", False, False, False, 1.0, "none"),
        ("nogate_quality",             "mid", "rs120", False, True,  False, 1.0, "none"),
        ("nogate_ath",                 "mid", "rs120", False, False, True,  1.0, "none"),
        ("nogate_sma100exit",          "mid", "rs120", False, False, False, 1.0, "sma100"),
        ("nogate_donch15exit",         "mid", "rs120", False, False, False, 1.0, "donch15"),
        ("nogate_quality_ath",         "mid", "rs120", False, True,  True,  1.0, "none"),
        ("nogate_ath_sma100",          "mid", "rs120", False, False, True,  1.0, "sma100"),
        ("nogate_qual_ath_sma100",     "mid", "rs120", False, True,  True,  1.0, "sma100"),
        ("nogate_buf22_ath_sma100",    "mid", "rs120", False, False, True,  1.5, "sma100"),
        ("nogate_qual_ath_donch15",    "mid", "rs120", False, True,  True,  1.0, "donch15"),
        ("FULL_gate_qual_ath_sma100",  "mid", "rs120", True,  True,  True,  1.5, "sma100"),
        ("n250_nogate_qual_ath_sma100","n250","rs120", False, True,  True,  1.0, "sma100"),
        ("mid_ret252_nogate_ath_sma100","mid","ret252",False, False, True,  1.0, "sma100"),
    ]
    for lbl, band, score, gate, qual, ath_f, bufm, exit_m in cfgs:
        ts = time.time()
        net = run(close, tv, ema, nbx_ema100, ind, band, score, 15, gate, qual, ath_f, bufm, exit_m, RT, 0.0)
        tax = run(close, tv, ema, nbx_ema100, ind, band, score, 15, gate, qual, ath_f, bufm, exit_m, RT, 0.20)
        m = net['modern']; fills = net['st']['fills'] or 1
        row = dict(config=lbl, band=band, score=score, gate=int(gate), quality=int(qual),
                   ath=int(ath_f), buffer=bufm, exit=exit_m,
                   net_cagr=round(net['cagr'], 1), post_tax20=round(tax['cagr'], 1),
                   maxdd=round(net['dd'], 1), sharpe=round(net['sharpe'], 2),
                   calmar=round(net['calmar'], 2) if pd.notna(net['calmar']) else "",
                   mult=round(net['mult'], 1), fills=net['st']['fills'],
                   avg_turn=round(net['st']['turn_sum'] / fills, 3),
                   stop_exits=net['st']['stop_exits'],
                   mod_cagr=round(m['cagr'], 1) if m else "",
                   mod_maxdd=round(m['dd'], 1) if m else "",
                   mod_calmar=round(m['calmar'], 2) if m else "")
        with open(out, "a", newline="") as f: csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print(f"  {lbl:28} net={row['net_cagr']:5.1f}% tax={row['post_tax20']:5.1f}% "
              f"DD={row['maxdd']:6.1f}% Cal={row['calmar']} turn={row['avg_turn']} "
              f"stops={row['stop_exits']} [{time.time()-ts:.0f}s]", flush=True)
        sys.stdout.flush()
    print(f"\nDone {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
