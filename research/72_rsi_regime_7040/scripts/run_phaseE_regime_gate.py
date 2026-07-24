#!/usr/bin/env python3
"""Phase E: add a market-regime gate (NIFTYBEES vs its 200-DMA) to the RSI-regime
slot portfolio. Hypothesis: the broad book's -45% DD is market-crash driven; going
to cash when NIFTY<200DMA should cut DD sharply while keeping the momentum return,
delivering BOTH goals (beat index >=1.5x AND lower DD). Ranks by net Calmar."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
SD = Path(__file__).resolve().parent
sys.path.insert(0, str(SD))
import rsi_regime_engine as E
import portfolio_engine as P


def market_gate(dates, ma=200):
    """risk_on[date] = NIFTYBEES close > SMA(ma). Causal (uses close up to that day)."""
    b = E.load('NIFTYBEES')
    sma = b['close'].rolling(ma).mean()
    on = (b['close'] > sma)
    return on.reindex(dates, method='ffill').fillna(False)


def run_gated(price_data, entry_th, exit_th, rsi_len, N, cost_bps, start, end,
              gate_ma=200, gate_mode='exit_all'):
    """Slot portfolio with a market regime gate. gate_mode:
       'off'      : no gate (== plain book)
       'block'    : risk-off blocks NEW entries only (held names ride)
       'exit_all' : risk-off liquidates everything and blocks entries."""
    all_dates = sorted(set().union(*[set(df.index) for df in price_data.values()]))
    all_dates = pd.DatetimeIndex(all_dates)
    if start: all_dates = all_dates[all_dates >= pd.Timestamp(start)]
    if end:   all_dates = all_dates[all_dates <= pd.Timestamp(end)]

    risk_on = market_gate(all_dates, gate_ma) if gate_mode != 'off' else None

    rsi_map, close_map, ff_map = {}, {}, {}
    for s, df in price_data.items():
        rsi_map[s] = E.rsi(df['close'], rsi_len)
        close_map[s] = df['close']
        ff_map[s] = df['close'].reindex(all_dates, method='ffill')

    rt = cost_bps / 10000.0
    cash = 1.0
    holds = {}   # sym -> {'shares':.., 'last':..}
    net = []; gross_mult = 1.0; gross = []
    gcost_track = 1.0
    for dt in all_dates:
        on = True if risk_on is None else bool(risk_on.get(dt, False))
        # mark to market
        nav = cash
        for s, h in holds.items():
            px = ff_map[s].get(dt, np.nan)
            if not np.isnan(px):
                h['last'] = px
            nav += h['shares'] * h['last']
        # EXITS: RSI<exit, no real bar handled by decision-on-real-close only
        to_exit = []
        for s, h in list(holds.items()):
            rv = rsi_map[s].get(dt, np.nan)
            real = close_map[s].get(dt, np.nan)
            force = (gate_mode == 'exit_all' and not on)
            if force or (not np.isnan(rv) and rv < exit_th and not np.isnan(real)):
                to_exit.append((s, real if not np.isnan(real) else h['last']))
        for s, px in to_exit:
            h = holds.pop(s)
            proceeds = h['shares'] * px
            cash += proceeds * (1 - rt)
            gcost_track *= (1 - rt)
        # ENTRIES: only if risk_on (or gate off / block handled)
        allow_entry = (gate_mode == 'off') or on
        if allow_entry and len(holds) < N:
            cands = []
            for s in price_data:
                if s in holds:
                    continue
                rv = rsi_map[s].get(dt, np.nan)
                real = close_map[s].get(dt, np.nan)
                if not np.isnan(rv) and rv >= entry_th and not np.isnan(real) and real > 0:
                    cands.append((rv, s, real))
            cands.sort(reverse=True)
            free = N - len(holds)
            # recompute nav for sizing
            nav_now = cash + sum(h['shares'] * h['last'] for h in holds.values())
            slot_cap = nav_now / N
            for rv, s, px in cands[:free]:
                if cash < slot_cap * 0.5:
                    break
                alloc = min(slot_cap, cash)
                shares = alloc / px
                cost = alloc * rt
                cash -= alloc
                cash -= cost
                gcost_track *= (1 - rt)
                holds[s] = {'shares': shares, 'last': px}
        nav = cash + sum(h['shares'] * h['last'] for h in holds.values())
        net.append((dt, nav))
        gross.append((dt, nav / gcost_track))
    net_eqs = pd.Series(dict(net)); gross_eqs = pd.Series(dict(gross))
    return gross_eqs, net_eqs


def line(net_eqs, gross_eqs, label, nnames, bench_m):
    m = E.metrics(net_eqs); mg = E.metrics(gross_eqs)
    beat = round(m['cagr'] / bench_m['cagr'], 2) if bench_m.get('cagr') else None
    lower = abs(m['maxdd']) < abs(bench_m['maxdd'])
    gate = (beat is not None and beat >= 1.5 and lower)
    print(f"{label:34s} netCAGR={m['cagr']:>6} gross={mg['cagr']:>6} DD={m['maxdd']:>7} "
          f"Calmar={m['calmar']:>5} Sharpe={m['sharpe']:>5} beat={beat}x lowerDD={lower} "
          f"GATE={'PASS' if gate else 'no'}", flush=True)
    return dict(label=label, n=nnames, net_cagr=m['cagr'], gross_cagr=mg['cagr'],
                maxdd=m['maxdd'], calmar=m['calmar'], sharpe=m['sharpe'],
                beat=beat, lowerDD=lower, gate_pass=gate)


def main():
    import csv as _csv
    n50 = [r['Symbol'].strip() for r in _csv.DictReader(open(str(E.DB.parent/'nifty50_official.csv')))]
    from run_phaseD_robustness import broad_universe
    broad = broad_universe()
    price50 = P.load_universe_prices(n50)
    priceB = P.load_universe_prices(broad)
    print(f"qualified nifty50={len(price50)} broad={len(priceB)}\n")

    def bench(start, end):
        return E.metrics(E.buyhold(E.load('NIFTYBEES', start=start, end=end)))
    bfull = bench('2015-01-01', None); boos = bench('2021-01-01', None)
    print(f"NIFTYBEES full CAGR {bfull['cagr']} DD {bfull['maxdd']} Calmar {bfull['calmar']}")
    print(f"NIFTYBEES OOS  CAGR {boos['cagr']} DD {boos['maxdd']} Calmar {boos['calmar']}\n")

    res = []
    cfg = dict(entry_th=60, exit_th=30, rsi_len=14, N=20, cost_bps=15)
    print("== BROAD universe: gate off vs block vs exit_all (full 2015-26) ==")
    for mode in ['off', 'block', 'exit_all']:
        g, n = run_gated(priceB, start='2015-01-01', end=None, gate_mode=mode, **cfg)
        res.append(line(n, g, f'broad_gate_{mode}', len(priceB), bfull))
    print("\n== NIFTY50: gate off vs exit_all (full) ==")
    for mode in ['off', 'exit_all']:
        g, n = run_gated(price50, start='2015-01-01', end=None, gate_mode=mode, **cfg)
        res.append(line(n, g, f'nifty50_gate_{mode}', len(price50), bfull))
    print("\n== BROAD gated exit_all: OOS 2021-26 ==")
    g, n = run_gated(priceB, start='2021-01-01', end=None, gate_mode='exit_all', **cfg)
    res.append(line(n, g, 'broad_gate_exit_all_OOS', len(priceB), boos))
    print("\n== BROAD gated, N & threshold neighborhood (full) ==")
    for N, en, ex in [(15,60,30),(25,60,30),(20,60,40),(20,70,40)]:
        c2 = dict(cfg); c2.update(N=N, entry_th=en, exit_th=ex)
        g, n = run_gated(priceB, start='2015-01-01', end=None, gate_mode='exit_all', **c2)
        res.append(line(n, g, f'broad_gate_N{N}_E{en}X{ex}', len(priceB), bfull))
    # cost stress on the headline gated broad book
    print("\n== cost stress (broad gated exit_all, full) ==")
    for cb in [30, 50]:
        c2 = dict(cfg); c2['cost_bps'] = cb
        g, n = run_gated(priceB, start='2015-01-01', end=None, gate_mode='exit_all', **c2)
        res.append(line(n, g, f'broad_gate_c{cb}', len(priceB), bfull))

    (SD.parent/'results'/'phaseE_regime_gate.json').write_text(
        json.dumps(dict(results=res, bench_full=bfull, bench_oos=boos), indent=2, default=str))
    print("\nsaved results/phaseE_regime_gate.json")


if __name__ == '__main__':
    main()
