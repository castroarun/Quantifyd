"""
G4 portfolio backtest (research/73) — weekly SuperTrend(10,3) long-only book.

Causal: signal at weekly close of week t -> fill at open of week t+1.
Config knobs: regime gate (NIFTYBEES>200DMA), 40/40/20 profit-booking, max positions,
per-name weight, cost, tax. Benchmark = NIFTYBEES buy & hold.

Outputs weekly equity curve + full metrics vs benchmark, per-year table.
"""
import os, sys, csv, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from st_weekly_engine import get_conn, load_daily, to_weekly, ROOT
sys.path.insert(0, ROOT)
from services.technical_indicators import calc_supertrend

RES = os.path.join(HERE, '..', 'results')
os.makedirs(RES, exist_ok=True)


def build_panel(conn, symbols, atr_period=10, multiplier=3.0):
    """Per-symbol weekly frame with open/close + ST dir + flip flags. Keyed by symbol."""
    panel = {}
    for sym in symbols:
        d = load_daily(conn, sym)
        if d.empty:
            continue
        w = to_weekly(d)
        if len(w) < 30:
            continue
        st, direction = calc_supertrend(w[['high', 'low', 'close']].assign(open=w['open']),
                                        atr_period, multiplier)
        w = w.copy()
        w['dir'] = direction.values
        w['flip_up'] = (w['dir'] == 1) & (w['dir'].shift(1) == -1)
        w['flip_dn'] = (w['dir'] == -1) & (w['dir'].shift(1) == 1)
        w['wk_gain'] = w['close'] / w['open'] - 1.0   # king-candle proxy on flip week
        panel[sym] = w
    return panel


def bench_weekly(conn, start, end):
    d = load_daily(conn, 'NIFTYBEES')
    w = to_weekly(d)
    w = w[(w.index >= pd.Timestamp(start)) & (w.index <= pd.Timestamp(end))]
    # 200-DMA on daily, sampled to weekly (Friday)
    d = d.copy()
    d['ma200'] = d['close'].rolling(200).mean()
    ma_w = d['ma200'].resample('W-FRI').last()
    risk_on = (w['close'] > ma_w.reindex(w.index)).fillna(False)
    return w, risk_on


def run_portfolio(panel, weeks, bench_w, risk_on, cfg):
    """
    cfg: dict(max_pos, weight, cost_side, gate, booking, idle_rate, tax)
    Returns weekly equity series (index=week dates) + trade log + stats.
    """
    cap0 = 1_000_000.0
    cash = cap0
    pos = {}   # sym -> dict(shares, entry_px, entry_wk, peak, booked)  booked in {0,1,2}
    max_pos = cfg['max_pos']
    weight = cfg['weight']
    cost = cfg['cost_side']
    idle_wk = (1 + cfg['idle_rate']) ** (1 / 52) - 1
    equity = []
    trades = []

    def price_at(sym, wk, field):
        w = panel[sym]
        if wk in w.index:
            v = w.at[wk, field]
            return float(v) if pd.notna(v) and v > 0 else None
        return None

    for wi in range(len(weeks) - 1):
        t = weeks[wi]
        t1 = weeks[wi + 1]          # execution week (fill at its open)
        # idle interest on cash
        cash *= (1 + idle_wk)

        # ---- gather signals from week t close ----
        exits, entries = [], []
        for sym, w in panel.items():
            if t not in w.index:
                continue
            if sym in pos:
                if bool(w.at[t, 'flip_dn']):
                    exits.append(sym)
            else:
                if bool(w.at[t, 'flip_up']):
                    entries.append((sym, float(w.at[t, 'wk_gain'])))

        # ---- profit-booking check (uses week t close, executes t1 open) ----
        booking_sells = []   # (sym, frac_of_original)
        if cfg['booking']:
            for sym, p in pos.items():
                if sym in exits:
                    continue
                c = price_at(sym, t, 'close')
                if c is None:
                    continue
                p['peak'] = max(p['peak'], c)
                r = c / p['entry_px']
                if p['booked'] == 0 and r >= 1.40:
                    booking_sells.append((sym, 0.40));
                elif p['booked'] == 1 and r >= 1.96:
                    booking_sells.append((sym, 0.40))

        # ---- EXECUTE at t1 open ----
        # exits (full)
        for sym in exits:
            px = price_at(sym, t1, 'open') or price_at(sym, t1, 'close') or price_at(sym, t, 'close')
            if px is None:
                continue
            p = pos.pop(sym)
            proceeds = p['shares'] * px * (1 - cost)
            gain = p['shares'] * (px - p['entry_px'])
            held_yrs = (t1 - p['entry_wk']).days / 365.25
            if cfg['tax'] and gain > 0:
                rate = 0.10 if held_yrs > 1 else 0.15
                proceeds -= gain * rate
            cash += proceeds
            trades.append(dict(sym=sym, entry=p['entry_wk'].date().isoformat(),
                               exit=t1.date().isoformat(), entry_px=p['entry_px'],
                               exit_px=px, reason='ST_FLIP', ret=px / p['entry_px'] - 1))
        # partial profit-booking sells
        for sym, frac in booking_sells:
            if sym not in pos:
                continue
            p = pos[sym]
            px = price_at(sym, t1, 'open') or price_at(sym, t, 'close')
            if px is None:
                continue
            sell_sh = p['orig_shares'] * frac
            sell_sh = min(sell_sh, p['shares'])
            proceeds = sell_sh * px * (1 - cost)
            gain = sell_sh * (px - p['entry_px'])
            held_yrs = (t1 - p['entry_wk']).days / 365.25
            if cfg['tax'] and gain > 0:
                proceeds -= gain * (0.10 if held_yrs > 1 else 0.15)
            cash += proceeds
            p['shares'] -= sell_sh
            p['booked'] += 1

        # entries: only if gate risk-on (or gate off)
        gate_ok = True
        if cfg['gate']:
            gate_ok = bool(risk_on.get(t, False))
        if gate_ok and entries:
            entries.sort(key=lambda x: -x[1])   # strongest flip-week gain first (king-candle)
            for sym, _ in entries:
                if len(pos) >= max_pos:
                    break
                px = price_at(sym, t1, 'open')
                if px is None:
                    continue
                # size off current NAV so per-name weight tracks the book
                nav_now = cash + positions_value(pos, panel, t)
                alloc = min(weight * nav_now, cash)
                if alloc < 1000:
                    continue
                shares = (alloc * (1 - cost)) / px   # cost as haircut on shares acquired
                cash -= alloc
                pos[sym] = dict(shares=shares, orig_shares=shares, entry_px=px,
                                entry_wk=t1, peak=px, booked=0)

        # ---- mark-to-market at t1 close for equity ----
        nav = cash + positions_value(pos, panel, t1)
        equity.append((t1, nav))

    eq = pd.Series({d: v for d, v in equity})
    return eq, trades


def positions_value(pos, panel, wk):
    tot = 0.0
    for sym, p in pos.items():
        w = panel[sym]
        # last available close <= wk
        sub = w.loc[:wk, 'close']
        if len(sub):
            px = sub.iloc[-1]
            if pd.notna(px):
                tot += p['shares'] * px
    return tot


def cap0_now(equity, cap0):
    return equity[-1][1] if equity else cap0


def metrics(eq, bench):
    eq = eq.dropna()
    bench = bench.reindex(eq.index).dropna()
    eq = eq.reindex(bench.index)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    rets = eq.pct_change().dropna()
    vol = rets.std() * np.sqrt(52)
    sharpe = (rets.mean() * 52) / (rets.std() * np.sqrt(52)) if rets.std() else 0
    downside = rets[rets < 0].std() * np.sqrt(52)
    sortino = (rets.mean() * 52) / downside if downside else 0
    dd = (eq / eq.cummax() - 1).min()
    calmar = cagr / abs(dd) if dd else 0
    # benchmark
    bcagr = (bench.iloc[-1] / bench.iloc[0]) ** (1 / yrs) - 1
    bdd = (bench / bench.cummax() - 1).min()
    return dict(cagr=100 * cagr, vol=100 * vol, sharpe=sharpe, sortino=sortino,
                maxdd=100 * dd, calmar=calmar, years=yrs,
                bench_cagr=100 * bcagr, bench_maxdd=100 * bdd,
                bench_calmar=bcagr / abs(bdd) if bdd else 0,
                final=eq.iloc[-1], total_x=eq.iloc[-1] / eq.iloc[0])


def per_year(eq, bench):
    out = []
    for y in range(eq.index[0].year, eq.index[-1].year + 1):
        e = eq[eq.index.year == y]
        b = bench.reindex(eq.index)[eq.index.year == y]
        if len(e) < 2:
            continue
        er = 100 * (e.iloc[-1] / e.iloc[0] - 1)
        br = 100 * (b.iloc[-1] / b.iloc[0] - 1) if len(b) >= 2 and pd.notna(b.iloc[0]) else float('nan')
        out.append((y, round(er, 1), round(br, 1), round(er - br, 1)))
    return out


def main():
    band = sys.argv[1] if len(sys.argv) > 1 else 'Nifty200'
    start = sys.argv[2] if len(sys.argv) > 2 else '2010-01-01'
    end = '2026-07-07'
    csvmap = {'Nifty50': 'nifty50_official.csv', 'Nifty200': 'nifty200_official.csv',
              'Midcap150': 'niftymidcap150_official.csv', 'Smallcap250': 'niftysmallcap250_official.csv'}
    conn = get_conn()
    syms = [s.strip() for s in pd.read_csv(os.path.join(ROOT, 'backtest_data', csvmap[band]))['Symbol'].dropna()]
    print(f'building panel for {band} ({len(syms)} names)...', flush=True)
    panel = build_panel(conn, syms)
    print(f'panel: {len(panel)} names with data', flush=True)
    bench_w, risk_on = bench_weekly(conn, start, end)
    conn.close()

    # common weekly calendar from benchmark
    weeks = list(bench_w.index)
    risk_on_d = {d: bool(v) for d, v in risk_on.items()}

    configs = {
        'core_nogate':   dict(max_pos=16, weight=0.0625, cost_side=0.0015, gate=False, booking=False, idle_rate=0.065, tax=True),
        'core_gate':     dict(max_pos=16, weight=0.0625, cost_side=0.0015, gate=True,  booking=False, idle_rate=0.065, tax=True),
        'book_gate':     dict(max_pos=16, weight=0.0625, cost_side=0.0015, gate=True,  booking=True,  idle_rate=0.065, tax=True),
    }
    summary = []
    for name, cfg in configs.items():
        eq, trades = run_portfolio(panel, weeks, bench_w['close'], risk_on_d, cfg)
        m = metrics(eq, bench_w['close'])
        py = per_year(eq, bench_w['close'])
        beat = sum(1 for _, er, br, ex in py if ex > 0)
        print(f'\n=== {band} / {name} ({start}+) ===', flush=True)
        print(f"  CAGR {m['cagr']:.1f}%  MaxDD {m['maxdd']:.1f}%  Calmar {m['calmar']:.2f}  "
              f"Sharpe {m['sharpe']:.2f}  Sortino {m['sortino']:.2f}  vol {m['vol']:.1f}%  "
              f"total {m['total_x']:.1f}x  trades={len(trades)}", flush=True)
        print(f"  BENCH NIFTYBEES: CAGR {m['bench_cagr']:.1f}%  MaxDD {m['bench_maxdd']:.1f}%  "
              f"Calmar {m['bench_calmar']:.2f}   beat-index {beat}/{len(py)} yrs", flush=True)
        print(f"  per-year (yr,strat%,idx%,excess%): {py}", flush=True)
        eq.to_csv(os.path.join(RES, f'g4_{band}_{name}_equity.csv'))
        summary.append(dict(band=band, config=name, start=start, **{k: round(v, 2) if isinstance(v, float) else v for k, v in m.items()}, beat_yrs=f'{beat}/{len(py)}', n_trades=len(trades)))
    pd.DataFrame(summary).to_csv(os.path.join(RES, f'g4_summary_{band}.csv'), index=False)
    print('\nDONE ->', os.path.join(RES, f'g4_summary_{band}.csv'), flush=True)


if __name__ == '__main__':
    main()
