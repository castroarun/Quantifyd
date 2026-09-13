"""
research/159 — backtest core for the v3 saucer -> shelf-breakout-near-ATH pattern.

Book (pre-registered, STATUS 11.2):
  v3 entries, NEXT-OPEN fill, 16 slots at 6.25% of NAV, Rs10L, NSE cash CNC,
  slot contention resolved by a seeded random draw, costs in bps per side,
  idle cash at 5.5% p.a., after-tax with Indian FY loss-netting
  (20% STCG / 12.5% LTCG above 365 days, losses carried forward).

Exit rules are CLOSE-based signals filled at the NEXT day's open -- the same convention
as the entry, so neither side gets a look-ahead advantage. Stated explicitly because a
close-signal filled at that same close would be unexecutable in practice.

Everything here is causal: indicators are computed on each symbol's own series and a
position is only ever marked with bars at or before the current day.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252.0
IDLE_YIELD = 0.055
STCG, LTCG = 0.20, 0.125
LTCG_DAYS = 365
SLOTS = 16
SLOT_PCT = 0.0625
START_CAPITAL = 1_000_000.0 * 10          # Rs 10 lakh... see note
START_CAPITAL = 1_000_000.0               # Rs 10,00,000 = Rs 10 lakh

EXITS = ['ST_7_3', 'ST_10_3', 'ST_14_4', 'DON20', 'DON10', 'SMA15', 'EMA50']


# --------------------------------------------------------------------- indicators
def supertrend_dir(high, low, close, period, mult):
    n = len(close)
    out = np.zeros(n, dtype=np.int8)
    if n <= period + 2:
        return out
    tr = np.empty(n); tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = np.full(n, np.nan)
    atr[period - 1] = np.nanmean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    hl2 = (high + low) / 2.0
    up, dn = hl2 + mult * atr, hl2 - mult * atr
    fu, fl = np.copy(up), np.copy(dn)
    d = np.ones(n, dtype=np.int8)
    for i in range(period + 1, n):
        fu[i] = up[i] if (up[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = dn[i] if (dn[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        d[i] = (-1 if close[i] < fl[i] else 1) if d[i - 1] == 1 else (1 if close[i] > fu[i] else -1)
    d[:period + 1] = 1
    return d


def exit_signals(o, h, l, c):
    """Boolean exit-signal arrays (True = the close-based rule says get out today)."""
    s = pd.Series(c)
    sig = {}
    sig['ST_7_3'] = supertrend_dir(h, l, c, 7, 3.0) == -1
    sig['ST_10_3'] = supertrend_dir(h, l, c, 10, 3.0) == -1
    sig['ST_14_4'] = supertrend_dir(h, l, c, 14, 4.0) == -1
    for nm, w in (('DON20', 20), ('DON10', 10)):
        lowest = s.rolling(w, min_periods=w).min().shift(1).to_numpy()
        sig[nm] = np.where(np.isfinite(lowest), c < lowest, False)
    sma15 = s.rolling(15, min_periods=15).mean().to_numpy()
    sig['SMA15'] = np.where(np.isfinite(sma15), c < sma15, False)
    ema50 = s.ewm(span=50, adjust=False, min_periods=50).mean().to_numpy()
    sig['EMA50'] = np.where(np.isfinite(ema50), c < ema50, False)
    return {k: np.asarray(v, dtype=bool) for k, v in sig.items()}


# --------------------------------------------------------------------- price panel
class Panel:
    """Per-symbol arrays aligned to a master trading calendar."""

    def __init__(self, con, symbols, calendar):
        self.cal = calendar
        self.pos = {d: i for i, d in enumerate(calendar)}
        self.n = len(calendar)
        self.open = {}
        self.close = {}
        self.sig = {}
        for sym in symbols:
            d = pd.read_sql_query(
                "SELECT date,open,high,low,close,volume FROM market_data_unified "
                "WHERE symbol=? AND timeframe='day' ORDER BY date", con, params=(sym,))
            d = d[(d['volume'] > 0) & (d['close'] > 0)]
            d = d.drop_duplicates(subset='date', keep='last')
            if len(d) < 60:
                continue
            idx = d['date'].map(self.pos)
            keep = idx.notna()
            d, idx = d[keep], idx[keep].astype(int).to_numpy()
            if len(d) < 60:
                continue
            o = d['open'].to_numpy(np.float64); h = d['high'].to_numpy(np.float64)
            l = d['low'].to_numpy(np.float64); c = d['close'].to_numpy(np.float64)
            sg = exit_signals(o, h, l, c)
            O = np.full(self.n, np.nan, np.float32); C = np.full(self.n, np.nan, np.float32)
            O[idx] = o; C[idx] = c
            # forward-fill the close so a holiday/missing bar marks at the last real
            # price instead of NaN; `open` keeps its NaNs (no bar = no fill possible)
            C = pd.Series(C).ffill().to_numpy(np.float32)
            self.open[sym] = O; self.close[sym] = C
            self.sig[sym] = {k: _scatter_bool(v, idx, self.n) for k, v in sg.items()}

    def has(self, sym):
        return sym in self.close


def _scatter_bool(v, idx, n):
    out = np.zeros(n, dtype=bool)
    out[idx] = v
    return out


# --------------------------------------------------------------------- simulation
def simulate(events, panel, cfg, seed):
    """
    events: list of dicts with keys symbol, entry_i (calendar index), entry_px
    cfg:    exit, hard_stop (bool), time_stop (int or 0), cost_bps,
            gate_ok (bool array over calendar or None)
    Returns (nav array, trades list)
    """
    rng = np.random.default_rng(seed)
    slots = cfg.get('slots', SLOTS)
    slot_pct = cfg.get('slot_pct', SLOT_PCT)
    n = panel.n
    cost = cfg['cost_bps'] / 10000.0
    exit_key = cfg['exit']
    hard = cfg.get('hard_stop', False)
    tstop = cfg.get('time_stop', 0)
    gate = cfg.get('gate_ok')
    # cfg['idle_yield'] lets a caller re-run a cell with the cash carry switched off,
    # so the equity contribution is visible separately (Arun asked for this).
    iy = cfg.get('idle_yield', IDLE_YIELD)
    daily_yield = (1.0 + iy) ** (1.0 / TRADING_DAYS) - 1.0

    by_day = {}
    for e in events:
        by_day.setdefault(e['entry_i'], []).append(e)

    cash = START_CAPITAL
    nav = np.full(n, np.nan)
    open_pos = []          # dicts
    trades = []
    pending_exit = []      # positions whose exit signal fired -> sell at next open
    # tax accounting
    fy_st, fy_lt = 0.0, 0.0
    carry_st = 0.0
    cur_fy = None

    for i in range(n):
        day = panel.cal[i]
        # ---- financial-year boundary: settle tax on 1 April -------------------------
        fy = int(day[:4]) - (1 if day[5:7] < '04' else 0)
        if cur_fy is None:
            cur_fy = fy
        elif fy != cur_fy:
            # Indian FY netting: losses offset gains (STCL against STCG first, then
            # LTCG); anything left over is carried forward. `carry` is <= 0.
            st, lt = fy_st, fy_lt
            pool = carry_st                  # carried-forward losses, <= 0
            if st < 0:
                pool += st; st = 0.0
            if lt < 0:
                pool += lt; lt = 0.0
            if pool < 0 and st > 0:          # absorb losses against STCG first
                use = min(st, -pool); st -= use; pool += use
            if pool < 0 and lt > 0:          # then against LTCG
                use = min(lt, -pool); lt -= use; pool += use
            carry_st = min(pool, 0.0)
            cash -= st * STCG + lt * LTCG
            fy_st = fy_lt = 0.0
            cur_fy = fy

        # ---- sell everything queued from yesterday's close signal -------------------
        still = []
        for p in pending_exit:
            px = panel.open[p['symbol']][i]
            if not np.isfinite(px):
                still.append(p); continue
            proceeds = p['shares'] * px * (1 - cost)
            cash += proceeds
            pnl = proceeds - p['cost_basis']
            held = i - p['entry_i']
            if held >= LTCG_DAYS * 252 / 365:
                fy_lt += pnl
            else:
                fy_st += pnl
            trades.append(dict(symbol=p['symbol'], entry_date=panel.cal[p['entry_i']],
                               exit_date=day, entry_px=p['entry_px'], exit_px=float(px),
                               shares=p['shares'], pnl=pnl,
                               ret_pct=100.0 * (proceeds / p['cost_basis'] - 1.0),
                               bars=held, reason=p['reason']))
        pending_exit = still

        # ---- new entries -----------------------------------------------------------
        cands = by_day.get(i, [])
        if cands and (gate is None or gate[i]):
            free = slots - len(open_pos)
            if free > 0:
                if len(cands) > free:
                    pick = rng.choice(len(cands), size=free, replace=False)
                    cands = [cands[j] for j in sorted(pick)]
                mv = sum(p['shares'] * _px(panel.close[p['symbol']], i) for p in open_pos)
                navnow = cash + mv
                for e in cands:
                    px = panel.open[e['symbol']][i]
                    if not np.isfinite(px) or px <= 0:
                        continue
                    alloc = navnow * slot_pct
                    shares = int(alloc // (px * (1 + cost)))
                    if shares <= 0:
                        continue
                    basis = shares * px * (1 + cost)
                    if basis > cash:
                        continue
                    cash -= basis
                    open_pos.append(dict(symbol=e['symbol'], shares=shares, entry_i=i,
                                         entry_px=float(px), cost_basis=basis,
                                         peak=float(px), reason=''))

        # ---- mark, then evaluate close-based exit signals ---------------------------
        mv = 0.0
        keep = []
        for p in open_pos:
            c = _px(panel.close[p['symbol']], i)
            mv += p['shares'] * c
            out = None
            if hard and c <= p['entry_px'] * 0.92:
                out = 'STOP8'
            elif tstop and (i - p['entry_i']) >= tstop:
                out = 'TIME'
            elif panel.sig[p['symbol']][exit_key][i]:
                out = exit_key
            if out and i + 1 < n:
                p['reason'] = out
                pending_exit.append(p)
            else:
                keep.append(p)
        open_pos = keep
        cash *= (1.0 + daily_yield)
        nav[i] = cash + mv

    # liquidate at the final close
    if open_pos:
        i = n - 1
        for p in open_pos:
            px = _px(panel.close[p['symbol']], i)
            proceeds = p['shares'] * px * (1 - cost)
            cash += proceeds
            trades.append(dict(symbol=p['symbol'], entry_date=panel.cal[p['entry_i']],
                               exit_date=panel.cal[i], entry_px=p['entry_px'],
                               exit_px=float(px), shares=p['shares'],
                               pnl=proceeds - p['cost_basis'],
                               ret_pct=100.0 * (proceeds / p['cost_basis'] - 1.0),
                               bars=i - p['entry_i'], reason='EOD'))
        nav[i] = cash
    return nav, trades


def _px(arr, i):
    v = arr[i]
    return float(v) if np.isfinite(v) else 0.0


# --------------------------------------------------------------------- metrics
def metrics(nav, cal, trades=None):
    nav = pd.Series(nav, index=pd.to_datetime(cal)).dropna()
    if len(nav) < 30 or nav.iloc[0] <= 0:
        return {}
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    dd = (nav / nav.cummax() - 1.0)
    mdd = dd.min()
    r = nav.pct_change().dropna()
    sharpe = (r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan
    out = dict(cagr=round(100 * cagr, 2), maxdd=round(100 * mdd, 2),
               calmar=round(cagr / abs(mdd), 3) if mdd < 0 else np.nan,
               sharpe=round(sharpe, 3), final=round(nav.iloc[-1], 0), years=round(yrs, 2))
    if trades is not None and trades:
        t = pd.DataFrame(trades)
        wins = t[t.pnl > 0]; loss = t[t.pnl <= 0]
        streak = mx = 0
        for p in t.sort_values('exit_date')['pnl']:
            streak = streak + 1 if p <= 0 else 0
            mx = max(mx, streak)
        out.update(trades=len(t),
                   win_rate=round(100 * len(wins) / len(t), 1),
                   avg_win=round(wins.ret_pct.mean(), 2) if len(wins) else np.nan,
                   avg_loss=round(loss.ret_pct.mean(), 2) if len(loss) else np.nan,
                   expectancy=round(t.ret_pct.mean(), 3),
                   max_loss_streak=mx,
                   trades_per_yr=round(len(t) / yrs, 1))
    return out
