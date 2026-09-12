# -*- coding: utf-8 -*-
"""research/164 — the research/161 book simulator, extended on THREE axes only:

  1. `slots`      how many concurrent positions the book may hold
  2. `slot_pct`   how much of NAV each new position is sized at
  3. `select`     which candidate wins a contested slot when more signals fire
                  on one day than there are free slots

Everything else is research/161's `bt_core.simulate` copied VERBATIM: next-open fills on
BOTH legs, SuperTrend(14,4) close trail, no hard stop, 25 bps a side, Indian FY tax
netting (20% STCG / 12.5% LTCG above 365 calendar days, losses carried forward), idle
cash credited DAILY at a post-tax rate and NEVER routed through the tax settlement.

Two bookkeeping additions that change no rule:
  * daily invested fraction (market value of open positions / NAV) — as research/163 did
  * contention accounting: how many days a slot limit actually BOUND, and how many
    qualifying signals were turned away because of it.

`select='random'` is byte-identical to research/161 — the rng is drawn only on days that
bind, in the same order, with the same call, so the incumbent path reproduces exactly.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252.0
STCG, LTCG = 0.20, 0.125
LTCG_DAYS = 365
START_CAPITAL = 1_000_000.0

# rank key -> (event field, descending?)   'random' handled separately
SELECT_RULES = {
    'random': None,                 # the incumbent AND the null control
    'rs':     ('rs252', True),      # IBD-style relative strength: best 12-month mover wins
    'ext':    ('ext_pct', False),   # least extended above the prior ATH wins
    'tv':     ('tv20_cr', True),    # deepest 20-day traded value wins
    'age':    ('x_bars', True),     # longest base (oldest prior ATH) wins
}


def _px(arr, i):
    v = arr[i]
    return float(v) if np.isfinite(v) else 0.0


def simulate(events, panel, cfg, seed):
    """Returns (nav, trades, inv, book) — book carries the contention accounting."""
    rng = np.random.default_rng(seed)
    slots = int(cfg.get('slots', 16))
    slot_pct = float(cfg.get('slot_pct', 0.0625))
    select = cfg.get('select', 'random')
    rule = SELECT_RULES[select]
    n = panel.n
    cost = cfg['cost_bps'] / 10000.0
    exit_key = cfg['exit']
    hard = cfg.get('hard_stop', False)
    tstop = cfg.get('time_stop', 0)
    gate = cfg.get('gate_ok')
    iy = cfg.get('idle_yield', 0.05)
    daily_yield = (1.0 + iy) ** (1.0 / TRADING_DAYS) - 1.0

    by_day = {}
    for e in events:
        by_day.setdefault(e['entry_i'], []).append(e)

    cash = START_CAPITAL
    nav = np.full(n, np.nan)
    inv = np.full(n, np.nan)
    open_pos = []
    trades = []
    pending_exit = []
    fy_st, fy_lt = 0.0, 0.0
    carry_st = 0.0
    cur_fy = None
    days_signal = days_bind = days_full = turned_away = 0
    cash_refused = entries_taken = 0

    for i in range(n):
        day = panel.cal[i]
        # ---- financial-year boundary: settle tax on 1 April ------------------------
        fy = int(day[:4]) - (1 if day[5:7] < '04' else 0)
        if cur_fy is None:
            cur_fy = fy
        elif fy != cur_fy:
            st, lt = fy_st, fy_lt
            pool = carry_st
            if st < 0:
                pool += st; st = 0.0
            if lt < 0:
                pool += lt; lt = 0.0
            if pool < 0 and st > 0:
                use = min(st, -pool); st -= use; pool += use
            if pool < 0 and lt > 0:
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
                               bars=held, reason=p['reason'],
                               notional=p['cost_basis'], tv20_cr=p['tv20_cr']))
        pending_exit = still

        # ---- new entries ------------------------------------------------------------
        cands = by_day.get(i, [])
        if cands:
            days_signal += 1
            free = slots - len(open_pos)
            if free <= 0:
                days_full += 1
                days_bind += 1
                turned_away += len(cands)
            elif len(cands) > free:
                days_bind += 1
                turned_away += len(cands) - free
        if cands and (gate is None or gate[i]):
            free = slots - len(open_pos)
            if free > 0:
                if len(cands) > free:
                    if rule is None:
                        pick = rng.choice(len(cands), size=free, replace=False)
                        cands = [cands[j] for j in sorted(pick)]
                    else:
                        key, desc = rule
                        sgn = -1.0 if desc else 1.0
                        cands = sorted(cands, key=lambda e: (sgn * e[key], e['symbol']))[:free]
                mv = sum(p['shares'] * _px(panel.close[p['symbol']], i) for p in open_pos)
                navnow = cash + mv
                for e in cands:
                    px = panel.open[e['symbol']][i]
                    if not np.isfinite(px) or px <= 0:
                        continue
                    alloc = navnow * slot_pct
                    shares = int(alloc // (px * (1 + cost)))
                    if shares <= 0:
                        cash_refused += 1
                        continue
                    basis = shares * px * (1 + cost)
                    if basis > cash:
                        # the SLOT was free but there was no cash to fill it. At full
                        # investment this, not the slot count, is the binding constraint.
                        cash_refused += 1
                        continue
                    cash -= basis
                    entries_taken += 1
                    open_pos.append(dict(symbol=e['symbol'], shares=shares, entry_i=i,
                                         entry_px=float(px), cost_basis=basis,
                                         peak=float(px), reason='',
                                         tv20_cr=float(e.get('tv20_cr', np.nan))))

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
        cash *= (1.0 + daily_yield)          # post-tax carry, never taxed again
        nav[i] = cash + mv
        inv[i] = mv / nav[i] if nav[i] > 0 else np.nan

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
                               bars=i - p['entry_i'], reason='EOD',
                               notional=p['cost_basis'], tv20_cr=p['tv20_cr']))
        nav[i] = cash
    book = dict(days_signal=days_signal, days_bind=days_bind, days_full=days_full,
                turned_away=turned_away, cash_refused=cash_refused,
                entries_taken=entries_taken)
    return nav, trades, inv, book


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
        # capacity: entry notional as a share of the name's own 20-day median traded value
        tv = t['tv20_cr'].to_numpy(float) * 1e7
        ok = np.isfinite(tv) & (tv > 0)
        if ok.any():
            frac = 100.0 * t['notional'].to_numpy(float)[ok] / tv[ok]
            out['pos_rs_med'] = round(float(np.median(t['notional'])), 0)
            out['cap_pct_med'] = round(float(np.median(frac)), 3)
            out['cap_pct_p95'] = round(float(np.percentile(frac, 95)), 3)
            out['cap_over1pct'] = round(100.0 * float((frac > 1.0).mean()), 1)
    return out


def window_dd(nav, cal, lo, hi):
    """Drawdown inside a window, measured from the running peak of the FULL curve."""
    s = pd.Series(nav, index=pd.to_datetime(cal)).dropna()
    dd = s / s.cummax() - 1.0
    sub = dd[(dd.index >= lo) & (dd.index <= hi)]
    return round(float(sub.min() * 100), 2) if len(sub) else np.nan


def window_cagr(nav, cal, lo, hi):
    idx = [i for i, d in enumerate(cal) if lo <= d <= hi]
    if not idx:
        return np.nan
    sl = slice(idx[0], idx[-1] + 1)
    m = metrics(np.asarray(nav)[sl], cal[sl])
    return m.get('cagr', np.nan)
