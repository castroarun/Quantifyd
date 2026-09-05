"""research/151 portfolio engine — VCP / N-day closing-high breakout.

Extends the research/142 Blue-Sky harness: same site mechanics (IBD-RS >= 70 percentile,
Rs 5cr 20d-median traded-value floor as of t-1, ETFs excluded, buy-stop AT the pivot
filled at max(pivot, open), stop and trail booked on the CLOSE), with three changes:

  1. PIVOT = rolling N-day maximum CLOSE (the VCP screen's pattern high) instead of the
     all-time-high close. N=30 is the parameterization that best reproduces their
     published VCP trade list (P1d: 25/37 exact pivot prices, 23/37 joint with the
     first-break date).
  2. SIZING can follow THEIR risk-based mechanic — position value = (risk% x equity) /
     stop%, capped at 30% of equity — or our fixed-%-of-NAV convention.
  3. Taxes are Indian FY (1 Apr - 31 Mar) with loss-netting and carry-forward:
     20% STCG, 12.5% LTCG (>365 days held), STCL offsets STCG then LTCG.

Idle cash accrues `cash_yield` (default 5% p.a.) daily. Costs are bps per side.
"""
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

CAPITAL = 1_000_000.0
TV_FLOOR = 5e7


@dataclass
class Cfg:
    pivot_n: int = 30              # rolling closing-high lookback (the pattern high)
    near_pct: float = 0.20         # prev close must be within this much BELOW the pivot
    rs_min: float = 70.0
    stop_pct: float = 0.08         # cut a loser at, close basis
    exit_kind: str = 'sma50'       # sma15 | sma20 | sma50 | sma150 | target25
    breakeven_at: float = 0.0      # 0 = off; else raise the stop to entry once +x
    slots: int = 5
    sizing: str = 'risk'           # risk | fixed
    risk_pct: float = 0.02
    cap_pct: float = 0.30
    size_pct: float = 0.0625       # only used when sizing == 'fixed'
    cost_bps: float = 25.0
    fill: str = 'realistic'        # realistic = max(pivot, open) | pivot | close
    gate: str = ''                 # '' = off, else index symbol for the weak-market gate
    gate_sma: int = 200
    tax: bool = True
    cash_yield: float = 0.05
    start: str = '2006-01-01'
    end: str = '2026-12-31'
    selection: str = 'random'      # random | rs
    capwin: float = 0.0            # >0: cap each trade's gross return at this (robustness)
    drop_topn: int = 0             # >0: delete the n best trades ex post (robustness)


def build_signal(F, dates, symbols, meta, cfg):
    """Return (TRIG bool matrix, PIV float matrix, TRAIL float matrix, RS matrix)."""
    close = pd.DataFrame(F['close'], index=dates, columns=symbols)
    tv20 = pd.DataFrame(F['tv20'], index=dates, columns=symbols)
    prev_close = close.shift(1)
    eligible = (tv20.shift(1) >= TV_FLOOR)
    eligible[meta['etfs']] = False

    r63 = close / close.shift(63) - 1
    r126 = close / close.shift(126) - 1
    r189 = close / close.shift(189) - 1
    r252 = close / close.shift(252) - 1
    score = (2 * r63 + r126 + r189 + r252).where(eligible)
    rs = (score.rank(axis=1, pct=True) * 100).shift(1)

    # rolling max CLOSE, strictly prior bars. pandas rolling skips NaN, so a missing
    # row cannot poison a max window (unlike a mean).
    piv = close.shift(1).rolling(cfg.pivot_n, min_periods=min(5, cfg.pivot_n)).max()
    prev_piv = piv.shift(1)

    setup = (eligible & (rs >= cfg.rs_min)
             & (prev_close < piv) & (prev_close >= (1 - cfg.near_pct) * piv))
    trig = setup & (close > piv) & (prev_close <= prev_piv) & piv.notna()

    if cfg.exit_kind.startswith('sma'):
        trail = pd.DataFrame(F[cfg.exit_kind], index=dates, columns=symbols)
    else:
        trail = pd.DataFrame(np.full(close.shape, np.nan, dtype='float32'),
                             index=dates, columns=symbols)
    return (trig.fillna(False).values, piv.values.astype('float32'),
            trail.values.astype('float32'), rs.values.astype('float32'))


def weak_array(F, dates, symbols, cfg):
    if not cfg.gate:
        return np.zeros(len(dates), dtype=bool)
    if cfg.gate not in symbols:
        raise SystemExit(f'gate series {cfg.gate} not in frames')
    s = pd.Series(F['close'][:, symbols.index(cfg.gate)], index=dates).dropna()
    weak = (s < s.rolling(cfg.gate_sma, min_periods=cfg.gate_sma // 2).mean()).shift(1)
    return weak.reindex(dates).ffill().fillna(False).to_numpy(dtype=bool)


def _fy(d):
    """Indian financial year label: FY starting 1 April."""
    return d.year if d.month >= 4 else d.year - 1


def simulate(seed, cfg, days_idx, dates, C, H, O, PIV, TRAIL, TRIG, RS, weak_arr):
    rng = np.random.default_rng(seed)
    cost = cfg.cost_bps / 10000.0
    cash = CAPITAL
    positions = []            # [col, entry_i, buy, qty, peak_close]
    trades = []
    equity = np.empty(len(days_idx))
    passed_up = 0
    st_pnl = lt_pnl = carry = 0.0
    cur_fy = _fy(dates[days_idx[0]])
    ydaily = 1.0 + cfg.cash_yield / 252.0
    frac = min(cfg.risk_pct / cfg.stop_pct, cfg.cap_pct) if cfg.sizing == 'risk' else cfg.size_pct
    tgt = 1.25 if cfg.exit_kind == 'target25' else None

    for k, i in enumerate(days_idx):
        if cfg.cash_yield and cash > 0:
            cash *= ydaily
        fy = _fy(dates[i])
        if cfg.tax and fy != cur_fy:
            st, lt = st_pnl + carry, lt_pnl
            if st < 0:
                lt += st
                st = 0.0
            carry = min(lt, 0.0)
            lt = max(lt, 0.0)
            cash -= 0.20 * max(st, 0.0) + 0.125 * lt
            st_pnl = lt_pnl = 0.0
            cur_fy = fy

        # ---- entries
        if not weak_arr[i] and len(positions) < cfg.slots:
            cand = np.nonzero(TRIG[i])[0]
            if len(cand):
                mtm = sum(q * (C[i, c] if C[i, c] == C[i, c] else b)
                          for c, _, b, q, _ in positions)
                eq = cash + mtm
                cand = (rng.permutation(cand) if cfg.selection == 'random'
                        else cand[np.argsort(-np.nan_to_num(RS[i, cand]))])
                for c in cand:
                    if len(positions) >= cfg.slots:
                        passed_up += 1
                        continue
                    piv = float(PIV[i, c])
                    if cfg.fill == 'pivot':
                        fillpx = piv
                    elif cfg.fill == 'close':
                        fillpx = float(C[i, c])
                    else:
                        fillpx = max(piv, float(O[i, c]))
                    if not np.isfinite(fillpx) or fillpx <= 0:
                        continue
                    qty = int(frac * eq / fillpx)
                    if qty < 1 or cash < qty * fillpx * (1 + cost):
                        passed_up += 1
                        continue
                    cash -= qty * fillpx * (1 + cost)
                    positions.append([c, i, fillpx, qty, fillpx])

        # ---- exits, on the close
        still = []
        for pos in positions:
            c, ei, b, q, peak = pos
            cl = C[i, c]
            if cl != cl:
                still.append(pos)
                continue
            cl = float(cl)
            if cl > peak:
                pos[4] = peak = cl
            reason = None
            floor = b * (1 - cfg.stop_pct)
            if cfg.breakeven_at and peak >= b * (1 + cfg.breakeven_at):
                floor = max(floor, b)
            if cl <= floor:
                reason = 'stop'
            elif tgt is not None and cl >= b * tgt:
                reason = 'target'
            elif i > ei and TRAIL[i, c] == TRAIL[i, c] and cl < TRAIL[i, c]:
                reason = 'trail'
            if reason:
                gross = cl / b - 1
                if cfg.capwin and gross > cfg.capwin:
                    cl = b * (1 + cfg.capwin)
                cash += q * cl * (1 - cost)
                pnl = q * (cl - b) - q * (cl + b) * cost
                held = (dates[i] - dates[ei]).days
                if held > 365:
                    lt_pnl += pnl
                else:
                    st_pnl += pnl
                trades.append((c, ei, i, b, cl, reason, pnl, held))
            else:
                still.append(pos)
        positions = still
        mtm = sum(q * (C[i, c] if C[i, c] == C[i, c] else b)
                  for c, _, b, q, _ in positions)
        equity[k] = cash + mtm

    last = days_idx[-1]
    for c, ei, b, q, _ in positions:
        cl = C[last, c]
        cl = float(cl) if cl == cl else b
        trades.append((c, ei, last, b, cl, 'open_marked', q * (cl - b),
                       (dates[last] - dates[ei]).days))
    return equity, trades, passed_up


def stats(equity, dates_used, trades, capital=CAPITAL):
    e = pd.Series(equity, index=dates_used)
    yrs = (dates_used[-1] - dates_used[0]).days / 365.25
    cagr = (e.iloc[-1] / capital) ** (1 / yrs) - 1 if e.iloc[-1] > 0 else -1.0
    dd = float((e / e.cummax() - 1).min())
    r = np.array([t[4] / t[3] - 1 for t in trades]) if trades else np.array([0.0])
    wins, losses = r[r > 0], r[r <= 0]
    # longest losing streak
    streak = best = 0
    for x in sorted(trades, key=lambda t: t[2]):
        if x[4] <= x[3]:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    yearly = e.groupby(e.index.year).last()
    yr = yearly.pct_change()
    yr.iloc[0] = yearly.iloc[0] / capital - 1
    iy = {}
    for y, g in e.groupby(e.index.year):
        run = g.cummax()
        iy[int(y)] = round(float((g / run - 1).min()) * 100, 1)
    return dict(final=float(e.iloc[-1]), x=float(e.iloc[-1] / capital), cagr=cagr * 100,
                dd=dd * 100, calmar=(cagr * 100) / abs(dd * 100) if dd else np.nan,
                n=len(trades), win=float((r > 0).mean() * 100),
                avg_win=float(wins.mean() * 100) if len(wins) else 0.0,
                avg_loss=float(losses.mean() * 100) if len(losses) else 0.0,
                mean=float(r.mean() * 100), median=float(np.median(r) * 100),
                streak=best, tpy=len(trades) / yrs,
                yearly={int(k): round(v * 100, 1) for k, v in yr.items()},
                intra_dd=iy), e
