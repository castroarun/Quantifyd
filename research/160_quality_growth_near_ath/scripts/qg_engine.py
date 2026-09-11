# -*- coding: utf-8 -*-
"""research/160 — quality-growth near all-time-high: the positional backtest ENGINE.

CAUSALITY CONTRACT (binding, playbook 5A). Every decision in this engine is taken on a
bar's CLOSE and filled at the NEXT day's OPEN (default) or at that same CLOSE (the second,
labelled convention). There is no intraday level, no `high >= level` trigger and no
same-bar fill anywhere: the `high`/`low` arrays are not even loaded by the simulator. The
trigger/fill trap that cost r/142 42 CAGR points is therefore structurally impossible here
rather than merely guarded against. Exits obey the identical contract.

Everything else is an axis: the near-ATH state and its threshold, the liquidity floor, the
point-in-time fundamental mask and its missing-data policy, the entry mechanic, the ranking
axis, the hysteresis buffer, the exit stack, the index regime gate, the book size, the
costs, the taxes and the idle-cash yield.

Reads the panel cache built by qg_panel.py. Writes ONE row per cell, incrementally and
resume-safely, to --out. Never touches market_data.db (the panel builder opens it
read-only) or anything under services/.

See QUALITY_GROWTH_NEAR_ATH_ENGINE_BUILD_STATUS.md section 7 for the calling contract and
the exact column list.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
MCAP_SNAP = ROOT / 'research' / '142_bananapatterns_replication' / 'results' / 'mcap_snapshot.json'

CAPITAL = 10_000_000.0
STCG, LTCG = 0.20, 0.125
TRADING_DAYS = 252.0

COLUMNS = [
    'label', 'start', 'end', 'entry', 'cadence', 'rank', 'slots', 'buffer', 'state', 'k',
    'tv_floor', 'mask', 'mask_missing', 'exits', 'index_gate', 'gate_action', 'fill',
    'cost_bps', 'tax', 'cash_yield', 'max_position_pct', 'n_paths', 'path_kind',
    'cagr_gross', 'cagr_net', 'cagr_net_tax',
    'cagr_net_tax_min', 'cagr_net_tax_max', 'cagr_net_tax_worstpath',
    'maxdd', 'maxdd_worst', 'calmar', 'sharpe',
    'trades_per_yr', 'win_rate', 'avg_win_pct', 'avg_loss_pct', 'expectancy_net_pct',
    'max_losing_streak', 'turnover_x_nav_yr', 'avg_pct_invested', 'capacity_ratio',
    'n_trades', 'final_x', 'yearly',
]


# ----------------------------------------------------------------------------- panel ---
class Panel:
    """Lazy accessor over the qg_panel.py npz. Frames are decompressed on first touch and
    cached, so a sweep that only uses sma50 never pays for sma200."""

    def __init__(self, path, shift=0):
        self.path = str(path)
        self._z = np.load(self.path)
        self.dates = np.asarray(self._z['dates'])
        self.syms = [str(s) for s in np.asarray(self._z['syms'])]
        self.is_fund = np.asarray(self._z['is_fund'])
        self.meta = json.loads(str(np.asarray(self._z['meta'])[0]))
        self.sym_ix = {s: i for i, s in enumerate(self.syms)}
        self.dt = pd.to_datetime(pd.Index(self.dates))
        self.shift = int(shift)
        self._cache = {}

    @classmethod
    def load(cls, path, shift=0):
        return cls(path, shift=shift)

    def f(self, key):
        """A [dates x syms] float32 frame. With shift=n every price/feature is moved n
        rows LATER, which is the look-ahead probe: a correct engine must not reproduce the
        unshifted result."""
        if key not in self._cache:
            a = np.asarray(self._z[key])
            if self.shift:
                b = np.full_like(a, np.nan)
                b[self.shift:] = a[:-self.shift]
                a = b
            self._cache[key] = a
        return self._cache[key]

    def prev(self, key):
        k = '__prev__' + key
        if k not in self._cache:
            a = self.f(key)
            b = np.full_like(a, np.nan)
            b[1:] = a[:-1]
            self._cache[k] = b
        return self._cache[k]

    def series(self, sym):
        """Close series of one symbol as a pandas Series (benchmarks, gates)."""
        j = self.sym_ix.get(sym)
        if j is None:
            return None
        s = pd.Series(self.f('close')[:, j], index=self.dt)
        return s.dropna()


class Derived:
    """Per-process derived state. The heavy frames already live in the panel; this holds
    the market-cap proxy and nothing else. Kept as a class because the study agent's
    calling contract names it."""

    def __init__(self, panel: Panel, mcap_snapshot=MCAP_SNAP):
        self.panel = panel
        self.shares = None
        self.mcap_known = 0
        p = Path(mcap_snapshot)
        if p.exists():
            snap = json.load(open(p))
            sh = np.full(len(panel.syms), np.nan)
            for s, v in snap.items():
                j = panel.sym_ix.get(s)
                if j is not None and isinstance(v, dict) and v.get('mcap') and v.get('px'):
                    sh[j] = v['mcap'] / v['px']
            self.shares = sh
            self.mcap_known = int(np.isfinite(sh).sum())


# ------------------------------------------------------------------------------- cell ---
@dataclass
class Cell:
    label: str = 'cell'
    start: str = '2010-01-01'
    end: str = '2026-09-10'
    entry: str = 'rebalance'            # rebalance | first_qualify | ath_breakout
    cadence: str = 'monthly'            # monthly | quarterly | semiannual | annual
    rank: str = 'rs'                    # rs | dist_ath | mcap_asc | mcap_desc | tv_desc | random
    slots: int = 15
    buffer: float = 1.5                 # hysteresis: keep a holding ranked within buffer*N
    retain: str = 'strict'              # strict: must still qualify | loose: state not required
    state: str = 'near'                 # near | new_ath
    k: float = 0.90                     # close >= k * causal ATH close
    tv_floor: float = 2.0               # Rs crore, 20d median traded value
    mask: str = ''                      # npz eligibility mask ('' = price-only arm)
    mask_missing: str = 'fail'          # fail | pass
    exits: str = 'none'
    index_gate: str = 'none'            # none | nifty200sma | niftybees100sma_weekly
    gate_action: str = 'block_new'      # block_new | liquidate_all
    fill: str = 'next_open'             # next_open | same_close
    cost_bps: float = 25.0
    tax: bool = True
    cash_yield: float = 0.05
    max_position_pct: float = 0.30
    capital: float = CAPITAL
    reentry: str = 'transition'         # transition | any  (daily entry modes)
    stale_exit_days: int = 60           # liquidate a name that has not printed a close
                                        # for this many sessions (suspension/delisting).
                                        # 0 = never, which carries dead names at their
                                        # last traded price for the rest of the run.
    offsets: int = 0                    # 0/1 = single path; 12 = the offset ensemble
    seeds: int = 0                      # 0/1 = single path; 30 = the seed ensemble
    arms: str = 'all'                   # all = gross/net/net_tax | tax = the net-of-tax arm only


def parse_exits(spec):
    """'sma_trail:50,peak_dd:25' -> dict. Unknown families raise rather than silently no-op."""
    out = dict(hard_stop=None, peak_dd=None, sma_trail=None, donch=None, months=None,
               fund_fail=False)
    known = {'none', 'fund_fail', 'sma_trail', 'peak_dd', 'donchian_low', 'time', 'hard_stop'}
    for part in (spec or 'none').split(','):
        part = part.strip()
        if not part:
            continue
        name, _, arg = part.partition(':')
        if name not in known:
            raise SystemExit('unknown exit family %r (known: %s)' % (name, sorted(known)))
        if name == 'none':
            continue
        if name == 'fund_fail':
            out['fund_fail'] = True
        elif name == 'sma_trail':
            out['sma_trail'] = int(arg)
        elif name == 'peak_dd':
            out['peak_dd'] = float(arg) / 100.0
        elif name == 'donchian_low':
            out['donch'] = int(arg)
        elif name == 'time':
            out['months'] = int(arg)
        elif name == 'hard_stop':
            out['hard_stop'] = float(arg) / 100.0
    return out


# -------------------------------------------------------------------------- mask/gate ---
def mask_meta(path):
    z = np.load(path)
    # numpy 2.x has no min/max ufunc loop for <U dtypes; sort the python strings instead
    d = sorted(str(x)[:10] for x in z['dates'])
    return dict(first=d[0], last=d[-1], n_rows=len(d),
                n_cols=len(z['cols']), true_frac=float(np.asarray(z['mask']).mean()))


def load_mask(path, panel: Panel, missing='fail'):
    """npz {dates:<U10 (1st of month), cols, mask[bool]} -> daily bool [nd x ns].

    A mask row is in force from its own date until the next one (forward fill). Symbols the
    mask has never heard of, and all days before its first row, take the MISSING POLICY —
    reported both ways because the gap between them is the coverage bias."""
    z = np.load(path)
    mdates = np.asarray([str(x)[:10] for x in z['dates']])
    mcols = [str(x) for x in z['cols']]
    mk = np.asarray(z['mask']).astype(bool)
    order = np.argsort(mdates)
    mdates, mk = mdates[order], mk[order]

    default = (missing == 'pass')
    out = np.full((len(panel.dates), len(panel.syms)), default, dtype=bool)
    row = np.searchsorted(mdates, panel.dates, side='right') - 1     # row in force per day
    have = row >= 0
    src_j, dst_j = [], []
    for i, s in enumerate(mcols):
        j = panel.sym_ix.get(s)
        if j is not None:
            src_j.append(i)
            dst_j.append(j)
    if src_j:
        sub = mk[np.clip(row, 0, None)][:, np.asarray(src_j)]
        sub[~have] = default
        out[:, np.asarray(dst_j)] = sub
    return out, len(src_j), len(mcols)


def build_gate(panel: Panel, kind):
    """True on days new entries are blocked / the book is liquidated. Computed on the
    dropna'd index series (never on the union index — that is how a phantom row silently
    disabled r/142's gate) and shifted one day so today's decision uses yesterday's state."""
    nd = len(panel.dates)
    if kind == 'none':
        return np.zeros(nd, dtype=bool)
    if kind == 'nifty200sma':
        s = panel.series('NIFTY50')
        if s is None or len(s) < 250:
            raise SystemExit('NIFTY50 series unavailable for --index-gate nifty200sma')
        weak = (s < s.rolling(200, min_periods=200).mean())
    elif kind == 'niftybees100sma_weekly':
        s = panel.series('NIFTYBEES')
        if s is None:
            raise SystemExit('NIFTYBEES series unavailable')
        wk = s.resample('W-FRI').last().dropna()
        weak_w = (wk < wk.rolling(100, min_periods=100).mean())
        weak = weak_w.reindex(s.index.union(weak_w.index)).ffill().reindex(s.index)
    else:
        raise SystemExit('unknown --index-gate %r' % kind)
    weak = weak.shift(1)
    full = weak.reindex(panel.dt).ffill()
    return full.fillna(False).to_numpy().astype(bool)


# ------------------------------------------------------------------------- simulation ---
def _fy_of(ts):
    """Indian financial year label: 1 Apr .. 31 Mar."""
    return ts.year if ts.month >= 4 else ts.year - 1


def _simulate(ctx, path_id, cost, use_tax):
    """One path. ctx carries every per-cell array; path_id is a rebalance-day offset
    (rebalance mode) or a tie-break seed (daily modes)."""
    cell = ctx['cell']
    C, O = ctx['C'], ctx['O']
    ATH, TV = ctx['ATH'], ctx['TV']
    QUAL, MASKD = ctx['QUAL'], ctx['MASKD']
    RV = ctx['RV']
    SMA, DL = ctx['SMA'], ctx['DL']
    weak = ctx['weak']
    days = ctx['days']
    dts = ctx['dts']
    ex = ctx['ex']
    slots, buf = int(cell.slots), float(cell.buffer)
    tgt_pct = min(1.0 / slots, float(cell.max_position_pct))
    next_open = (cell.fill == 'next_open')
    liq_all = (cell.gate_action == 'liquidate_all')
    stale_n = int(cell.stale_exit_days)
    rng = np.random.default_rng(10_000 + path_id)

    if cell.entry == 'rebalance':
        reb = ctx['reb_days'][path_id]              # set of indices into the full calendar
    else:
        reb = set()
        NEWQ = ctx['NEWQ']

    cash = float(cell.capital)
    y_day = 1.0 + float(cell.cash_yield) / TRADING_DAYS
    pos = {}   # col -> [entry_i, entry_px, qty, peak_close, last_px, tv_at_entry, last_trade_i]
    pend_sell, pend_buy = [], []
    trades = []
    equity = np.empty(len(days))
    invested = np.empty(len(days))
    sell_notional = 0.0
    fy_pool, fy_carry = 0.0, 0.0
    cur_fy = _fy_of(dts[0])
    missed_buy = missed_sell = 0

    for k, i in enumerate(days):
        # ---- 1. execute what was decided yesterday, at today's OPEN -------------------
        if next_open and (pend_sell or pend_buy):
            keep_s = []
            for col, reason, age in pend_sell:
                if col not in pos:
                    continue
                px = O[i, col]
                if not np.isfinite(px):
                    px = C[i, col]
                if not np.isfinite(px):
                    # the name did not trade: a market order cannot fill. Carry the order,
                    # but never forever - after 10 sessions force the exit at the last
                    # known price so a dead symbol cannot sit in the book for years.
                    if age < 10:
                        keep_s.append((col, reason, age + 1))
                        continue
                    px = pos[col][4]
                    missed_sell += 1
                ei, bpx, qty, _pk, _lp, tve, _lt = pos.pop(col)
                proceeds = qty * px * (1 - cost)
                cash += proceeds
                sell_notional += qty * px
                if use_tax:
                    basis = qty * bpx * (1 + cost)
                    held = (dts[k] - ctx['dt_all'][ei]).days
                    fy_pool += (LTCG if held > 365 else STCG) * (proceeds - basis)
                trades.append((col, ei, i, bpx, float(px), qty, reason, tve))
            pend_sell = keep_s
            for col, qty in pend_buy:
                px = O[i, col]
                if not np.isfinite(px):
                    missed_buy += 1
                    continue
                if col in pos or len(pos) >= slots:
                    continue
                need = qty * px * (1 + cost)
                if need > cash:
                    qty = int(cash / (px * (1 + cost)))
                    need = qty * px * (1 + cost)
                if qty < 1:
                    missed_buy += 1
                    continue
                cash -= need
                pos[col] = [i, float(px), qty, float(px), float(px),
                            float(TV[i, col]) if np.isfinite(TV[i, col]) else np.nan, i]
            pend_buy = []

        # ---- 2. idle cash + tax settlement at the FY boundary -------------------------
        if cash > 0:
            cash *= y_day
        fy = _fy_of(dts[k])
        if use_tax and fy != cur_fy:
            total = fy_pool + fy_carry
            if total > 0:
                cash -= total
                fy_carry = 0.0
            else:
                fy_carry = total
            fy_pool = 0.0
            cur_fy = fy

        is_reb = i in reb
        gate_weak = bool(weak[i])

        # ---- 3. EXIT decisions on today's CLOSE ---------------------------------------
        if pos:
            for col in list(pos.keys()):
                p = pos[col]
                c = C[i, col]
                if not np.isfinite(c):
                    # The name did not print a close. No decision is possible on a price
                    # that does not exist - but a position must not sit at a frozen price
                    # forever either, which is how a delisted name flatters a hold-forever
                    # arm. After stale_exit_days sessions without a print, liquidate at
                    # the last known price.
                    if stale_n and (i - p[6]) >= stale_n and not any(col == s_[0] for s_ in pend_sell):
                        pend_sell.append((col, 'stale', 0))
                    continue
                p[6] = i
                p[4] = float(c)
                if c > p[3]:
                    p[3] = float(c)
                if any(col == s[0] for s in pend_sell):
                    continue
                reason = None
                if liq_all and gate_weak:
                    reason = 'index_gate'
                elif ex['hard_stop'] and c <= p[1] * (1 - ex['hard_stop']):
                    reason = 'hard_stop'
                elif ex['peak_dd'] and c <= p[3] * (1 - ex['peak_dd']):
                    reason = 'peak_dd'
                elif ex['sma_trail'] and i > p[0] and np.isfinite(SMA[i, col]) and c < SMA[i, col]:
                    reason = 'sma_trail'
                elif ex['donch'] and i > p[0] and np.isfinite(DL[i, col]) and c < DL[i, col]:
                    reason = 'donchian_low'
                elif ex['months'] and (dts[k] - ctx['dt_all'][p[0]]).days >= ex['months'] * 30.44:
                    reason = 'time'
                elif ex['fund_fail'] and not MASKD[i, col] and (cell.entry != 'rebalance' or is_reb):
                    reason = 'fund_fail'
                if reason:
                    pend_sell.append((col, reason, 0))

        # ---- 4. ENTRY decisions on today's CLOSE --------------------------------------
        selling = {s[0] for s in pend_sell}
        if cell.entry == 'rebalance' and is_reb:
            cand = np.nonzero(QUAL[i])[0]
            if len(cand):
                cand = _order(cand, RV, i, cell.rank, rng)
                rank_of = {int(c): r for r, c in enumerate(cand)}
                keep_n = int(math.ceil(buf * slots))
                for col in list(pos.keys()):
                    if col in selling:
                        continue
                    r = rank_of.get(col)
                    ok = (r is not None and r < keep_n)
                    if not ok and cell.retain == 'loose':
                        # 'loose': the near-ATH STATE is not required to keep a name, but
                        # liquidity, the mask and the ranking still are.
                        ok = bool(ctx['ELIG'][i, col] and MASKD[i, col] and
                                  _loose_rank(col, cand, RV, i, cell.rank) < keep_n)
                    if not ok:
                        pend_sell.append((col, 'rebalance_out', 0))
                        selling.add(col)
                held_after = len([c for c in pos if c not in selling])
                free = slots - held_after - len(pend_buy)
                if free > 0 and not (gate_weak and not liq_all):
                    nav = cash + sum(p[2] * (C[i, c] if np.isfinite(C[i, c]) else p[4])
                                     for c, p in pos.items())
                    tgt = tgt_pct * nav
                    for col in cand:
                        if free <= 0:
                            break
                        col = int(col)
                        if col in pos and col not in selling:
                            continue
                        px = C[i, col]
                        if not np.isfinite(px):
                            continue
                        qty = int(tgt / px)
                        if qty < 1:
                            continue
                        pend_buy.append((col, qty))
                        free -= 1
        elif cell.entry in ('first_qualify', 'ath_breakout'):
            if not (gate_weak and not liq_all):
                free = slots - len([c for c in pos if c not in selling]) - len(pend_buy)
                if free > 0:
                    src = NEWQ if cell.reentry == 'transition' else QUAL
                    cand = np.nonzero(src[i])[0]
                    if len(cand):
                        cand = _order(cand, RV, i, cell.rank, rng)
                        nav = cash + sum(p[2] * (C[i, c] if np.isfinite(C[i, c]) else p[4])
                                         for c, p in pos.items())
                        tgt = tgt_pct * nav
                        for col in cand:
                            if free <= 0:
                                break
                            col = int(col)
                            if col in pos:
                                continue
                            px = C[i, col]
                            if not np.isfinite(px):
                                continue
                            qty = int(tgt / px)
                            if qty < 1:
                                continue
                            pend_buy.append((col, qty))
                            free -= 1

        # ---- 5. same-close execution (the second labelled convention) -----------------
        if not next_open and (pend_sell or pend_buy):
            for col, reason, _age in pend_sell:
                if col not in pos:
                    continue
                px = C[i, col]
                if not np.isfinite(px):
                    px = pos[col][4]
                ei, bpx, qty, _pk, _lp, tve, _lt = pos.pop(col)
                proceeds = qty * px * (1 - cost)
                cash += proceeds
                sell_notional += qty * px
                if use_tax:
                    basis = qty * bpx * (1 + cost)
                    held = (dts[k] - ctx['dt_all'][ei]).days
                    fy_pool += (LTCG if held > 365 else STCG) * (proceeds - basis)
                trades.append((col, ei, i, bpx, float(px), qty, reason, tve))
            pend_sell = []
            for col, qty in pend_buy:
                px = C[i, col]
                if not np.isfinite(px) or col in pos or len(pos) >= slots:
                    continue
                need = qty * px * (1 + cost)
                if need > cash:
                    qty = int(cash / (px * (1 + cost)))
                    need = qty * px * (1 + cost)
                if qty < 1:
                    missed_buy += 1
                    continue
                cash -= need
                pos[col] = [i, float(px), qty, float(px), float(px),
                            float(TV[i, col]) if np.isfinite(TV[i, col]) else np.nan, i]
            pend_buy = []

        # ---- 6. mark to market --------------------------------------------------------
        mtm = 0.0
        for c, p in pos.items():
            px = C[i, c]
            if np.isfinite(px):
                p[4] = float(px)
            mtm += p[2] * p[4]
        equity[k] = cash + mtm
        invested[k] = mtm / equity[k] if equity[k] > 0 else 0.0

    # ---- close the book at the last close ---------------------------------------------
    last = days[-1]
    for col, p in list(pos.items()):
        px = C[last, col] if np.isfinite(C[last, col]) else p[4]
        trades.append((col, p[0], last, p[1], float(px), p[2], 'open_marked', p[5]))
    if use_tax:
        total = fy_pool + fy_carry
        if total > 0:
            equity[-1] -= total                 # settle the accrued realized pool
    return dict(equity=equity, invested=invested, trades=trades,
                sell_notional=sell_notional, missed_buy=missed_buy)


def _order(cand, RV, i, rank, rng):
    """Rank the candidate columns. A random permutation is applied FIRST so that ties are
    broken randomly and the seed ensemble actually measures path dependence."""
    cand = rng.permutation(cand)
    if rank == 'random' or RV is None:
        return cand
    v = RV[i, cand]
    v = np.where(np.isfinite(v), v, -np.inf)
    return cand[np.argsort(-v, kind='stable')]


def _loose_rank(col, cand, RV, i, rank):
    if RV is None:
        return 10 ** 9
    v = RV[i, col]
    if not np.isfinite(v):
        return 10 ** 9
    return int(np.sum(np.nan_to_num(RV[i, cand], nan=-np.inf) > v))


# ----------------------------------------------------------------------------- metrics --
def _year_stats(nav):
    """r/154 convention: a year's drawdown is measured from the running peak of the FULL
    curve, never from the year's first bar (that error reported -2.4% where the truth was
    -16.5%)."""
    peak = nav.cummax()
    dd = nav / peak - 1.0
    out = {}
    for yr, seg in nav.groupby(nav.index.year):
        prev = nav[nav.index.year < yr]
        base = prev.iloc[-1] if len(prev) else seg.iloc[0]
        out[int(yr)] = [round(float(seg.iloc[-1] / base - 1.0) * 100, 2),
                        round(float(dd[dd.index.year == yr].min()) * 100, 2)]
    return out


def _path_stats(res, dts_used, capital, cost, cash_yield):
    e = pd.Series(res['equity'], index=dts_used)
    yrs = max((dts_used[-1] - dts_used[0]).days / 365.25, 1e-9)
    cagr = (e.iloc[-1] / capital) ** (1 / yrs) - 1
    dd = float((e / e.cummax() - 1).min())
    r = e.pct_change().dropna()
    rf_d = cash_yield / TRADING_DAYS
    sharpe = float((r.mean() - rf_d) / r.std() * math.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan

    tr = res['trades']
    rets = np.array([(t[4] * (1 - cost)) / (t[3] * (1 + cost)) - 1 for t in tr]) if tr else np.array([])
    wins = rets[rets > 0]
    loss = rets[rets <= 0]
    streak = best = 0
    for t in sorted(tr, key=lambda x: x[2]):
        rr = (t[4] * (1 - cost)) / (t[3] * (1 + cost)) - 1
        streak = streak + 1 if rr <= 0 else 0
        best = max(best, streak)
    notional = [t[3] * t[5] for t in tr]
    tves = [t[7] for t in tr if np.isfinite(t[7]) and t[7] > 0]
    cap = (float(np.median(notional)) / float(np.median(tves))) if notional and tves else np.nan

    return dict(
        cagr=cagr * 100, maxdd=dd * 100,
        calmar=(cagr * 100) / abs(dd * 100) if dd < 0 else np.nan,
        sharpe=sharpe, n_trades=len(tr), trades_per_yr=len(tr) / yrs,
        win_rate=float((rets > 0).mean() * 100) if len(rets) else np.nan,
        avg_win_pct=float(wins.mean() * 100) if len(wins) else np.nan,
        avg_loss_pct=float(loss.mean() * 100) if len(loss) else np.nan,
        expectancy_net_pct=float(rets.mean() * 100) if len(rets) else np.nan,
        max_losing_streak=best,
        turnover_x_nav_yr=res['sell_notional'] / float(e.mean()) / yrs,
        avg_pct_invested=float(np.mean(res['invested']) * 100),
        capacity_ratio=cap, final_x=float(e.iloc[-1] / capital),
        yearly=_year_stats(e), curve=e, years=yrs)


# ------------------------------------------------------------------------------ driver --
def _reb_days(dates, days, cadence, n_offsets):
    months = {'monthly': set(range(1, 13)), 'quarterly': {1, 4, 7, 10},
              'semiannual': {1, 7}, 'annual': {1}}[cadence]
    ym = [(int(dates[i][:4]), int(dates[i][5:7])) for i in days]
    firsts = []
    prev = None
    for p, key in enumerate(ym):
        if key != prev:
            if key[1] in months:
                firsts.append(p)
            prev = key
    out = {}
    for off in range(max(n_offsets, 1)):
        sel = set()
        for p in firsts:
            q = p + off
            if q < len(days) and ym[q][1] == ym[p][1]:      # stay inside the month
                sel.add(days[q])
            elif p < len(days):
                sel.add(days[min(p + off, len(days) - 1)])
        out[off] = sel
    return out


def run_cell(panel: Panel, der: Derived, cell: Cell, verbose=True):
    t0 = time.time()
    C, O = panel.f('close'), panel.f('open')
    ATH, TV = panel.f('athc'), panel.f('tv20')
    ex = parse_exits(cell.exits)

    ELIG = (TV >= cell.tv_floor * 1e7) & np.isfinite(C)
    ELIG[:, np.asarray(panel.is_fund)] = False
    if cell.state == 'near':
        ST = C >= cell.k * ATH
    elif cell.state == 'new_ath':
        ST = C > panel.prev('athc')
    else:
        raise SystemExit('unknown --state %r' % cell.state)
    ST &= np.isfinite(ATH)

    if cell.mask:
        MASKD, n_hit, n_cols = load_mask(cell.mask, panel, cell.mask_missing)
        mm = mask_meta(cell.mask)
        if verbose:
            print('mask %s: rows %s..%s (%d monthly), %d cols, %d matched the panel, '
                  'true %.3f%%; missing=%s'
                  % (Path(cell.mask).name, mm['first'], mm['last'], mm['n_rows'],
                     mm['n_cols'], n_hit, mm['true_frac'] * 100, cell.mask_missing),
                  flush=True)
        # A window that starts before the mask does is NOT a fundamentals test over that
        # stretch: every day before the first mask row takes the missing policy wholesale
        # (all-cash under 'fail', price-only under 'pass'). Found 2026-09-11 running the
        # DATA-LEG's arun_strict mask (starts 2015-01) over a 2010 window: the 'fail' arm
        # returned 6.1%, which was the idle-cash yield, not a result.
        pre = sum(1 for d in panel.dates if cell.start <= str(d) < mm['first'])
        tot = sum(1 for d in panel.dates if cell.start <= str(d) <= cell.end)
        if tot and pre / tot > 0.02:
            print('  !! WARNING: %.0f%% of the window (%s..%s) precedes the mask\'s first '
                  'row %s. Those days are decided by --mask-missing=%s alone, not by any '
                  'fundamental evidence. Align --start to the mask, or read this arm as a '
                  'coverage artefact.' % (100.0 * pre / tot, cell.start, mm['first'],
                                          mm['first'], cell.mask_missing), flush=True)
    else:
        MASKD = np.ones_like(ELIG, dtype=bool)

    QUAL = ELIG & ST & MASKD
    if cell.entry == 'ath_breakout':
        QUAL = QUAL & (C > panel.prev('athc'))

    RV = None
    if cell.rank == 'rs':
        RV = panel.f('score')
    elif cell.rank == 'dist_ath':
        RV = C / ATH
    elif cell.rank == 'tv_desc':
        RV = TV
    elif cell.rank in ('mcap_asc', 'mcap_desc'):
        if der.shares is None:
            raise SystemExit('mcap ranking needs %s' % MCAP_SNAP)
        m = (C * der.shares[None, :].astype(np.float32))
        RV = m if cell.rank == 'mcap_desc' else -m
    elif cell.rank != 'random':
        raise SystemExit('unknown --rank %r' % cell.rank)

    SMA = panel.f('sma%d' % ex['sma_trail']) if ex['sma_trail'] else None
    DL = panel.f('donch_low%d' % ex['donch']) if ex['donch'] else None
    weak = build_gate(panel, cell.index_gate)

    days = np.array([i for i, d in enumerate(panel.dates)
                     if cell.start <= str(d) <= cell.end], dtype=np.int64)
    if len(days) < 30:
        raise SystemExit('window %s..%s has only %d sessions' % (cell.start, cell.end, len(days)))
    if verbose:
        pre_mask = int((ELIG[days] & ST[days]).sum())
        post = int(QUAL[days].sum())
        print('  qualifying name-days in the window: %d liquid+state -> %d after the mask '
              '(%.1f%% survive), %.1f candidates per session'
              % (pre_mask, post, 100.0 * post / max(pre_mask, 1), post / len(days)),
              flush=True)
        if post / len(days) < cell.slots:
            print('  !! NOTE: fewer qualifying names per session (%.1f) than slots (%d): '
                  'this book cannot stay fully invested, and its return is partly the '
                  'idle-cash yield. Report avg_pct_invested alongside the CAGR.'
                  % (post / len(days), cell.slots), flush=True)
    dt_all = panel.dt
    dts = dt_all[days]

    ctx = dict(cell=cell, C=C, O=O, ATH=ATH, TV=TV, QUAL=QUAL, MASKD=MASKD, ELIG=ELIG,
               RV=RV, SMA=SMA, DL=DL, weak=weak, days=days, dts=dts, dt_all=dt_all, ex=ex)
    if cell.entry == 'rebalance':
        n_off = max(int(cell.offsets), 1)
        rd = _reb_days(panel.dates, days, cell.cadence, n_off)
        if int(cell.seeds) > 1 and int(cell.offsets) <= 1:
            # a seed ensemble ON a rebalance book: the calendar is held at offset 0 and only
            # the tie-break / random selection varies. Used for the random-selection null.
            ctx['reb_days'] = {p: rd[0] for p in range(int(cell.seeds))}
            path_ids, kind = list(range(int(cell.seeds))), 'seed'
        else:
            ctx['reb_days'] = rd
            path_ids, kind = list(range(n_off)), 'offset'
    else:
        NEWQ = np.zeros_like(QUAL)
        NEWQ[1:] = QUAL[1:] & ~QUAL[:-1]
        NEWQ[0] = QUAL[0]
        # Day one of the TRADING window is a book initialisation, not a transition: every
        # name that already qualifies is a candidate. Without this the book can only ever
        # buy names that turn eligible AFTER the start date, and a window beginning inside
        # a long bull run would start permanently empty.
        NEWQ[days[0]] |= QUAL[days[0]]
        ctx['NEWQ'] = NEWQ
        n_seed = max(int(cell.seeds), 1)
        path_ids, kind = list(range(n_seed)), 'seed'

    cost = cell.cost_bps / 10000.0
    arms = ['net_tax'] if cell.arms == 'tax' else ['gross', 'net', 'net_tax']
    per_path, curves, trade_dump = [], {}, []
    for pid in path_ids:
        got = {}
        for arm in arms:
            c_ = 0.0 if arm == 'gross' else cost
            tax_ = bool(cell.tax) and arm == 'net_tax'
            res = _simulate(ctx, pid, c_, tax_)
            st = _path_stats(res, dts, cell.capital, c_, cell.cash_yield)
            got[arm] = st
            if pid == path_ids[0] and arm == 'net_tax':
                trade_dump = [dict(
                    symbol=panel.syms[t[0]], entry_date=str(panel.dates[t[1]]),
                    exit_date=str(panel.dates[t[2]]), entry_px=round(t[3], 2),
                    exit_px=round(t[4], 2), qty=t[5], reason=t[6],
                    held_days=(panel.dt[t[2]] - panel.dt[t[1]]).days,
                    ret_net_pct=round(((t[4] * (1 - c_)) / (t[3] * (1 + c_)) - 1) * 100, 2))
                    for t in sorted(res['trades'], key=lambda x: x[1])]
        base = got['net_tax']
        row = dict(path=pid, kind=kind,
                   cagr_gross=got.get('gross', base)['cagr'],
                   cagr_net=got.get('net', base)['cagr'],
                   cagr_net_tax=base['cagr'])
        for kk in ('maxdd', 'calmar', 'sharpe', 'n_trades', 'trades_per_yr', 'win_rate',
                   'avg_win_pct', 'avg_loss_pct', 'expectancy_net_pct', 'max_losing_streak',
                   'turnover_x_nav_yr', 'avg_pct_invested', 'capacity_ratio', 'final_x'):
            row[kk] = base[kk]
        row['yearly'] = base['yearly']
        per_path.append(row)
        curves['%s%d' % (kind, pid)] = base['curve']
        if verbose:
            print('  %s %-2d  CAGR net-tax %6.2f%%  DD %7.2f%%  Calmar %5.2f  trades %4d  '
                  '(%.1fs)' % (kind, pid, base['cagr'], base['maxdd'], base['calmar'],
                               base['n_trades'], time.time() - t0), flush=True)

    med = lambda key: float(np.nanmedian([p[key] for p in per_path]))
    ct = [p['cagr_net_tax'] for p in per_path]
    worst_ix = int(np.argmin(ct))
    years = sorted({y for p in per_path for y in p['yearly']})
    yearly = {str(y): [round(float(np.nanmedian([p['yearly'][y][0] for p in per_path if y in p['yearly']])), 2),
                       round(float(np.nanmedian([p['yearly'][y][1] for p in per_path if y in p['yearly']])), 2)]
              for y in years}

    row = {c: '' for c in COLUMNS}
    for f in fields(Cell):
        if f.name in row:
            row[f.name] = getattr(cell, f.name)
    row.update(
        n_paths=len(per_path), path_kind=kind,
        cagr_gross=round(med('cagr_gross'), 2), cagr_net=round(med('cagr_net'), 2),
        cagr_net_tax=round(med('cagr_net_tax'), 2),
        cagr_net_tax_min=round(float(np.min(ct)), 2),
        cagr_net_tax_max=round(float(np.max(ct)), 2),
        cagr_net_tax_worstpath=round(float(ct[worst_ix]), 2),
        maxdd=round(med('maxdd'), 2),
        maxdd_worst=round(float(np.min([p['maxdd'] for p in per_path])), 2),
        calmar=round(med('calmar'), 2), sharpe=round(med('sharpe'), 2),
        trades_per_yr=round(med('trades_per_yr'), 1), win_rate=round(med('win_rate'), 1),
        avg_win_pct=round(med('avg_win_pct'), 2), avg_loss_pct=round(med('avg_loss_pct'), 2),
        expectancy_net_pct=round(med('expectancy_net_pct'), 3),
        max_losing_streak=int(med('max_losing_streak')) if np.isfinite(med('max_losing_streak')) else 0,
        turnover_x_nav_yr=round(med('turnover_x_nav_yr'), 2),
        avg_pct_invested=round(med('avg_pct_invested'), 1),
        capacity_ratio=round(med('capacity_ratio'), 4),
        n_trades=int(med('n_trades')), final_x=round(med('final_x'), 2),
        yearly=json.dumps(yearly))
    return dict(row=row, paths=per_path, curves=curves, trades=trade_dump,
                seconds=time.time() - t0)


def paired_diff(a, b, key='cagr_net_tax'):
    """A vs B on the SAME path. Unpaired medians lie at small n (r/158: a gate looked like a
    winner on 10-seed medians and lost on 20 of 30 paired paths)."""
    pa = {p['path']: p[key] for p in a['paths']}
    pb = {p['path']: p[key] for p in b['paths']}
    common = sorted(set(pa) & set(pb))
    d = [pa[p] - pb[p] for p in common]
    return dict(n=len(d), median_delta=float(np.median(d)) if d else float('nan'),
                a_wins=int(sum(1 for x in d if x > 0)), deltas=d)


# --------------------------------------------------------------------------------- CLI --
def append_row(out, row):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow(row)


def done_labels(out):
    p = Path(out)
    if not p.exists():
        return set()
    with open(p) as f:
        return {r['label'] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--panel', default=str(STUDY / 'results' / 'panel_2000.npz'))
    ap.add_argument('--out', default=str(STUDY / 'results' / 'cells.csv'))
    ap.add_argument('--grid', default=None, help='JSON list of cell dicts; one process, '
                                                 'panel + derived frames built once')
    ap.add_argument('--dump-equity', action='store_true')
    ap.add_argument('--dump-trades', action='store_true')
    ap.add_argument('--shift-test', type=int, default=0,
                    help='look-ahead probe: move ALL price data N days later')
    ap.add_argument('--quiet', action='store_true')
    # `from __future__ import annotations` makes f.type a STRING, so map it explicitly
    # rather than trusting it to be a callable.
    TYPES = {'str': str, 'int': int, 'float': float, 'bool': bool}
    for f in fields(Cell):
        ty = TYPES[f.type if isinstance(f.type, str) else f.type.__name__]
        if ty is bool:
            ap.add_argument('--' + f.name.replace('_', '-'), dest=f.name,
                            action='store_true', default=None)
            ap.add_argument('--no-' + f.name.replace('_', '-'), dest=f.name,
                            action='store_false', default=None)
        else:
            ap.add_argument('--' + f.name.replace('_', '-'), dest=f.name, type=ty,
                            default=None)
    a = ap.parse_args()

    panel = Panel.load(a.panel, shift=a.shift_test)
    der = Derived(panel)
    if not a.quiet:
        print('panel %s  %d dates %s..%s  %d symbols (%d funds/indices)  mcap proxy %d'
              % (Path(a.panel).name, len(panel.dates), panel.dates[0], panel.dates[-1],
                 len(panel.syms), int(panel.is_fund.sum()), der.mcap_known), flush=True)
        if a.shift_test:
            print('SHIFT TEST ACTIVE: all price data moved %d day(s) later' % a.shift_test)

    cli = {f.name: getattr(a, f.name) for f in fields(Cell)
           if getattr(a, f.name, None) is not None}
    if a.grid:
        cells = [Cell(**{**cli, **c}) for c in json.load(open(a.grid))]
    else:
        cells = [Cell(**cli)]

    have = done_labels(a.out)
    for cell in cells:
        if cell.label in have:
            print('skip %s (already in %s)' % (cell.label, a.out), flush=True)
            continue
        print('\n=== %s ===' % cell.label, flush=True)
        r = run_cell(panel, der, cell, verbose=not a.quiet)
        append_row(a.out, r['row'])
        if a.dump_equity:
            pd.DataFrame(r['curves']).to_csv(STUDY / 'results' / ('%s_equity.csv' % cell.label))
        if a.dump_trades and r.get('trades'):
            pd.DataFrame(r['trades']).to_csv(
                STUDY / 'results' / ('%s_trades.csv' % cell.label), index=False)
        print('%s: CAGR gross %.2f / net %.2f / after-tax %.2f  DD %.2f (worst %.2f)  '
              'Calmar %.2f  %d paths in %.0fs'
              % (cell.label, r['row']['cagr_gross'], r['row']['cagr_net'],
                 r['row']['cagr_net_tax'], r['row']['maxdd'], r['row']['maxdd_worst'],
                 r['row']['calmar'], r['row']['n_paths'], r['seconds']), flush=True)


if __name__ == '__main__':
    main()
