# -*- coding: utf-8 -*-
"""research/170 Part A -- generate this study's engine from research/160's FROZEN copy.

research/160's qg_engine.py is never edited. This script copies it and applies a small,
auditable set of EXACT-STRING patches, refusing to write anything if any patch fails to
match or matches more than once. Same discipline as research/162's patch_engine.py.

What the patches add, and why each one is needed for Arun's question:

  1. COLUMNS       -- the new output columns.
  2. sale reason   -- a monthly rebalance sale is split into 'reb_rank' (the name still
                      qualifies, its RS rank simply fell outside the leeway band -- the ONLY
                      kind of sale a wider rank leeway can ever prevent) and 'reb_state'
                      (it no longer qualifies at all: below the near-ATH band, below the
                      liquidity floor, or off the screen -- no leeway can save it).
  3. tax in rupees -- the FY-netting block already computes the tax; it just never recorded
                      the total. A leeway is supposed to pay for itself in tax, so the tax
                      has to be a number, not an inference.
  4. holding stats -- average holding period and the share of closed trades held beyond 365
                      days, i.e. the share taxed at 12.5% instead of 20%.
  5. plumbing      -- _path_stats needs the full date index to compute (3) and (4).

Run:  venv/bin/python3 research/170_.../scripts/patch_engine170.py
Idempotent: re-running regenerates the file from source.
"""
import sys
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'research' / '160_quality_growth_near_ath' / 'scripts' / 'qg_engine.py'
DST = Path(__file__).resolve().parent / 'qg_engine170.py'

PATCHES = []


def P(name, old, new):
    PATCHES.append((name, old, new))


# ---------------------------------------------------------------- 1. output columns ----
P('columns', """    'n_trades', 'final_x', 'yearly',
]""", """    'n_trades', 'final_x', 'yearly',
    # --- research/170 additions -------------------------------------------------------
    'retain', 'keep_n', 'tax_paid', 'tax_pct_nav_yr', 'avg_hold_days', 'pct_trades_ltcg',
    'sells_rank', 'sells_state', 'sells_other', 'pct_sells_rank',
]""")

# ------------------------------------------------------- 2. split the sale reason ------
P('sale_reason', """                    if not ok:
                        pend_sell.append((col, 'rebalance_out', 0))
                        selling.add(col)""",
  """                    if not ok:
                        # research/170: a monthly rebalance sells for two different reasons
                        # and only ONE of them is a rank. 'reb_rank' = the name is still in
                        # today's qualifying set (liquid, inside the near-ATH band, passing
                        # the screen) but its RS rank fell outside ceil(buffer*N) -- this is
                        # the only sale a wider rank leeway can ever prevent. 'reb_state' =
                        # the name is not in today's qualifying set at all, so no leeway of
                        # any width would have kept it.
                        why = 'reb_rank' if (r is not None) else 'reb_state'
                        pend_sell.append((col, why, 0))
                        selling.add(col)""")

# ------------------------------------------------------------- 3. tax paid, rupees -----
P('tax_init', """    sell_notional = 0.0
    fy_pool, fy_carry = 0.0, 0.0""",
  """    sell_notional = 0.0
    tax_paid = 0.0                              # research/170: rupees actually paid
    fy_pool, fy_carry = 0.0, 0.0""")

P('tax_settle', """            if total > 0:
                cash -= total
                fy_carry = 0.0
            else:
                fy_carry = total
            fy_pool = 0.0
            cur_fy = fy""",
  """            if total > 0:
                cash -= total
                tax_paid += total
                fy_carry = 0.0
            else:
                fy_carry = total
            fy_pool = 0.0
            cur_fy = fy""")

P('tax_final', """    if use_tax:
        total = fy_pool + fy_carry
        if total > 0:
            equity[-1] -= total                 # settle the accrued realized pool
    return dict(equity=equity, invested=invested, trades=trades,
                sell_notional=sell_notional, missed_buy=missed_buy)""",
  """    if use_tax:
        total = fy_pool + fy_carry
        if total > 0:
            equity[-1] -= total                 # settle the accrued realized pool
            tax_paid += total
    return dict(equity=equity, invested=invested, trades=trades,
                sell_notional=sell_notional, missed_buy=missed_buy, tax_paid=tax_paid)""")

# -------------------------------------------------- 4/5. holding stats + plumbing ------
P('stats_sig', """def _path_stats(res, dts_used, capital, cost, cash_yield):""",
  """def _path_stats(res, dts_used, capital, cost, cash_yield, dt_all=None):""")

P('stats_call', """            st = _path_stats(res, dts, cell.capital, c_, cell.cash_yield)""",
  """            st = _path_stats(res, dts, cell.capital, c_, cell.cash_yield, ctx['dt_all'])""")

P('stats_body', """    return dict(
        cagr=cagr * 100, maxdd=dd * 100,""",
  """    # --- research/170: what a wider rank leeway actually buys ---------------------------
    # Fewer sales -> longer holds -> more of the book's gains taxed at 12.5% instead of 20%,
    # and fewer rupees of tax. None of that is visible in CAGR alone, which is why Arun's
    # question needs these columns and not just a ranking.
    closed = [t for t in tr if t[6] != 'open_marked']
    if dt_all is not None and closed:
        hold = np.array([(dt_all[t[2]] - dt_all[t[1]]).days for t in closed], float)
        n_lt = int((hold > 365).sum())
    else:
        hold, n_lt = np.array([]), 0
    n_rank = sum(1 for t in closed if t[6] == 'reb_rank')
    n_state = sum(1 for t in closed if t[6] == 'reb_state')
    n_cl = max(len(closed), 1)
    tp = float(res.get('tax_paid', 0.0))
    return dict(
        tax_paid=tp, tax_pct_nav_yr=tp / float(e.mean()) / yrs * 100.0,
        avg_hold_days=float(np.mean(hold)) if len(hold) else float('nan'),
        pct_trades_ltcg=100.0 * n_lt / n_cl,
        sells_rank=n_rank, sells_state=n_state, sells_other=len(closed) - n_rank - n_state,
        pct_sells_rank=100.0 * n_rank / n_cl,
        cagr=cagr * 100, maxdd=dd * 100,""")

P('row_keys', """        for kk in ('maxdd', 'calmar', 'sharpe', 'n_trades', 'trades_per_yr', 'win_rate',
                   'avg_win_pct', 'avg_loss_pct', 'expectancy_net_pct', 'max_losing_streak',
                   'turnover_x_nav_yr', 'avg_pct_invested', 'capacity_ratio', 'final_x'):""",
  """        for kk in ('maxdd', 'calmar', 'sharpe', 'n_trades', 'trades_per_yr', 'win_rate',
                   'avg_win_pct', 'avg_loss_pct', 'expectancy_net_pct', 'max_losing_streak',
                   'turnover_x_nav_yr', 'avg_pct_invested', 'capacity_ratio', 'final_x',
                   'tax_paid', 'tax_pct_nav_yr', 'avg_hold_days', 'pct_trades_ltcg',
                   'sells_rank', 'sells_state', 'sells_other', 'pct_sells_rank'):""")

P('row_update', """        n_trades=int(med('n_trades')), final_x=round(med('final_x'), 2),
        yearly=json.dumps(yearly))""",
  """        n_trades=int(med('n_trades')), final_x=round(med('final_x'), 2),
        retain=cell.retain,
        keep_n=int(math.ceil(float(cell.buffer) * int(cell.slots))),
        tax_paid=round(med('tax_paid'), 0),
        tax_pct_nav_yr=round(med('tax_pct_nav_yr'), 3),
        avg_hold_days=round(med('avg_hold_days'), 1),
        pct_trades_ltcg=round(med('pct_trades_ltcg'), 1),
        sells_rank=int(med('sells_rank')), sells_state=int(med('sells_state')),
        sells_other=int(med('sells_other')),
        pct_sells_rank=round(med('pct_sells_rank'), 1),
        yearly=json.dumps(yearly))""")

P('docstring', '''"""research/160 — quality-growth near all-time-high: the positional backtest ENGINE.''',
  '''"""research/170 Part A -- GENERATED from research/160's qg_engine.py by patch_engine170.py.

DO NOT EDIT THIS FILE. Edit the patch script and regenerate. The original docstring follows.

research/160 - quality-growth near all-time-high: the positional backtest ENGINE.''')


def main():
    src = SRC.read_text(encoding='utf-8')
    out = src
    for name, old, new in PATCHES:
        n = out.count(old)
        if n != 1:
            print('PATCH FAILED: %r matched %d times (expected 1). Nothing written.'
                  % (name, n))
            sys.exit(1)
        out = out.replace(old, new)
    DST.write_text(out, encoding='utf-8')
    print('wrote %s (%d bytes) from %s (%d bytes); %d patches applied'
          % (DST, len(out), SRC, len(src), len(PATCHES)))


if __name__ == '__main__':
    main()
