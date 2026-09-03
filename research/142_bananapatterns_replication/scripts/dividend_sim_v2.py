"""Smoothed (progressive) quarterly dividend simulation — v2, per Arun 03-Sep-2026.

Changes vs v1 (which paid a volatile 40% of new profit):
  - baseline entitlement = 30% of new profit above HWM (not 40)
  - HARD CAP on each payout: last paid dividend x (1 + g) per quarter
    (the dividend line can only step up incrementally, never spike)
  - surplus entitlement above the cap goes to the LIQUID-ETF reserve
    (reserve earns ~6% p.a., credited quarterly)
  - dry / weak quarters: reserve tops the payout up toward the cap line,
    so income keeps flowing through droughts as long as the reserve lasts
  - if the reserve cannot fund the line, the payout falls to whatever is
    available, and the cap re-bases from that lower payout (a "dividend
    cut", like a real fund) — capital is NEVER invaded
  - HWM still ratchets: profit is only ever counted once

Run for several cap-growth rates g and compare to v1 on smoothness.
"""
import pandas as pd
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
eq = pd.read_csv(STUDY / 'results' / 'replica_tax_trail20_equity.csv',
                 index_col=0, parse_dates=True)
term = eq.iloc[-1]
seed = (term - term.median()).abs().idxmin()
nav = eq[seed].loc['2016-01-01':]
nav = nav / nav.iloc[0] * 1_000_000

q_end = nav.resample('QE').last().dropna()
rets = q_end.pct_change().fillna(0.0)

RES_RATE_Q = 0.06 / 4          # liquid-ETF yield on the reserve, per quarter


def run(baseline=0.30, g=0.10, label=''):
    book, hwm, reserve = 1_000_000.0, 1_000_000.0, 0.0
    cap = None                 # smoothed dividend line; set by first payout
    rows = []
    for dt, r in list(rets.items())[1:]:
        book *= (1 + r)
        reserve *= (1 + RES_RATE_Q)
        new_profit = max(0.0, book - hwm)
        entitlement = baseline * new_profit
        if cap is None:
            target = entitlement            # first payout seeds the line
        else:
            target = cap * (1 + g)
        # profit funds the payout first; reserve tops up toward the line
        from_profit = min(entitlement, target)
        from_reserve = min(reserve, max(0.0, target - from_profit))
        paid = from_profit + from_reserve
        surplus = entitlement - from_profit  # boom overflow -> reserve
        reserve += surplus - from_reserve
        book -= entitlement                  # entitlement leaves the book either way
        if new_profit > 0:
            hwm = book
        if paid > 0:
            cap = paid                       # line re-bases on what was actually paid
        rows.append(dict(quarter=f'{dt.year}-Q{(dt.month - 1) // 3 + 1}',
                         new_profit=round(new_profit),
                         entitlement=round(entitlement),
                         paid=round(paid),
                         from_reserve=round(from_reserve),
                         to_reserve=round(max(0.0, surplus - from_reserve)),
                         reserve_bal=round(reserve),
                         nav=round(book), hwm=round(hwm)))
    df = pd.DataFrame(rows)
    pays = df.paid[df.paid > 0]
    print(f'--- {label} (baseline {baseline:.0%}, cap growth {g:.0%}/qtr) ---')
    print(f'total paid Rs {df.paid.sum():,.0f} | ending NAV Rs {book:,.0f} | '
          f'reserve Rs {reserve:,.0f}')
    print(f'zero quarters: {(df.paid == 0).sum()}/{len(df)} | '
          f'paid-quarter range Rs {pays.min():,.0f} .. Rs {pays.max():,.0f} | '
          f'max/median ratio {pays.max() / pays.median():.1f}x')
    return df


dfA = run(0.30, 0.10, 'B: smoothed g=10%')
dfB = run(0.30, 0.05, 'C: smoothed g=5%')
dfC = run(0.30, 0.075, 'D: smoothed g=7.5%')

for tag, df in [('g10', dfA), ('g5', dfB), ('g75', dfC)]:
    df.to_csv(STUDY / 'results' / f'dividend_sim_v2_{tag}.csv', index=False)

print('\nQuarter-by-quarter, g=7.5%:')
print(dfC.to_string(index=False))
