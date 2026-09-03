"""Quarterly dividend simulation (HWM + equalization reserve) on the after-tax
trail-20 book (median seed), rescaled to Rs 10,00,000 on 2016-01-01.

Policy (Arun, 03-Sep-2026):
  - Quarter-end declaration (Mar/Jun/Sep/Dec last trading day)
  - new_profit = max(0, NAV - HWM)
  - gross distribution = 40% of new_profit  (leaves the book)
      -> 90% paid to the investor, 10% into the equalization reserve
  - dry quarter (no new profit): pay from reserve, capped at 50% of the
    trailing-4-quarter average paid dividend
  - HWM ratchets to the post-distribution NAV after any profitable quarter
  - capital is NEVER invaded: payouts come only from profit or the reserve
"""
import pandas as pd
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
eq = pd.read_csv(STUDY / 'results' / 'replica_tax_trail20_equity.csv',
                 index_col=0, parse_dates=True)
term = eq.iloc[-1]
seed = (term - term.median()).abs().idxmin()
nav = eq[seed]
nav = nav.loc['2016-01-01':]
nav = nav / nav.iloc[0] * 1_000_000        # Rs 10L at Jan-2016

q_end = nav.resample('QE').last().dropna()
rets = q_end.pct_change().fillna(0.0)

book = 1_000_000.0
hwm = 1_000_000.0
reserve = 0.0
paid_hist = []
rows = []
for dt, r in list(rets.items())[1:]:
    book *= (1 + r)
    new_profit = max(0.0, book - hwm)
    if new_profit > 0:
        gross = 0.40 * new_profit
        to_res = 0.10 * gross
        paid = gross - to_res
        reserve += to_res
        book -= gross
        hwm = book
        src = 'profit'
    else:
        avg4 = sum(paid_hist[-4:]) / max(1, len(paid_hist[-4:]))
        paid = min(reserve, 0.5 * avg4)
        reserve -= paid
        to_res = 0.0
        src = 'reserve' if paid > 0 else 'skipped'
    paid_hist.append(paid)
    rows.append(dict(quarter=f'{dt.year}-Q{(dt.month - 1) // 3 + 1}',
                     nav_pre=round(book + (0.40 * new_profit if new_profit > 0 else 0), 0),
                     new_profit=round(new_profit, 0),
                     dividend_paid=round(paid, 0), source=src,
                     to_reserve=round(to_res, 0), reserve_bal=round(reserve, 0),
                     hwm=round(hwm, 0)))

df = pd.DataFrame(rows)
df.to_csv(STUDY / 'results' / 'dividend_sim.csv', index=False)
print(df.to_string(index=False))
tot = df.dividend_paid.sum()
print(f'\nTOTAL dividends paid over {len(df)} quarters: Rs {tot:,.0f}')
print(f'Ending book NAV (after all distributions): Rs {book:,.0f}')
print(f'Reserve balance at end: Rs {reserve:,.0f}')
counter = float(nav.iloc[-1])
print(f'Counterfactual NAV with NO dividends: Rs {counter:,.0f}')
print(f'(dividends + ending book + reserve = Rs {tot + book + reserve:,.0f} — the gap vs the '
      f'counterfactual is the compounding the payouts gave up)')
