"""Publish the adopted dividend policy's 10-year simulation to the app.

Source: research/142_bananapatterns_replication/results/dividend_sim_v2_b25_g75.csv
(policy variant E — 25% of new profit above the flow-adjusted HWM, payout capped
at last dividend +7.5%/qtr, surplus to a 6% p.a. equalization reserve).

Writes static/app/dividend_sim.json in the year-row shape the Sleeves page renders:
one row per year with the four quarterly payouts, the year's new profit above the
HWM, and the reserve balance at year end.
"""
import csv
import json
from collections import OrderedDict
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
SRC = ROOT / 'research' / '142_bananapatterns_replication' / 'results' / 'dividend_sim_v2_b25_g75.csv'
OUT = ROOT / 'static' / 'app' / 'dividend_sim.json'


def main():
    years = OrderedDict()
    tot_paid = tot_profit = 0.0
    last_nav = last_reserve = 0.0
    for r in csv.DictReader(open(SRC)):
        y, q = r['quarter'].split('-Q')
        y, q = int(y), int(q)
        row = years.setdefault(y, dict(year=y, q=[None, None, None, None],
                                       src=['', '', '', ''], profit=0.0, reserve=0.0))
        paid = float(r['paid'])
        row['q'][q - 1] = round(paid)
        # a quarter is reserve-funded when the reserve supplied most of the payout
        row['src'][q - 1] = 'reserve' if float(r['from_reserve']) > paid / 2 and paid > 0 else 'profit'
        row['profit'] += float(r['new_profit'])
        row['reserve'] = round(float(r['reserve_bal']))
        last_nav = float(r['nav'])
        last_reserve = float(r['reserve_bal'])
        tot_paid += paid
        tot_profit += float(r['new_profit'])
    rows = []
    for y, row in years.items():
        row['profit'] = round(row['profit'])
        row['total'] = round(sum(x or 0 for x in row['q']))
        rows.append(row)
    payload = dict(
        policy='25% of new profit above the flow-adjusted high-water mark; payout capped at '
               'last dividend +7.5%/quarter; surplus to a liquid equalization reserve (~6% p.a.) '
               'that bridges profitless quarters; capital never invaded.',
        seed_capital=1000000, start='2016-01', end=rows[-1]['year'],
        rows=rows, total_paid=round(tot_paid), total_profit=round(tot_profit),
        end_nav=round(last_nav), end_reserve=round(last_reserve),
        note='Simulated on the after-tax Open Alpha trail-20 equity curve (median seed) rescaled '
             'to Rs 10,00,000 at Jan-2016 — a policy rehearsal, not a record of payments made. '
             'The first real declaration is 30-Sep-2026.')
    json.dump(payload, open(OUT, 'w'), indent=1)
    print(f'wrote {OUT}: {len(rows)} years, total paid Rs {tot_paid:,.0f}, '
          f'end NAV Rs {last_nav:,.0f}, reserve Rs {last_reserve:,.0f}')


if __name__ == '__main__':
    main()
