"""
G2 sweep (research/91): after G1 killed the baseline, test the two levers most
likely to matter for the user's thesis -- (a) HOLD the trend (overnight + SMA-cross
exit vs the tight intraday ATR-extend), (b) TIMEFRAME (5/15/30-min + daily) -- plus
a stricter 'rising 20 SMA' (slope over 10 bars). Pooled over 12 deep names, both
directions, gross AND net. Prepares each (name,tf) once; backtests are then cheap.
"""
import os, sys, csv, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.WARNING)
from sma_pullback_engine import load_bars, resample_intraday, add_indicators, backtest, summarize

OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'g2_sweep.csv'))
FIELDS = ['cell', 'tf', 'exit', 'hold', 'slope_lb', 'direction', 'n', 'gross_exp_pct',
          'net_exp_pct', 'net_exp_R', 'win_pct', 'pf', 't_stat', 'avg_hold', 'total_net_pct']
NAMES = ['NIFTY50', 'BANKNIFTY', 'HDFCBANK', 'ICICIBANK', 'RELIANCE', 'INFY',
         'TCS', 'SBIN', 'ITC', 'HINDUNILVR', 'KOTAKBANK', 'BHARTIARTL']
TFS = [('5minute', None, '5m'), ('5minute', '15min', '15m'),
       ('5minute', '30min', '30m'), ('day', None, '1d')]
EXITS = [('ext25', dict(exit_mode='extend', atr_mult=2.5)),
         ('ext35', dict(exit_mode='extend', atr_mult=3.5)),
         ('cross', dict(exit_mode='sma_cross'))]
COST = 5.0

# ---- prepare (name, tf) dataframes once (load + resample + indicators)
print('preparing dataframes...', flush=True)
PREP = {}
for tf, rs, tag in TFS:
    for nm in NAMES:
        df = load_bars(nm, tf)
        if df.empty:
            continue
        if rs:
            df = resample_intraday(df, rs)
        df = add_indicators(df)
        PREP[(nm, tag)] = df
    print(f'  {tag}: {sum(1 for k in PREP if k[1]==tag)} names ready', flush=True)

done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        done = {(r['cell'], r['direction']) for r in csv.DictReader(f)}
    print(f'resume: {len(done)} cells done', flush=True)
else:
    with open(OUT, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()

# ---- build cells
cells = []
for tf, rs, tag in TFS:
    holds = ['overnight'] if tag == '1d' else ['intraday', 'overnight']
    for exlabel, exkw in EXITS:
        for hold in holds:
            for slope_lb in (3, 10):
                cells.append((tag, exlabel, exkw, hold, slope_lb))

print(f'\n{len(cells)*2} cells (x both dirs). cost={COST}bps\n', flush=True)
best = []
for ci, (tag, exlabel, exkw, hold, slope_lb) in enumerate(cells, 1):
    for d in ('long', 'short'):
        cellid = f'{tag}_{exlabel}_{hold}_sl{slope_lb}'
        if (cellid, d) in done:
            continue
        t0 = time.time()
        pooled = []
        for nm in NAMES:
            df = PREP.get((nm, tag))
            if df is None:
                continue
            try:
                pooled.extend(backtest(df, direction=d, slope_lb=slope_lb, near_pct=0.005,
                                       entry_win=3, hold=hold, cost_bps=COST, **exkw))
            except Exception as e:
                print(f'  {nm} {cellid} {d} ERR {e}', flush=True)
        s = summarize(pooled)
        row = {'cell': cellid, 'tf': tag, 'exit': exlabel, 'hold': hold,
               'slope_lb': slope_lb, 'direction': d,
               **{f: s.get(f) for f in FIELDS if f not in
                  ('cell', 'tf', 'exit', 'hold', 'slope_lb', 'direction')}}
        with open(OUT, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        best.append((s.get('net_exp_R') or -9, s.get('gross_exp_pct') or -9, cellid, d, s))
        print(f'[{ci}/{len(cells)}] {cellid:26s} {d:5s} {time.time()-t0:4.0f}s | '
              f'n={s.get("n",0):5d} gross={s.get("gross_exp_pct")}% net={s.get("net_exp_pct")}% '
              f'R={s.get("net_exp_R")} t={s.get("t_stat")}', flush=True)

print('\n=== TOP 8 by net expectancy (R) ===', flush=True)
for R, g, cid, d, s in sorted(best, reverse=True)[:8]:
    print(f'  {cid:26s} {d:5s} gross={g}% net={s.get("net_exp_pct")}% R={R} t={s.get("t_stat")} n={s.get("n")}', flush=True)
print('\nGATE: any cell with gross>0 AND net>0 AND t>=3 advances to G3. Else NO EDGE.', flush=True)
