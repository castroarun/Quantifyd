"""Generate the client tearsheet for the research/73 Nifty200 weekly-ST winner."""
import os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from st_weekly_engine import get_conn, load_daily, to_weekly, ROOT
sys.path.insert(0, os.path.join(ROOT, 'research', '_utilities'))
from tearsheet import generate_tearsheet

RES = os.path.join(HERE, '..', 'results')

# strategy NAV
eq = pd.read_csv(os.path.join(RES, 'g4_Nifty200_core_nogate_equity.csv'), index_col=0, parse_dates=True).iloc[:, 0]
eq.name = 'strat'
# benchmark NAV — NIFTYBEES weekly close aligned to strat dates
conn = get_conn()
nb = to_weekly(load_daily(conn, 'NIFTYBEES'))['close']
conn.close()
nb = nb.reindex(eq.index).ffill().bfill()

meta = dict(
    bench='NIFTYBEES (Nifty 50 TR proxy)',
    tag='BACKTEST · net of 0.30% RT cost + STCG/LTCG',
    disclosures=(
        'Weekly SuperTrend(10,3) long-only trend book on the CURRENT Nifty 200 (survivorship-biased — '
        'today’s members applied to the past; delisted names absent). Blind ST-flip exit, no regime '
        'gate, no profit-booking (both tested and found to HURT). Max 16 concurrent names @6.25% target, '
        'idle cash @6.5%. Signal at weekly close → fill next-week open (no look-ahead). Net of 0.15%/side '
        'cost + STCG 15%/LTCG 10%. 2010–2026. Returns are tail-driven (top-10%% of trades ≈ 78%% of '
        'return) and regime-dependent. Not competitive with the regime-gated midcap momentum book '
        '(Calmar ~1.7); merit is simplicity + beating passive at index-like DD. Past performance is not '
        'indicative of future results.'
    ),
)
import shutil
generate_tearsheet(eq, nb, 'Weekly SuperTrend (10,3) — Nifty 200 Trend Book', meta=meta, out_dir=RES)
slug = 'weekly-supertrend-nifty200'
os.replace(os.path.join(RES, 'tearsheet.png'), os.path.join(RES, slug + '.png'))
os.replace(os.path.join(RES, 'tearsheet.html'), os.path.join(RES, slug + '.html'))
# copy PNG into frontend/public for /app publishing
pub = os.path.join(ROOT, 'frontend', 'public', slug + '.png')
shutil.copy(os.path.join(RES, slug + '.png'), pub)
print('tearsheet ->', slug + '.png', '| published copy ->', pub)
