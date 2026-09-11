# -*- coding: utf-8 -*-
"""The Open Alpha universe lets gold, silver and index ETFs through as stock candidates.

The filter is (BEES|ETF|LIQUID|GILT|SENSEX|NIF[A-Z]*50). It was written against the ETF
names that existed when r/142 was built, and the 2023-2025 wave of gold and silver funds is
named nothing like any of them: EGOLD, GOLD1, GOLDADD, GOLDAXIS, GOLDBETA, GOLDCASE,
GROWWGOLD, HDFCGOLD, LICMFGOLD, TATAGOLD, ESILVER, SILVER1, SILVERADD, SILVERAG,
SILVERBETA, HDFCSILVER, SBISILVER, TATSILV - plus MON100, MAFANG, ICICIB22, METAL,
MODEFENCE.

An Indian equity momentum book buying gold funds is not a small tidiness problem. Gold ran
hard through the 2024-2026 window and those names are the reason one overlay arm printed
32% - it was not measuring a fundamental screen, it was measuring a gold allocation.

THE HARD PART IS NOT OVER-MATCHING. SKYGOLD, GOLDIAM, SILVERTUC, GOLDTECH and similar are
real jewellery and engineering companies with real earnings, and a lazy /GOLD|SILVER/ would
delete them from the universe - a silent, permanent loss of legitimate candidates that
nobody would notice. So the pattern is anchored and enumerated rather than substring-based,
and the check below prints BOTH what it newly excludes and what it must keep.

This prints and proposes. It writes nothing.
"""
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/158_oa_arming_width/scripts'))
import oa_entry_mechanics as em          # noqa: E402

NEW = re.compile(
    r'(BEES|ETF|LIQUID|GILT|SENSEX|NIF[A-Z]*50'
    r'|^GOLD$|^GOLD[0-9]|^EGOLD$|^GOLDADD$|^GOLDAXIS$|^GOLDBETA$|^GOLDCASE$'
    r'|^GROWWGOLD$|^HDFCGOLD$|^LICMFGOLD$|^TATAGOLD$|^AXISGOLD$|^QGOLDHALF$'
    r'|^SILVER$|^SILVER[0-9]|^ESILVER$|^SILVERADD$|^SILVERAG$|^SILVERBETA$'
    r'|^HDFCSILVER$|^SBISILVER$|^TATSILV$|^SILVERIETF$'
    r'|^MON100$|^MAFANG$|^ICICIB22$|^METAL$|^MODEFENCE$|^HNGSNGBEES$)')

con = sqlite3.connect('file:%s?mode=ro' % (ROOT / 'backtest_data/market_data.db'), uri=True)
syms = sorted(r[0] for r in con.execute(
    "select distinct symbol from market_data_unified where timeframe='day'"))
con.close()

old_hit = {s for s in syms if em.ETF_RE.search(s)}
new_hit = {s for s in syms if NEW.search(s)}
added = sorted(new_hit - old_hit)
lost = sorted(old_hit - new_hit)

print('universe symbols           : %d' % len(syms))
print('excluded by the OLD filter : %d' % len(old_hit))
print('excluded by the NEW filter : %d' % len(new_hit))
print()
print('NEWLY EXCLUDED (%d) - these were being traded as if they were companies:' % len(added))
for s in added:
    print('   ', s)
if lost:
    print()
    print('WARNING - the new filter DROPS these previous exclusions:', lost)

print()
KEEP = ['SKYGOLD', 'GOLDIAM', 'SILVERTUC', 'GOLDTECH', 'GOLDSTAR', 'GOLDENTOBC',
        'VIPULLTD', 'RAJESHEXPO', 'TITAN', 'KALYANKJIL', 'PCJEWELLER', 'THANGAMAYL']
present = [s for s in KEEP if s in syms]
wrong = [s for s in present if NEW.search(s)]
print('real companies checked : %s' % ', '.join(present))
print('wrongly excluded       : %s' % (', '.join(wrong) if wrong else 'none'))

miss = [s for s in syms if re.search(r'GOLD|SILVER|SLVR', s) and s not in new_hit]
print()
print('still in the universe with a gold/silver-ish name (%d) - eyeball for strays:' % len(miss))
print('   ', ', '.join(miss))
