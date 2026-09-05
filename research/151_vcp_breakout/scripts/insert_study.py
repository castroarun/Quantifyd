"""Insert the research/151 BacktestStudy entry into frontend/src/data/backtests.ts."""
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
p = ROOT / 'frontend' / 'src' / 'data' / 'backtests.ts'
entry = (ROOT / 'research' / '151_vcp_breakout' / 'scripts' / 'study_entry.ts').read_text(encoding='utf-8')
s = p.read_text(encoding='utf-8')
anchor = 'export const BACKTEST_STUDIES: BacktestStudy[] = [' + chr(10)
assert anchor in s, 'anchor missing'
if 'vcp-breakout-research151' in s:
    print('already inserted; nothing to do')
else:
    s = s.replace(anchor, anchor + entry, 1)
    p.write_text(s, encoding='utf-8')
    print('inserted; file now', len(s), 'chars')
