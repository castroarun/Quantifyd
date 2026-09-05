# -*- coding: utf-8 -*-
"""ops_center.py REVIEWS strings are written straight into JSON - they are NOT
%-formatted (confirmed at the render loop). So a doubled %% renders literally as '%%'.
Fix the doubled percent signs in the research/153 entry and in the sentence research/153
added to the four-sleeve review. Idempotent; touches nothing else."""
from pathlib import Path

P = Path("/home/arun/quantifyd/research/111_sensex_manual_mgmt/scripts/ops_center.py")
s = P.read_text(encoding="utf-8")

start = s.index('    ("research/153 IPO Base - adoption call')
end = s.index('Published at /app/backtest/ipo-base-breakout-research153."),', start)
block = s[start:end]
s = s[:start] + block.replace("%%", "%") + s[end:]

HOOK = "THIRD CANDIDATE ADDED 2026-09-05: research/153 IPO-Base sleeve"
i = s.index(HOOK)
j = s.index("the gold-only null as the binding comparison.", i)
s = s[:i] + s[i:j].replace("%%", "%") + s[j:]

P.write_text(s, encoding="utf-8")
print("fixed doubled percent signs in the research/153 entries")
