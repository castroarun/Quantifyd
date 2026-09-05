# -*- coding: utf-8 -*-
"""The BacktestStudy.system field is {intro, rows: KV[]}, not the SystemRules shape.
Rewrite the research/153 entry's system block accordingly. Idempotent."""
from pathlib import Path

P = Path("/home/arun/quantifyd/frontend/src/data/backtests.ts")
s = P.read_text(encoding="utf-8")
if "sharedCoreTitle: 'Shared core - the decoded bananapatterns" not in s:
    print("nothing to fix")
    raise SystemExit(0)

start = s.index("      sharedCoreTitle: 'Shared core - the decoded bananapatterns")
end = s.index("    },\n    conditions: {\n      intro: 'Pre-registered in the STATUS document", start)
block = s[start:end]
# keep only the sharedCore KV rows, renamed to `rows`
inner_start = block.index("      sharedCore: [")
inner_end = block.index("      ],", inner_start) + len("      ],\n")
rows = block[inner_start:inner_end].replace("sharedCore: [", "rows: [", 1)
s = s[:start] + rows + s[end:]
P.write_text(s, encoding="utf-8")
print("fixed system block")
