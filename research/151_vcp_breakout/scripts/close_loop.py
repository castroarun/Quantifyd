"""Close the research/151 loop on the VPS: ops review entry, labs reference, INDEX, TODO."""
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')

# ---------------------------------------------------------------- ops registry
ops = ROOT / 'research' / '111_sensex_manual_mgmt' / 'scripts' / 'ops_center.py'
s = ops.read_text(encoding='utf-8')
marker = 'research/151'
if marker in s:
    print('ops_center: already registered')
else:
    entry = (
        '    ("BananaPatterns screen family - re-open ONLY on new evidence '
        '(research/151 VCP verdict: NO EDGE)",\n'
        '     "2027-03-05", "SCHEDULED",\n'
        '     "research/151 killed the site VCP screen: their published trades contain NO '
        'volatility contraction (pivot ages 1-157 bars, 11/37 bases with zero contractions), '
        'the null control shows a SHORTER pivot lookback always scores better (2-day null '
        'Calmar 2.63 vs 30-day 1.28) so the pattern subtracts value, correlation to the live '
        'Open Alpha book is 0.749 daily / 0.759 monthly (bar <0.40), and the best blend weight '
        'adds +0.033 Calmar against a +0.10 bar while LOSING to a plain cash sleeve. '
        'RE-OPEN CRITERION (pre-registered): only if the site publishes an explicit, '
        'reproducible VCP definition (contraction count, tightness ratio, base depth, volume '
        'condition) OR supplies a trade list that our 30-day closing-high reconstruction fails '
        'to explain in a NEW way. Absent that, cite this study and decline. Artifacts: '
        'research/151_vcp_breakout/results/RESULTS.md, p1d_family_scan.csv, p6g_cells.csv, '
        'vcp_adopted_spec.json, vcp_equity_seeds.csv; published at '
        '/app/backtest/vcp-breakout-research151."),\n'
    )
    s = s.replace('REVIEWS = [\n', 'REVIEWS = [\n' + entry, 1)
    ops.write_text(s, encoding='utf-8')
    print('ops_center: review registered')

# ------------------------------------------------------- labs & jobs reference
labs = ROOT / 'docs' / 'LABS_AND_JOBS_REFERENCE.md'
if labs.exists():
    t = labs.read_text(encoding='utf-8')
    if 'research/151' not in t:
        t += (
            '\n## research/151 — BananaPatterns "VCP" screen (review due 2027-03-05)\n\n'
            'Verdict **NO EDGE**. The screen reproduces the site exit engine exactly (31/32 '
            'trades) but its "volatility contraction pattern" is absent from its own published '
            'trades, a null control shows the pattern subtracts value, and the book correlates '
            '0.75 with the live Open Alpha sleeve and loses the blend test to plain cash. '
            'Re-open only on a published, reproducible VCP definition. '
            'Study: `research/151_vcp_breakout/results/RESULTS.md`; '
            'page: `/app/backtest/vcp-breakout-research151`.\n'
        )
        labs.write_text(t, encoding='utf-8')
        print('labs reference: appended')
    else:
        print('labs reference: already present')
else:
    print('labs reference: file missing, skipped')

# ------------------------------------------------------------------- INDEX.md
idx = ROOT / 'research' / 'INDEX.md'
t = idx.read_text(encoding='utf-8')
if '151_vcp_breakout' in t:
    print('INDEX: already present')
else:
    lines = t.splitlines()
    row = (
        '| 151 | [BananaPatterns "VCP" screen](151_vcp_breakout/results/RESULTS.md) - Arun: '
        'test the site VCP screen (5 positions, cut a loser at 7%, trail 50-day, risk 2%, gate '
        'off) claiming Rs 10L -> Rs 2.6Cr = 25.99x / +72.1% CAGR / -14.8% worst fall over '
        '2020-25. Replication gate first (40-trade ground truth from one of their own VCP runs), '
        'then claim, then strategy; r/142 engine extended with their risk-based sizing '
        '(risk/stop capped at 30%), Indian-FY tax netting and a null control on the pivot '
        'lookback | daily 2004-06 -> 2026-09 (2,321 symbols); their window 2020-25 plus '
        '2012-2026 and 2006-2026; ~230 cells, 10-seed scan / 30-seed adoption; after tax, '
        '25/40/60 bps, idle cash 5% | **Three verdicts. (1) RULES - PARTIAL (62.2%)**: their '
        'EXIT engine reproduces 31/32 ground-truth trades exactly (8% stop on the CLOSE + exit '
        'at the close breaking the 50-SMA) - the same engine r/142 decoded behind Blue Sky. '
        'Their entry pivot is an exact prior CLOSE in 36/37 but is NOT the ATH close (median 6% '
        'below it) and contains **no volatility contraction at all**: pivot ages 1-157 bars, '
        '11/37 bases with zero measurable contractions, volume ratio 0.27-1.53, and no fixed '
        'lookback can fit them (needs N>=157 and N<=11 at once). Best of 68 reconstructions = '
        '30-day rolling closing high. **(2) CLAIM - REFUTED**: their own dials replayed honestly '
        'give 32.4% CAGR [6.5..61.6 across seeds] at -34.5% DD; trade count matches (121 vs 164) '
        'so the machine is right and the number is not; their -14.8% worst fall is unreachable '
        '(best path -21%). Their risk sizing (2%/7% = 28.6% per position) is what makes their '
        'single path so wild. **(3) STRATEGY - NO EDGE, killed by its own null control**: '
        'shrinking the pivot lookback toward no-pattern monotonically IMPROVES the book (30d '
        'Calmar 1.28 -> 10d 1.59 -> 3d 1.91 -> 2d 2.63), so the screen subtracts value; the stop '
        'is inert (6/8/10/15%/none all identical - the trail always fires first) and only RS>=70 '
        'does real work. Portfolio: corr **0.749 daily / 0.759 monthly to the live Open Alpha '
        'book** (bar <0.40), best blend weight adds +0.033 Calmar (bar +0.10) with a WORSE '
        'drawdown, every heavier weight loses the paired test (25% wins 1 of 30 paths), and a '
        'plain CASH sleeve beats it at every weight; adding it worsens 2008, 2018 and 2022H1 '
        'alike. Standalone it buys +2.1pp CAGR over OA for +16pp of drawdown and dies on cost '
        '(Calmar 0.90 / 0.71 / 0.51 at 25 / 40 / 60 bps on ~37x NAV annual turnover) | '
        '**NO EDGE - rejected; published at /app/backtest/vcp-breakout-research151** |')
    # insert after the last table row that starts with '| 1' (numeric studies)
    ins = None
    for i, ln in enumerate(lines):
        if ln.startswith('| 150 ') or ln.startswith('|150'):
            ins = i + 1
    if ins is None:
        for i, ln in enumerate(lines):
            if ln.startswith('|') and '---' in ln:
                ins = i + 1
                break
    lines.insert(ins if ins is not None else len(lines), row)
    idx.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('INDEX: row inserted at line', ins)

# -------------------------------------------------------------------- TODO.md
todo = ROOT / 'TODO.md'
t = todo.read_text(encoding='utf-8')
if 'research/151' in t:
    print('TODO: already present')
else:
    block = (
        '\n### research/151 — BananaPatterns "VCP" screen — DONE 2026-09-05, verdict NO EDGE\n\n'
        '- Replication gate PARTIAL (62.2% joint match). Their exit engine reproduces 31/32 '
        'ground-truth trades exactly; their entry pivot is an exact prior close but carries no '
        'volatility-contraction structure, and no fixed lookback can fit it.\n'
        '- Published claim (25.99x / +72.1% CAGR / -14.8% worst fall) REFUTED: 32.4% CAGR '
        '[6.5..61.6] at -34.5% on their own dials, after tax and costs, 30 seeds.\n'
        '- Killed by its own null control: shrinking the pivot lookback toward no-pattern-at-all '
        'monotonically improves the book, so the screen subtracts value.\n'
        '- Portfolio: corr 0.749 to the live Open Alpha book (bar <0.40); best blend weight adds '
        '+0.033 Calmar (bar +0.10) and loses to a plain cash sleeve at the same weight.\n'
        '- Published at `/app/backtest/vcp-breakout-research151`. Deliverables for study r/154 in '
        'place: `research/151_vcp_breakout/results/vcp_equity_seeds.csv` (30 after-tax daily '
        'curves) and `vcp_adopted_spec.json`.\n'
        '- Dated obligation registered in the Ops & Review Centre: re-open only on a published, '
        'reproducible VCP definition (due 2027-03-05).\n'
    )
    todo.write_text(t.rstrip() + '\n' + block, encoding='utf-8')
    print('TODO: appended')
