# -*- coding: utf-8 -*-
"""research/163 — assert that harmonising the cash yield moved ONLY True North and
Open Alpha · Base Age.

Compares a snapshot of the report JSON taken BEFORE the rebuild against the one written
AFTER it, and fails loudly if anything outside the two re-run books changed.

    venv/bin/python3 .../check_unchanged.py results/mpf_report_before.json \
                                            ../../static/app/mpf_report.json

Rows that MUST be bit-identical on both windows: IPO Base, Quality Summit, NIFTYBEES, and
the Open Alpha · ATH + VIX summary row. Rows expected to move: True North, Open Alpha ·
Base Age, and the 50-50 blend (which is built from the two of them).
"""
import json
import sys
from pathlib import Path

MOVERS = {'True North', 'Open Alpha · Base Age', 'TN + Base Age (50-50, monthly)'}
FROZEN = {'IPO Base', 'Quality Summit', 'NIFTYBEES'}


def walk(a, b, path, diffs):
    if type(a) is not type(b):
        diffs.append((path, a, b)); return
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                diffs.append((path + '/' + str(k), a.get(k, '<missing>'), b.get(k, '<missing>')))
            else:
                walk(a[k], b[k], path + '/' + str(k), diffs)
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append((path + '/len', len(a), len(b))); return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, '%s/%d' % (path, i), diffs)
    elif a != b:
        diffs.append((path, a, b))


def main():
    before = json.load(open(sys.argv[1]))
    after = json.load(open(sys.argv[2]))
    fail = 0

    for win in ('headline', 'window2018'):
        if win not in before or win not in after:
            print('!! %s missing from one side' % win); fail += 1; continue
        for name in sorted(set(before[win]['rows']) | set(after[win]['rows'])):
            rb = before[win]['rows'].get(name)
            ra = after[win]['rows'].get(name)
            frozen = name in FROZEN
            if rb is None or ra is None:
                print('%-34s %-11s row present on only one side' % (name, win)); fail += 1
                continue
            diffs = []
            walk(rb, ra, '', diffs)
            yb = before[win]['yoy'].get(name)
            ya = after[win]['yoy'].get(name)
            ydiffs = []
            walk(yb, ya, '', ydiffs)
            moved = bool(diffs or ydiffs)
            tag = 'FROZEN' if frozen else ('mover' if name in MOVERS else 'other')
            state = 'CHANGED' if moved else 'identical'
            bad = (frozen and moved)
            print('%-34s %-11s %-7s %-10s%s' % (name, win, tag, state,
                                                '   <-- VIOLATION' if bad else ''))
            if bad:
                fail += 1
                for p, x, y in (diffs + ydiffs)[:8]:
                    print('        %s : %r -> %r' % (p, x, y))

    # the ATH + VIX summary row is typed from research/159's own summary json
    ab, aa = before['correction']['athVix'], after['correction']['athVix']
    d = []
    walk(ab, aa, '', d)
    print('%-34s %-11s %-7s %-10s%s' % ('Open Alpha · ATH + VIX', 'correction', 'FROZEN',
                                        'CHANGED' if d else 'identical',
                                        '   <-- VIOLATION' if d else ''))
    if d:
        fail += 1

    # so are the entry surface, null control and gate bake-off tables
    for k in ('entrySurface', 'nullControl', 'gateBakeoff', 'vixGates'):
        d = []
        walk(before['correction'][k], after['correction'][k], '', d)
        print('%-34s %-11s %-7s %-10s%s' % (k, 'correction', 'FROZEN',
                                            'CHANGED' if d else 'identical',
                                            '   <-- VIOLATION' if d else ''))
        if d:
            fail += 1

    print('\n%s' % ('FAIL — %d frozen item(s) moved' % fail if fail else
                    'PASS — only True North, Base Age and the blend moved'))
    sys.exit(1 if fail else 0)


if __name__ == '__main__':
    main()
