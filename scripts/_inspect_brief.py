"""Tiny one-off: inspect /tmp/brief_test.json from the latest API run."""
import json, sys

p = sys.argv[1] if len(sys.argv) > 1 else '/tmp/brief_test.json'
b = json.load(open(p))
if 'error' in b:
    print('ERROR:', b['error'])
    sys.exit(1)

print('Date:', b['date'])
print('Bias:', b['bias']['label'], 'score:', b['bias']['score'])
print('Summary:', b['bias']['summary'][:240])
print()
print('Market data fetched:')
for k, v in (b.get('market') or {}).items():
    if v:
        pct = v.get('pct')
        pct_str = ('%.2f%%' % pct) if pct is not None else 'n/a'
        print('  %-12s last=%s pct=%s' % (k, v.get('last'), pct_str))
    else:
        print('  %-12s FAILED' % k)
print()
print('Holdings today:', len(b['holdings']['today']))
for ev in b['holdings']['today'][:10]:
    print('  -', ev.get('tradingsymbol'), ev.get('event_type'), '·',
          ev.get('purpose') or ev.get('details') or '')
print('Holdings upcoming (7d):', len(b['holdings']['upcoming']))
for ev in b['holdings']['upcoming'][:10]:
    print('  -', ev.get('event_date'), ev.get('tradingsymbol'),
          ev.get('event_type'), '·',
          (ev.get('purpose') or ev.get('details') or '')[:60])
print()
print('F&O ban total:', b['fno_ban']['count'])
print('F&O ban in holdings:', b['fno_ban']['in_holdings'])
print()
print('Strategy outlook:')
for k, v in b['strategy_outlook'].items():
    print('  %s: %s' % (k, v))
