# -*- coding: utf-8 -*-
"""Add a top-of-page SENSEX LIVE card to Nas.tsx.

The 3 SENSEX 9:16 systems trade real money on Wed/Thu but had no UI — their live legs were invisible
(only the separate research/82 PAPER book showed at the bottom). This adds a LIVE card at the TOP
(above the Trade Book) with a LIVE marking in the MODE column, fed by /app/sensex_live.json (written
by the standalone scripts/sensex_live_writer.py). Frontend-only -> no backend restart.
"""
import io
import sys

F = "frontend/src/pages/Nas.tsx"
s = io.open(F, encoding="utf-8").read()
if "SensexLiveCard" in s:
    print("already patched"); sys.exit(0)

COMPONENT = r'''/* ---------- SENSEX LIVE book (real-money 9:16 executors) ----------
   The 3 SENSEX 9:16 systems (ATM / ATM2 / ATM4) trade REAL MONEY on Wed/Thu — SENSEX's 0/1-DTE,
   the days NIFTY has nothing to harvest and NIFTY real money is switched off. This renders their
   live legs at the TOP of the page with a LIVE marking in the MODE column. Data comes from a
   standalone writer (scripts/sensex_live_writer.py -> /app/sensex_live.json) so it needs no backend
   restart while positions are open. Refreshes every 8s. */
function SensexLiveCard() {
  const [d, setD] = useState<any>(null);
  useEffect(() => {
    const load = () => fetch(`/app/sensex_live.json?t=${Date.now()}`, { cache: 'no-store' })
      .then((r) => r.json()).then(setD).catch(() => {});
    load();
    const id = setInterval(load, 8000);
    return () => clearInterval(id);
  }, []);
  if (!d) return null;
  const rs = (n: number) => `${n >= 0 ? '+' : '−'}₹${Math.abs(Math.round(n)).toLocaleString('en-IN')}`;
  const cc = (n: number) => (n >= 0 ? '#3fb950' : '#f85149');
  const withLegs = (d.systems || []).filter((x: any) => (x.legs || []).length);
  const cell: React.CSSProperties = { fontSize: 11, padding: '3px 8px', textAlign: 'right',
    borderTop: '1px solid var(--line)', fontVariantNumeric: 'tabular-nums' };
  const head: React.CSSProperties = { fontSize: 9.5, color: 'var(--ink-faint, #6e7681)', padding: '2px 8px',
    textAlign: 'right', textTransform: 'uppercase', letterSpacing: '0.04em' };
  const modePill = (m: string) => (
    <span style={{ display: 'inline-block', fontSize: 9.5, fontWeight: 800, letterSpacing: '0.05em',
      padding: '1px 7px', borderRadius: 4,
      background: m === 'live' ? 'rgba(63,185,80,0.16)' : 'rgba(139,148,158,0.14)',
      color: m === 'live' ? '#3fb950' : 'var(--ink-muted)',
      border: `1px solid ${m === 'live' ? 'rgba(63,185,80,0.4)' : 'var(--line)'}` }}>
      {m === 'live' ? 'LIVE' : (m || '').toUpperCase()}
    </span>
  );
  const GRID = '54px 68px 46px 60px 60px 60px 1fr 78px 82px';

  return (
    <section className={styles.sectionBlock} style={{ marginTop: 14,
      border: '1px solid rgba(63,185,80,0.35)', borderRadius: 10, padding: '12px 14px',
      background: 'rgba(63,185,80,0.03)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4, flexWrap: 'wrap' }}>
        <span className="section-title">SENSEX · LIVE</span>
        {modePill('live')}
        <span style={{ fontSize: 11, color: 'var(--ink-muted)' }}>{d.expiry}</span>
        {d.spot != null && (
          <span style={{ fontSize: 11, color: 'var(--ink-muted)' }}>spot {d.spot.toLocaleString('en-IN')}</span>
        )}
        <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--ink-muted)' }}>
          day <b style={{ color: cc(d.day_pnl) }}>{rs(d.day_pnl)}</b>
        </span>
      </div>
      <div style={{ fontSize: 10, color: 'var(--ink-faint, #6e7681)', marginBottom: 8 }}>
        Real-money 9:16 systems on SENSEX (Wed/Thu 0/1-DTE) · updated {d.generated_at}
      </div>

      {!withLegs.length && (
        <div style={{ fontSize: 11.5, color: 'var(--ink-faint, #6e7681)',
          border: '1px solid var(--line)', borderRadius: 8, padding: '10px 12px' }}>
          No live SENSEX positions right now.
          {(d.systems || []).some((x: any) => x.mode === 'flat')
            ? ' (a system was margin-rejected / flat today)' : ''}
        </div>
      )}

      {withLegs.map((sy: any, si: number) => (
        <div key={sy.key} style={{ marginTop: si ? 12 : 2 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 2 }}>
            <span style={{ fontSize: 12, fontWeight: 700 }}>{sy.label}</span>
            {modePill(sy.mode)}
            <span style={{ marginLeft: 'auto', fontSize: 12, fontWeight: 800, color: cc(sy.pnl) }}>
              {rs(sy.pnl)}
            </span>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <div style={{ display: 'grid', gridTemplateColumns: GRID, gap: 8, minWidth: 620,
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' }}>
              <span style={{ ...head, textAlign: 'left' }}>C/P</span><span style={head}>STRIKE</span>
              <span style={head}>QTY</span><span style={{ ...head, textAlign: 'left' }}>MODE</span>
              <span style={head}>ENTRY</span><span style={head}>LTP</span>
              <span style={{ ...head, textAlign: 'left' }}>ARM</span><span style={head}>STATUS</span>
              <span style={head}>P&amp;L</span>
              {(sy.legs || []).map((l: any, li: number) => (
                <React.Fragment key={li}>
                  <span style={{ ...cell, textAlign: 'left', fontWeight: 700, color: '#f85149' }}>SELL {l.cp}</span>
                  <span style={cell}>{l.strike}</span>
                  <span style={cell}>{l.qty}</span>
                  <span style={{ ...cell, textAlign: 'left' }}>{modePill(l.mode)}</span>
                  <span style={cell}>{l.entry}</span>
                  <span style={cell}>{l.ltp ?? '—'}</span>
                  <span style={{ ...cell, textAlign: 'left',
                    color: (l.arm || '').includes('UNARMED') ? '#f85149' : 'var(--ink-muted)' }}>{l.arm}</span>
                  <span style={{ ...cell, color: l.status === 'ACTIVE' ? '#3fb950' : 'var(--ink-muted)' }}>{l.status}</span>
                  <span style={{ ...cell, fontWeight: 700, color: cc(l.pnl) }}>{rs(l.pnl)}</span>
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>
      ))}
    </section>
  );
}


'''

# 1) insert the component before the paper-book component
ANCHOR_DEF = "/* ---------- SENSEX expiry-day paper book (research/82) ----------"
if s.count(ANCHOR_DEF) != 1:
    print("FAIL def anchor (%d)" % s.count(ANCHOR_DEF)); sys.exit(1)
s = s.replace(ANCHOR_DEF, COMPONENT + ANCHOR_DEF, 1)

# 2) render it at the TOP — right after the chart, above the Trade Book
ANCHOR_REN = "      <NiftyChart />\n\n      {/* Trade Book"
if s.count(ANCHOR_REN) != 1:
    print("FAIL render anchor (%d)" % s.count(ANCHOR_REN)); sys.exit(1)
s = s.replace(ANCHOR_REN, "      <NiftyChart />\n\n      <SensexLiveCard />\n\n      {/* Trade Book", 1)

io.open(F, "w", encoding="utf-8").write(s)
print("SensexLiveCard added + rendered at top")
