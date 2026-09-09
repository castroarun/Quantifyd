import { useEffect, useState, lazy, Suspense } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

/* THE PORTFOLIO (/app/portfolio) — the three books and the desk that funds them,
   behind one nav entry and four tabs.

   They were four separate sidebar rows, which is wrong twice over: it is one portfolio,
   and the sidebar had grown long enough that the books were hard to find in it. Press
   0-3 to move between them (0 is the desk, where the portfolio opens); the tab is in
   the URL (?tab=oa) so a particular book can
   still be linked to and reloaded directly.

   Each tab is a lazy chunk, so opening the portfolio does not download all four. */

const MomentumPaper = lazy(() => import('./MomentumPaper'));
const BlueskyPaper = lazy(() => import('./BlueskyPaper'));
const IpoPaper = lazy(() => import('./IpoPaper'));
const CapitalDesk = lazy(() => import('./CapitalDesk'));

type TabId = 'tn' | 'oa' | 'ipo' | 'cd';

const TABS: { id: TabId; n: number; label: string; sub: string }[] = [
  { id: 'tn', n: 1, label: 'True North', sub: 'Nifty-200 momentum · LIVE' },
  { id: 'oa', n: 2, label: 'Open Alpha', sub: 'ATH breakout · LIVE' },
  { id: 'ipo', n: 3, label: 'IPO Base', sub: 'recent listings · paper' },
  /* 0, not 4: the desk is where the portfolio opens, so its key sits beside 1-2-3
     rather than after them. The tab keeps its place at the end of the strip - moving it
     would shuffle three shortcuts people already have in their fingers. */
  { id: 'cd', n: 0, label: 'Capital Desk', sub: 'money in and out · targets' },
];

export default function Portfolio() {
  const nav = useNavigate();
  const loc = useLocation();
  const urlTab = new URLSearchParams(loc.search).get('tab') as TabId | null;
  /* The desk is the default: arriving at the portfolio, the question is what the whole
     thing is worth and what happens next, not how one book is doing. A ?tab= link still
     wins, so a bookmarked book opens on that book. */
  const [tab, setTab] = useState<TabId>(
    TABS.some((t) => t.id === urlTab) ? (urlTab as TabId) : 'cd');

  const go = (id: TabId) => {
    setTab(id);
    nav(`/portfolio?tab=${id}`, { replace: true });
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      /* Never steal a digit someone is typing into a field — the Capital Desk has an
         amount box, and a deposit of "2" silently becoming a tab change would be the
         worst kind of bug to ship on a page that moves money. */
      const el = document.activeElement as HTMLElement | null;
      const tag = el?.tagName?.toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || el?.isContentEditable) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const hit = TABS.find((t) => e.key === String(t.n));
      if (hit) { e.preventDefault(); go(hit.id); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);                                   // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div>
      <div role="tablist" aria-label="Portfolio books" style={{
        display: 'flex', gap: 4, flexWrap: 'wrap', padding: '2px 22px 0',
        borderBottom: '1px solid var(--hairline,rgba(0,0,0,0.1))',
        background: 'var(--surface,#fff)', position: 'sticky', top: 0, zIndex: 20,
        marginBottom: 14,        /* the strip's border was touching the heading below it */
      }}>
        {TABS.map((t) => {
          const on = t.id === tab;
          return (
            <button key={t.id} role="tab" aria-selected={on} onClick={() => go(t.id)}
              title={`${t.label} — press ${t.n}`}
              style={{
                appearance: 'none', cursor: 'pointer', textAlign: 'left',
                border: 'none', borderBottom: `2px solid ${on ? 'var(--accent,#2563EB)' : 'transparent'}`,
                background: 'transparent', padding: '8px 14px 10px',
                color: on ? 'var(--ink,#1B1B1A)' : 'var(--ink-muted,#8a8a85)',
                font: `${on ? 600 : 500} 13.5px/1.25 inherit`,
              }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                {t.label}
                <kbd style={{
                  font: '500 9.5px/1 ui-monospace,monospace', padding: '2px 4px',
                  borderRadius: 3, border: '1px solid var(--hairline,rgba(0,0,0,0.16))',
                  color: 'var(--ink-muted,#8a8a85)',
                }}>{t.n}</kbd>
              </span>
              <span style={{ display: 'block', fontSize: 10.5, fontWeight: 400,
                             color: 'var(--ink-muted,#8a8a85)', marginTop: 2 }}>{t.sub}</span>
            </button>
          );
        })}
      </div>

      <Suspense fallback={<div style={{ padding: '28px 22px', fontSize: 13,
                                        color: 'var(--ink-muted,#8a8a85)' }}>Loading…</div>}>
        {tab === 'tn' && <MomentumPaper />}
        {tab === 'oa' && <BlueskyPaper />}
        {tab === 'ipo' && <IpoPaper />}
        {tab === 'cd' && <CapitalDesk />}
      </Suspense>
    </div>
  );
}
