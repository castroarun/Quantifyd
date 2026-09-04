/**
 * Global search — one box in the TopBar, on every page.
 *
 * Ranking answers the question the box is for: "where is this thing I am
 * looking at?" So matches ON THE CURRENT PAGE come first, scanned live out of
 * the rendered DOM (which means anything on screen is searchable — table rows,
 * legs, strikes, chips — with no per-page wiring). Destinations (pages,
 * systems, published studies) follow underneath, favourites ranked first
 * among them. Focusing the empty box lists your favourites.
 *
 * Display-only: it reads the DOM and navigates. It touches no trading state.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import styles from './GlobalSearch.module.css';
import { destinations, type Destination } from './searchIndex';
import { isFavourite, useFavourites } from './favourites';
import StarButton from './StarButton';

const MAX_NODES = 8000;
const MAX_PAGE_HITS = 30;
const MAX_DEST_HITS = 14;
const HIT_CLASS = 'qf-search-hit';

interface PageHit {
  kind: 'page';
  id: string;
  text: string;
  context: string;
  el: HTMLElement;
  score: number;
}

interface DestHit {
  kind: 'dest';
  id: string;
  dest: Destination;
  score: number;
}

type Hit = PageHit | DestHit;

const clean = (s: string) => s.replace(/\s+/g, ' ').trim();

/** Nearest heading above this element — tells you which card/section it sits in. */
function sectionContext(el: Element): string {
  let cur: Element | null = el;
  for (let hops = 0; cur && hops < 10; hops += 1) {
    let sib: Element | null = cur.previousElementSibling;
    for (let n = 0; sib && n < 6; n += 1) {
      if (/^H[1-6]$/.test(sib.tagName)) {
        const t = clean(sib.textContent ?? '');
        if (t) return t.slice(0, 60);
      }
      sib = sib.previousElementSibling;
    }
    cur = cur.parentElement;
  }
  return '';
}

/** For a table cell, "row label · column header" beats a generic section name. */
function tableContext(el: Element): string {
  const cell = el.closest('td,th');
  if (!cell) return '';
  const row = cell.parentElement;
  const table = cell.closest('table');
  if (!row || !table) return '';
  const idx = Array.prototype.indexOf.call(row.children, cell);
  const headRow = table.querySelector('thead tr');
  const head = headRow?.children?.[idx];
  const col = head ? clean(head.textContent ?? '').slice(0, 30) : '';
  const first = row.children[0];
  const rowLabel = first && first !== cell ? clean(first.textContent ?? '').slice(0, 30) : '';
  return [rowLabel, col].filter(Boolean).join(' · ');
}

function scoreText(text: string, tokens: string[], tag: string): number {
  const lower = text.toLowerCase();
  let s = 0;
  const i = lower.indexOf(tokens[0]);
  if (i === 0) s += 60;
  else if (i > 0 && /[^a-z0-9]/.test(lower[i - 1])) s += 42;
  else s += 16;
  // Shorter text is a more precise hit than a paragraph that happens to contain it
  s += Math.max(0, 32 - Math.floor(text.length / 6));
  if (/^H[1-3]$/.test(tag)) s += 26;
  else if (/^H[4-6]$/.test(tag)) s += 16;
  else if (tag === 'TH' || tag === 'BUTTON' || tag === 'A') s += 12;
  return s;
}

function scanCurrentPage(tokens: string[]): PageHit[] {
  const root = document.querySelector('[data-search-root]');
  if (!root) return [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const hits: PageHit[] = [];
  const seen = new Set<string>();
  let scanned = 0;
  let node = walker.nextNode();
  while (node) {
    scanned += 1;
    if (scanned > MAX_NODES) break;
    const raw = node.nodeValue;
    if (raw && raw.trim().length > 1) {
      const text = clean(raw);
      if (text.length <= 400) {
        const lower = text.toLowerCase();
        if (tokens.every((t) => lower.includes(t))) {
          const el = node.parentElement as HTMLElement | null;
          if (el && !el.closest('[data-search-ui]') && el.getClientRects().length > 0) {
            const context = tableContext(el) || sectionContext(el);
            const key = `${context}|${text}`;
            if (!seen.has(key)) {
              seen.add(key);
              hits.push({
                kind: 'page',
                id: `p${hits.length}:${key}`,
                text,
                context,
                el,
                score: scoreText(text, tokens, el.tagName),
              });
            }
          }
        }
      }
    }
    node = walker.nextNode();
  }
  return hits.sort((a, b) => b.score - a.score).slice(0, MAX_PAGE_HITS);
}

function scanDestinations(tokens: string[]): DestHit[] {
  const out: DestHit[] = [];
  for (const d of destinations()) {
    const label = d.label.toLowerCase();
    const hay = `${label} ${(d.hint ?? '').toLowerCase()} ${(d.keywords ?? '').toLowerCase()}`;
    if (!tokens.every((t) => hay.includes(t))) continue;
    let s = 0;
    const i = label.indexOf(tokens[0]);
    if (i === 0) s += 70;
    else if (i > 0) s += 45;
    else s += 12;
    if (d.group === 'Pages') s += 14;
    else if (d.group === 'Systems') s += 8;
    if (isFavourite(d.to)) s += 20; // a page you starred is the one you meant
    s += Math.max(0, 20 - Math.floor(d.label.length / 4));
    out.push({ kind: 'dest', id: d.id, dest: d, score: s });
  }
  return out.sort((a, b) => b.score - a.score).slice(0, MAX_DEST_HITS);
}

/** Snippet centred on the match, with the matched run wrapped in <mark>. */
function Snippet({ text, token }: { text: string; token: string }) {
  const lower = text.toLowerCase();
  const at = token ? lower.indexOf(token) : -1;
  if (at < 0) return <>{text.length > 120 ? `${text.slice(0, 120)}…` : text}</>;
  const start = Math.max(0, at - 34);
  const head = (start > 0 ? '…' : '') + text.slice(start, at);
  const hit = text.slice(at, at + token.length);
  const tailRaw = text.slice(at + token.length);
  const tail = tailRaw.length > 70 ? `${tailRaw.slice(0, 70)}…` : tailRaw;
  return (
    <>
      {head}
      <mark className={styles.mark}>{hit}</mark>
      {tail}
    </>
  );
}

export default function GlobalSearch() {
  const [query, setQuery] = useState('');
  const [debounced, setDebounced] = useState('');
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const wrapRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const flashRef = useRef<HTMLElement | null>(null);
  const navigate = useNavigate();
  const location = useLocation();
  const favs = useFavourites();

  // Debounce so a fast typist does not trigger a DOM walk per keystroke
  useEffect(() => {
    const id = setTimeout(() => setDebounced(query.trim().toLowerCase()), 90);
    return () => clearTimeout(id);
  }, [query]);

  const { pageHits, destHits, tokens } = useMemo(() => {
    const t = debounced.split(/\s+/).filter(Boolean);
    if (!t.length) return { pageHits: [] as PageHit[], destHits: [] as DestHit[], tokens: t };
    return { pageHits: scanCurrentPage(t), destHits: scanDestinations(t), tokens: t };
    // location.pathname: a new page means the DOM to scan has changed
    // favs: starred pages rank higher, so the list must rebuild when they change
  }, [debounced, location.pathname, favs]);

  /** Empty box, focused: your favourites are the whole list. */
  const favHits: DestHit[] = useMemo(
    () =>
      debounced
        ? []
        : favs.map((f) => ({
            kind: 'dest' as const,
            id: `fav:${f.to}`,
            dest: { id: `fav:${f.to}`, label: f.label, group: 'Pages' as const, to: f.to },
            score: 0,
          })),
    [favs, debounced],
  );

  const hits: Hit[] = useMemo(
    () => (debounced ? [...pageHits, ...destHits] : favHits),
    [debounced, pageHits, destHits, favHits],
  );

  const indexById = useMemo(() => {
    const m = new Map<string, number>();
    hits.forEach((h, i) => m.set(h.id, i));
    return m;
  }, [hits]);

  useEffect(() => setActive(0), [debounced, hits.length]);

  // A new page means new content — drop the panel rather than show stale hits
  useEffect(() => {
    setOpen(false);
    setQuery('');
  }, [location.pathname]);

  // Keep the keyboard-selected row visible inside the panel
  useEffect(() => {
    const el = panelRef.current?.querySelector(`[data-idx="${active}"]`) as HTMLElement | null;
    el?.scrollIntoView({ block: 'nearest' });
  }, [active]);

  const clearFlash = useCallback(() => {
    if (flashRef.current) {
      flashRef.current.classList.remove(HIT_CLASS);
      flashRef.current = null;
    }
  }, []);

  const activate = useCallback(
    (hit: Hit) => {
      if (hit.kind === 'dest') {
        setOpen(false);
        setQuery('');
        inputRef.current?.blur();
        navigate(hit.dest.to);
        return;
      }
      if (!document.contains(hit.el)) return;
      clearFlash();
      hit.el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      hit.el.classList.add(HIT_CLASS);
      flashRef.current = hit.el;
      window.setTimeout(clearFlash, 2200);
      setOpen(false);
      inputRef.current?.blur();
    },
    [navigate, clearFlash],
  );

  // Ctrl/Cmd+K or "/" focuses the box from anywhere
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = document.activeElement as HTMLElement | null;
      const typing = !!el && (el.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName));
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        inputRef.current?.focus();
        inputRef.current?.select();
        return;
      }
      if (e.key === '/' && !typing && !e.ctrlKey && !e.metaKey && !e.altKey) {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // Click outside closes the panel
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, []);

  useEffect(() => clearFlash, [clearFlash]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Escape') {
      if (query) {
        setQuery('');
        setOpen(false);
      } else {
        inputRef.current?.blur();
      }
      return;
    }
    if (!open || !hits.length) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActive((a) => (a + 1) % hits.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActive((a) => (a - 1 + hits.length) % hits.length);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const hit = hits[active];
      if (hit) activate(hit);
    }
  };

  const token = tokens[0] ?? '';
  const showPanel = open && (debounced.length > 0 || favHits.length > 0);

  const destGroups = useMemo(() => {
    const groups: Array<{ name: string; items: DestHit[] }> = [];
    for (const h of destHits) {
      const g = groups.find((x) => x.name === h.dest.group);
      if (g) g.items.push(h);
      else groups.push({ name: h.dest.group, items: [h] });
    }
    return groups;
  }, [destHits]);

  const destRow = (h: DestHit) => {
    const idx = indexById.get(h.id) ?? -1;
    return (
      <div
        key={h.id}
        data-idx={idx}
        className={`${styles.row} ${idx === active ? styles.rowActive : ''}`}
        onMouseEnter={() => setActive(idx)}
      >
        <button type="button" className={styles.rowMain} onClick={() => activate(h)}>
          <span className={styles.itemText}>
            <Snippet text={h.dest.label} token={token} />
            {h.dest.hotkey && <kbd className={styles.itemKbd}>{h.dest.hotkey}</kbd>}
          </span>
          {h.dest.hint && <span className={styles.itemHint}>{h.dest.hint}</span>}
        </button>
        <StarButton compact target={{ to: h.dest.to, label: h.dest.label }} />
      </div>
    );
  };

  return (
    <div
      className={styles.wrap}
      ref={wrapRef}
      data-search-ui
      onClick={() => inputRef.current?.focus()}
    >
      <svg className={styles.icon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
        <circle cx="11" cy="11" r="7" />
        <path d="m20 20-3.2-3.2" />
      </svg>
      <input
        ref={inputRef}
        className={styles.input}
        value={query}
        placeholder="Search"
        aria-label="Search this page and the app"
        spellCheck={false}
        autoComplete="off"
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={onKeyDown}
      />
      {query ? (
        <button
          type="button"
          className={styles.clear}
          aria-label="Clear search"
          onClick={() => {
            setQuery('');
            setOpen(false);
            inputRef.current?.focus();
          }}
        >
          ×
        </button>
      ) : (
        <kbd className={styles.kbd}>/</kbd>
      )}

      {showPanel && (
        <div className={styles.panel} ref={panelRef} onMouseDown={(e) => e.preventDefault()}>
          {!debounced ? (
            <div className={styles.group}>
              <div className={styles.groupHead}>
                <span>Favourites</span>
                <span className={styles.count}>{favHits.length}</span>
              </div>
              {favHits.map(destRow)}
            </div>
          ) : (
            <>
              {!hits.length && <div className={styles.empty}>No matches for “{query}”</div>}

              {pageHits.length > 0 && (
                <div className={styles.group}>
                  <div className={styles.groupHead}>
                    <span>On this page</span>
                    <span className={styles.count}>{pageHits.length}</span>
                  </div>
                  {pageHits.map((h) => {
                    const idx = indexById.get(h.id) ?? -1;
                    return (
                      <button
                        type="button"
                        key={h.id}
                        data-idx={idx}
                        className={`${styles.item} ${idx === active ? styles.itemActive : ''}`}
                        onMouseEnter={() => setActive(idx)}
                        onClick={() => activate(h)}
                      >
                        <span className={styles.itemText}>
                          <Snippet text={h.text} token={token} />
                        </span>
                        {h.context && <span className={styles.itemHint}>{h.context}</span>}
                      </button>
                    );
                  })}
                </div>
              )}

              {destGroups.map((g) => (
                <div className={styles.group} key={g.name}>
                  <div className={styles.groupHead}>
                    <span>{g.name}</span>
                    <span className={styles.count}>{g.items.length}</span>
                  </div>
                  {g.items.map(destRow)}
                </div>
              ))}
            </>
          )}

          <div className={styles.foot}>
            <span>↑↓ move · ↵ open · esc close</span>
            <span>
              {!debounced
                ? 'star a page to pin it here'
                : pageHits.length
                  ? 'page matches first'
                  : 'no match on this page'}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
