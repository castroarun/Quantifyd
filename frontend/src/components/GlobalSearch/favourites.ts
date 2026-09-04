/**
 * Favourite pages — a tiny localStorage store shared by the TopBar star, the
 * search panel and the Sidebar's Favourites section.
 *
 * It is a browser preference, not app state: no backend, no trading data. The
 * cached snapshot is what makes it safe for useSyncExternalStore (the getter
 * must return a referentially stable value between actual changes).
 */
import { useSyncExternalStore } from 'react';
import { destinations } from './searchIndex';

const KEY = 'qf.favourites.v1';
const EVENT = 'qf-favourites-changed';
const MAX = 12;

export interface Favourite {
  to: string;
  label: string;
}

let cache: Favourite[] | null = null;

function read(): Favourite[] {
  if (cache) return cache;
  try {
    const raw = localStorage.getItem(KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    cache = Array.isArray(parsed)
      ? parsed.filter((x: unknown): x is Favourite =>
          !!x && typeof (x as Favourite).to === 'string' && typeof (x as Favourite).label === 'string')
      : [];
  } catch {
    cache = [];
  }
  return cache;
}

function commit(next: Favourite[]) {
  cache = next;
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* private mode / quota — the in-memory list still works for this session */
  }
  window.dispatchEvent(new Event(EVENT));
}

export function listFavourites(): Favourite[] {
  return read();
}

export function isFavourite(to: string): boolean {
  return read().some((f) => f.to === to);
}

/** Add when absent, remove when present. Oldest drops past MAX. */
export function toggleFavourite(fav: Favourite): void {
  const cur = read();
  const next = cur.some((f) => f.to === fav.to)
    ? cur.filter((f) => f.to !== fav.to)
    : [...cur, fav].slice(-MAX);
  commit(next);
}

function subscribe(cb: () => void): () => void {
  const onStorage = (e: StorageEvent) => {
    if (e.key === KEY) {
      cache = null; // another tab changed it — re-read on next snapshot
      cb();
    }
  };
  window.addEventListener(EVENT, cb);
  window.addEventListener('storage', onStorage);
  return () => {
    window.removeEventListener(EVENT, cb);
    window.removeEventListener('storage', onStorage);
  };
}

const EMPTY: Favourite[] = [];

export function useFavourites(): Favourite[] {
  return useSyncExternalStore(subscribe, read, () => EMPTY);
}

/**
 * What "this page" means for the star button. An exact route match keeps the
 * real name (including a study title); a deeper path (/journal/day/...) folds
 * up to its parent page so favourites stay stable and re-visitable.
 */
export function resolvePage(pathname: string): Favourite | null {
  const all = destinations();
  const exact = all.find((d) => d.to === pathname);
  if (exact) return { to: exact.to, label: exact.label };
  let best: { to: string; label: string } | null = null;
  for (const d of all) {
    if (pathname.startsWith(`${d.to}/`) && (!best || d.to.length > best.to.length)) {
      best = { to: d.to, label: d.label };
    }
  }
  return best;
}
