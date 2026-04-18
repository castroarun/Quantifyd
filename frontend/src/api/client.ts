// Thin fetch wrapper with session cookies and 401 → /login redirect.

const BASE = ''; // same origin

function buildUrl(path: string): string {
  if (path.startsWith('http')) return path;
  return BASE + path;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(buildUrl(path), {
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...(init.headers || {}),
    },
    ...init,
  });

  if (res.status === 401) {
    // Session invalid — kick back to login.
    if (!window.location.pathname.endsWith('/login')) {
      window.location.href = '/app/login';
    }
    throw new Error('Unauthorized');
  }

  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body && typeof body === 'object' && 'error' in body) message = String(body.error);
      else if (body && 'message' in body) message = String(body.message);
    } catch {
      /* ignore */
    }
    throw new Error(message);
  }

  // Some endpoints return empty bodies on POST; guard JSON parse.
  const text = await res.text();
  if (!text) return {} as T;
  return JSON.parse(text) as T;
}

export function apiGet<T>(path: string): Promise<T> {
  return request<T>(path, { method: 'GET' });
}

export function apiPost<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}
