import { useEffect, useRef } from 'react';

/** Subscribe to an SSE stream with automatic reconnection on errors. */
export function useSSE<T = unknown>(
  url: string | null,
  onMessage: (payload: T) => void,
  deps: unknown[] = [],
) {
  const cbRef = useRef(onMessage);
  cbRef.current = onMessage;

  useEffect(() => {
    if (!url) return;
    let source: EventSource | null = null;
    let reconnectTimer: number | null = null;
    let closed = false;

    const connect = () => {
      if (closed) return;
      try {
        source = new EventSource(url, { withCredentials: true });
      } catch {
        reconnectTimer = window.setTimeout(connect, 3000);
        return;
      }

      source.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data) as T;
          cbRef.current(data);
        } catch {
          /* ignore malformed payloads */
        }
      };

      source.onerror = () => {
        source?.close();
        source = null;
        if (!closed) {
          reconnectTimer = window.setTimeout(connect, 3000);
        }
      };
    };

    connect();

    return () => {
      closed = true;
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      source?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, ...deps]);
}
