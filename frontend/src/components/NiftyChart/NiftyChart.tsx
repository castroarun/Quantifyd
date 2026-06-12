import { useEffect, useRef } from 'react';

// Live NIFTY 50 5-min candlestick chart — TradingView advanced-chart widget.
// Frontend-only embed (no backend), shown above the NAS Trade Book. Matches the
// 5m + ATR/MA look the user uses on TradingView.
export default function NiftyChart() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.innerHTML = '';
    const inner = document.createElement('div');
    inner.className = 'tradingview-widget-container__widget';
    inner.style.height = '100%';
    inner.style.width = '100%';
    el.appendChild(inner);

    const script = document.createElement('script');
    script.src =
      'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
    script.type = 'text/javascript';
    script.async = true;
    script.innerHTML = JSON.stringify({
      autosize: true,
      symbol: 'NSE:NIFTY',
      interval: '5',
      timezone: 'Asia/Kolkata',
      theme: 'dark',
      style: '1',
      locale: 'en',
      hide_side_toolbar: true,
      allow_symbol_change: false,
      save_image: false,
      studies: ['STD;SMA', 'STD;Average%True%Range'],
      support_host: 'https://www.tradingview.com',
    });
    el.appendChild(script);

    return () => {
      if (el) el.innerHTML = '';
    };
  }, []);

  return (
    <section style={{ margin: '0 0 18px' }}>
      <div
        style={{
          fontSize: 12,
          color: 'var(--ink-muted, #8b93a1)',
          margin: '0 0 8px',
          letterSpacing: 0.3,
        }}
      >
        NIFTY 50 · live 5-min
      </div>
      <div
        ref={ref}
        className="tradingview-widget-container"
        style={{
          height: 380,
          width: '100%',
          borderRadius: 12,
          overflow: 'hidden',
          border: '1px solid var(--border, #232936)',
        }}
      />
    </section>
  );
}
