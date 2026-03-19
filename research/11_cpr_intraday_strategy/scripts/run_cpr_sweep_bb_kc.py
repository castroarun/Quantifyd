"""
CPR Intraday Sweep: Bollinger Bands & Keltner Channel experiments.

Groups:
  A) Bollinger Bands only  (12 configs)
  B) Keltner Channels only (12 configs)
  C) BB + KC squeeze        (6 configs)

Total: 30 configs
"""

import sys, os, csv, time, json, logging, io, contextlib

# Suppress noisy logs
logging.disable(logging.WARNING)

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from services.cpr_intraday_engine import CPRIntradayEngine, CPRIntradayConfig

# ── Constants ──────────────────────────────────────────────────────────

SYMBOLS = [
    'BHARTIARTL', 'HDFCBANK', 'HINDUNILVR', 'ICICIBANK', 'INFY',
    'ITC', 'KOTAKBANK', 'RELIANCE', 'SBIN', 'TCS',
]
START_DATE = '2024-01-01'
END_DATE = '2025-10-27'

OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'cpr_sweep_bb_kc.csv')

FIELDNAMES = [
    'label', 'use_bb', 'bb_period', 'bb_std', 'use_kc', 'kc_period',
    'kc_mult', 'total_trades', 'win_rate', 'total_pnl', 'pnl_pct',
    'profit_factor', 'max_drawdown', 'sharpe', 'avg_trades_per_day',
    'exit_reasons',
]

# Wide CPR base settings
BASE = dict(
    symbols=SYMBOLS,
    start_date=START_DATE,
    end_date=END_DATE,
    narrow_cpr_threshold=3.0,
    cpr_proximity_pct=3.0,
    max_wick_pct=30.0,
    st_period=7,
    st_multiplier=3.0,
)

# ── Build configs ─────────────────────────────────────────────────────

configs = []

# Group A: Bollinger Bands only
for bb_p in [10, 20, 30]:
    for bb_s in [1.5, 2.0, 2.5, 3.0]:
        label = f"BB_p{bb_p}_s{bb_s}"
        params = dict(
            **BASE,
            use_bollinger=True,
            bb_period=bb_p,
            bb_std=bb_s,
            use_kc=False,
        )
        configs.append((label, params))

# Group B: Keltner Channels only
for kc_p in [10, 20, 30]:
    for kc_m in [1.0, 1.5, 2.0, 2.5]:
        label = f"KC_p{kc_p}_m{kc_m}"
        params = dict(
            **BASE,
            use_bollinger=False,
            use_kc=True,
            kc_period=kc_p,
            kc_multiplier=kc_m,
        )
        configs.append((label, params))

# Group C: BB + KC squeeze combos
squeeze_combos = [
    (10, 2.0, 10, 1.5),
    (20, 2.0, 20, 1.5),
    (20, 2.5, 20, 2.0),
    (30, 2.0, 30, 1.5),
    (20, 1.5, 20, 1.0),
    (10, 1.5, 10, 1.0),
]
for bb_p, bb_s, kc_p, kc_m in squeeze_combos:
    label = f"SQZ_bb{bb_p}s{bb_s}_kc{kc_p}m{kc_m}"
    params = dict(
        **BASE,
        use_bollinger=True,
        bb_period=bb_p,
        bb_std=bb_s,
        use_kc=True,
        kc_period=kc_p,
        kc_multiplier=kc_m,
    )
    configs.append((label, params))

# ── Skip already-completed configs ────────────────────────────────────

done = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, newline='') as f:
        done = {row['label'] for row in csv.DictReader(f)}
    print(f"Skipping {len(done)} already-completed configs")
else:
    # Write header
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

total = len(configs)
pending = [(l, p) for l, p in configs if l not in done]
print(f"Total configs: {total} | Pending: {len(pending)}")

if not pending:
    print("All configs already done. Exiting.")
    sys.exit(0)

# ── Preload data once ────────────────────────────────────────────────

print("Preloading data...", flush=True)
t0 = time.time()
daily_data, five_min_data = CPRIntradayEngine.preload_data(
    SYMBOLS, START_DATE, END_DATE
)
print(f"Data loaded in {time.time() - t0:.1f}s "
      f"({len(daily_data)} daily, {len(five_min_data)} 5min)", flush=True)

# ── Run sweep (suppress engine verbose output) ───────────────────────

for i, (label, params) in enumerate(pending, 1):
    print(f"[{i}/{len(pending)}] {label} ...", end='', flush=True)
    t1 = time.time()

    cfg = CPRIntradayConfig(**params)
    engine = CPRIntradayEngine(
        cfg,
        preloaded_daily=daily_data,
        preloaded_5min=five_min_data,
    )

    # Suppress engine's verbose progress prints
    with contextlib.redirect_stdout(io.StringIO()):
        result = engine.run()

    elapsed = time.time() - t1

    row = {
        'label': label,
        'use_bb': params.get('use_bollinger', False),
        'bb_period': params.get('bb_period', ''),
        'bb_std': params.get('bb_std', ''),
        'use_kc': params.get('use_kc', False),
        'kc_period': params.get('kc_period', ''),
        'kc_mult': params.get('kc_multiplier', ''),
        'total_trades': result.total_trades,
        'win_rate': round(result.win_rate, 2),
        'total_pnl': round(result.total_pnl, 2),
        'pnl_pct': round(result.total_pnl_pct, 2),
        'profit_factor': round(result.profit_factor, 2),
        'max_drawdown': round(result.max_drawdown, 2),
        'sharpe': round(result.sharpe_ratio, 2),
        'avg_trades_per_day': round(result.avg_trades_per_day, 2),
        'exit_reasons': json.dumps(result.exit_reason_counts),
    }

    # Write incrementally
    with open(OUTPUT_CSV, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

    print(
        f" {elapsed:.0f}s | Trades={result.total_trades} "
        f"WR={result.win_rate:.1f}% PF={result.profit_factor:.2f} "
        f"PnL={result.total_pnl:,.0f} DD={result.max_drawdown:.2f}%",
        flush=True,
    )

print(f"\nDone. Results in {OUTPUT_CSV}")
