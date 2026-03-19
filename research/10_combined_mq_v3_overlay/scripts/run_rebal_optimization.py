import logging
logging.disable(logging.WARNING)

import time
import csv
import sys
from itertools import product
from collections import defaultdict

sys.path.insert(0, r'c:\Users\Castro\Documents\Projects\Covered_Calls')
from services.mq_backtest_engine import MQBacktestEngine, BacktestResult
from services.mq_portfolio import MQBacktestConfig

LOG_PATH = "c:/Users/Castro/Documents/Projects/Covered_Calls/optimization_progress.log"
def log(msg):
    print(msg)
    sys.stdout.flush()
    with open(LOG_PATH, "a") as lf:
        lf.write(msg + chr(10))

with open(LOG_PATH, "w") as lf:
    lf.write("")


# Parameter grid
portfolio_sizes = [30]
equity_allocs = [0.85, 0.90, 0.95]
rebalance_schedules = {
    "semi": [1, 7],
    "quarterly": [1, 4, 7, 10],
    "bi-monthly": [1, 3, 5, 7, 9, 11],
}
hard_stop_losses = [0.25]
ath_drawdowns = [0.15, 0.20]
ath_proximities = [0.05, 0.10]

combos = list(product(
    portfolio_sizes,
    equity_allocs,
    rebalance_schedules.items(),
    hard_stop_losses,
    ath_drawdowns,
    ath_proximities,
))

total = len(combos)
log(f"Total combinations: {total}")
log(f"Starting optimization...\n")

log("Preloading universe and price data...")
t0 = time.time()
base_config = MQBacktestConfig(
    portfolio_size=30,
    trailing_stop_loss=True,
    daily_ath_drawdown_exit=True,
)
seed_engine = MQBacktestEngine(config=base_config)
seed_result = seed_engine.run()
preloaded_universe = seed_engine.universe
preloaded_price_data = seed_engine.price_data
load_time = time.time() - t0
log(f"Data preloaded in {load_time:.1f}s\n")

results = []
start_time = time.time()

for i, (ps, eq, (rebal_name, rebal_months), hsl, ath_dd, ath_prox) in enumerate(combos, 1):
    config = MQBacktestConfig(
        portfolio_size=ps,
        equity_allocation_pct=eq,
        debt_reserve_pct=round(1.0 - eq, 4),
        rebalance_months=rebal_months,
        hard_stop_loss=hsl,
        rebalance_ath_drawdown=ath_dd,
        ath_proximity_threshold=ath_prox,
        trailing_stop_loss=True,
        daily_ath_drawdown_exit=True,
    )

    engine = MQBacktestEngine(
        config=config,
        preloaded_universe=preloaded_universe,
        preloaded_price_data=preloaded_price_data,
    )

    try:
        result = engine.run()
        results.append({
            "run": i,
            "portfolio_size": ps,
            "equity_alloc": eq,
            "rebalance": rebal_name,
            "rebalance_months": str(rebal_months),
            "hard_stop_loss": hsl,
            "ath_drawdown": ath_dd,
            "ath_proximity": ath_prox,
            "cagr": result.cagr,
            "total_return": result.total_return_pct,
            "sharpe": result.sharpe_ratio,
            "sortino": result.sortino_ratio,
            "max_drawdown": result.max_drawdown,
            "calmar": result.calmar_ratio,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "final_value": result.final_value,
        })
    except Exception as e:
        results.append({
            "run": i,
            "portfolio_size": ps,
            "equity_alloc": eq,
            "rebalance": rebal_name,
            "rebalance_months": str(rebal_months),
            "hard_stop_loss": hsl,
            "ath_drawdown": ath_dd,
            "ath_proximity": ath_prox,
            "cagr": None,
            "total_return": None,
            "sharpe": None,
            "sortino": None,
            "max_drawdown": None,
            "calmar": None,
            "win_rate": None,
            "total_trades": None,
            "final_value": None,
            "error": str(e),
        })

    if i % 5 == 0 or i == total:
        elapsed = time.time() - start_time
        rate = i / elapsed if elapsed > 0 else 0
        eta = (total - i) / rate if rate > 0 else 0
        log(f"  [{i:>3}/{total}] {elapsed:.0f}s elapsed | {rate:.1f} runs/s | ETA {eta:.0f}s"); sys.stdout.flush()

elapsed_total = time.time() - start_time
print(f"\nCompleted {total} runs in {elapsed_total:.1f}s ({total/elapsed_total:.1f} runs/s)\n")

csv_path = r"c:\Users\Castro\Documents\Projects\Covered_Calls\optimization_agent2_rebal.csv"
valid_results = [r for r in results if r.get("cagr") is not None]
failed = total - len(valid_results)

if valid_results:
    fieldnames = list(valid_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in valid_results:
            writer.writerow(row)
    print(f"Saved {len(valid_results)} results to {csv_path}")
    if failed:
        print(f"  ({failed} runs failed)")

print("\n" + "=" * 120)
print("TOP 20 CONFIGURATIONS BY CAGR")
print("=" * 120)

sorted_results = sorted(valid_results, key=lambda x: x["cagr"], reverse=True)

hdr = f"{'#':>3}  {'PS':>3}  {'EqAlloc':>7}  {'Rebalance':<12}  {'HSL':>5}  {'ATH_DD':>6}  {'ATH_Prox':>8}  {'CAGR':>8}  {'Return':>8}  {'Sharpe':>7}  {'Sortino':>8}  {'MaxDD':>7}  {'Calmar':>7}  {'WinRate':>7}  {'Trades':>6}"
print(hdr)
print("-" * 120)

for rank, r in enumerate(sorted_results[:20], 1):
    line = f"{rank:>3}  {r['portfolio_size']:>3}  {r['equity_alloc']:>7.2f}  {r['rebalance']:<12}  {r['hard_stop_loss']:>5.2f}  {r['ath_drawdown']:>6.2f}  {r['ath_proximity']:>8.2f}  {r['cagr']:>7.2f}%  {r['total_return']:>7.1f}%  {r['sharpe']:>7.3f}  {r['sortino']:>8.3f}  {r['max_drawdown']:>6.1f}%  {r['calmar']:>7.3f}  {r['win_rate']:>6.1f}%  {r['total_trades']:>6}"
    print(line)

print("\n" + "=" * 80)
print("REBALANCE FREQUENCY IMPACT ON CAGR (AVERAGE)")
print("=" * 80)

rebal_stats = defaultdict(list)
for r in valid_results:
    rebal_stats[r["rebalance"]].append(r["cagr"])

print(f"{'Rebalance Type':<15}  {'Count':>6}  {'Avg CAGR':>10}  {'Min CAGR':>10}  {'Max CAGR':>10}  {'Std Dev':>10}")
print("-" * 80)

for rebal_type in ["semi", "quarterly", "bi-monthly"]:
    if rebal_type in rebal_stats:
        vals = rebal_stats[rebal_type]
        n = len(vals)
        avg = sum(vals) / n
        mn = min(vals)
        mx = max(vals)
        variance = sum((v - avg) ** 2 for v in vals) / n
        std = variance ** 0.5
        print(f"{rebal_type:<15}  {n:>6}  {avg:>9.2f}%  {mn:>9.2f}%  {mx:>9.2f}%  {std:>9.2f}%")

print("\n" + "=" * 80)
print("EQUITY ALLOCATION IMPACT ON CAGR (AVERAGE)")
print("=" * 80)

eq_stats = defaultdict(list)
for r in valid_results:
    eq_stats[r["equity_alloc"]].append(r["cagr"])

print(f"{'Equity Alloc':>12}  {'Count':>6}  {'Avg CAGR':>10}  {'Min CAGR':>10}  {'Max CAGR':>10}")
print("-" * 60)
for eq in sorted(eq_stats.keys()):
    vals = eq_stats[eq]
    n = len(vals)
    avg = sum(vals) / n
    mn = min(vals)
    mx = max(vals)
    print(f"{eq:>11.0%}  {n:>6}  {avg:>9.2f}%  {mn:>9.2f}%  {mx:>9.2f}%")

print("\n" + "=" * 80)
print("PORTFOLIO SIZE IMPACT ON CAGR (AVERAGE)")
print("=" * 80)

ps_stats = defaultdict(list)
ps_sharpe = defaultdict(list)
for r in valid_results:
    ps_stats[r["portfolio_size"]].append(r["cagr"])
    ps_sharpe[r["portfolio_size"]].append(r["sharpe"])

print(f"{'Port Size':>10}  {'Count':>6}  {'Avg CAGR':>10}  {'Avg Sharpe':>11}")
print("-" * 50)
for ps in sorted(ps_stats.keys()):
    vals = ps_stats[ps]
    sharpes = ps_sharpe[ps]
    avg_cagr = sum(vals) / len(vals)
    avg_sharpe = sum(sharpes) / len(sharpes)
    print(f"{ps:>10}  {len(vals):>6}  {avg_cagr:>9.2f}%  {avg_sharpe:>10.3f}")

print("\nDone.")