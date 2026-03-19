import subprocess, sys, os, json, time, csv

# Focused grid: 3 MQ bases x 5 V3 configs = 15 runs
mq_bases = [
    dict(ps=30, hsl=0.20, ath=0.15, label="PS30_OPT"),
    dict(ps=30, hsl=0.15, ath=0.10, label="PS30_AGG"),
    dict(ps=20, hsl=0.20, ath=0.15, label="PS20_OPT"),
]

v3_configs = [
    dict(v3_pct=0.00, system="PRIMARY", trail=20.0, max_c=0,  label="NO_V3"),
    dict(v3_pct=0.20, system="PRIMARY", trail=20.0, max_c=5,  label="V3_20_PRIM"),
    dict(v3_pct=0.30, system="PRIMARY", trail=20.0, max_c=5,  label="V3_30_PRIM"),
    dict(v3_pct=0.20, system="SNIPER",  trail=20.0, max_c=5,  label="V3_20_SNIP"),
    dict(v3_pct=0.20, system="PRIMARY", trail=15.0, max_c=10, label="V3_20_T15_C10"),
]

configs = []
for mq in mq_bases:
    for v3 in v3_configs:
        if v3["v3_pct"] == 0.0:
            mq_pct = 0.90
            debt_pct = 0.10
        else:
            mq_pct = 1.0 - v3["v3_pct"] - 0.05
            debt_pct = 0.05
        lbl = mq["label"] + "_" + v3["label"]
        configs.append(dict(
            label=lbl,
            portfolio_size=mq["ps"],
            hard_stop_loss=mq["hsl"],
            rebalance_ath_drawdown=mq["ath"],
            mq_capital_pct=mq_pct,
            v3_capital_pct=v3["v3_pct"],
            debt_capital_pct=debt_pct,
            v3_system_name=v3["system"],
            v3_trail_pct=v3["trail"],
            v3_max_concurrent=v3["max_c"],
        ))

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimization_agent4_combined.csv")
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_combined_worker.py")

# Remove old CSV
if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)

total = len(configs)
overall_start = time.time()
print("=" * 80)
print("COMBINED MQ + V3 OVERLAY OPTIMIZATION (subprocess mode)")
print("Total configurations:", total)
print("=" * 80)
print()
sys.stdout.flush()

for i, cfg_dict in enumerate(configs):
    run_start = time.time()
    label = cfg_dict["label"]
    cfg_json = json.dumps(cfg_dict)

    # Run worker as subprocess (clean memory per run)
    proc = subprocess.run(
        [sys.executable, WORKER, cfg_json, OUTPUT_CSV],
        capture_output=True, text=True, timeout=900,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    run_time = time.time() - run_start
    elapsed = time.time() - overall_start
    avg_per_run = elapsed / (i + 1)
    remaining = avg_per_run * (total - i - 1)

    # Parse worker output
    stdout = proc.stdout.strip()
    if stdout.startswith("OK|"):
        parts = stdout.split("|")
        print("[{:2d}/{}] {} | {} | {:.1f}s | ETA {:.0f}m".format(
            i+1, total, parts[1], " | ".join(parts[2:6]), run_time, remaining/60))
    else:
        stderr_short = proc.stderr[:200] if proc.stderr else ""
        print("[{:2d}/{}] {} | FAIL: {} | {:.1f}s".format(
            i+1, total, label, stdout[:80] or stderr_short[:80], run_time))
    sys.stdout.flush()

total_time = time.time() - overall_start
print()
print("All {} runs completed in {:.1f} minutes ({:.1f}s avg)".format(total, total_time/60, total_time/total))
print()

# Read and display results
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "r") as f:
        reader = csv.DictReader(f)
        results = list(reader)

    # Convert numeric fields
    for r in results:
        for k in ["combined_cagr", "combined_sharpe", "combined_max_dd", "combined_calmar",
                   "v3_trades", "v3_win_rate", "v3_total_pnl", "combined_final",
                   "mq_cagr", "mq_sharpe", "mq_max_dd"]:
            try:
                r[k] = float(r[k])
            except (ValueError, KeyError):
                r[k] = 0

    results_sorted = sorted(results, key=lambda r: r["combined_cagr"], reverse=True)

    print("=" * 120)
    print("TOP 15 CONFIGURATIONS BY COMBINED CAGR")
    print("=" * 120)
    for idx, row in enumerate(results_sorted[:15]):
        print("#{:2d} {:40s} CAGR={:6.2f} Sharpe={:5.2f} MaxDD={:5.2f} Calmar={:5.2f} V3trades={:3.0f} V3WR={:5.1f} V3PnL={:>12s} Final={:>14s}".format(
            idx+1, row["label"], row["combined_cagr"], row["combined_sharpe"],
            row["combined_max_dd"], row["combined_calmar"], row["v3_trades"],
            row["v3_win_rate"], "{:,.0f}".format(row["v3_total_pnl"]),
            "{:,.0f}".format(row["combined_final"])))
    print()

    # Baselines
    print("=" * 80)
    print("MQ-ONLY BASELINES (NO V3 OVERLAY)")
    print("=" * 80)
    baselines = [r for r in results_sorted if "NO_V3" in r["label"]]
    baseline_map = {}
    for row in baselines:
        mq_key = row["label"].replace("_NO_V3", "")
        baseline_map[mq_key] = row["combined_cagr"]
        print("  {:40s} CAGR={:6.2f} Sharpe={:6.2f} MaxDD={:6.2f}".format(
            row["label"], row["combined_cagr"], row["combined_sharpe"], row["combined_max_dd"]))
    print()

    # Value-add
    print("=" * 80)
    print("V3 OVERLAY VALUE-ADD ANALYSIS")
    print("=" * 80)
    for row in results_sorted[:15]:
        if "NO_V3" in row["label"]:
            continue
        mq_key = row["label"].rsplit("_V3", 1)[0]
        base_cagr = baseline_map.get(mq_key, 0)
        delta = row["combined_cagr"] - base_cagr
        sign = "+" if delta >= 0 else ""
        print("  {:40s} Combined={:6.2f} Baseline={:6.2f} Delta={}{:.2f}".format(
            row["label"], row["combined_cagr"], base_cagr, sign, delta))
    print()

    # Add rank to CSV
    CSV_HEADERS = ["rank"] + list(results[0].keys())
    for idx, row in enumerate(results_sorted):
        row["rank"] = idx + 1
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for row in results_sorted:
            writer.writerow(row)
    print("Results saved to:", OUTPUT_CSV)
    print("Total rows:", len(results_sorted))
else:
    print("ERROR: No results CSV found!")
