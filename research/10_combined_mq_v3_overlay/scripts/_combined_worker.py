import sys, os, json, time, logging, csv

logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.combined_mq_v3_engine import CombinedMQV3Engine, CombinedConfig

# Read config from command line arg (JSON)
cfg_json = sys.argv[1]
cfg_dict = json.loads(cfg_json)
output_csv = sys.argv[2]

run_start = time.time()
label = cfg_dict["label"]

try:
    config = CombinedConfig(
        start_date="2023-01-01",
        end_date="2025-12-31",
        initial_capital=10_000_000,
        mq_capital_pct=cfg_dict["mq_capital_pct"],
        v3_capital_pct=cfg_dict["v3_capital_pct"],
        debt_capital_pct=cfg_dict["debt_capital_pct"],
        v3_system_name=cfg_dict["v3_system_name"],
        v3_trail_pct=cfg_dict["v3_trail_pct"],
        v3_max_concurrent=cfg_dict["v3_max_concurrent"],
        portfolio_size=cfg_dict["portfolio_size"],
        hard_stop_loss=cfg_dict["hard_stop_loss"],
        rebalance_ath_drawdown=cfg_dict["rebalance_ath_drawdown"],
    )

    engine = CombinedMQV3Engine(config)
    result = engine.run()
    run_time = time.time() - run_start

    row = {
        "label": label,
        "portfolio_size": cfg_dict["portfolio_size"],
        "hard_stop_loss": cfg_dict["hard_stop_loss"],
        "ath_drawdown": cfg_dict["rebalance_ath_drawdown"],
        "mq_pct": round(cfg_dict["mq_capital_pct"] * 100, 1),
        "v3_pct": round(cfg_dict["v3_capital_pct"] * 100, 1),
        "debt_pct": round(cfg_dict["debt_capital_pct"] * 100, 1),
        "v3_system": cfg_dict["v3_system_name"],
        "v3_trail": cfg_dict["v3_trail_pct"],
        "v3_max_concurrent": cfg_dict["v3_max_concurrent"],
        "mq_cagr": result.mq_result.cagr,
        "mq_sharpe": result.mq_result.sharpe_ratio,
        "mq_max_dd": result.mq_result.max_drawdown,
        "v3_trades": result.v3_total_trades,
        "v3_win_rate": result.v3_win_rate,
        "v3_total_pnl": round(result.v3_total_pnl, 0),
        "v3_profit_factor": result.v3_profit_factor,
        "combined_cagr": result.combined_cagr,
        "combined_sharpe": result.combined_sharpe,
        "combined_max_dd": result.combined_max_drawdown,
        "combined_calmar": result.combined_calmar,
        "combined_final": round(result.combined_final, 0),
        "combined_total_return_pct": result.combined_total_return_pct,
    }

    # Append to CSV
    headers = list(row.keys())
    file_exists = os.path.exists(output_csv)
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print("OK|{}|CAGR={:.2f}|Sharpe={:.2f}|MaxDD={:.2f}|V3={:d}|{:.1f}s".format(
        label, result.combined_cagr, result.combined_sharpe,
        result.combined_max_drawdown, result.v3_total_trades, run_time))

except Exception as e:
    import traceback
    run_time = time.time() - run_start
    print("ERR|{}|{}|{:.1f}s".format(label, str(e)[:80], run_time))
    traceback.print_exc()
