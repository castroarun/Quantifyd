"""Ingest AlgoTest trade exports into backtest_data/algotest_studies.db.

One row per option-strategy trade (the parent row of each straddle), keeping GROSS
P/L and premium turnover separately so the cost model stays a query-time parameter
rather than being baked into the stored data.

Filenames:  algotest_<SL>SL.csv            -> NIFTY
            algotest_<SL>SL_<Index>.csv    -> that index
"""
import csv, os, re, sqlite3, sys, glob, datetime as dt
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB   = os.path.join(ROOT, "backtest_data", "algotest_studies.db")

# lot size per index -> lots = Quantity / lot_size
LOT = {"NIFTY": 65, "SENSEX": 20, "BANKNIFTY": 15}

# Ex-ante event days: Union Budget + General-Election result/reaction.
EVENT = {"2021-02-01","2022-02-01","2023-02-01","2024-02-01","2024-07-23",
         "2025-02-01","2026-02-01","2026-02-02","2024-06-03","2024-06-04"}

DDL = """
CREATE TABLE IF NOT EXISTS at_runs (
  run_id TEXT PRIMARY KEY, index_name TEXT, sl_pct REAL, sl_unit TEXT,
  entry_time TEXT, exit_time TEXT, square_off TEXT, trail_be TEXT,
  qty INTEGER, lot_size INTEGER, lots REAL,
  n_trades INTEGER, period_start TEXT, period_end TEXT,
  gross_total REAL, turnover_total REAL, source_file TEXT, loaded_at TEXT);
CREATE TABLE IF NOT EXISTS at_trades (
  run_id TEXT, entry_date TEXT, year INTEGER, weekday TEXT, dte INTEGER,
  strike REAL, entry_time TEXT, exit_time TEXT,
  gross REAL, turnover REAL, n_legs INTEGER, is_event INTEGER,
  PRIMARY KEY (run_id, entry_date, strike));
CREATE INDEX IF NOT EXISTS ix_at_trades_run  ON at_trades(run_id);
CREATE INDEX IF NOT EXISTS ix_at_trades_dte  ON at_trades(run_id, dte);
CREATE INDEX IF NOT EXISTS ix_at_trades_year ON at_trades(run_id, year);
"""
F = lambda s: float(s) if str(s).strip() else 0.0

def parse(path):
    base = os.path.basename(path)
    m = re.match(r"algotest_(\d+)SL(?:_(\w+))?\.csv$", base, re.I)
    if not m: return None
    sl = float(m.group(1))
    idx = (m.group(2) or "NIFTY").upper()
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    kids = defaultdict(list)
    for r in rows:
        if "." in r["Index"]: kids[r["Index"].split(".")[0]].append(r)
    trades, qty = [], None
    for r in rows:
        if "." in r["Index"]: continue
        k = kids.get(r["Index"], [])
        if not k: continue
        ed = dt.date.fromisoformat(r["Entry-Date"])
        xd = dt.date.fromisoformat(k[0]["ExpiryDate"])
        turn = sum((F(x["Entry-Price"]) + F(x["ExitPrice"])) * float(x["Quantity"]) for x in k)
        qty = qty or int(float(k[0]["Quantity"]))
        trades.append(dict(entry_date=ed.isoformat(), year=ed.year, weekday=ed.strftime("%a"),
                           dte=(xd-ed).days, strike=float(k[0]["StrikePrice"]),
                           entry_time=r["Entry-Time"], exit_time=r["ExitTime"],
                           gross=F(r["P/L"]), turnover=turn, n_legs=len(k),
                           is_event=1 if ed.isoformat() in EVENT else 0))
    if not trades: return None
    lot = LOT.get(idx, 1)
    ds = sorted(t["entry_date"] for t in trades)
    run_id = f"{idx}-{int(sl)}SL"
    run = dict(run_id=run_id, index_name=idx, sl_pct=sl, sl_unit="Percent(%)",
               entry_time=trades[0]["entry_time"], exit_time="15:15:00",
               square_off="Partial", trail_be="ON (All Legs)",
               qty=qty, lot_size=lot, lots=round(qty/lot, 2),
               n_trades=len(trades), period_start=ds[0], period_end=ds[-1],
               gross_total=sum(t["gross"] for t in trades),
               turnover_total=sum(t["turnover"] for t in trades),
               source_file=base, loaded_at=dt.datetime.now().isoformat(timespec="seconds"))
    return run, trades

def main(src):
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = sqlite3.connect(DB); con.executescript(DDL)
    files = sorted(glob.glob(os.path.join(src, "algotest_*.csv")))
    nr = nt = 0
    for p in files:
        got = parse(p)
        if not got:
            print(f"  skip {os.path.basename(p)}"); continue
        run, trades = got
        con.execute("DELETE FROM at_runs   WHERE run_id=?", (run["run_id"],))
        con.execute("DELETE FROM at_trades WHERE run_id=?", (run["run_id"],))
        con.execute("INSERT INTO at_runs ({}) VALUES ({})".format(
            ",".join(run), ",".join("?"*len(run))), tuple(run.values()))
        con.executemany(
            "INSERT OR REPLACE INTO at_trades (run_id,entry_date,year,weekday,dte,strike,"
            "entry_time,exit_time,gross,turnover,n_legs,is_event) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [(run["run_id"], t["entry_date"], t["year"], t["weekday"], t["dte"], t["strike"],
              t["entry_time"], t["exit_time"], t["gross"], t["turnover"], t["n_legs"], t["is_event"])
             for t in trades])
        nr += 1; nt += len(trades)
        print(f"  {run['run_id']:>14}  {len(trades):>5} trades  {run['period_start']}..{run['period_end']}"
              f"  qty={run['qty']} ({run['lots']} lots)")
    con.commit()
    print(f"\n{nr} runs / {nt} trades -> {DB}")
    for row in con.execute("SELECT index_name, COUNT(*), MIN(sl_pct), MAX(sl_pct) FROM at_runs GROUP BY index_name"):
        print(f"  {row[0]}: {row[1]} runs, SL {row[2]:.0f}%..{row[3]:.0f}%")
    con.close()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads"))
