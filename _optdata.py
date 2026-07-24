import sqlite3, glob, os
for db in ["backtest_data/nas_trading.db","backtest_data/nas_916_otm_trading.db"]:
    cx=sqlite3.connect(db); cx.row_factory=sqlite3.Row
    for tbl in ["nas_option_snapshots","nas_mtm_snapshots"]:
        try:
            n=cx.execute("SELECT COUNT(*) FROM "+tbl).fetchone()[0]
            cols=[d[1] for d in cx.execute("PRAGMA table_info("+tbl+")").fetchall()]
            print(os.path.basename(db),tbl,"rows=",n,"cols=",cols)
            if n:
                r=cx.execute("SELECT * FROM "+tbl+" ORDER BY rowid DESC LIMIT 1").fetchone()
                print("   last row:", dict(r))
        except Exception as e: print(os.path.basename(db),tbl,"ERR",e)
    cx.close()
# options in central market_data_unified?
cx=sqlite3.connect("backtest_data/market_data.db")
for pat in ["%CE","%PE"]:
    n=cx.execute("SELECT COUNT(DISTINCT symbol) FROM market_data_unified WHERE symbol LIKE ?",(pat,)).fetchone()[0]
    print("market_data_unified option-like symbols",pat,"=",n)
cx.close()