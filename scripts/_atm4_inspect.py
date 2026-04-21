import sqlite3
for db in ['backtest_data/nas_916_atm4_trading.db', 'backtest_data/nas_atm4_trading.db']:
    print(f'=== {db} ===')
    c = sqlite3.connect('/home/arun/quantifyd/' + db)
    tbls = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(' tables:', tbls)
    for t in tbls:
        try:
            n = c.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
            if n:
                print(f'   {t}: {n} rows')
        except Exception as e:
            print(f'   {t}: err {e}')
