import sqlite3
import os

DB = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', '..', 'backtest_data', 'market_data.db'))
con = sqlite3.connect(DB)
cur = con.cursor()
cur.execute("SELECT symbol FROM market_data_unified WHERE timeframe='5minute' AND symbol LIKE '%NIFTY%' GROUP BY symbol")
print('NIFTY:', cur.fetchall())
cur.execute("SELECT COUNT(DISTINCT symbol) FROM market_data_unified WHERE timeframe='5minute' AND date>='2024-03-18'")
print('5min syms:', cur.fetchone()[0])
cur.execute("SELECT symbol, COUNT(*) AS n FROM market_data_unified WHERE timeframe='5minute' AND date>='2024-03-18' GROUP BY symbol HAVING n>=20000 ORDER BY n DESC LIMIT 5")
print('top 5:', cur.fetchall())
con.close()
