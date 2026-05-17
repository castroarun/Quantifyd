"""Phase 08 — fetch official daily history for NIFTY 100 / NIFTY MIDCAP
150 / NIFTY SMALLCAP 250 from niftyindices.com (public, no credentials),
then build the full strategy-vs-benchmarks YoY table.

niftyindices serves history via a POST JSON endpoint that needs a
browser-like session (cookies from a prior GET). We chunk by 6-month
windows (the API caps long ranges), assemble daily CLOSE, save CSVs,
and regenerate the YoY comparison. Honest: if a series can't be
fetched for a span, that span is left blank — never fabricated.
"""
from __future__ import annotations
import json, time, subprocess
from pathlib import Path
import pandas as pd
# NOTE: python `requests` fails on this Windows box with SSLError (cert
# revocation check). Project-sanctioned workaround (CLAUDE.md): shell out
# to `curl --ssl-no-revoke` (uses Windows cert store; does NOT disable
# TLS verification).

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)

INDICES = {
    "NIFTY 100": "nifty100",
    "NIFTY MIDCAP 150": "niftymidcap150",
    "NIFTY SMALLCAP 250": "niftysmallcap250",
}
START, END = pd.Timestamp("2014-01-01"), pd.Timestamp("2026-12-31")
HDRS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Content-Type": "application/json; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www.niftyindices.com/reports/historical-data",
    "Origin": "https://www.niftyindices.com",
}
EP = ("https://www.niftyindices.com/Backpage.aspx/"
      "getHistoricaldatatabletoString")


def six_month_chunks(s, e):
    cur = s
    while cur <= e:
        nxt = min(cur + pd.DateOffset(months=6) - pd.Timedelta(days=1), e)
        yield cur, nxt
        cur = nxt + pd.Timedelta(days=1)


def _curl_post(name, a, b):
    cinfo = ("{'name':'%s','startDate':'%s','endDate':'%s',"
             "'indexName':'%s'}" % (name, a.strftime("%d-%b-%Y"),
                                    b.strftime("%d-%b-%Y"), name))
    body = json.dumps({"cinfo": cinfo})
    cmd = ["curl", "--ssl-no-revoke", "-s", "-m", "18", "-X", "POST", EP,
           "-H", "Content-Type: application/json; charset=UTF-8",
           "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                 "AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
           "-H", "X-Requested-With: XMLHttpRequest",
           "-H", "Referer: https://www.niftyindices.com/reports/"
                 "historical-data",
           "-H", "Origin: https://www.niftyindices.com",
           "--data", body]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return out.stdout


def fetch_index(sess, name):
    """Walk BACKWARD year-by-year from END to START. niftyindices
    serves recent years fast but STALLS on pre-inception ranges
    (12s timeout -> 0 rows). Stop after 2 consecutive empty years =
    inception reached; earlier years honestly left blank."""
    rows = []
    for yr in range(START.year, END.year + 1):
        a = pd.Timestamp(f"{yr}-01-01")
        b = pd.Timestamp(f"{yr}-12-31")
        got = 0
        # up to 4 attempts with backoff — interior 0s are transient
        # niftyindices stalls, NOT missing data. Do not early-stop.
        for attempt in range(4):
            try:
                txt = _curl_post(name, a, b)
                d = json.loads(txt).get("d", "") if txt.strip() else ""
                data = json.loads(d) if isinstance(d, str) and d else d
                for rec in data or []:
                    dt = rec.get("HistoricalDate") or rec.get("Date")
                    cl = rec.get("CLOSE") or rec.get("Close")
                    if dt and cl not in (None, "", "-"):
                        rows.append((pd.to_datetime(dt, dayfirst=True,
                                     errors="coerce"),
                                     float(str(cl).replace(",", ""))))
                        got += 1
                if got > 0:
                    break
            except Exception as e:
                if attempt == 3:
                    print(f"  ! {name} {yr} err {repr(e)[:60]}", flush=True)
            time.sleep(2 + attempt * 2)
        print(f"  {name} {yr}: {got} days"
              f"{' (UNAVAILABLE after 4 tries)' if got == 0 else ''}",
              flush=True)
        time.sleep(0.3)
    if not rows:
        return None
    s = (pd.Series({d: c for d, c in rows if pd.notna(d)})
         .sort_index())
    s = s[~s.index.duplicated(keep="last")]
    return s


def main():
    series = {}
    for name, slug in INDICES.items():
        print(f"Fetching {name} ...", flush=True)
        s = fetch_index(None, name)
        if s is None or s.empty:
            print(f"  {name}: NO DATA (left blank, not fabricated)")
            continue
        s.to_csv(RES / f"idx_{slug}.csv", header=["close"])
        print(f"  {name}: {len(s)} days {s.index.min().date()}.."
              f"{s.index.max().date()}", flush=True)
        series[name] = s

    # ---- strategy gross per-year + Nifty 50 (NIFTYBEES) + fetched ----
    nav = pd.read_csv(RES / "phase04_chosen_gross_nav.csv",
                      parse_dates=["date"]).set_index("date")["nav"]
    sy = nav.groupby(nav.index.year).last()
    sret = sy.pct_change()
    sret.iloc[0] = sy.iloc[0] / 1.0 - 1

    import sqlite3
    ROOT = Path(__file__).resolve().parents[3]
    con = sqlite3.connect(ROOT / "backtest_data" / "market_data.db")
    nb = pd.read_sql("SELECT date,close FROM market_data_unified WHERE "
                     "timeframe='day' AND symbol='NIFTYBEES' AND close "
                     "IS NOT NULL ORDER BY date", con,
                     parse_dates=["date"]).set_index("date")["close"]
    con.close()

    def yearly_ret(s):
        y = s.groupby(s.index.year).last()
        return (y.pct_change() * 100).round(1)

    tbl = pd.DataFrame({"Strategy_gross_%": (sret * 100).round(1)})
    tbl["Nifty50_%"] = yearly_ret(nb)
    for name, s in series.items():
        col = name.replace("NIFTY ", "Nifty").replace(" ", "")
        tbl[col + "_%"] = yearly_ret(s)
    tbl = tbl.loc[2014:2026]
    tbl.to_csv(RES / "yoy_vs_all_benchmarks.csv")
    print("\n=== STRATEGY (gross) vs BENCHMARKS — yearly % ===")
    print(tbl.to_string())

    # CAGRs for whatever we have
    def cagr(s, lo, hi):
        s = s[(s.index >= lo) & (s.index <= hi)].dropna()
        if len(s) < 2:
            return float("nan")
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        return ((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100
    lo, hi = nav.index[0], nav.index[-1]
    print(f"\nCAGR {lo.date()}..{hi.date()}:")
    print(f"  Strategy(gross) "
          f"{((nav.iloc[-1]/nav.iloc[0])**(365.25/((hi-lo).days))-1)*100:.1f}%"
          f"  Nifty50 {cagr(nb,lo,hi):.1f}%")
    for name, s in series.items():
        print(f"  {name}: {cagr(s,lo,hi):.1f}%")


if __name__ == "__main__":
    main()
