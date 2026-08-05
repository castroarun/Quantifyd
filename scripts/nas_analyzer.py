#!/usr/bin/env python3
"""NAS periodic system analyzer — automated performance / drift / health review of the live NAS book.
Reads each system's trade log, computes a per-system + book scorecard, recent-vs-all drift (edge decay),
and a RED/AMBER/GREEN flag with a one-line read. Writes:
  - static/app/nas_analyzer.json   (for the /app/reports page)
  - backtest_data/reports/nas_analyzer_<DATE>.md  (durable dated report)
Cron: post-close daily (15:45 IST). Deterministic — no Claude needed; the judgment layer can read the JSON."""
import sqlite3
import json
import os
import datetime

ROOT = "/home/arun/quantifyd"
SYSTEMS = [
    ("nas_916_atm", "NIFTY 9:16 ATM (trail)"),
    ("nas_916_atm2", "NIFTY 9:16 ATM2 (move-stop)"),
    ("nas_916_atm4", "NIFTY 9:16 ATM4 (roll)"),
    ("sensex_atm", "SENSEX ATM (trail)"),
    ("sensex_atm2", "SENSEX ATM2 (move-stop)"),
    ("sensex_atm4", "SENSEX ATM4 (roll)"),
]
RECENT_DAYS = 10  # recent window = last N distinct trade-dates


def stats(pnls):
    if not pnls:
        return dict(n=0, total=0, avg=0, win=0, avgwin=0, avgloss=0, pf=0, worst=0, best=0)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    sw, sl = sum(wins), abs(sum(losses)) or 1
    return dict(
        n=len(pnls), total=round(sum(pnls)), avg=round(sum(pnls) / len(pnls)),
        win=round(100 * len(wins) / len(pnls)),
        avgwin=round(sw / len(wins)) if wins else 0,
        avgloss=round(sum(losses) / len(losses)) if losses else 0,
        pf=round(sw / sl, 2), worst=round(min(pnls)), best=round(max(pnls)),
    )


def analyze_system(db):
    path = "%s/backtest_data/%s_trading.db" % (ROOT, db)
    if not os.path.exists(path):
        return None
    c = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    try:
        rows = c.execute("SELECT trade_date, net_pnl FROM nas_atm_trades "
                         "WHERE exit_reason NOT LIKE 'COUNTERFACTUAL%' OR exit_reason IS NULL "
                         "ORDER BY trade_date").fetchall()
    except Exception:
        rows = []
    c.close()
    if not rows:
        return dict(all=stats([]), recent=stats([]), rag="—", read="no trades logged")
    all_p = [r[1] or 0 for r in rows]
    dates = sorted({r[0] for r in rows})
    recent_dates = set(dates[-RECENT_DAYS:])
    recent_p = [r[1] or 0 for r in rows if r[0] in recent_dates]
    A, R = stats(all_p), stats(recent_p)
    # RAG: edge health = recent avg vs full avg
    if R["avg"] < 0:
        rag, read = "RED", "recent %d trades net %+d — losing lately" % (R["n"], R["total"])
    elif A["avg"] > 0 and R["avg"] < 0.5 * A["avg"]:
        rag, read = "AMBER", "recent avg %+d/tr vs all %+d/tr — edge softening" % (R["avg"], A["avg"])
    elif R["avg"] >= A["avg"] > 0:
        rag, read = "GREEN", "recent %+d/tr >= all %+d/tr — holding up" % (R["avg"], A["avg"])
    else:
        rag, read = "AMBER", "recent %+d/tr — watch" % R["avg"]
    return dict(all=A, recent=R, rag=rag, read=read)


def main():
    now = datetime.datetime.now()
    out = {"generated": now.strftime("%Y-%m-%d %H:%M:%S"), "recent_days": RECENT_DAYS, "systems": []}
    book_all, book_recent = 0, 0
    for db, label in SYSTEMS:
        s = analyze_system(db)
        if s is None:
            continue
        s.update(db=db, label=label)
        out["systems"].append(s)
        book_all += s["all"]["total"]
        book_recent += s["recent"]["total"]
    out["book"] = {"all_total": book_all, "recent_total": book_recent}

    reds = [s for s in out["systems"] if s["rag"] == "RED"]
    ambers = [s for s in out["systems"] if s["rag"] == "AMBER"]
    out["verdict"] = "RED" if reds else ("AMBER" if ambers else "GREEN")

    os.makedirs("%s/static/app" % ROOT, exist_ok=True)
    os.makedirs("%s/backtest_data/reports" % ROOT, exist_ok=True)
    with open("%s/static/app/nas_analyzer.json" % ROOT, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    # durable markdown
    md = ["# NAS System Analyzer — %s" % now.strftime("%Y-%m-%d %H:%M IST"),
          "", "**Book verdict: %s** · all-time net %+d · last-%d-day net %+d" %
          (out["verdict"], book_all, RECENT_DAYS, book_recent), "",
          "| System | RAG | all net | all avg/tr | win% | PF | recent net | recent avg | read |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in out["systems"]:
        A, R = s["all"], s["recent"]
        md.append("| %s | **%s** | %+d | %+d | %d%% | %.2f | %+d | %+d | %s |" %
                  (s["label"], s["rag"], A["total"], A["avg"], A["win"], A["pf"], R["total"], R["avg"], s["read"]))
    mdpath = "%s/backtest_data/reports/nas_analyzer_%s.md" % (ROOT, now.strftime("%Y-%m-%d"))
    with open(mdpath, "w") as f:
        f.write("\n".join(md) + "\n")
    print("verdict %s | book all %+d recent %+d | wrote nas_analyzer.json + %s" %
          (out["verdict"], book_all, book_recent, os.path.basename(mdpath)))


if __name__ == "__main__":
    main()
