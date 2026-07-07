import json, shutil, datetime as dt
MP = "backtest_data/nas_day_matrix.json"
m = json.load(open(MP)); s = m["systems"]
for k in ["nas_916_atm", "nas_916_atm2", "nas_916_atm4"]:      # 916 -> live all weekdays
    s[k]["live"] = True
    for d in ["0", "1", "2", "3", "4"]:
        s[k]["dte"][d] = True
for k in ["nas_atm", "nas_atm2", "nas_atm4"]:                   # squeeze -> paper (live off, shadow kept)
    s[k]["live"] = False
shutil.copy(MP, MP + ".bak_916allday_live")
json.dump(m, open(MP, "w"), indent=2)
MM = "backtest_data/nas_master_mode.json"
shutil.copy(MM, MM + ".bak_prelive")
json.dump({"mode": "live"}, open(MM, "w"))
print("APPLIED: 916 all-weekday live; squeeze live=False; master-mode=live\n")

from services.nas_day_matrix import gate, load
m2 = load()
S916 = ["nas_916_atm", "nas_916_atm2", "nas_916_atm4"]
SSQ = ["nas_atm", "nas_atm2", "nas_atm4"]
allok = True
for d, lbl in [(dt.date(2026,7,8),"Wed"),(dt.date(2026,7,9),"Thu"),(dt.date(2026,7,10),"Fri"),(dt.date(2026,7,13),"Mon"),(dt.date(2026,7,14),"Tue")]:
    def mode(k):
        g = gate(k, d, None, m2, gap=(0.0, False, False)); return g["mode"] if g["allow"] else "off"
    live = [k for k in S916+SSQ if mode(k) == "live"]
    sq = [mode(k) for k in SSQ]
    ok = set(live) == set(S916) and all(x == "paper" for x in sq)
    allok = allok and ok
    print("%s %s: LIVE=%s | squeeze=%s  %s" % (lbl, d, [x.replace("nas_916_","916-") for x in live], sq, "OK" if ok else "*** FAIL ***"))
print("\nMASTER-MODE:", json.load(open(MM)))
import config as C
print("916 live lots:", [getattr(C,n).get("lots_per_leg") for n in ["NAS_916_ATM_DEFAULTS","NAS_916_ATM2_DEFAULTS","NAS_916_ATM4_DEFAULTS"]], "(paper-shadow lots 10)")
print("\n=== VERDICT:", "ALL CHECKS PASS — only the 3x 916 go live, every weekday; squeeze paper" if allok else "*** CHECK FAILED — REVIEW ***", "===")
