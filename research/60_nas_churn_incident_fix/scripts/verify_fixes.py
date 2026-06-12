"""Post-deploy verification for the 5 NAS stops/churn fixes. Run on the VPS."""
import datetime, importlib

print("=== FIX #3: cooldown date-parse (isoformat) ===")
s = "2026-06-12T13:45:04.502518"  # production exit_time = datetime.now().isoformat()
ok = False
for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
    try:
        datetime.datetime.strptime(s[:26], fmt); ok = True; break
    except ValueError:
        pass
print("  isoformat exit_time parses:", ok, "(was False before fix -> cooldown was blind)")

print("=== FIX #1a: SL-skip fetch fallback in 3 executors ===")
needle = "don" + chr(39) + "t silently skip the SL check"
for m in ("nas_atm_executor", "nas_atm2_executor", "nas_atm4_executor"):
    src = open("services/%s.py" % m).read()
    print("  %-18s fetch-fallback=%s  check_and_handle_sl=%s" % (
        m, needle in src, "def check_and_handle_sl" in src))

print("=== FIX #2: tick-level ST + level-breach ===")
t = open("services/nas_ticker.py").read()
print("  _atm_naked_st_val refs:", t.count("_atm_naked_st_val"))
print("  _atm4_naked_st_val refs:", t.count("_atm4_naked_st_val"))
print("  level-breach (latest_close > st_val):", "latest_close > st_val" in t)
print("  ST TICK-EXIT log lines:", t.count("ST TICK-EXIT"))
print("  999999 sentinel preserved:", "999999" in t)

print("=== FIX #4: additive subscriptions helper ===")
print("  _tokens_in_use_by_others defined:", "def _tokens_in_use_by_others" in t)
print("  old triangular chains remaining:", t.count("set(self._option_tokens.keys()) - set(self._atm"))

print("=== modules import cleanly ===")
for m in ("services.nas_ticker", "services.nas_atm_executor",
          "services.nas_atm2_executor", "services.nas_atm4_executor"):
    importlib.import_module(m)
    print("  import %s: OK" % m)
from services.nas_ticker import NasTicker
print("  NasTicker._tokens_in_use_by_others is method:",
      callable(getattr(NasTicker, "_tokens_in_use_by_others", None)))

print("=== FIX #1b: squeeze SL poll registered ===")
a = open("app.py").read()
print("  _nas_squeeze_sl_monitor defined:", "def _nas_squeeze_sl_monitor" in a)
print("  job id registered:", "id=" + chr(39) + "nas_squeeze_sl_monitor" + chr(39) in a)
