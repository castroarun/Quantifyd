"""2026-06-05: lock NAS enabled flags explicitly so they persist across restart.
Squeeze OFF (8 variants exceed ~52L margin; squeeze rejected + re-spammed every 5-min). 9:16 ON
(holds today's live positions). Post-spread overrides so 916 (which spreads the squeeze base) stays on."""
import io, py_compile, sys

PATH = "config.py"
s = io.open(PATH, encoding="utf-8").read()
ANCHOR = ("    'entry_start_time': '09:16',\n}\n\n"
          "# --- ORB: Opening Range Breakout (Cash Equity Intraday) ---")
BLOCK = ("    'entry_start_time': '09:16',\n}\n\n"
         "# 2026-06-05: NAS enabled flags LOCKED (post-spread). Squeeze OFF — over-margin: all 8\n"
         "# variants exceed ~Rs52L; squeeze entries rejected (insufficient funds) and re-spammed\n"
         "# every 5-min candle. 9:16 family ON — it holds the day's live positions. Re-enable the\n"
         "# squeeze when the book is funded / re-sized.\n"
         "NAS_DEFAULTS['enabled'] = False\n"
         "NAS_ATM_DEFAULTS['enabled'] = False\n"
         "NAS_ATM2_DEFAULTS['enabled'] = False\n"
         "NAS_ATM4_DEFAULTS['enabled'] = False\n"
         "NAS_916_OTM_DEFAULTS['enabled'] = True\n"
         "NAS_916_ATM_DEFAULTS['enabled'] = True\n"
         "NAS_916_ATM2_DEFAULTS['enabled'] = True\n"
         "NAS_916_ATM4_DEFAULTS['enabled'] = True\n\n"
         "# --- ORB: Opening Range Breakout (Cash Equity Intraday) ---")

if "NAS enabled flags LOCKED" in s:
    print("already patched"); sys.exit(0)
n = s.count(ANCHOR)
if n != 1:
    print("ABORT: anchor count %d (need 1)" % n); sys.exit(1)
s = s.replace(ANCHOR, BLOCK)
io.open(PATH, "w", encoding="utf-8").write(s)
py_compile.compile(PATH, doraise=True)
import importlib, config
importlib.reload(config)
print("OK enabled:",
      "Sq-OTM", config.NAS_DEFAULTS.get("enabled"),
      "| Sq-ATM", config.NAS_ATM_DEFAULTS.get("enabled"),
      "| 916-OTM", config.NAS_916_OTM_DEFAULTS.get("enabled"),
      "| 916-ATM", config.NAS_916_ATM_DEFAULTS.get("enabled"))
