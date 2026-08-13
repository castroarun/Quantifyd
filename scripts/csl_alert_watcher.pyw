"""TB-CSL trade ALERTS — Windows desktop watcher (research/111 paper/real books).

Polls the app's csl_paper.json every 30s; every ENTRY / EXIT event from the paper
(and future real-money) books pops an ALWAYS-ON-TOP dialog that stays on screen
until you press OK or "Open trades page" (which jumps to /app/straddles#csl-paper).

Runs headless via pythonw (no console). Auto-started at login by a shortcut in
shell:startup (csl_alerts.bat). Last-seen pointer: %LOCALAPPDATA%/csl_alerts_seen.json
Stdlib only — no pip installs needed."""
import json, os, time, traceback, urllib.request, webbrowser

BASE = "http://94.136.185.54:5000"
URL = BASE + "/app/csl_paper.json"
PAGE = BASE + "/app/straddles#csl-paper"
SEEN = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "csl_alerts_seen.json")
POLL = 30  # seconds


def load_seen():
    try:
        return json.load(open(SEEN)).get("last_ts", "")
    except Exception:
        return ""


def save_seen(ts):
    try:
        json.dump({"last_ts": ts}, open(SEEN, "w"))
    except Exception:
        pass


def show_alert(events):
    """One sticky dialog listing all new events. Blocks until the user acts."""
    import tkinter as tk
    root = tk.Tk()
    root.title("TB-CSL trade alert")
    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()
    root.configure(bg="#101418")
    root.protocol("WM_DELETE_WINDOW", lambda: None)  # no X-close: must use a button
    pad = {"padx": 16, "pady": 6}
    tk.Label(root, text="TB-CSL  •  %d new event%s" % (len(events), "s" if len(events) > 1 else ""),
             font=("Segoe UI", 12, "bold"), fg="#E8B04B", bg="#101418").pack(**pad)
    for ev in events:
        col = "#D9534F" if ev.get("source") == "REAL" else "#4BAE7F"
        head = "%s  %s  [%s %s]" % (ev.get("ts", ""), ev.get("book", ""),
                                    "LIVE REAL" if ev.get("source") == "REAL" else "LIVE PAPER",
                                    ev.get("type", ""))
        tk.Label(root, text=head, font=("Segoe UI", 10, "bold"), fg=col, bg="#101418").pack(anchor="w", padx=16)
        tk.Label(root, text=ev.get("msg", ""), font=("Segoe UI", 10), fg="#DDE3EA",
                 bg="#101418", wraplength=520, justify="left").pack(anchor="w", padx=24, pady=(0, 6))
    fr = tk.Frame(root, bg="#101418"); fr.pack(pady=12)

    def open_page():
        webbrowser.open(PAGE)
        root.destroy()

    tk.Button(fr, text="Open trades page", command=open_page, font=("Segoe UI", 10, "bold"),
              bg="#2B5D8A", fg="white", padx=14, pady=4).pack(side="left", padx=8)
    tk.Button(fr, text="OK", command=root.destroy, font=("Segoe UI", 10),
              bg="#333A42", fg="white", padx=22, pady=4).pack(side="left", padx=8)
    # re-assert topmost every 2s in case another window steals it
    def keep_top():
        try:
            root.attributes("-topmost", True); root.after(2000, keep_top)
        except Exception:
            pass
    keep_top()
    root.mainloop()


def main():
    last = load_seen()
    while True:
        try:
            with urllib.request.urlopen(URL + "?t=%d" % time.time(), timeout=15) as r:
                d = json.load(r)
            evs = d.get("events", [])
            new = [e for e in evs if e.get("ts", "") > last]
            if new:
                last = max(e["ts"] for e in new)
                save_seen(last)
                show_alert(new)          # blocks until user acts; that's the point
        except Exception:
            traceback.print_exc()        # invisible under pythonw; harmless
        time.sleep(POLL)


if __name__ == "__main__":
    if not load_seen():                  # first ever run: don't replay history
        try:
            with urllib.request.urlopen(URL, timeout=15) as r:
                evs = json.load(r).get("events", [])
            if evs:
                save_seen(max(e.get("ts", "") for e in evs))
        except Exception:
            pass
    main()
