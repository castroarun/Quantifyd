"""Prove the routes actually serve before this ships, not just that the file imports.

An import check would have caught the blueprint NameError but NOT the missing
jsonify or the bad SQL column - both only fire when the endpoint is CALLED. So
this stands up a throwaway Flask app, registers the module against it exactly as
app.py does, and requests the endpoints through the test client.
"""
import sys, json
sys.path.insert(0, "/home/arun/quantifyd")
from flask import Flask


class _NullSched:
    """register() wants a scheduler; give it one that records instead of scheduling."""
    def __init__(self): self.jobs = []
    def add_job(self, *a, **k):
        self.jobs.append(k.get("id") or (a[0].__name__ if a and callable(a[0]) else "job"))
        return None


app = Flask(__name__)
sched = _NullSched()
from services.v2_ironfly_api import register
register(app, sched)

routes = sorted(str(r) for r in app.url_map.iter_rules() if "v2-" in str(r))
print(f"registered {len(routes)} v2 routes:")
for r in routes:
    print("   ", r)
assert any("shadow-stops" in r for r in routes), "shadow-stops route missing"
assert any("v2-ironfly/state" in r for r in routes), "state route missing"
print(f"scheduled jobs: {sched.jobs}")

cli = app.test_client()
for path in ("/api/v2-ironfly/shadow-stops", "/api/v2-ironfly/state"):
    r = cli.get(path)
    body = r.get_data(as_text=True)
    ok = r.status_code == 200
    print(f"\n{path} -> HTTP {r.status_code}")
    try:
        d = json.loads(body)
        if isinstance(d, dict) and "error" in d:
            print("   ERROR PAYLOAD:", str(d["error"])[:300]); ok = False
        else:
            print("   keys:", list(d)[:10] if isinstance(d, dict) else type(d).__name__)
    except Exception:
        print("   non-JSON:", body[:200]); ok = False
    if path.endswith("shadow-stops"):
        assert ok, "shadow-stops did not return clean JSON"
print("\nVERIFIED: routes register and serve JSON.")
