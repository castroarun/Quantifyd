# Laptop Replacement Recovery — Zero-Loss Procedure

**Goal:** if the current laptop dies, a brand-new laptop should be productive
in under 30 minutes with no data loss and no broken pointers.

This is achievable because **the VPS at `94.136.185.54` is canonical for
everything that matters** — code is on GitHub, data lives on VPS, live
trading runs on VPS, backtests run on VPS. The laptop is dev-only.

---

## What's on the laptop that's *not* recoverable from VPS or GitHub

Three things, all small and easy to recreate:

| Item | Location | Recreation cost |
|---|---|---|
| GitHub PAT | `C:\Users\Castro\.claude\CLAUDE.md` | Generate new at https://github.com/settings/tokens |
| SSH keypair to VPS | `~/.ssh/id_rsa` + `id_rsa.pub` | `ssh-keygen -t rsa -b 4096`, then push pub key to VPS via password auth (one-time) |
| VPS sudo password | env var or memory | Already on VPS (`/etc/sudoers.d/*` allowlists), this is just for shell login |

**Everything else** (code, market data, position state, equity history,
config files, scheduled jobs) lives on VPS or GitHub.

---

## Step-by-step recovery on a fresh laptop

### 0. Install prerequisites (Windows 10/11)

- Git for Windows (includes Git Bash + OpenSSH)
- Python 3.12+ from python.org
- Node.js 20+ for the React frontend
- VS Code (optional, for editing)
- Claude Code CLI (optional)

### 1. Clone the repo

```powershell
mkdir C:\Users\<NewUser>\Documents\Projects
cd C:\Users\<NewUser>\Documents\Projects
git clone https://github.com/castroarun/Quantifyd.git Covered_Calls
cd Covered_Calls
```

### 2. Generate fresh SSH key + authorize on VPS

```powershell
ssh-keygen -t rsa -b 4096 -f $env:USERPROFILE\.ssh\id_rsa -N ""
# One-time: paste pub key into VPS authorized_keys (use VPS password for this single bootstrap)
type $env:USERPROFILE\.ssh\id_rsa.pub | ssh arun@94.136.185.54 "cat >> ~/.ssh/authorized_keys"
# Verify passwordless access:
ssh -o BatchMode=yes arun@94.136.185.54 'echo OK'
```

### 3. Install Python deps + build frontend

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
cd frontend
npm install
npm run build
cd ..
```

### 4. (Optional) Pull a snapshot of `market_data.db` for offline backtest dev

The VPS has the canonical DB. The laptop only needs a copy if you want to
**develop new strategy code** against real historical data without an
internet round-trip every query.

```powershell
python scripts/pull_market_data_from_vps.py
```

This rsyncs `/home/arun/quantifyd/backtest_data/market_data.db` from the VPS
to the laptop's `backtest_data/market_data.db`. Re-run anytime to refresh.

If you skip this step, laptop code that touches `market_data.db` will simply
fail until you run a backtest on VPS or pull the snapshot. **Live trading
is unaffected** — it runs on VPS where the DB always exists.

### 5. (Optional) Install Claude Code memory + global instructions

If you want Claude Code on the new laptop to have the same context:

- Copy `C:\Users\<OldUser>\.claude\CLAUDE.md` from a previous backup
  *or* re-run `/init` after the first `claude` invocation in the project
- Memory files at `C:\Users\<OldUser>\.claude\projects\<project-id>\memory\`
  rebuild over time as you work — they're not strictly required

### 6. Verify

```powershell
# Verify VPS is healthy
ssh arun@94.136.185.54 'systemctl is-active quantifyd'   # → active
ssh arun@94.136.185.54 'curl -s http://127.0.0.1:5000/api/n500m/state | head -c 200'

# Verify laptop dev workflow works
.\venv\Scripts\activate
python -c "from services.n500m_configs import load_all_configs; print(len(load_all_configs()))"
```

---

## What you should *never* do on the new laptop

1. **Run Kite data downloads.** `services/data_manager.py` will refuse
   (raises `RuntimeError`) unless `ALLOW_LOCAL_DATA_WRITE=1` is set.
   If you need fresh data, run the download on VPS via paramiko.

2. **Run a backtest on laptop without VPS being unavailable.** Use VPS for
   sweeps. They're slower but uninterrupted. Laptop sweeps die from
   suspends and OOMs (proven 2026-05-07).

3. **Hold live position state in laptop memory.** Live trading runs only
   on VPS. The laptop is for editing code that gets pushed to VPS.

---

## Architectural invariants (so you can audit at a glance)

| Concern | Convention |
|---|---|
| Code | GitHub `castroarun/Quantifyd` (canonical) |
| Market data DB | VPS `/home/arun/quantifyd/backtest_data/market_data.db` (canonical, 4.85 GB+, growing) |
| Live trading services (ORB, MST, KC6, N500M) | VPS systemd `quantifyd.service` |
| Backtest sweeps (any `research/<NN>/scripts/run_*.py`) | VPS, launched via paramiko/ssh |
| In-laptop paths in code | None (all relative or env-var-driven) |
| Hardcoded `C:\Users\...` paths in services/ | None as of 2026-05-07 |
| Kite token | VPS at `backtest_data/access_token.json`, refreshed daily 08:55 IST cron |
| GitHub PAT | Per-user, regenerated on laptop replacement |
| SSH keys | Per-user, regenerated on laptop replacement |

If anything in the production code violates these invariants, fix it —
otherwise the laptop crash story breaks.
