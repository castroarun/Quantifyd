# Scripts

## Backup / Restore

### Backup `backtest_data/` to a GitHub release

```bash
export GITHUB_TOKEN=ghp_...       # needs repo scope
scripts/backup_to_github_release.sh
```

Creates a tarball excluding `access_token.json` and `fundamentals_cache/`, then
uploads it as an asset on a new GitHub release tagged `backup-YYYYMMDD_HHMMSS`
(marked as pre-release so it doesn't clutter the "latest" view).

- Typical compressed size: ~1.2 GB (down from ~4.7 GB uncompressed).
- Assets > 1.9 GB are auto-split into `*.part` files and reassembled on restore.
- Safe to run unattended (e.g. from cron — see below).

### Restore from a GitHub release

```bash
export GITHUB_TOKEN=ghp_...
scripts/restore_from_github_release.sh                    # latest backup-* tag
scripts/restore_from_github_release.sh backup-20260420_161530
scripts/restore_from_github_release.sh backup-20260420_161530 --force
```

Downloads all release assets, reassembles if split, extracts into the repo root.

### Cron (VPS) — daily backup after market close

Add to `crontab -e`:

```
# Backup backtest_data/ daily at 16:00 IST (Mon-Fri)
0 16 * * 1-5 GITHUB_TOKEN=ghp_... /home/arun/quantifyd/scripts/backup_to_github_release.sh >> /home/arun/quantifyd/logs/backup.log 2>&1
```

Keep the token in a restricted env file and source it rather than hardcoding in
the crontab if possible:

```
0 16 * * 1-5 . /home/arun/.github_token && /home/arun/quantifyd/scripts/backup_to_github_release.sh >> /home/arun/quantifyd/logs/backup.log 2>&1
```

Where `/home/arun/.github_token` (chmod 600) contains `export GITHUB_TOKEN=ghp_...`.

### What's committed to git vs uploaded as release

| Path | Where | Why |
|---|---|---|
| `backtest_data/*_trading.db` | git | Live trading state, ~100 KB each, version history useful |
| `backtest_data/backtest_results.db`, `mq_agent.db`, `tactical_pool.db` | git | Small strategy outputs |
| `backtest_data/*.json` | git | Optimization results, instrument tokens, config |
| `backtest_data/market_data.db` | release | 2.1 GB, rebuildable from Kite |
| `backtest_data/options_data.db` | release | 1.4 GB, rebuildable |
| `backtest_data/*.pkl` | release | 500-800 MB each, derived data |
| `backtest_data/fundamentals_cache/` | neither | Rebuildable from screener/Kite |
| `backtest_data/access_token.json` | neither | **Secret** — never commit |
