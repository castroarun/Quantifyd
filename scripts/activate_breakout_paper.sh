#!/bin/bash
# One-shot activation of the Breakout ₹10L paper book (research/71).
# Fired by cron at 15:36 IST (after market close) so the backend restart is legal.
# Restarts quantifyd (registers /api/breakout-paper/* + the 15:45 daily job), then seeds
# the book (all-cash start; takes today's entry only if NIFTY>200DMA). Self-removes its cron.
LOG=/tmp/breakout_paper_activate.log
echo "=== activate $(date '+%F %T %Z') ===" >> "$LOG"
sudo /bin/systemctl restart quantifyd >> "$LOG" 2>&1
sleep 25
cd /home/arun/quantifyd && ./venv/bin/python3 -c "import services.breakout_paper as bp; print('SEED', bp.seed())" >> "$LOG" 2>&1
echo "state check:" >> "$LOG"
cd /home/arun/quantifyd && ./venv/bin/python3 -c "import services.breakout_paper as bp,json; st=bp.get_state(); print(json.dumps({k:st[k] for k in ['seeded','gate','nav','cash','n_holdings','data_asof','today_candidates']}, default=str))" >> "$LOG" 2>&1
# self-remove this one-shot cron line
crontab -l 2>/dev/null | grep -v activate_breakout_paper.sh | crontab -
echo "done, cron removed $(date '+%T')" >> "$LOG"
