#!/usr/bin/env bash
# Launch Phase F (vol-BO) and Phase G (CCRB) sweeps on VPS, fully detached.
#
# Designed to be invoked once via:
#   ssh arun@94.136.185.54 'bash /home/arun/quantifyd/scripts/launch_phase_fg_on_vps.sh'
#
# Single-instance enforcement — refuses to start a second copy. Cleans up
# its own pid files on completion.
#
# Both sweeps run in parallel (independent runners, independent CSVs).
#
# Logs:
#   /tmp/phase_f_volbo.log
#   /tmp/phase_g_ccrb.log
#
# Resume: re-run the same command. The runners are idempotent (skip-set
# keyed on (sym, tf, variant, dir, date) in the existing signals CSVs).

set -euo pipefail

cd /home/arun/quantifyd

PHASE_F_PID=/tmp/phase_f_volbo.pid
PHASE_G_PID=/tmp/phase_g_ccrb.pid
PYTHON=venv/bin/python3

abort_if_running() {
    local pidfile="$1"; local label="$2"
    if [ -f "$pidfile" ]; then
        local pid
        pid=$(cat "$pidfile" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[launch] $label already running (PID $pid). Refusing to launch a second copy."
            echo "[launch] To monitor: tail -f /tmp/${label,,}.log"
            echo "[launch] To kill:    kill $pid && rm $pidfile"
            return 1
        fi
        # Stale pidfile
        rm -f "$pidfile"
    fi
    return 0
}

# Per-instance check
abort_if_running "$PHASE_F_PID" "phase_f_volbo" || exit 2
abort_if_running "$PHASE_G_PID" "phase_g_ccrb"  || exit 2

# --- Phase F (vol-BO) ---
nohup "$PYTHON" research/34_nifty500_expansion/scripts/run_volbo_500.py \
    > /tmp/phase_f_volbo.log 2>&1 < /dev/null &
echo $! > "$PHASE_F_PID"
echo "[launch] Phase F (vol-BO) started — PID $(cat $PHASE_F_PID)"

# --- Phase G (CCRB) ---
nohup "$PYTHON" research/34_nifty500_expansion/scripts/run_ccrb_500.py \
    > /tmp/phase_g_ccrb.log 2>&1 < /dev/null &
echo $! > "$PHASE_G_PID"
echo "[launch] Phase G (CCRB)   started — PID $(cat $PHASE_G_PID)"

echo
echo "[launch] Both sweeps launched. Tail logs:"
echo "  ssh arun@94.136.185.54 'tail -f /tmp/phase_f_volbo.log'"
echo "  ssh arun@94.136.185.54 'tail -f /tmp/phase_g_ccrb.log'"
echo
echo "[launch] STATUS files (auto-updated by runners):"
echo "  research/34_nifty500_expansion/VOLBO_RUN_PROGRESS.md"
echo "  research/34_nifty500_expansion/CCRB_RUN_PROGRESS.md"
