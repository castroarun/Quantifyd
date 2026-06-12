---
description: Run the NAS Live Guardian — validate the live options book, hunt the 06-12 failure modes, report issues by itself.
argument-hint: "[monitor] (loop every 5 min) | [firedrill] | (blank = one validation pass)"
allowed-tools: Bash, Read, Grep, Glob, Task
---

Invoke the **nas-live-guardian** agent to validate the live NAS options book on the VPS.

Arguments: `$ARGUMENTS`

Interpret the argument:
- **blank or `firedrill`** → run a single full validation pass (use `--firedrill`). Spawn the
  `nas-live-guardian` agent with: "Run one full NAS Live Guardian pass with --firedrill,
  interpret every check, investigate any FAIL or unexpected WARN, and report the verdict with
  per-variant open legs, combined day-P&L, and any issue + root-cause + recommended action."
- **`monitor`** → continuous monitoring. Spawn the `nas-live-guardian` agent with: "Monitor the
  live NAS book every ~5 minutes while the market is open (09:15–15:30 IST). First pass with
  --firedrill, subsequent passes without. Report a tight status each cycle and escalate any
  FAIL/unexpected-WARN immediately with evidence + recommended action (freeze if live money is
  at risk)." Use ScheduleWakeup to pace the 5-minute cadence.

Always run via the agent so the investigation logic and the 06-12 failure-mode knowledge are
applied — do not just print the raw harness output. The harness itself is
`scripts/nas_live_guardian.py` on the VPS (`ssh arun@94.136.185.54 'cd /home/arun/quantifyd &&
./venv/bin/python3 scripts/nas_live_guardian.py --firedrill'`).
