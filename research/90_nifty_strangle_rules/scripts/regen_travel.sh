#!/bin/bash
cd /home/arun/quantifyd
venv/bin/python research/90_nifty_strangle_rules/scripts/run_replay_nsrw.py > /tmp/nsrw_regen.log 2>&1
venv/bin/python research/90_nifty_strangle_rules/scripts/run_replay_paths.py >> /tmp/nsrw_regen.log 2>&1
venv/bin/python research/90_nifty_strangle_rules/scripts/inject_travel.py >> /tmp/nsrw_regen.log 2>&1
