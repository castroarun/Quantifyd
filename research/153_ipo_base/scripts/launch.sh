#!/bin/bash
cd /home/arun/quantifyd
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec nice -n 10 venv/bin/python -u research/153_ipo_base/scripts/ipo_sweep.py "$@"
