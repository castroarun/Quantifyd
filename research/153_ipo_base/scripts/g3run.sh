#!/bin/bash
cd /home/arun/quantifyd
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
R=research/153_ipo_base
for t in adopted mid wide; do
  nice -n 10 venv/bin/python -u $R/scripts/ipo_g3.py $R/results/spec_$t.json $t > $R/results/g3_$t.log 2>&1
  echo "=== $t done ==="
done
echo ALLDONE
