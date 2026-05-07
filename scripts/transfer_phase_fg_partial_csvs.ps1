# Transfer Phase F + G partial CSVs from laptop to VPS.
#
# Why: laptop's local sweeps were 132/218 (vol-BO) and 56/218 (CCRB)
#      complete before the processes crashed today. Transferring the
#      partial CSVs to VPS lets the resumed sweep skip already-processed
#      stocks/cells (skip-set is keyed on (sym, tf, variant, dir, date))
#      and avoids ~5 hours of re-compute.
#
# Run this BEFORE invoking launch_phase_fg_on_vps.sh.
#
# To cold-start on VPS instead (no laptop transfer), skip this script.
# Cost: ~3-4h extra vol-BO + ~5-6h extra CCRB at the start of each run.

$ErrorActionPreference = "Stop"
$VPS = "arun@94.136.185.54"
$VpsResultsDir = "/home/arun/quantifyd/research/34_nifty500_expansion/results"
$LocalResultsDir = "C:\Users\Castro\Documents\Projects\Covered_Calls\research\34_nifty500_expansion\results"

# Sanity check — confirm VPS results dir exists
$dirCheck = ssh -o ConnectTimeout=10 $VPS "[ -d '$VpsResultsDir' ] && echo OK || echo MISSING"
if ($dirCheck -ne "OK") {
    Write-Error "VPS results dir missing: $VpsResultsDir"
    exit 1
}

# vol-BO partial
$volboCsv = "$LocalResultsDir\volbo_signals.csv"
if (Test-Path $volboCsv) {
    $size = (Get-Item $volboCsv).Length
    Write-Host "[transfer] volbo_signals.csv: $([math]::Round($size/1MB,1)) MB"
    scp -C $volboCsv "${VPS}:${VpsResultsDir}/volbo_signals.csv"
} else {
    Write-Host "[transfer] no volbo_signals.csv on laptop — VPS will cold-start"
}

# CCRB partial
$ccrbCsv = "$LocalResultsDir\ccrb_signals.csv"
if (Test-Path $ccrbCsv) {
    $size = (Get-Item $ccrbCsv).Length
    Write-Host "[transfer] ccrb_signals.csv: $([math]::Round($size/1MB,1)) MB"
    scp -C $ccrbCsv "${VPS}:${VpsResultsDir}/ccrb_signals.csv"
} else {
    Write-Host "[transfer] no ccrb_signals.csv on laptop — VPS will cold-start"
}

Write-Host
Write-Host "[transfer] done. To launch sweeps on VPS:"
Write-Host "  ssh $VPS 'bash /home/arun/quantifyd/scripts/launch_phase_fg_on_vps.sh'"
