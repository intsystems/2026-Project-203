#!/usr/bin/env bash
# =============================================================================
# The eight "hero" runs of the paper on the NanoGPT speedrun (record #40),
# 8xH100, one run per optimizer at its starting learning rate.
#
#   bash run_all.sh                      # all eight, full length (~25 min total)
#   bash run_all.sh Muon SignMuon        # only these
#   NANOGPT_ITERS=200 bash run_all.sh    # short smoke pass first (recommended)
#   NPROC=1 SCRIPT=train_gpt_a100.py bash run_all.sh    # single A100/H100
#
# Learning rates come from OPTIMIZER_CONFIG in the training script (all "round"
# numbers, anchored on record #40's Muon lr=0.06 -- see the comment there for
# why the sign family starts at 0.03 rather than at some 1e-4).  Override a
# single run with SIGNMUON_LR=... and one optimizer name.
#
# Logs land in $LOG_DIR (default ./logs) as "<Opt>_lr<lr>_<hash>.txt".
# Turn them into data with:   python parse_logs.py logs -o results
# =============================================================================
set -uo pipefail

SCRIPT="${SCRIPT:-train_gpt.py}"
NPROC="${NPROC:-8}"
LOG_DIR="${LOG_DIR:-logs}"
export LOG_DIR

# Order: cheapest-to-interpret first. Muon leads so a broken environment is
# caught against the known record-#40 curve before any method under study runs.
ALL_OPTS=(
  Muon              # reference, == record #40 exactly (lr 0.06)
  SignSGD           # reference baseline
  SignMuon          # sign after  the LMO
  MuonUSign         # sign before the LMO
  MuonSign          # sign on both sides
  EF21-SignMuon     # EF21 on the LMO direction  (the Thm-4 diverging one)
  EF21-MuonUSign    # EF21 on the momentum, uplink only
  EF21-MuonSign     # bidirectional EF21
)
OPTS=("${@:-}")
if [ -z "${1:-}" ]; then OPTS=("${ALL_OPTS[@]}"); fi

mkdir -p "$LOG_DIR"
echo "script=$SCRIPT  nproc=$NPROC  log_dir=$LOG_DIR  iters=${NANOGPT_ITERS:-<full>}"
echo "runs: ${OPTS[*]}"
echo

failed=()
for opt in "${OPTS[@]}"; do
  echo "=============================================================="
  echo "  $opt   ($(date '+%H:%M:%S'))"
  echo "=============================================================="
  if [ "$NPROC" = "1" ] && [ "$SCRIPT" = "train_gpt_a100.py" ]; then
    SIGNMUON_OPT="$opt" python "$SCRIPT"
  else
    SIGNMUON_OPT="$opt" torchrun --standalone --nproc_per_node="$NPROC" "$SCRIPT"
  fi
  # A diverging method exits non-zero only on a real crash; the training script
  # aborts cleanly (and logs DIVERGED) when the loss goes non-finite, which is a
  # RESULT, not a failure. Keep going either way, and report at the end.
  status=$?
  [ $status -ne 0 ] && failed+=("$opt (exit $status)")
  echo
done

echo "=============================================================="
if [ ${#failed[@]} -eq 0 ]; then
  echo "all runs finished"
else
  echo "runs that exited non-zero: ${failed[*]}"
fi
echo "now:  python parse_logs.py $LOG_DIR -o results  &&  python plot_runs.py results/steps.csv"
