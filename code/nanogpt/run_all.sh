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

# --- preflight -------------------------------------------------------------
# A missing dependency or a missing data shard fails identically for every run,
# so check ONCE on a single process instead of discovering it eight times over
# eight torchrun teardowns.
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  echo "--- preflight ---"
  python - "$SCRIPT" <<'PY' || { echo "preflight FAILED -- fix the above before running"; exit 1; }
import glob, importlib, os, sys
script = sys.argv[1]
need = ["torch", "numpy", "huggingface_hub"]
if script == "train_gpt.py":
    need.append("kernels")            # Flash Attention 3 fetch (8xH100 build only)
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    print(f"  MISSING python packages: {', '.join(missing)}")
    print(f"  fix: pip install -r requirements.txt   (or just: pip install {' '.join(missing)})")
    if "kernels" in missing:
        print("  no FA3 available? -> NPROC=8 SCRIPT=train_gpt_a100.py bash run_all.sh")
    sys.exit(1)
import torch
print(f"  torch {torch.__version__}, cuda {torch.version.cuda}, "
      f"{torch.cuda.device_count()} GPU(s) visible")
data = os.environ.get("DATA_PATH", ".")
tr = glob.glob(os.path.join(data, "data/fineweb10B/fineweb_train_*.bin"))
va = glob.glob(os.path.join(data, "data/fineweb10B/fineweb_val_*.bin"))
print(f"  data: {len(tr)} train shard(s), {len(va)} val shard(s)")
if not tr or not va:
    print("  MISSING data. fix: python data/cached_fineweb10B.py 9")
    sys.exit(1)
# a full run consumes ~611M tokens (2330 steps x 262144); each shard is 100M
iters = int(os.environ.get("NANOGPT_ITERS", 2290)) + 40
need_shards = -(-iters * 262144 // 100_000_000)
if len(tr) < need_shards:
    print(f"  WARNING: {iters} steps need ~{need_shards} train shards, found {len(tr)};")
    print("           the loader will wrap around and re-see tokens.")
if script == "train_gpt.py":
    # Actually FETCH the kernel. `kernels` downloads a PREBUILT Flash Attention 3
    # binary keyed to the exact torch version/ABI, so "import kernels works" does
    # NOT imply "FA3 resolves for this torch" -- and the real fetch happens deep in
    # model construction, i.e. once per run, after compile setup. Pay for it once
    # here (it is cached afterwards) instead of eight times.
    from kernels import get_kernel
    try:
        get_kernel("varunneal/flash-attention-3")
        print("  FA3 kernel resolved for this torch build")
    except Exception as exc:
        print(f"  FA3 FETCH FAILED: {type(exc).__name__}: {exc}")
        print(f"  this torch is {torch.__version__}; the hub keys prebuilts by torch")
        print("  version/ABI, and an nvcr.io dev build (2.10.0a0+gitXXXX) may have no")
        print("  matching variant. Two ways forward:")
        print("    1) get the torch the hub has prebuilts for:")
        print("         pip install -r requirements.txt      # torch==2.10 from PyPI")
        print("    2) skip FA3 entirely -- FlexAttention instead, runs on 8xH100:")
        print("         NPROC=8 SCRIPT=train_gpt_a100.py bash run_all.sh")
        print("       (FA3->FlexAttention and FP8->bf16 are both OUTSIDE the optimizer,")
        print("        so the sign/EF21 comparison is unaffected; only wall-clock differs)")
        sys.exit(1)
PY
  echo "--- preflight OK ---"
  echo
fi

# Sign-terminated methods, for the optional second-LR probe below. Their starting
# LR is the least certain number in the whole experiment (see OPTIMIZER_CONFIG in
# the training script): the LMO five are pinned to record #40's own 0.06, but the
# sign three rest on a spectral-norm argument and on Mishra et al.'s tuned value
# mapped through our per-layer scaling, which together bracket 0.01-0.04.
# SIGN_PROBE_LR=0.01 adds one extra run per sign method at that LR.
SIGN_OPTS=" SignMuon MuonSign SignSGD "

run_one() {   # $1 = optimizer, $2 = lr override ("" for the configured default)
  local opt="$1" lr="$2" label="$1"
  [ -n "$lr" ] && label="$1 @ lr=$lr"
  echo "=============================================================="
  echo "  $label   ($(date '+%H:%M:%S'))"
  echo "=============================================================="
  if [ "$NPROC" = "1" ] && [ "$SCRIPT" = "train_gpt_a100.py" ]; then
    SIGNMUON_OPT="$opt" ${lr:+SIGNMUON_LR=$lr} python "$SCRIPT"
  else
    SIGNMUON_OPT="$opt" ${lr:+SIGNMUON_LR=$lr} \
      torchrun --standalone --nproc_per_node="$NPROC" "$SCRIPT"
  fi
  # A diverging method exits non-zero only on a real CRASH: the training script
  # aborts cleanly (and logs DIVERGED) when the loss goes non-finite, which is a
  # RESULT, not a failure.
  local status=$?
  if [ $status -ne 0 ]; then
    failed+=("$label (exit $status)")
  else
    succeeded=$((succeeded + 1))
  fi
  echo
}

failed=()
succeeded=0
for opt in "${OPTS[@]}"; do
  run_one "$opt" ""
  # Systematic failures (missing module, no GPU, OOM) kill every run identically.
  # Do not spend the rest of the rented machine rediscovering that.
  if [ ${#failed[@]} -ge 2 ] && [ $succeeded -eq 0 ]; then
    echo "!! the first ${#failed[@]} runs both failed and none succeeded -- stopping."
    echo "!! this is an environment problem, not an optimizer result. Fix it, then rerun."
    break
  fi
done

if [ -n "${SIGN_PROBE_LR:-}" ] && [ $succeeded -gt 0 ]; then
  for opt in "${OPTS[@]}"; do
    case "$SIGN_OPTS" in *" $opt "*) run_one "$opt" "$SIGN_PROBE_LR";; esac
  done
fi

echo "=============================================================="
if [ ${#failed[@]} -eq 0 ]; then
  echo "all $succeeded run(s) finished"
else
  echo "$succeeded run(s) ok; exited non-zero: ${failed[*]}"
fi
echo "now:  python parse_logs.py $LOG_DIR -o results  &&  python plot_runs.py results/steps.csv"
