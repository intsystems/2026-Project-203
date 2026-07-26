#!/usr/bin/env bash
# =============================================================================
# Build a working Python environment for the record-#40 NanoGPT runs.
#
#   bash setup_env.sh                 # auto: keep an existing usable torch
#   FRESH_TORCH=1 bash setup_env.sh   # force a clean venv with its own torch
#   VENV=/path/to/venv bash setup_env.sh
#
# Then:   source .venv/bin/activate && bash run_all.sh
#
# -----------------------------------------------------------------------------
# Why this script exists (two real failure modes on a rented Debian/NGC box)
# -----------------------------------------------------------------------------
# 1. `pip install -r requirements.txt` into the SYSTEM python tries to upgrade a
#    distro-managed package and dies with
#
#        Cannot uninstall cryptography 41.0.7
#        The package's contents are unknown: no RECORD file was found
#
#    apt-installed packages have no pip RECORD, so pip cannot remove them. The fix
#    is never to install into the system python: everything below goes into a venv.
#
# 2. The `cryptography` in that chain comes from `kernels`. Since 0.14 `kernels`
#    requires `huggingface-hub>=1.10` (a MAJOR bump), and since 0.16 also
#    `sigstore>=4`, which drags in cryptography / pyOpenSSL / rfc3161-client.
#    `kernels<0.13` needs only `huggingface_hub<2.0`, `packaging`, `pyyaml` --
#    no sigstore, no cryptography, nothing that can collide with apt. It is also
#    much closer to what record #40 actually ran (Oct 2025). So we pin it.
#
# -----------------------------------------------------------------------------
# Why we do NOT reinstall torch by default
# -----------------------------------------------------------------------------
# The FA3 kernel repo (varunneal/flash-attention-3) ships prebuilts for
#     torch 2.8 / 2.9 / 2.10 / 2.11 / 2.12  x  cu126 / cu128 / cu130  x  x86_64 / aarch64
# (verified against the hub on 2026-07-26). So ANY CUDA torch in 2.8-2.12 resolves
# FA3 -- there is no reason to spend ~3 GB replacing a container's torch, and
# `torch==2.10` was never load-bearing. For the record: record #40 itself ran
# torch 2.10.0.dev20250926+cu126 (a nightly), Python 3.10.12, Triton 3.5.0 -- i.e.
# not even the PyPI `torch==2.10` release that upstream's requirements.txt pins.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/.venv}"
PYTHON="${PYTHON:-python3}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.10.*}"   # only used when we install torch ourselves
MIN_TORCH="2.8"
MAX_TORCH="2.12"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "interpreter"
command -v "$PYTHON" >/dev/null || { echo "no '$PYTHON' on PATH; set PYTHON=..."; exit 1; }
"$PYTHON" -c 'import sys; print(sys.executable, sys.version)'
"$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info>=(3,10) else 1)' \
  || { echo "need Python >= 3.10 (record #40 ran 3.10.12)"; exit 1; }

# --- does the system/container already have a torch we can reuse? -------------
REUSE_TORCH=0
if [ "${FRESH_TORCH:-0}" != "1" ]; then
  if "$PYTHON" - "$MIN_TORCH" "$MAX_TORCH" <<'PY'
import sys
lo, hi = (tuple(int(x) for x in a.split(".")) for a in sys.argv[1:3])
try:
    import torch
except Exception:
    raise SystemExit(1)
v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
raise SystemExit(0 if lo <= v <= hi and torch.version.cuda else 1)
PY
  then REUSE_TORCH=1; fi
fi

say "virtualenv -> $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  VENV_ARGS=()
  if [ "$REUSE_TORCH" = "1" ]; then
    echo "reusing the existing torch (--system-site-packages, torch is NOT reinstalled)"
    VENV_ARGS=(--system-site-packages)
  else
    echo "no usable CUDA torch in $MIN_TORCH..$MAX_TORCH -> clean venv, installing $TORCH_SPEC"
  fi
  # Debian splits `venv` into its own apt package and the error it prints when the
  # package is absent is easy to miss inside a longer log, so say it plainly.
  if ! "$PYTHON" -m venv ${VENV_ARGS[@]+"${VENV_ARGS[@]}"} "$VENV" 2>/dev/null; then
    echo "  'python -m venv' failed. On Debian/Ubuntu that usually means:"
    echo "      sudo apt-get update && sudo apt-get install -y python3-venv"
    echo "  (or install uv / virtualenv and point VENV= at an env you made yourself)"
    exit 1
  fi
else
  echo "already exists, reusing it"
fi
VPY="$VENV/bin/python"
"$VPY" -m pip install --quiet --upgrade pip setuptools wheel

# --- dependencies -------------------------------------------------------------
# Install ONLY what is actually missing or wrong-versioned. That matters beyond
# tidiness: with --system-site-packages, force-reinstalling `numpy` would shadow the
# container's numpy with a possibly different major version, and a torch built
# against the other one breaks on import. Leaving a satisfied dependency alone is
# the safe default; we only ever add.
say "dependencies"
PKGS=()   # declare before mapfile: `set -u` + an undeclared array is a hard error
mapfile -t PKGS < <("$VPY" - <<'PY'
import importlib.util as u
def ver(mod):
    try:
        from importlib.metadata import version
        return version(mod)
    except Exception:
        return None
def tup(s):
    out = []
    for p in (s or "0").split("."):
        try: out.append(int(p))
        except ValueError: break
    return tuple(out)
if u.find_spec("numpy") is None: print("numpy")
if u.find_spec("tqdm") is None: print("tqdm")
if u.find_spec("huggingface_hub") is None: print("huggingface-hub<2.0,>=0.26.0")
k = ver("kernels") if u.find_spec("kernels") else None
if k is None or not ((0, 9) <= tup(k) < (0, 13)):
    print("kernels>=0.9,<0.13")          # see the header: >=0.14 drags in sigstore
PY
)
if [ "$REUSE_TORCH" != "1" ]; then PKGS+=("$TORCH_SPEC"); fi
if [ ${#PKGS[@]} -eq 0 ]; then
  echo "  everything already satisfied"
else
  echo "  installing: ${PKGS[*]}"
  # First try a plain install. If pip decides it must UPGRADE something it can see
  # in the system tree, it will try to uninstall an apt package and fail (failure
  # mode 1 in the header) -- then retry with --ignore-installed, which puts a fresh
  # copy in the venv that shadows the system one instead of touching it.
  if ! "$VPY" -m pip install "${PKGS[@]}"; then
    echo
    echo "  plain install failed (most likely pip tried to replace an apt-managed"
    echo "  package). Retrying with --ignore-installed, which only ADDS to the venv."
    "$VPY" -m pip install --ignore-installed "${PKGS[@]}"
  fi
fi

# --- verify -------------------------------------------------------------------
say "verify"
"$VPY" - <<'PY'
import sys
import torch
print(f"  python  {sys.version.split()[0]}")
print(f"  torch   {torch.__version__}  (CUDA {torch.version.cuda}), "
      f"{torch.cuda.device_count()} GPU(s) visible")
if not torch.version.cuda:
    print("  !! this torch is CPU-only -- rerun with FRESH_TORCH=1"); raise SystemExit(1)
import kernels
print(f"  kernels {getattr(kernels, '__version__', '?')}")
# Actually FETCH the kernel: "import kernels works" does NOT imply "FA3 resolves
# for this torch". The real fetch otherwise happens deep in model construction,
# i.e. once per run. Pay for it here; it is cached afterwards.
from kernels import get_kernel
try:
    get_kernel("varunneal/flash-attention-3")
    print("  FA3     resolved and cached for this torch build")
except Exception as exc:
    v = torch.__version__.split("+")[0].split(".")
    print(f"  !! FA3 FETCH FAILED: {type(exc).__name__}: {exc}")
    print(f"  !! this torch wants build variant torch{v[0]}{v[1]}-cxx11-cu"
          f"{(torch.version.cuda or '').replace('.','')}-x86_64-linux")
    print("  !! the hub has torch28..torch212 x cu126/cu128/cu130. If yours is outside")
    print("  !! that, rerun with:  FRESH_TORCH=1 bash setup_env.sh")
    print("  !! or skip FA3 entirely (FlexAttention, still 8 GPUs):")
    print("  !!     NPROC=8 SCRIPT=train_gpt_a100.py bash run_all.sh")
    raise SystemExit(1)
PY

say "data"
if compgen -G "${DATA_PATH:-.}/data/fineweb10B/fineweb_train_*.bin" >/dev/null; then
  n=$(ls "${DATA_PATH:-.}"/data/fineweb10B/fineweb_train_*.bin 2>/dev/null | wc -l)
  echo "  $n train shard(s) present"
else
  echo "  no shards yet. A full run needs 7 (2330 steps x 262144 tokens ~ 611M):"
  echo "      $VPY data/cached_fineweb10B.py 9"
fi

say "ready"
cat <<EOF
  source $VENV/bin/activate
  NANOGPT_ITERS=200 bash run_all.sh     # smoke pass first
  bash run_all.sh                       # the eight hero runs

  torch was $([ "$REUSE_TORCH" = 1 ] && echo "REUSED from the system/container" || echo "installed into the venv")
  no system/apt package was modified.
EOF
