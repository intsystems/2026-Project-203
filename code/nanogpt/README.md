# SignMuon / MuonSign / EF21 optimizers on the NanoGPT speedrun (record #40)

This folder plugs the paper's six matrix-aware 1-bit optimizers — plus `SignSGD`
and a reference `Muon` — into [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt),
in the **distributed data-parallel** setting (*not* the federated setting).

The base is **record #40** (`2025-10-04_Backout`, PR #140, 2.358 min on 8×H100),
the **last record before NorMuon** (#41). Everything up to #40 — Polar Express
orthogonalization, Flash Attention 3, YaRN, sparse attention gate, token smearing,
dropped first attn/MLP layers, BF16 cross-entropy, the backout skip — is included,
while **Muon is still a clean, separable optimizer** whose orthogonalization step is
exactly where the sign/EF21 variants inject. This deliberately avoids NorMuon and
everything co-tuned to it (#41→#84), so there is **no degradation from NorMuon-specific
hyperparameter tuning**. See "Why record #40" below.

## Files

| file | what it is |
|------|------------|
| `signmuon_optimizers.py` | The eight optimizers. **Single source of truth, shared by both training scripts**, pure-torch (Polar Express LMO, no Triton), unit-testable off-GPU. |
| `train_gpt.py` | The **8×H100** script: record #40 verbatim (Flash Attention 3 + FP8) with a `SIGNMUON_OPT=` selector. Use for the final runs. |
| `train_gpt_a100.py` | The **single-A100** build. Identical to `train_gpt.py` except the two Hopper-only pieces (FA3, FP8) are swapped for Ampere-safe equivalents. Every difference is a `# ===== [A100 DIFF #k] ...` banner. |
| `test_signmuon_optimizers.py` | Portable CPU test: each optimizer's update recurrence == the numpy paper reference (`../counterexamples/optimizers.py`). |
| `test_distributed_sharding.py` | gloo/CPU test: the sharded `step()` == a single-process centralized run. |
| `data/cached_fineweb10B.py` | Downloads the pre-tokenized FineWeb-10B GPT-2 tokens (same stream for every record). |
| `requirements.txt` | Python deps (mirrors upstream; `torch==2.10`). |
| `train_gpt_rec40_reference.py` | The **verbatim record-#40 source** (from its run log), for provenance / diffing. Not wired to the optimizer knob. |
| `lesha_nanogpt.py`, `train_nanogpt.py` | **Superseded** classic-record scaffolding, kept for reference only. |

## Setup

```bash
cd code/nanogpt
pip install -r requirements.txt
# download e.g. the first 900M training tokens (chunks) + the val chunk
python data/cached_fineweb10B.py 9        # use a larger arg (up to 103) for full runs
```

The scripts read `data/fineweb10B/fineweb_{train,val}_*.bin` relative to the current
directory (or set `DATA_PATH=/abs/path` to point elsewhere).

## Running

Pick the optimizer for the hidden matrix weights via `SIGNMUON_OPT` (embeddings,
scalars and the LM head always use `DistAdam`, as in every Muon speedrun; the small
attention/smear/skip **gates** ride with the selected optimizer, exactly as in #40):

```bash
# --- final run on 8×H100 (record #40, FA3 + FP8) ---
SIGNMUON_OPT=EF21-MuonUSign torchrun --standalone --nproc_per_node=8 train_gpt.py

# --- single A100 (imitates the 8×H100 run; ~8× slower wall-clock, same curves) ---
SIGNMUON_OPT=EF21-MuonUSign python train_gpt_a100.py
#   or:  SIGNMUON_OPT=EF21-MuonUSign torchrun --standalone --nproc_per_node=1 train_gpt_a100.py
```

Valid `SIGNMUON_OPT`: `SignMuon`, `EF21-SignMuon`, `MuonUSign`, `MuonUDSign`,
`EF21-MuonUSign`, `EF21-MuonUDSign`, `SignSGD`, `Muon` (default `Muon`).
Sweep overrides: `SIGNMUON_LR`, `SIGNMUON_MOMENTUM`, `SIGNMUON_WD`.

Each logged run records `train_gpt*.py` **and** `signmuon_optimizers.py` verbatim, so a
log fully reproduces the optimizer definitions even though they are imported.

## How the single A100 imitates 8×H100

**The batch/step math is unchanged — this is record #40's own "fewer GPUs" path.**
Record #40 sets `grad_accum_steps = 8 // world_size` and, per optimizer step,
accumulates that many micro-batches before stepping. On one GPU (`world_size == 1`)
it runs `grad_accum_steps == 8` micro-batches, so it sees the **same global batch, the
same tokens, and the same number of optimizer steps** as the 8-rank run. The missing
`1/8` factor in the optimizers' `reduce_scatter(op=AVG)` (a no-op on one process) washes
out because **Muon and Adam are scale-invariant** (Muon's Polar Express normalizes the
update; Adam is `m/√v`). Upstream states this explicitly: *"To run experiments on fewer
GPUs, simply modify `--nproc_per_node`. This should not change the behavior of the
training."*

**The optimizer under study is byte-for-byte identical on A100 and 8×H100.** Both scripts
import the same pure-torch `signmuon_optimizers`, so the sign/EF21 update geometry (momentum,
LMO placement, error-feedback estimators) is the same object on both cards.

**The only residual A100 ↔ 8×H100 differences are outside the optimizer**, and are the
three `[A100 DIFF #k]` banners in `train_gpt_a100.py`:

| # | 8×H100 (`train_gpt.py`) | single A100 (`train_gpt_a100.py`) | touches |
|---|---|---|---|
| 1 | 8 ranks | 1 rank, `grad_accum_steps=8` (built into #40) + torchrun env defaults | nothing (same batch/steps) |
| 2 | Flash Attention 3 (Hopper-only) | **FlexAttention** with a block mask reproducing #40's mask *exactly* (per-document causal + `bm_size`-token left window; documents delimited by BOS=50256; RoPE/YaRN unchanged) | attention kernel numerics only |
| 3 | FP8 lm_head matmul (Hopper-only) | **bf16** lm_head (`DISABLE_FP8=1`) | lm_head (a DistAdam param), not the methods under study |

So A100 curves should track the 8×H100 curves closely, differing only by (a) the
FlexAttention-vs-FA3 attention kernel and (b) the bf16-vs-FP8 lm_head — neither of which
changes the sign/EF21 geometry.

### ⚠️ Validate the attention swap once, on a GPU

FA3 → FlexAttention is the one change that could not be verified on a CPU-only laptop.
The masks are mathematically equivalent (same causal + token window + per-document
structure), but **before trusting long A100 sweeps, do a short A/B**: run
`train_gpt_a100.py` on a **single H100** (FlexAttention) and `train_gpt.py` on 8×H100
(FA3), same `SIGNMUON_OPT`, and confirm the first few hundred steps' loss curves overlap.
`train_gpt_a100.py` is device-agnostic, so it runs on H100 too — that isolates the
attention-kernel difference from everything else.

Notes: (a) an 80 GB A100 is assumed — one 32 768-token micro-batch fits (same as one H100
rank); (b) `create_block_mask` is built eagerly per micro-batch, so A100 validation passes
are slower — fine, wall-clock is not the goal here; if `_compile=True` in
`build_flex_block_mask` errors on your torch build, set it to `False`.

## The eight optimizers

Let `M` be the (Nesterov EMA) momentum of the **averaged** gradient and `PE(·)` the Muon
Polar Express orthogonalization (record #40's LMO; approximate polar factor `U Vᵀ`). Every
rule is verbatim the centralized algorithm boxes of the paper and their numpy reference in
[`../counterexamples/optimizers.py`](../counterexamples/optimizers.py).

| name | update | final step |
|------|--------|------------|
| `SignMuon` | `X ← X − η·sign(PE(M))` | sign |
| `EF21-SignMuon` | `d_est ← d_est + mean\|D−d_est\|·sign(D−d_est)`, `D=PE(M)`; `X ← X − η·d_est` | LMO |
| `MuonUSign` (= MuonSign) | `X ← X − η·PE(sign(M))` | LMO |
| `MuonUDSign` | `X ← X − η·sign(PE(sign(M)))` | sign |
| `EF21-MuonUSign` | `g_est ← g_est + mean\|M−g_est\|·sign(M−g_est)`; `X ← X − η·PE(g_est)` | LMO |
| `EF21-MuonUDSign` | uplink EF on `g_est` → exact `X ← X − η·PE(g_est)`; downlink EF compresses `X−W` into the broadcast model `W` | LMO |
| `SignSGD` | `X ← X − η·sign(M)` | sign |
| `Muon` | `X ← X − η·PE(M)` (reference, no compression; == record #40) | LMO |

Sign-**terminated** steps (`SignMuon`, `MuonUDSign`, `SignSGD`) move every weight by `≈ η`
each step and use **no** fan-in lr scaling, so they need a **much smaller `η`** than the
LMO-terminated methods. The defaults in `train_gpt.py:OPTIMIZER_CONFIG` reflect this but are
only starting points (except `Muon`, which equals record #40 exactly: `lr=0.06`,
`momentum=0.95`, `weight_decay=0.0`). **Retune `η` per method for a real run.**

**Merged attention weight.** Record #40 stores Q/K/V/O in one `[hdim, 4·dim]` parameter but
always uses it as `.view(4, hdim, dim)`. The optimizer therefore orthogonalizes the four
`[hdim, dim]` blocks independently (detected via the model's `.module == "attn"` tag);
same-shaped MLP weights are orthogonalized as a single matrix. This is localized to the LMO
call — every other op (momentum, sign, EF21) is elementwise.

## Distributed ≠ federated

**Federated** (`../federated_algorithms.py`): each client keeps its own momentum/EF21
estimator, compresses *its own* update, and the server aggregates the *compressed* messages.

**Distributed data-parallel** (here): one logical model; `reduce_scatter(op=AVG)` gives one
owning rank the true mean gradient; that rank runs the ordinary **centralized** rule
(momentum/LMO/sign/EF21 via `self.state[p]`); `all_gather` broadcasts the updated parameter.
So the 1-bit "compression" is a property of the update **rule**, reproduced exactly, not of
the wire transport — the honest distributed analog of the centralized algorithms the paper
analyzes. `EF21-MuonUDSign` keeps an exact server model `X` in state and lets the live
params be the sign-compressed broadcast model `W`; `swap_in_exact()`/`swap_out_exact()`
expose `X` for validation (the scripts report val loss on `X`).

## Testing

Both tests run on CPU with only `torch` (+ `numpy` for the math test); no GPUs.

```bash
# 1) update recurrence == numpy paper reference (../counterexamples/optimizers.py)
SIGNMUON_NO_COMPILE=1 python test_signmuon_optimizers.py

# 2) sharded step() == single-process centralized run (gloo, 4 ranks)
python test_distributed_sharding.py          # or: python test_distributed_sharding.py 2
```

## Why record #40 (and not the current NorMuon record)

The current upstream record (#84, 1.320 min) fuses Muon into a `NorMuonAndAdam` optimizer
(NorMuon = Muon + an Adafactor-style per-neuron variance normalization) with cautious weight
decay, a batch-size/seq-len curriculum, paired-head Muon, MTP and heavy Triton kernels — and
~15 records of hyperparameters co-tuned to NorMuon. There is no clean single orthogonalization
step to host the sign/EF21 rules, and swapping NorMuon out while keeping NorMuon-tuned LR/WD
would degrade results. **Record #40 is the principled cut**: the last record *before* NorMuon,
so it carries all the non-NorMuon architecture/kernel improvements while keeping a separable
Muon whose LMO the paper's variants slot into unchanged, tuned for plain Muon. Porting onto a
later NorMuon-based record is possible but is a separate, larger effort with real degradation
risk (see the discussion in the project notes).
