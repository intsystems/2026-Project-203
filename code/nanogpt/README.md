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

Code lives here; **every artefact of a run lives in [`../results/nanogpt/`](../results/nanogpt/)**,
beside `results/synthetic/` and `results/federated/`, so all four experiments are
read the same way. Start at [`../results/nanogpt/SUMMARY.md`](../results/nanogpt/SUMMARY.md).

| file | what it is |
|------|------------|
| `signmuon_optimizers.py` | The eight optimizers. **Single source of truth, shared by both training scripts**, pure-torch (Polar Express LMO, no Triton), unit-testable off-GPU. |
| `train_gpt.py` | The **8×H100** script: record #40 (Flash Attention 3 + FP8) with its `Muon` replaced by the `SIGNMUON_OPT=` selector. Use for the final runs. |
| `train_gpt_a100.py` | The **single-A100** build. Identical to `train_gpt.py` except the two Hopper-only pieces (FA3, FP8) are swapped for Ampere-safe equivalents. Every difference is a `# ===== [A100 DIFF #k] ...` banner. |
| `test_signmuon_optimizers.py` | Portable CPU test: each optimizer's update recurrence == the numpy paper reference (`../counterexamples/optimizers.py`), the per-layer EF21 compressor scale on the merged `qkvo_w`, **and** the per-layer LR multipliers (== record #40's aspect factor for the LMO family, unit gain for both families, agreement with `../common/lr_scaling.py`). |
| `test_distributed_sharding.py` | gloo/CPU test: the sharded `step()` == a single-process centralized run, over both padding regimes. |
| `setup_env.sh` | Builds the venv from `requirements.txt` and verifies it with a real FA3 fetch. **Start here.** |
| `run_all.sh` | Launches the eight hero runs, one per optimizer, at their starting learning rates. |
| `parse_logs.py` | Raw logs -> `runs.csv` (one row per run) + `steps.csv` (tidy per-step) + `diagnostics.csv` + `runs.json`. |
| `make_tables.py` | Those CSVs -> `SUMMARY.md`, `MANIFEST.json`, and the two LaTeX table bodies. This is what makes `tab:nanogpt` and `tab:nanogpt_diag` *derived* numbers rather than transcribed ones. |
| `plot_article.py` | The three nanoGPT figures the paper includes (`fig:nanogpt`, `fig:nanogpt_appendix`, `fig:nanogpt_diag`). Run IDs are pinned; see the module docstring for which and why. |
| `plot_runs.py` | Exploratory loss-vs-steps and loss-vs-time figures from `steps.csv`, with record #40's own curve drawn behind every validation figure. Not used by the paper. |
| `reference_record40.csv` | Record #40's published validation curve, averaged over its **five** upstream 8×H100 logs (`3.2780 ± 0.0009` at step 2330, `140.7 s`). The pass/fail line for the `Muon` control arm. |
| `data/cached_fineweb10B.py` | Downloads the pre-tokenized FineWeb-10B GPT-2 tokens (same stream for every record). |
| `requirements.txt` | Python deps (mirrors upstream; `torch==2.10`). |
| `train_gpt_rec40_reference.py` | The **verbatim record-#40 source** (from its run log), for provenance / diffing. Not wired to the optimizer knob. |

Everything except the two training scripts is standard-library or
matplotlib-only, so the whole analysis path runs on a laptop with no GPU and no
torch; only `run_all.sh` needs the cluster.

## Setup

```bash
cd code/nanogpt
bash setup_env.sh                         # venv + deps + a real FA3 fetch, verified
source .venv/bin/activate
python data/cached_fineweb10B.py 9        # ~900M train tokens + the val chunk
```

The venv is not optional, and neither is its own torch. Both pins are in
`requirements.txt`:

* **Never `pip install` into the system python.** apt-installed packages carry no pip
  RECORD, so pip cannot remove them: `Cannot uninstall cryptography 41.0.7 -- no RECORD
  file was found`.
* **`kernels<0.13`.** 0.14+ requires `huggingface-hub>=1.10` (a major bump) and 0.16+ also
  `sigstore>=4`, which pulls cryptography / pyOpenSSL / rfc3161-client -- that is where
  the collision above comes from. `<0.13` fetches FA3 identically.
* **`torch==2.10.0` from PyPI, not the container's.** `kernels` downloads a *prebuilt* FA3
  binary and the hub has **cu126 / cu128 / cu130** variants only (for torch 2.8-2.12,
  x86_64 and aarch64). An NGC container ships a **CUDA 13.1** build
  (`2.10.0a0+...nv26.01`), which would need a `cu131` variant that does not exist, so the
  container torch simply cannot run the FA3 path. PyPI's `torch==2.10.0` is a cu128 build
  -- variant `torch210-cxx11-cu128-x86_64-linux`, which does exist -- and cu128 runs fine
  on the newer driver.

For provenance, record #40 itself ran **torch `2.10.0.dev20250926+cu126`, Python 3.10.12,
Triton 3.5.0, CUDA 12.6**; the eight runs in the paper ran **torch `2.10.0+cu128`
(PyPI, CUDA 12.8), Python 3.12.3**, on an 8×H100 SXM node with driver `595.71.05`.
Those are read back out of the logs by `make_tables.py` and land in
[`../results/nanogpt/MANIFEST.json`](../results/nanogpt/MANIFEST.json).

If FA3 still will not resolve, skip it -- FlexAttention instead, still 8 GPUs:

```bash
NPROC=8 SCRIPT=train_gpt_a100.py bash run_all.sh
```

That is a genuine fallback, not a downgrade of the experiment: FA3 -> FlexAttention and
FP8 -> bf16 lm_head are both **outside** the optimizer, so the sign/EF21 comparison is
unchanged and only the absolute wall-clock differs from record #40.

`run_all.sh` re-runs exactly this check itself -- imports, GPU count, data shards, **and a
real FA3 fetch** -- on one process before launching anything (`SKIP_PREFLIGHT=1` to
bypass), and stops the sweep if the first runs all fail, rather than reproducing the same
environment error eight times. It drives everything through one interpreter
(`$PYTHON -m torch.distributed.run`, not a bare `torchrun`), so the environment it
validates is the one that trains; set `PYTHON=/path/to/venv/bin/python` to be explicit.

The scripts read `data/fineweb10B/fineweb_{train,val}_*.bin` relative to the current
directory (or set `DATA_PATH=/abs/path` to point elsewhere). A full run consumes ~611M
tokens (2330 steps x 262144), so 7+ shards.

## Running

Pick the optimizer for the hidden matrix weights via `SIGNMUON_OPT` (embeddings,
scalars and the LM head always use `DistAdam`, as in every Muon speedrun; the small
attention/smear/skip **gates** ride with the selected optimizer, exactly as in #40):

```bash
# --- all eight, one run each, at their starting LRs (this is the main experiment) ---
bash run_all.sh
NANOGPT_ITERS=200 bash run_all.sh        # cheap smoke pass first -- do this once

# --- a single method on 8×H100 (record #40, FA3 + FP8) ---
SIGNMUON_OPT=EF21-MuonUSign torchrun --standalone --nproc_per_node=8 train_gpt.py

# --- single A100 (imitates the 8×H100 run; ~8× slower wall-clock, same curves) ---
SIGNMUON_OPT=EF21-MuonUSign python train_gpt_a100.py
#   or:  NPROC=1 SCRIPT=train_gpt_a100.py bash run_all.sh
```

Valid `SIGNMUON_OPT`: `SignMuon`, `EF21-SignMuon`, `MuonUSign`, `MuonSign`,
`EF21-MuonUSign`, `EF21-MuonSign`, `SignSGD`, `Muon` (default `Muon`).

| env var | meaning |
|---|---|
| `SIGNMUON_LR`, `SIGNMUON_MOMENTUM`, `SIGNMUON_WD` | override the method's hyperparameters |
| `SIGNMUON_LR_SCALING` | per-layer rule: `unit-gain` (default), `semantic`, `mup`, `legacy`, `none` |
| `SIGNMUON_SEED` | RNG seed (default `0`); see **Provenance** — the eight logged runs predate it |
| `SIGNMUON_RUN_ID` | override the log name (default `<Opt>_lr<lr>_<hash>`) |
| `NANOGPT_ITERS`, `NANOGPT_VAL_EVERY` | shorten a run / change the validation cadence |
| `LOG_DIR`, `DATA_PATH` | where logs are written / where the `.bin` shards live |
| `SIGN_PROBE_LR`, `SKIP_PREFLIGHT` | `run_all.sh` only: extra sign-method runs at that LR / skip the environment check |

Each logged run records `train_gpt*.py` **and** `signmuon_optimizers.py` verbatim, so a
log fully reproduces the optimizer definitions even though they are imported. It also
carries a machine-readable `RUNMETA {...}` / `RUNEND {...}` JSON header, the per-layer
LR multiplier table, **per-step training loss**, per-validation compressor diagnostics,
and the usual validation points.
A run whose loss goes non-finite logs `DIVERGED` and stops -- that is a result, not a
crash, and the analysis tooling reports it as such.

`EF21-MuonSign` is the one method with two models, and each validation reports
**both**: `val_loss` is the exact server model `X` (the iterate the convergence
corollary bounds) and `val_loss_W` is the sign-compressed broadcast model `W` (the
iterate every gradient was actually evaluated at). They separate by ~2.2 nats here,
which is the point of printing both.

### Validate the port before reading any result

`run_all.sh` runs `Muon` first for exactly this reason: it **is** record #40's optimizer,
so it must land on record #40's published curve. Those numbers are checked in
(`reference_record40.csv`, mean of upstream's five 8×H100 logs) and drawn behind every
validation figure:

| step | 250 | 500 | 1000 | 1500 | 2000 | 2330 |
|---|---|---|---|---|---|---|
| upstream val loss | 4.0898 | 3.8203 | 3.5766 | 3.4509 | 3.3307 | **3.2780 ± 0.0009** |

A `Muon` run outside roughly `3.278 ± 0.003` at step 2330 means the **port** is broken, not
the optimizer — fix that before interpreting the other seven arms. Wall-clock *will* be
somewhat above the record's `140.7 s` / `60.4 ms per step`: this port replaces #40's Triton
kernels and its batched sharded transport with a pure-torch, per-parameter equivalent
(identical arithmetic, more kernel launches). All eight methods pay that cost equally, so
every cross-method comparison — including loss-vs-time — stays fair.

## Results, and how to regenerate them

Everything a run produces is under [`../results/nanogpt/`](../results/nanogpt/):

| path | what it is |
|---|---|
| `SUMMARY.md` | **read this first.** Both paper tables, the record-#40 control check, the wall-clock spread, and the provenance table. |
| `logs/` | the eight raw speedrun logs (each embeds its own source; ~300 kB apiece) |
| `runs.csv` | one row per run: final/best val loss (and `_w` for the broadcast model), `steps_to_<target>` / `ms_to_<target>`, ms/step, peak memory, seed, GPU |
| `steps.csv` | tidy `run_id, optimizer, lr, lr_scaling, step, wallclock_ms, train_loss, val_loss, val_loss_w` |
| `diagnostics.csv` | one row per (run, validation, layer type): compressor contraction `alpha_up`/`alpha_dn`, estimator lag, server/broadcast gap, per-block `mean\|grad\|` |
| `runs.json` | `runs.csv` plus the raw `RUNMETA`/`RUNEND` dicts |
| `MANIFEST.json` | per-run provenance: environment, and the SHA-256 of the source each log embeds |
| `tab_nanogpt.tex`, `tab_nanogpt_diag.tex` | the two LaTeX table bodies, for diffing against the paper |
| `figures/` | `fig_nanogpt_{main,appendix,diag}.{pdf,png}` |

Three commands rebuild all of it from the logs, on any machine, with no GPU:

```bash
cd code/nanogpt
python parse_logs.py        # logs        -> runs.csv, steps.csv, diagnostics.csv, runs.json
python make_tables.py       # those CSVs  -> SUMMARY.md, MANIFEST.json, tab_nanogpt*.tex
python plot_article.py      # those CSVs  -> figures/fig_nanogpt_{main,appendix,diag}.*
```

`parse_logs.py` also prints the summary table to the terminal. `steps_to_<target>` is
linearly interpolated between validation points, so it does not depend on where the
(coarse, every-250-step) validation grid happens to fall; the `_w` columns repeat the
crossing on `EF21-MuonSign`'s broadcast model, because its `X` column never reaches
these targets and a single set of columns would report "never" for a method that in
fact trains normally at the iterate its gradients are taken at.

The headline numbers, for reference — full table in `SUMMARY.md`:

| method | step | η₀ | val. loss | steps to 3.35 |
|---|---|---|---|---|
| `Muon` (== record #40) | lmo | 0.06 | **3.2785** | 1.00× |
| `EF21-SignMuon` | lmo | 0.06 | 3.2860 | 1.02× |
| `SignMuon` | sign | 0.03 | 3.2881 | 1.02× |
| `MuonUSign` | lmo | 0.06 | 3.2959 | 1.05× |
| `EF21-MuonUSign` | lmo | 0.06 | 3.3203 | 1.13× |
| `EF21-MuonSign` (`W`) | lmo | 0.06 | 3.3213 | 1.14× |
| `MuonSign` | sign | 0.03 | 3.3249 | 1.14× |
| `SignSGD` | sign | 0.03 | 3.4049 | — |
| `EF21-MuonSign` (`X`) | lmo | 0.06 | 5.5198 | — |

For exploratory work rather than the paper's figures:

```bash
python plot_runs.py                                     # -> figures/exploratory/
python plot_runs.py --both-themes --anytime --minutes
```

`plot_runs.py` writes four figures (PDF + PNG; eight with `--both-themes`):
{validation, training} loss vs {steps, wall-clock}. **Steps** compares the methods as
*optimizers* (same data, same number of updates); **wall-clock** compares them as
*systems*, on the speedrun's own `train_time` clock, with validation and compilation
excluded. The default is the raw curve, not the running-minimum "anytime best"
envelope used in some published figures — an envelope is monotone by construction and
hides exactly the instability the sign methods are being tested for; `--anytime`
overlays it dashed if you want the comparison. `--metric perplexity` relabels to
`exp(loss)`; `--only Muon SignMuon` restricts the figure. Its output is gitignored;
the paper's three figures are not.

## Provenance: which build produced which log

A speedrun log embeds its own source, so this is checkable rather than asserted, and
`make_tables.py` checks it: `MANIFEST.json` carries the SHA-256 of `train_gpt.py` and
of `signmuon_optimizers.py` *as recorded in each log*, and `SUMMARY.md` labels the
distinct builds. There are two, and one difference from the working tree. None of it
invalidates the comparison, but all of it is worth knowing before you re-run anything.

**Build B — the five non-EF21 arms** (`Muon`, `SignSGD`, `SignMuon`, `MuonUSign`,
`MuonSign`). The first vintage. No compressor diagnostics, and the EF21 compressor's
`mean|·|` was taken over the whole merged `qkvo_w` rather than per Q/K/V/O block.

**Build A — the three EF21 arms**, re-run after that second point was fixed. The fix
is confined to `_scaled_sign`, which **only the EF21 methods call**: `Muon` never
compresses, and the other four transmit a bare entrywise `sign`, which is scale-free
and therefore identical whether the tensor is read as one layer or four. So the five
build-B arms are unaffected by it and were not re-run — the tables mix vintages, but
not in a way that changes any of their numbers. Both vintages ran on the same node on
the same day; `MANIFEST.json` shows the identical torch and driver.

**The working tree differs from both, in one function.** `sign_pm1` now maps exact
zeros to an independent random ±1, as the paper's convention requires (a transmitted
`0` would make the alphabet ternary and break the one-bit accounting); the logged
builds used `torch.sign`, i.e. `sign(0) = 0`. It bites in exactly one place on this
model: record #40 zero-initializes `c_proj` and the `O` block of `qkvo_w`, so at step
0 the gradient of every `c_fc` and of `Q/K/V` is *exactly* zero, and the four
sign-terminated methods now take a full random ±1 step there instead of no step at
all. The three EF21 methods are unaffected even at step 0, because their compressor
scale `mean|Δ|` is then zero as well. A re-run will therefore not reproduce these
curves bit-for-bit; it should reproduce them to well within the ±0.0009 seed spread
record #40 reports.

**The eight runs are unseeded.** Upstream modded-nanogpt seeds nothing, and neither
did this port when they were made — `MANIFEST.json` records `seed: null` for all
eight, and that means "whatever torch drew at process start", not "seed 0".
`SIGNMUON_SEED` (default `0`) now pins it. Only the RNG is pinned: cuDNN/cuBLAS
autotuning and the reduction order are left alone, because forcing deterministic
kernels would change the very wall-clock the table reports.


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
| `MuonUSign` | `X ← X − η·PE(sign(M))` | LMO |
| `MuonSign` | `X ← X − η·sign(PE(sign(M)))` | sign |
| `EF21-MuonUSign` | `g_est ← g_est + mean\|M−g_est\|·sign(M−g_est)`; `X ← X − η·PE(g_est)` | LMO |
| `EF21-MuonSign` | uplink EF on `g_est` → exact `X ← X − η·PE(g_est)`; downlink EF compresses `X−W` into the broadcast model `W` | LMO |
| `SignSGD` | `X ← X − η·sign(M)` | sign |
| `Muon` | `X ← X − η·PE(M)` (reference, no compression; == record #40) | LMO |

## Per-layer learning rates, and why one `η₀` fits all eight

The two families produce step matrices whose norms scale *differently* with shape, so a
single global `η` cannot be right for both. The paper's appendix (`app:lrscale`) fixes
this with the **unit-gain** criterion: the RMS gain of a step matrix `s ∈ R^{m×n}` on an
isotropic input is exactly `γ(s) = ‖s‖_F / √m`, so demanding equal per-step gain on every
layer gives one formula, `λ = √fan_out / ‖s‖_F`, and two closed forms:

| family | methods | `‖s‖_F` | `λ` |
|---|---|---|---|
| `lmo` | `Muon`, `MuonUSign`, `EF21-MuonUSign`, `EF21-MuonSign`, `EF21-SignMuon` | `√min(m,n)` | `√max(1, m/n)` |
| `sign` | `SignMuon`, `MuonSign`, `SignSGD` | `√(mn)` | `1/√fan_in` |

Two things make this the right default here:

1. **The `lmo` line *is* record #40.** `√max(1, m/n)` is character-for-character Keller
   Jordan's shipped aspect factor, which record #40 computes as
   `max(1, p.size(-2)/p.size(-1))**0.5`. On every shape #40 uses — gates `[1,12]` and
   `[6,12]`, the attention blocks `[768,768]`, the MLP matrices `[768,3072]` — the two
   agree exactly (all evaluate to `1.0`), so **`Muon` here is the record verbatim**. A test
   pins this (`test_lmo_family_matches_record40_aspect_factor`).
2. **`η₀` now means one thing for all eight methods:** the per-step RMS gain. At
   `η₀ = 0.06` every method takes the same per-entry RMS step on every layer — e.g.
   `1.08e-3` on the MLP matrices, `1.73e-2` on the gates — whether the step is
   `PE(·)` or `± 1`. This is what makes a learning rate transferable between the
   reference and the methods under study, and it is why the sign family's starting `η`
   is `0.03` rather than the `3e-4` an unscaled implementation needs.

**Starting learning rates** (`train_gpt.py:OPTIMIZER_CONFIG`, all "round" to one
significant digit):

| methods | `η₀` | why |
|---|---|---|
| `Muon`, `MuonUSign`, `EF21-MuonUSign`, `EF21-MuonSign`, `EF21-SignMuon` | **0.06** | the reference's own value — not a guess |
| `SignMuon`, `MuonSign`, `SignSGD` | **0.03** | spectral discount on an entrywise-uniform step |

**The LMO five sit at the reference's LR on purpose.** Their final step is an orthogonal
matrix (or, for the EF21 pair, an error-feedback estimate of one), so it has the same
spectral *and* Frobenius norm as Muon's and there is nothing to rescale. It also makes
the paper's contrasts matched-hyperparameter ones: `Muon` vs `MuonUSign` (what does a
1-bit uplink cost?) and `EF21-SignMuon` vs `EF21-MuonUSign`/`EF21-MuonSign` (Thm 4: EF21
on the LMO *output* diverges, EF21 on the momentum does not) differ only in the rule.

**`EF21-SignMuon` belongs with the LMO five, not with `SignMuon`.** Error feedback is
exactly what undoes the 1-bit quantization: `d_est` is a full-precision accumulator
tracking `PE(M)`, so the step regains the LMO's magnitude — its operator norm starts at
~1.1x Muon's and decays toward 1.0 as `d_est` tracks `D`. Running it at Muon's own LR is
the only way its divergence reads as the *rule's* fault rather than the step size's.

**Where 0.03 comes from.** A sign step is entrywise uniform, so at equal RMS gain it is
spectrally more aggressive than an orthogonal one — and the framework these methods are
analysed in (Gluon / EF21-Muon) is a *spectral*-norm framework, so that is the norm to
match. Three independent routes agree:

| route | number |
|---|---|
| spectral matching: `‖λ·sign(·)‖_op/‖λ·PE(·)‖_op` = 1.40 (mlp) / 1.86 (attn); one `η₀` must satisfy the tighter | `0.06/1.86 = 0.032` |
| Mishra et al.'s tuned nanoGPT value, mapped in: their Alg. 1 has **no** shape factor, so `η=1e-3` on `d=384` is a global unscaled LR → `η₀ ≈ 0.023`; correcting for their broken schedule (`warmup_iters=2000 > max_iters=1500`, so their LR only ramps 0→7.5e-4) and our 8.5x batch | `≈ 0.032` |
| Lion's "3–10x smaller than AdamW", decomposed: AdamW's `m/√v` is ~0.3 per entry vs a sign step's 1.0, so ~3x of that is pure norm — which unit-gain already handles; residual robustness discount 1–3x | `0.02 – 0.06` |

If you expected `1e-4`: that is a **global, unscaled** LR from standard-batch,
long-schedule training. Per weight entry, `η₀=0.03` here means `5.4e-4` (mlp) to `1.1e-3`
(attn) — *half* what Muon itself takes at the record's `0.06`. If `1e-4` per entry were
right for this model, Muon at `0.06` would be ~10x too large too, and it is the record.
This codebase runs far more aggressively than standard GPT-2 training: 2330 steps,
262k tokens/step, Muon at `0.06` (vs ~0.02 typical), Adam at `0.008` with `lr_mul=75` on
the embeddings.

**Confidence.** The LMO five are pinned by the record. The sign three are the one
uncertain number and the evidence brackets `0.01–0.04`; `SIGN_PROBE_LR=0.01 bash
run_all.sh` adds one extra run per sign method at the downside. Tuning ladder:
`0.01, 0.02, 0.03, 0.06, 0.1, 0.2` (`SIGNMUON_LR=0.1 SIGNMUON_OPT=SignMuon ...`). The
multiplier table actually in force is printed into every log.

**One inherited caveat.** The rule reads `(fan_out, fan_in)` off the *stored* tensor, and
record #40 stores the MLP `c_fc` transposed (`[dim, hdim]`, used as `x @ c_fc`) so it can
share a `reduce_scatter` with the attention weight. For `c_fc` the semantic fan-in/fan-out
are therefore swapped, and both families get a 2× smaller multiplier there than a
`[fan_out, fan_in]` reading would give. That is what record #40 itself does for Muon, so it
is the default; `SIGNMUON_LR_SCALING=semantic` corrects it (and thereby moves the Muon
baseline off the record), and `none`/`legacy`/`mup` are there for the ablation.

**Merged attention weight.** Record #40 stores Q/K/V/O in one `[hdim, 4·dim]` parameter but
always uses it as `.view(4, hdim, dim)`. The optimizer therefore orthogonalizes the four
`[hdim, dim]` blocks independently (detected via the model's `.module == "attn"` tag);
same-shaped MLP weights are orthogonalized as a single matrix. This is localized to the LMO
call — every other op (momentum, sign, EF21) is elementwise.

## Distributed ≠ federated

**Federated** (`../federated/algorithms.py`): each client keeps its own momentum/EF21
estimator, compresses *its own* update, and the server aggregates the *compressed* messages.

**Distributed data-parallel** (here): one logical model; `reduce_scatter(op=AVG)` gives one
owning rank the true mean gradient; that rank runs the ordinary **centralized** rule
(momentum/LMO/sign/EF21 via `self.state[p]`); `all_gather` broadcasts the updated parameter.
So the 1-bit "compression" is a property of the update **rule**, reproduced exactly, not of
the wire transport — the honest distributed analog of the centralized algorithms the paper
analyzes. `EF21-MuonSign` keeps an exact server model `X` in state and lets the live
params be the sign-compressed broadcast model `W`; `swap_in_exact()`/`swap_out_exact()`
expose `X` for validation, and the scripts report the loss at **both** models on the
same validation tokens.

## Testing

Both tests run on CPU with only `torch` (+ `numpy` for the math test); no GPUs.

```bash
# 1) update recurrence == numpy paper reference (../counterexamples/optimizers.py),
#    plus the per-layer LR multipliers
SIGNMUON_NO_COMPILE=1 python test_signmuon_optimizers.py

# 2) sharded step() == single-process centralized run.  Runs a PORTABLE simulation of
#    world sizes 1/2/4/8 first (no gloo, no multiprocessing -- works on Windows), then
#    the real-collectives gloo test if the platform supports it.
python test_distributed_sharding.py          # SIGNMUON_SKIP_GLOO=1 for the simulation only
```

Run **both** before renting the machine — test (2) covers the two padding regimes that
record #40's real parameter counts hit on 8 ranks (a group shorter than `world_size`, and
a group of 10 spanning two chunks the second of which is partial). A rank that owns a
parameter in an early chunk but nothing in the last one needs a *fresh* scratch buffer for
the padded `reduce_scatter`; reusing the earlier one silently zeroes an already-averaged
gradient, which on 8 GPUs would have frozen six of the ten `attn_gate` matrices for the
whole run without any error message.

Neither test drives an input with an exact zero in it, so both are insensitive to the
`sign_pm1` tie-breaking described under **Provenance**; the compressor tests do exercise
it indirectly, since a residual confined to one Q/K/V/O block leaves the other three
blocks at `mean|Δ| = 0` and the random signs there must still be scaled away to nothing.

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
