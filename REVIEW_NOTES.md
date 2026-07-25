# Code review notes — `code/` vs. `aaai_article/v2_SignMuon_AAAI.tex`

Review date: 2026-07-25. The article is treated as canonical; where the code looked
more defensible the disagreement is flagged rather than silently resolved.

Nothing here was run on a GPU. `code/tests/test_code.py` is the CPU test suite that
validates the torch rewrites — **run it before trusting any of this**
(`python3 -m tests.test_code` from `code/`). The numpy-only parts (counterexamples) were
executed and are reported as verified.

---

## 1. The counterexamples and the *implemented* LMO — RESOLVED, no action

> **Decision (Alexey, 2026-07-25): not an issue.** Newton–Schulz is only one way to
> approximate the idealized LMO. The counterexamples run the **exact** LMO, which is
> what the theorems are stated about; deep-learning runs use Newton–Schulz, which is
> what practitioners do. The published constants (`σ₁ = 1000`, `M = 100`) **stay as
> they are** and the code defaults are unchanged. The rest of this section is kept
> only as a record of what was measured.

Theorems 1–3 are stated for the **exact** Muon LMO `polar(Y) = UVᵀ`. The code that
trains networks uses 5 iterations of the Newton–Schulz quintic in bfloat16. These are
different maps, and the theorems' quantity of interest is a sign pattern, so an entry
of `polar(Y)` that is small in magnitude can flip.

Verified with `code/counterexamples/verify_ns_oracle.py`:

| | published instance | exact oracle | Newton–Schulz, 5 steps |
| :--- | :--- | :--- | :--- |
| Thm 1 SignMuon | `σ₁ = 1000` | ascends, `−42468/103` | **descends, `+3470.95`** |
| Thm 2 MuonUSign | `M = 100` | ascends, `−13.89` | **descends, `+5.86`** |
| Thm 3 MuonSign | `M = 100` | ascends, `−76` | ascends, `−76` |

Two of the three instances behave differently under the approximate oracle. Since
the theorems concern the exact oracle, this is a property of the approximation, not
a gap in the proofs.

### What was measured

*Theorem 1*: `G = σ₁u₁v₁ᵀ + O` has `cond(G) = σ₁ + 1`. At `σ₁ = 1000` five
Newton–Schulz steps cannot lift the `O` component out of the noise floor at all
(the iteration amplifies small singular values by ≈3.44 per step, so it needs
`log(σ₁)/log(3.44) ≈ 6` steps just to resolve them). The construction works for
*any* `σ₁ > 532/43 = 12.37`, with the closed form

    ⟨G, sign(polar(G))⟩ = (−43·σ₁ + 532)/103,

so `σ₁` was simply chosen far larger than it needs to be.

*Theorem 2*: the mismatch entry is present under Newton–Schulz too, but shallower
(`−0.05` at 5 steps vs `−0.2425` exact), which raises the threshold on `M` from
`42.7` to `216.5`. `M = 100` sits between them.

For the record, these constants ascend under the exact oracle *and* Newton–Schulz
for every step count in `{5, 6, 8, 10, 20}` in both float32 and bfloat16 — **not
applied**, kept only in case the question ever comes up in review:

| | change | new exact value |
| :--- | :--- | :--- |
| Thm 1 | `σ₁ = 1000 → 100` | `−3768/103 ≈ −36.58` |
| Thm 2 | `M = 100 → 500` | `−110.90` |
| Thm 3 | `M = 100 → 500` | `−476` |

Only the constants change; the proofs are "exhibit a `G` with `⟨G, s(G)⟩ < 0`", so
their structure is untouched. With these values, running all eight methods on the
two instances with the **bfloat16 5-step oracle** gives exactly the right verdicts:
SignMuon diverges on instance 1, MuonUSign and MuonSign diverge on instance 2, and
Muon, SignSGD, EF21-MuonUSign and EF21-MuonSign all descend
(`verify_ns_oracle.py --trajectories`).

`signmuon_counterexample(sigma1=...)` is now a parameter and the counterexample
optimizers accept `lmo=make_lmo("ns", steps=5, dtype="bfloat16")`, so either oracle
can be run on demand. **Defaults are unchanged**: the figures reproduce bit-for-bit.

---

## 2. Methods the paper claims but the code did not have

`tab:fed_master` lists six federated methods; the code had three. Table 1
(`tab:synthetic_results`), Table 2 (`tab:cifar_central`), Table 4
(`tab:exp_2`) and Table 5 (`tab:exp_3`) all have `-` entries for **MuonUSign**.
The reason was concrete: `optimizers.py` defined a class named `MuonSign` that was
actually the paper's MuonUSign, and **no CLI ever exposed it**.

Now implemented and wired to both entry points:

| | centralized | federated |
| :--- | :--- | :--- |
| SignMuon | was ✓ | was ✓ |
| MuonUSign | **added** (was present but unreachable, and misnamed) | **added** |
| MuonSign (both sides) | **added** | **added** |
| EF21-SignMuon | **added** | **added** |
| EF21-MuonUSign | was ✓ (`EF_USignMuon`) | was ✓ |
| EF21-MuonSign | was ✓ (`EF_UDSignMuon`) | was ✓ |

So every `-` in the paper's tables is now fillable, and `ef21signmuon` is available
as the *negative* baseline the paper's Theorem 4 predicts should diverge — worth
plotting, since demonstrating a predicted failure is strong evidence.

**Naming**: the code now uses the paper's names everywhere, including
`counterexamples/` and `nanogpt/` (which used `MuonUDSign` / `EF21-MuonUDSign`).
Legacy CLI spellings still resolve. There is deliberately **no** alias for the old
`MuonSign` class name, since resolving it silently would swap the algorithm rather
than the label. `nanogpt/train_nanogpt.py` (marked superseded) was updated for the
same reason — it imported `MuonSign` from `code/optimizers.py` and would otherwise
have quietly started running a different method. Notebooks were left alone: they
reference existing saved run directories by name.

---

## 3. Bugs fixed

1. **`zeropower_via_newtonschulz5` corrupted its input.** `X = G.to(dtype=bfloat16)`
   returns `G` itself when the dtype already matches, and the next line was
   `X /= norm` — an in-place divide on the caller's gradient buffer. Harmless with
   fp32 gradients, silently wrong with a bf16 model. Now normalizes out-of-place
   and clones when aliasing. `test_newton_schulz_does_not_mutate_input` covers it.
2. **`federated_muon` applied weight decay twice** — coupled into the client
   gradient *and* decoupled on the server (`param.mul_(1 - lr*wd)`).
3. **`federated_signmuon_client` folded a default `wd = 1e-4` into the gradient
   before the LMO**, while the EF21 paths used pure gradients. So SignMuon's LMO saw
   a shrinkage-perturbed gradient and EF21's did not.
4. **`--algorithm signmuon` crashed.** It was the default *and* the README's
   documented value, but argparse's `choices` list only had `signmuon_cl`.
   (Defaults bypass validation, so it worked until anyone typed it explicitly.)
5. **`train_mnist.build_optimizers` returned the string `"Incorrect optimizer
   name"`** on an unknown optimizer, which then hit `opt_main, opt_aux = ...` and
   raised an unrelated unpacking error. Also `main.py` offered `--optimizer
   ef_usignmuon` for MNIST, which routed straight into that branch.
6. **`train_cifar` built an AdamW for SignSGD and threw it away** (`return opt, None`).
7. **Multi-seed runs silently overwrote each other.** `run_name` did not contain the
   seed and `save_run` began with `shutil.rmtree(run_dir)`, so a five-seed sweep
   kept one seed. See §5.
8. **The synthetic benchmark scored EF21-MuonSign on the wrong model.** `p.data`
   holds the compressed broadcast model `W`; the convergence theory bounds the exact
   model `X`. `using_exact()` existed but was never called there. The reported
   3564 iterations is an `f(W)` number.
9. `federated_adam` logged `Round {r}` instead of `r+1`.
10. `federated_algorithms.py` imported `sklearn.cluster.KMeans` (never used) — a
    hard dependency that would break the import on a machine without scikit-learn.
11. **Every client re-read the first `n_steps` batches of a fresh epoch each round**
    (the iterator was rebuilt per round). With `shuffle=True` the samples were at
    least random, but a client's shard was never swept. The iterator now persists.
12. `norm_weight` removed. It rescaled every parameter to `‖p‖ = √numel` on every
    step, which is not part of any published method and would wreck training if the
    flag were ever passed. It was plumbed through both configs and the nanogpt CLI.

---

## 4. Fairness problems in the federated comparison (this invalidates Tables 4–5)

You chose "uniform per paper, re-run later". The seven near-duplicate
implementations are now **one parameterized driver** (`run_federated`), so these
can't recur:

* **Cosine LR decay was applied to `sgd`, `muon`, `ef21_muon` and `ef_ud_muon` but
  not to `signsgd`, `signmuon_client` or `adam`.** SignMuon was compared against
  EF21-MuonUSign with one of them on a decaying schedule and the other not.
* **1-D parameters (biases, BatchNorm scales) went through the sign/Muon rule**,
  not AdamW. The paper explicitly says otherwise, and the centralized code did
  follow the paper — so the two settings disagreed. Only the *last two tensors*
  were treated as the head.
* **SignSGD signed every parameter** including the head, while the Muon family got
  an AdamW head. Under a "same 1-bit budget" claim that is not apples-to-apples.
* Weight decay was applied inconsistently per method (see bugs 2–3).

All four are now uniform for every method. **Consequence: the federated numbers in
`tab:exp_2` and `tab:exp_3` no longer correspond to the code and need re-running.**
The centralized path is unchanged by default (`--head-adamw auto` reproduces the old
behaviour), so `tab:cifar_central` still stands.

One more thing to disclose rather than change: **BatchNorm running statistics are
never updated in the federated setting.** Local models are discarded each round and
BN runs in inference mode during accumulation, so the statistics stay at their
initialization `(0, 1)` for the entire run, in training and evaluation alike. BN is
effectively a fixed normalization with learnable affine parameters. This is
self-consistent (no train/test mismatch) and explains why federated CNN2 accuracy
sits where it does; it deserves one sentence in the reproducibility appendix. A
`--live-bn-stats` flag now exists but changes the numbers.

---

## 5. Multi-seed base (no runners written, as agreed)

* `utils.seed_everything` — one implementation, was duplicated in both mains.
* Seeded `torch.Generator` per DataLoader (`seed + client_id` federated), so data
  order and model init are independently reproducible.
* `saves/<run_name>/seed<seed>/` layout; `save_run` never deletes.
* **`utils.History` records the x-axis explicitly** and stores only evaluated
  points. The old format wrote one entry per round and forward-filled the previous
  value whenever `eval_freq > 1`, which turns unmeasured rounds into flat plateaus
  and makes a cross-seed mean meaningless. `aggregate.py` reads the legacy format
  too, with a warning.
* `aggregate.py` groups runs by config-minus-seed (ignoring seed / run_name /
  device / data path), averages pointwise on the common step indices, and reports
  mean ± sample std. It flags unequal seed counts and single-seed groups instead of
  printing a misleading `± 0`.
* Accuracies are recorded in **percent** in both settings (they disagreed before).

The paper currently says: *"All experiments use a single fixed random seed
(default 0)... Each reported number corresponds to a single such run."* For an
oral/spotlight this should become mean ± std over ≥3 seeds, at least for the
headline comparisons where the gaps are small — SignMuon `84.57%` vs
EF21-MuonUSign `84.59%` (Table 5) is a 0.02-point difference currently supported by
one run each, and the paper draws a conclusion from it ("performs on par").

---

## 6. Paper issues found while cross-checking (your call on each)

**a. Sign error, line 475.** The main text writes

    s_t^(j) = sign(A(M_t^(j)))

but `A` is defined as the *argmin*, so `A(M) = −UVᵀ` and `sign(A(M)) = −sign(UVᵀ)`.
Subtracting it ascends. Should be `sign(−A(M_t^(j)))`, or just `sign(D_t^(j))`.
Equation (13) and Algorithm `alg:fed_workerlmo` are both correct; this is an
isolated slip in the federated prose.

**b. Grid table (`tab:grid_search`) does not match anything that was run**, and two
reported optima are off their stated grid:

| method | reported `η` | stated grid | on grid? |
| :--- | :--- | :--- | :--- |
| Muon | `0.0065` | `[5e-3, 2e-2]` step `1e-3` | **no** |
| EF21-MuonUSign | `0.0033` | `[5e-3, 2e-2]` step `1e-4` | **no** (below range) |
| EF21-MuonSign | `0.0028` | `[5e-3, 2e-2]` step `1e-4` | **no** (below range) |
| SignMuon | `0.0002` | `[1e-4, 1e-3]` step `1e-4` | yes |
| SignSGD | `0.00015` | `[5e-5, 2e-4]` step `1e-5` | yes |
| SGD / Adam | `0.1` / `0.07` | `[1e-2, 1e-1]` step `1e-2` | yes |

The code's actual EF21 grid was `np.arange(0.0020, 0.0030, 1e-4)` — which contains
`0.0028` but not `0.0033`, so that number came from a run whose grid isn't in the
repo at all. This is the kind of thing a careful reviewer checks. Either amend the
table to the grids that were really used, or re-run: `synthetic_benchmark.py` now
has one documented grid per method (all eight, `--lr-grid` to override) and its
defaults are annotated with exactly which rows are and are not reproducible from
the paper's table.

**c. Momentum convention is inconsistent within the paper.** Equation (13) in the
main text uses heavy-ball `M_t = μM_{t-1} + G_t`; every algorithm box uses EMA
`μM_{t-1} + (1−μ)G_t`. I verified these give **identical** trajectories for all
eight methods — `M^EMA = (1−μ)·M^HB` exactly, and `sign`, `polar` and the EF21
recursion are all positively homogeneous, so the `(1−μ)` cancels
(`test_gradient_scale_invariance`). So this is cosmetic, but a reader will notice.
Suggest EMA everywhere plus a one-line remark that the two coincide — the remark is
worth having anyway, because it also tells the reader that `μ` does not need
retuning when `η` is retuned.

**d. Algorithm 1 (`MuonLMO`) omits Muon's aspect-ratio factor** `√max(1, m/n)`,
which the code applies (as does reference Muon). It is invisible to SignMuon and
MuonSign (they sign the output) but does rescale the step of Muon, MuonUSign and
EF21-MuonUSign, and it is *not* absorbable into `η` because layers have different
aspect ratios. Either add the line to Algorithm 1 or state that it is omitted.
`muon_lmo(..., scale_aspect=False)` gives the exact `UVᵀ` of the theory.

**e. Prose fixes.** Line 494 is garbled (`"effectively address: the baseline
majority-vote scheme and the error-feedback variant -- effectively address"`,
duplicated). Line 339: `"The same idea is used for create MuonSign. It also presents
in Appendix\ref{app:alg})."` — broken grammar and an unbalanced paren. Six
`Appendix\ref{...}` are missing the space and render as "AppendixA"
(lines 339, 452, 486, 494, 542, 626); line 584 separately reads "reported in
Appendix in Table~\ref{...}". Table 5's MuonUSign row reads `-%`.

**f. `bfloat16` precision deserves a sentence.** The reported experiments
orthogonalize in bfloat16 (~3 decimal digits). For the methods that sign the LMO
*output*, entries of `polar(M)` near zero can flip, so the iteration counts in
Table 1 on a deterministic 500×500 problem carry a precision-dependent component.
`--lmo-dtype float32` is now available; worth reporting which was used.

---

## 7. Refactor summary

`code/` is now a Python package, one subdirectory per experiment family, and every
command runs from `code/` as `python3 -m <package>.<module>`:

```
code/
├── REPRODUCE.md          exact command for every paper table and figure
├── aggregate.py          multi-seed mean +/- std
├── common/               optimizers.py, models.py, utils.py
├── centralized/          main.py, train.py, data.py
├── federated/            main.py, algorithms.py, data.py, grid.py
├── synthetic/            benchmark.py
├── counterexamples/      exact-LMO theorems + figures
├── nanogpt/              speedrun adaptation (unchanged)
├── tests/                test_code.py
├── notebooks/            plotting only
└── results/              all output: {centralized,federated,synthetic}/
```

All output now goes to one `results/` tree (was `saves/`, `saves_federated/`,
`saves_synthetic_01/`, `project/saves_synthetic_001/`, `output_grid/`), so
`aggregate.py` sweeps everything in one scan. `aggregate.py` still reads the old
directories too, so pre-reorganization runs are not lost.

Module-level changes:

* `common/optimizers.py`: six copy-pasted `step()` methods → one base class with a
  `_direction` hook; eight methods, ~30 lines each of genuinely different logic.
* `federated/algorithms.py`: 1631 → ~490 lines. Seven `local_train_*` +
  `federated_*` pairs, each re-implementing gradient accumulation, cosine
  annealing, evaluation and logging → one `run_federated` driven by a `MethodSpec`
  (`lmo` × `uplink` × `downlink`), which is literally the paper's `tab:fed_master`.
  Adding a method is now one dict entry. ~450 lines of commented-out dead code
  removed (it is in git history).
* `train_cifar.py` + `train_mnist.py` → `centralized/train.py` (they were ~90%
  identical and had drifted apart in three ways).
* `synthetic_benchmark_grad.py` + `synthetic_benchmark_conclusion.py` →
  `synthetic/benchmark.py --mode grid|final`, all eight methods, `--device`
  instead of a hardcoded `cuda:2`/`cuda:3`.
* `federated_efud_grid.py` → `federated/grid.py`, any method, any grid.
* New: `common/utils.py`, `aggregate.py`, `tests/test_code.py`,
  `counterexamples/verify_ns_oracle.py`, and **`REPRODUCE.md`** — one command per
  paper table/figure, with the published hyperparameters filled in, so a reviewer
  can regenerate any artifact without reading the code.

The load-bearing test is `test_federated_one_client_equals_centralized`: the
federated driver with `N = 1` must reproduce the centralized optimizer exactly, for
all eight matrix rules. That is what stops the two files from drifting apart again.

---

## 8. What I did not touch

* Notebooks — moved to `code/notebooks/` and renamed
  (`plot_{centralized,federated,synthetic,counterexample}.ipynb`) but *not* edited.
  They still read the old `saves*` paths and the old `EF-UDSignMuon/` directory
  names, so they need repointing at `results/` before the next round of figures.
* `nanogpt/train_gpt.py`, `train_gpt_a100.py`, `signmuon_optimizers.py` — renamed
  only; the logic is the current, tested speedrun code. (`train_nanogpt.py`, which is
  superseded, also had its import repointed to `common.optimizers`.)
* `scrap/` — no `UDSign` references remained.
* The paper itself. Every recommendation above is unapplied and reversible.
