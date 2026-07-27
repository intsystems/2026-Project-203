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

---

# Federated overhaul — 2026-07-27

Follow-up review, focused on the federated experiments. §4 above said Tables 4–5
needed re-running for *fairness* reasons; this pass found that the protocol
itself was the larger problem, and brought it into line with the centralized one.

## 9. The federated learning rates were tuned on the test set

`federated/grid.py` launched each configuration, scanned the training log for the
accuracy printed at `--eval-round`, and ranked by it. That accuracy was **test**
accuracy: the federated setting had no validation split at all — `partition_data`
split the CIFAR test set across clients and that was the only held-out data. With
ten methods and several rates each, the rates in Table 5 rest on dozens of peeks
at the test set.

This is squarely at odds with what the paper says two sections earlier about the
centralized protocol ("Selection uses a fixed 45k/5k train/validation partition
and validation accuracy alone… **the test set is never consulted during
tuning**"). A reviewer who reads both sections will notice.

**Fixed.** `federated/data.py` now holds out the same 5000 images the centralized
path holds out (same `val_seed = 12345`, same permutation), *before* the client
partition, so no client ever sees one. `--split tune` reports validation accuracy
and **never loads the test set**; `--split full` partitions all 50k and reports
test accuracy. `federated/tune.py` replaces `grid.py` with the 1–2–5 lattice,
equal budget per method, and a boundary check; `grid.py` is now a stub that
fails with a pointer rather than silently running the old protocol.

## 10. Per-layer learning rates, and why they matter more here

The TODO at the top of `federated/algorithms.py` is done: the driver takes
`--lr-scaling`, applies `η_layer = η₀·λ(family, shape)` and switches
`scale_aspect` off inside `muon_lmo`, exactly as the centralized path does. A
`family` field on `MethodSpec` carries the classification rather than inferring
it.

Measured on CNN2, at one global rate and `η = 1`, the realized step norms are

| method | conv1 | conv2 | fc1 |
| :--- | ---: | ---: | ---: |
| SignMuon / MuonSign / SignSGD, `‖s‖_F/√min(m,n)` | 8.7 | 40.0 | 63.7 |
| Muon / MuonUSign / EF21-*, same normalization | ~0.5 | ~0.5 | ~0.4 |

so the sign family's steps are 8.7×–63.7× the LMO family's *and* span 7.3×
among themselves. CNN2's three matrix parameters have `fan_in` 75, 1600, 4608,
giving the sign multiplier a **7.8× spread** — wider than ResNet-18's, and
without ResNet-18's problem of one shape holding 63% of the parameters. If the
centralized `--log-gain` measurement comes out ambiguous, CNN2 is the better
instrument and this is where to repeat it.

Grid anchors are *transported* rather than re-guessed:
`anchor(rule) = anchor_legacy · geomean(λ_legacy)/geomean(λ_rule)`, evaluated on
the model's own shapes. Under `unit-gain` on CNN2 that leaves the LMO family
untouched and multiplies the sign family by 28.65. `python3 -m federated.tune
--stage anchors` prints the table.

`--lr-scaling legacy` reproduces the old convention bit for bit
(`test_federated_legacy_rule_is_the_old_convention` pins it), so nothing
published is lost.

## 11. Bugs and inconsistencies fixed in this pass

1. **The auxiliary AdamW was weight-decayed federated but not centrally.**
   `centralized.train.build_optimizers` passes `weight_decay=0.0` for the
   auxiliary group ("`--weight-decay` describes the matrix parameters only");
   the federated driver passed the run's `weight_decay`. So the two settings
   were running different algorithms under one name. Note that
   `test_the_two_drivers_agree_on_the_weight_decay_convention` did **not** catch
   this: its "centralized reference" is hand-rolled inside the test and decayed
   the auxiliary group too, i.e. the test pinned the federated driver to a
   convention the real centralized path does not use. Both are corrected.
2. **`weight_decay` defaulted to `5e-4` federated and `0` centrally**, and the
   paper's protocol paragraph says 0 is the primary setting. Now 0 in both, with
   `5e-4` as an ablation phase.
3. **The `adam` baseline used coupled `torch.optim.Adam`** while every other
   method used decoupled decay, and centrally the same baseline is AdamW. Now
   AdamW when the convention is decoupled, so the baseline is not handicapped by
   an Adam-vs-AdamW difference on top of the comparison being made.
4. **Per-client buffers round-tripped across PCIe every round.**
   `buffer_device` defaulted to `"cpu"` and `federated/main.py` never overrode
   it, so each client's momentum and EF21 buffers were copied device→host→device
   once per round — ~240k transfers over a 2000-round, 10-client run, to save a
   few tens of MB of VRAM. The default is now the compute device.
5. **Empty client shards surfaced as `cannot reshape tensor of 0 elements`**
   from inside the model, several call frames from the cause. Now rejected at
   partition time with a message that names the clients.
6. **Dead code in the Dirichlet partition.** The class indices were re-fetched
   and re-shuffled at the end of every class iteration, doing nothing but
   advancing the global RNG. The partition also drew from the *global* numpy RNG;
   it now takes an explicit generator (`--partition-seed`, defaulting to the run
   seed), so it is reproducible on its own.
7. **`requirements.txt` was empty**, while `REPRODUCE.md` opened with
   `pip install -r requirements.txt`.
8. **Progress lines used `\r` with `end=""`**, so a log had one enormous line and
   anything reading it incrementally blocked until an evaluation round. Now
   line-oriented.

## 12. The uplink alphabet is ternary — two wrong claims in the paper

`sign(0) = 0`, so a client transmits a symbol from `{−1, 0, +1}`, not `{−1, +1}`.
This is not a corner case: `polar(M)` has an exactly-zero column wherever `M`
does, and `M` does wherever a feature was zero across the whole local batch —
after ReLU and MaxPool, common. Measured on CNN2 with the real driver, **8–17% of
transmitted entries are zero every round**.

Two paper claims are affected.

**(a) "1 bit per parameter" / "32× reduction".** At a 10% zero rate the symbol
entropy is `H(0.45, 0.10, 0.45) ≈ 1.37` bits, so the real reduction is ~23×, not
32×. The driver now records `uplink_zero_frac` and prints the implied
bits/parameter in the run summary. `--uplink-zeros {random,positive}` maps the
zeros to `±1` and makes the channel a genuine one bit; it costs nothing in
expected descent, because a zero LMO entry carries no directional information.
The EF21 uplink is deliberately left ternary — `α·sign(Δ)` is zero exactly where
the estimator is on target, and forcing it off by `α` is the mechanism Theorem 4
exploits.

**(b) "the aggregate is equal to +1 or −1 in each component"** (main text). False
as implemented, and — note — *not* rescued by using an odd client count, since a
zero vote is not `±1`. Measured alignment `A = E[⟨truth, ŝ⟩]/(mn)` over 400k
coordinates, each client correct with probability 0.65 and silent with
probability `q`:

| `N` | tie % (`q`=0) | `A` (`q`=0) | tie % (`q`=0.10) | `A` (`q`=0.10) |
| ---: | ---: | ---: | ---: | ---: |
| 9 | 0.00 | 0.6573 | 8.38 | 0.6183 |
| 10 | 15.32 | 0.6558 | 9.23 | 0.6410 |
| 11 | 0.00 | 0.7029 | 7.22 | 0.6676 |
| 15 | 0.00 | 0.7747 | 5.54 | 0.7406 |

**Action for the paper:** state the measured zero rate and the implied bit cost,
and either restrict the `±1` sentence or say that ties abstain.

**The client count is now 11, not 10.** At a strictly `±1` uplink an even count is
wasted: `N = 10` delivers the same alignment as `N = 9` and differs only in tying
~15% of coordinates, because an even vote's extra voter is never decisive. At
CNN2's real zero rate that parity effect washes out and alignment is simply
monotone in `N` — but 11 still beats 10 (0.668 vs 0.641), so 11 is the default.
The tie *rule* is irrelevant either way (0.6580 vs 0.6581 at `N = 10`); `zero`
abstains, `random` restores `‖s‖_F = √(mn)` by adding noise of matching size.

## 13. Table 5 is missing two of the six methods

`tab:fed_master` lists six federated methods and the abstract names MuonSign, but
Table 5 has SignMuon, MuonUSign (`-`), EF21-MuonUSign and EF21-MuonSign — no
**MuonSign** row and no **EF21-SignMuon** row. EF21-SignMuon is the method
Theorem 4 predicts should diverge, so it belongs there as a predicted-failure
baseline, exactly as it does in the nanoGPT table. All ten are now runnable and
the overnight driver covers all ten by default.

## 14. Speed

A federated run was documented at 3–6 h. The dominant costs were the default
`--eval_freq 1` (2000 evaluations of 10k images) and ten `DataLoader`s with
`num_workers=0` doing PIL decode and augmentation on the main thread.

* `--eval_freq` now defaults to 100, and the tuning driver derives it from the
  round count so that the tail mean always covers the last quarter of training
  at any horizon.
* `--loader gpu` (the default on CUDA) uploads the dataset once as `uint8`
  (153 MB for CIFAR-10 train) and does the random crop, flip and normalization as
  tensor ops on the device. The augmentation is torchvision's — zero-padded
  random 32×32 crop with `padding=4`, then a random horizontal flip — and
  `test_gpu_crop_and_flip_match_torchvision_distributionally` compares the
  per-row padding probability against `RandomCrop` itself over 2000 draws rather
  than just checking shapes. `--loader torch` keeps the old path.

## 16. Federated debug pass — three findings about the *motivation*

A full invariant sweep over the federated path (determinism, split disjointness,
shard coverage, EF21 bookkeeping, cosine schedule, driver plumbing) came back
clean. The problems that remain are not bugs in the code; they are places where
the experiment does not measure what the paper says it measures.

**a. There was only one full-precision Muon, and it is the control for the wrong
template.** `muon` orthogonalizes on the *worker* and the server averages the polar
factors. That is the right control for SignMuon and EF21-SignMuon. It is the wrong
one for MuonUSign, EF21-MuonUSign and EF21-MuonSign, whose uncompressed limit is
`polar(mean M)` — one LMO on the server. The two are not interchangeable, because
averaging near-orthogonal matrices shortens the step. Measured on a 120×4608
matrix, clients sharing a signal plus independent noise of relative size `q`,
reported as `‖·‖_F/√min(m,n)`:

| `q` | N=1 | N=3 | N=11 | N=21 |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.767 | 0.767 | 0.767 | 0.767 |
| 0.3 | 0.768 | 0.744 | 0.735 | 0.734 |
| 1.0 | 0.769 | 0.622 | **0.559** | 0.548 |
| 3.0 | 0.768 | 0.483 | 0.324 | 0.286 |

`polar(mean)` is flat to 0.3% from N=1 to N=21 at every noise level. So the
reference loses up to 27% of its step length at N=11 purely to averaging, by an
amount that drifts over training as the gradient-to-noise ratio changes — a tuned
`η₀` absorbs a constant handicap, not that one. Comparing the server-LMO family
against `muon` confounds "what does the 1-bit uplink cost?" with "what does
averaging orthogonal matrices cost?", and it flatters the compressed methods.

**Fixed**: `muonserver` added (one `MethodSpec` entry), anchored at Muon's rate,
in the tuner's method list and the overnight sweep. Each family now has the
uncompressed control from its own template.

**b. The "32× reduction" is an uplink-only figure, and SignMuon's downlink is
uncompressed.** Computed rather than quoted (`communication_bits`), on CNN2:

| method | uplink | downlink | **round trip** |
| :--- | ---: | ---: | ---: |
| SignMuon, MuonUSign, EF21-SignMuon, EF21-MuonUSign | 22× | 1× | **1.9×** |
| MuonSign, EF21-MuonSign | 22× | 29× | **25×** |

Three corrections: the ternary alphabet (1.37 bits, §12); the auxiliary group,
never compressed either way, which alone puts a *perfect* 1-bit uplink at 1.087
bits/parameter model-wide; and, dominating both, the full-precision broadcast.

**This is the argument for the bidirectional methods, not against the paper.** The
abstract's own framing — "compressing the downlink as well is where the guarantee
starts to cost something, and we measure what" — is exactly right. What is not
defensible is the sentence attaching ~32× to SignMuon. **Action:** quote the round
trip and let EF21-MuonSign carry the communication claim.

**c. The per-layer rule does not equalize the *realized* gain for the LMO family.**
The unit-gain derivation assumes the oracle returns `‖polar‖_F = √min(m,n)`
exactly. Newton–Schulz does not, and its error is shape-dependent, so the realized
gain `γ = ‖λ·s‖_F/√fan_out` — the quantity the rule exists to flatten — is not flat.
Measured on `muonserver`:

| dtype | `ns_steps` | gain spread | conv1 | conv2 | fc1 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| bfloat16 | 5 (default) | **1.34×** | 0.875 | 0.943 | 0.705 |
| bfloat16 | 7 | 1.11× | 0.888 | 0.940 | 0.846 |
| bfloat16 | 12 | 1.15× | 0.951 | 0.893 | 0.829 |
| float32 | 5 | 1.35× | 0.873 | 0.945 | 0.700 |
| float32 | 12 | 1.28× | 0.947 | 0.893 | 0.742 |

`fc1` (120×4608, aspect 38:1, 72% of the matrix parameters) realizes only 0.70 of
the assumed gain. The sign family is flat by construction (±1 steps) and sits near
1.08×. A tuned `η₀` is one constant and cannot absorb a per-layer factor; raising
`ns_steps` shrinks the error but never removes it, because the quintic oscillates
in a band around 1 rather than converging.

Left at `ns_steps = 5`: that is what reference Muon uses and what the published
numbers used, and 1.34× is under half a learning-rate lattice step (2–2.5×), so it
cannot flip a ranking. But the paper should say that the rule equalizes the
*intended* step exactly and the *realized* step to ~1.3× for the LMO family, the
gap being the oracle's rather than the rule's.

**Now measured, not assumed**: the per-layer profile is printed at round 1,
`gain_spread` is recorded at every evaluation, and the driver warns above 1.15×.
The honest reading is that unit-gain equalizes the *intended* step exactly (that is
`test_unit_gain_equalizes_the_per_step_gain_exactly`) and the *realized* step to
within ~25% for the LMO family, limited by the oracle rather than by the rule.

## 15. Still open (paper text, not code)

Unchanged from §6 above and re-confirmed against the current `.tex`:

* **line 458**, `s_t^(j) = sign(A(M_t^(j)))`: `A` is the *argmin*, so this
  ascends. Should be `sign(-A(·))` or `sign(D_t^(j))`. Equation (13) and both
  algorithm boxes are correct.
* **line 474** still reads `"... effectively address: the baseline majority-vote
  scheme and the error-feedback variant -- effectively address ..."`.
* six `Appendix\ref{...}` without a space (lines 469, 474, …) render as
  "AppendixA"; Table 5's MuonUSign row still reads `-%`.
* Algorithm 1 still omits Muon's `√max(1,m/n)` factor, which the code applies.
