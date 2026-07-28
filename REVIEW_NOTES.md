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

---

# Theory vetting — 2026-07-28

Full re-derivation of every theorem, lemma, corollary, proposition and remark in
`aaai_article/` (main text + the three `\input`ed appendices), in exact
rational/symbolic arithmetic where possible, plus an independent audit of every
citation of `EF_Muon_Friends/template.tex` (Gruntkowska et al., EF21-Muon) and a
code-vs-boxes cross-check. Experiments were out of scope except where a number
feeds a theoretical claim.

## A. What is confirmed correct

Everything load-bearing. Specifically re-derived and confirmed:

* **Prop. `prop:reduction`** — `M~_t = g_t G`, `g_t > 0` under both momentum
  rules; increment identity; divergence under non-summable steps.
* **Thm. 1** — `O` orthogonal; `O v1 = u1`; `<u1 v1^T, S> = -43/103`;
  `||O||_1 = 532/103`; `polar(G) = O` with `sigma = (1001,1,1,1)`;
  `<G, sign(polar(G))> = -42468/103`. The displayed numeric `G` matches
  `1000*u1 v1^T + O` in every entry to the printed precision.
* **Thms. 2-3** — `D_42 = -0.2425356250` (`= -1/sqrt(17)`); `sign(D)` disagrees
  with `S` at exactly `(4,2)` and nowhere else; `C = 10.3656412507`;
  `<G,D> = -13.8879213`; `<G, sign(D)> = -76` exactly.
* **Thm. 4 (EF21-SignMuon)** — the *entire* construction, in exact rationals:
  the alpha table `7/10, 21/20, 131/200, 24/25`; the period-2 cycle `d_B, d_A`
  from `t = 3` (checked to `t = 25`); every residual entrywise nonzero; the
  `(2,2)` average `+7/40` against the target `-7/25`; `X_0,X_1,X_2 = Z_1,Z_2,Z_3`;
  the constant shift `[[-7/20, 21/20],[-7/20,-7/20]]`; the residues `W_12 mod p_1`
  and `W_21 mod p_2` alternating `rho^+/rho^-` with the parity the proof needs;
  band disjointness including the wrap-around at `rho_2^- = 0`; ball disjointness
  `||Z_j - Z_k||_F >= 7/10 > 2r`; `max_t (X_t)_22 = 7/10` (so Rem.
  `rem:efsm_bdd` holds); the rescaling lemma; and `49/240` per period = `49/480`
  per step.
* **Thm. 4, the Nesterov branch** — this is the one step that could not be
  checked by hand, and it is right. Solving `H_3[0,1] = H_3[1,0]` symbolically
  in mu returns `T = 4(mu+1)(35mu^2+48mu+24)/(117mu(2mu+1))`, *identically equal*
  to the printed `T`. Under `mu = s/(1+s)` the two leading minors of `H_3` have
  numerators `702s^3+1291s^2+699s+120` and a degree-7 polynomial with
  coefficients `[600444, 2335316, 3919204, 3721944, 2156145, 751938, 144405,
  11700]` — all strictly positive, so `H_3 > 0` on `(0,1)` and
  `polar(M~_3) = Dbar^-`. `T > 0` throughout. Also confirmed:
  `Mbar^+- = Dbar^+- * [[24/25, -+7/25],[-+7/25, 337/300]]` with the second
  factor SPD and `det = 1`; singular values `3/4, 4/3`; `A_std = (1+mu)/(1-mu)`
  and `A_Nes = (1+mu)/((1-mu)(1+2mu))` both forced by
  `(1-mu)G^+- = Mbar^+- - mu Mbar^-+`. An independent float64 implementation
  written from the LaTeX alone reproduces `49/240` per period for
  `mu in {0, .5, .9, .95, .99}` x {standard, Nesterov} to `2.5e-14`.
* **Lem. `lem:signcontr`** — the identity, the `(1 - 1/d)` bound, the exact
  `alpha(Y) = (2 - ||Y||_0/d)||Y||_1^2/(d||Y||_F^2)`, the `2/pi` Gaussian limit,
  the `(2d-1)/d^2` 1-sparse value, and "`1/d` is an infimum, not attained".
* **Rem. `rem:normmismatch`** — the `Y_delta = (1-delta)I + delta*J` computation
  (`C(Y)-Y = (1-delta)(J/n - I)`, eigenvalues `0, -(1-delta)`,
  `||Y_delta||_{2->2} = 1+(n-1)delta`, ratio -> 1).
* **Rem. `rem:dimension`** — the downlink residual recursion is exactly what
  `central_alg_ud` produces.
* **Rem. `rem:gluon`** — the conic-combination bound
  `rhobar <= (sum_j alpha_j/rhobar^(j))^-1` is correct given the source's `rho`
  convention.
* **`eq:wd_ball`** — the decoupled-decay gain bound
  `gamma(W_t) <= max{gamma(W_0), 1/lambda_wd}`.
* **Unit-gain appendix** — `gamma(A) = ||A||_F/sqrt(m)`;
  `lambda_l = sqrt(m)/||s||_F`; the closed forms `sqrt(max(1,m/n))` and
  `1/sqrt(n)`; He/Kaiming gains `sqrt(2)` and `1/sqrt(3)`; the factor-13 span
  across ResNet-18; `sqrt(27) -> sqrt(512)`; the Nesterov-deviation inequality
  `E||grad f - M~_t||^2 <= (1-beta)^2 E||grad f - M_t||^2 + 3 beta^2 sigma^2`
  (correctly labelled an expectation, not a corollary).
* **`app:mishra`** — the internal algebra is right: the sign-error identity, the
  step from `<G, sign(polar(G))> < 0` to `sum |G_ij| q_ij > (1/2)||G||_1`,
  `R_T > G_T`, and the per-step increase `(eta/4)(42468/103)`.
* **The EF21-Muon reduction, against the actual source.** Every one of the eight
  citations lands on the right object. Def. 1, Alg. 3's loop order, the
  `beta = 1-mu` correspondence, the `K = T+1` index shift and the duplicated
  `X_0` term, Thm. 19's threshold `gamma_i <= (2 L_i^0 + 2 sqrt(zeta_i))^-1` and
  its `gamma`-weighted LHS, the sharp step `G^sharp = ||G||_* U V^T`, Rem. 23
  (exactly the two `alpha_P^-2` terms get the `rhobar_i^2` factor), App. D's
  `alpha > 1 - rhounder^2/rhobar^2` (so **`r_i`, not `sqrt(r_i)`**, and it really
  is App. D), dropout `alpha_P = p`, Top-K SVD
  `alpha_P = 1 - sigma_{K+1}^2/sigma_1^2`, `(rhounder_i, rhobar_i) =
  (1, sqrt(r_i))` from their Rem. 7, and the Assumption correspondence including
  "`f_j >= f_j^*` only for the `(L0,L1)` result". The source imposes **no**
  bounded-iterate, smoothness-region or client-sampling hypothesis, and `N = 1`
  is covered.
* **The `cor:l0l1` footnote is accurate.** Their printed second step-size
  condition in Thm. 24 *is* tighter than the display its own proof establishes,
  by exactly `K+1`. (The same slip is in their `p = 1` Thm. 6.) The correction is
  load-bearing: the printed version tightens with `K` and would kill the
  constant-`eta_i` claim.
* **Code vs. algorithm boxes** — no sign placement is swapped in any of the three
  independent implementations; the update is `X <- X - eta D` with `D = +U V^T`
  everywhere; EF21 residuals and `alpha = mean|Delta|` match; the exact-SVD LMO
  truncates to rank `r`; the NS quintic is an odd polynomial in `X` so it never
  fabricates a null-space completion; unit-gain is implemented as
  `eq:unit_gain_closed` with the paper's family assignment; `59/59` tests pass.

## B. Errors — things that are stated and are not true

> **Status 2026-07-28.** All of B1–B11 are fixed, as are C1–C5 and C7–C10,
> C13, C14. Still open and needing a decision rather than an edit: **C6** (the
> `{±1}` / "1 bit" / "32×" framing — the measured alphabet is ternary at 8–17%
> zeros, ~1.37 bits/symbol and ~23×, and changing the headline numbers is an
> author call), **C11** (hoisting `as:1`/`as:2` into the main text),
> **C12** (notation clashes on `mu` and `alpha`), all of **D** (needs the
> Mishra et al. paper, which is not in the repo) and the paper-side half of
> **E**. B1 was fixed by deleting the false clause rather than by raising `M`
> to 500: raising it would have changed Thm. 3's `-76` and the published
> `-13.89`, and forced a figure and code regeneration, for no gain given the
> theorems are stated over the exact oracle.

**B1. The Newton-Schulz verification claim in the proof of Thm. 2 is false.**
`v2_SignMuon_AAAI.tex:747` says of `<G,D> = -13.89`: *"which we verified for both
the exact polar factor and the Newton--Schulz approximation used in practice."*
At the paper's own 5-step quintic (`a,b,c` of Alg. 1, Frobenius normalisation)
the value is **`+5.86 > 0` — it descends.** Measured:

| NS steps | `D_42` | `C` | `<G,D>` at `M=100` |
| ---: | ---: | ---: | ---: |
| 3 | -0.00473 | 11.033 | **+10.56** |
| 5 | -0.05033 | 10.897 | **+5.86** |
| 6 | -0.38635 | 7.642 | -30.99 |
| 8 | -0.43300 | 8.021 | -35.28 |
| exact | -0.24254 | 10.366 | -13.89 |

It turns negative only from 6 NS steps, or at `M >~ 216` with 5. This is the
number `REVIEW_NOTES section 1` already recorded (`+5.86`); the decision taken
there was to state the theorems for the exact oracle and keep the constants —
which is fine — but the sentence claiming NS verification was never removed.
**Either delete the clause, or raise `M` to 500** (`<G,D> = -110.90` exact,
negative at every NS step count in `{5,6,8,10,20}`), which costs nothing since
the proof is "exhibit a `G`". Thm. 3 needs no change: `<G, sign(D)> = -76` under
every oracle.

**B2. "Compressing *before* the oracle removes the coupling" (line 428) is
wrong.** EF21-MuonUSign uses the *same* single magnitude
`alpha_t = mean|Delta_t|` coupling all coordinates (`eq:ef21_central`,
`central_alg_ef`). Nothing is decoupled. The appendix states the real reason
correctly (`ef21_musign_reduction.tex:136`): EF21-SignMuon "tracks the
non-Lipschitz polar factor and the momentum-tracking step breaks". That is
exactly what the counterexample exploits — `D_t` jumps by `Theta(1)` between
`Dbar^+` and `Dbar^-` regardless of `eta`, whereas the EF21 lemma needs a target
that moves `O(eta)` per round. The main text should say that, not "removes the
coupling".

**B3. Coupled weight decay: `L -> L + lambda_wd` is off by a factor `r_i`
(line 1194).** Adding `(lambda_wd/2)||W||_F^2` contributes
`grad h = lambda_wd W`, and under **Assumption `as:2`** (nuclear-dual /
spectral-primal) the sharp constant is
`||lambda_wd Z||_* <= lambda_wd * r_i * ||Z||_{2->2}` (tight at `Z = I_r`). So
the rates carry over with `L_i -> L_i + lambda_wd r_i`. `L -> L + lambda_wd` is
the Euclidean statement, and the paper's rates are not Euclidean.

**B4. The floor coefficient in `app:images_task` (line 824) has one power of
`||S||_F` too many.** Balancing `eq:descent_lemma` gives
`eta rho ||grad F||_F ||D||_F = eta^2 L ||D||_F^2 / 2`, i.e.
`||grad F||_F = eta L ||S||_F / (2 rho)`. The text writes `L ||S||_F^2 / 2rho`.
(The predicted slope of 1 and the "gap attributable to rho alone" remark are
unaffected.)

**B5. "...cannot shrink `W_t` at all" (line 1192) contradicts the same
paragraph.** Fourteen lines later: "the update degenerates towards
`-sign(W_t)`" — which does shrink `||W||`. The correct claim is that coupled
decay cannot change the *step length* (fixed by `eq:two_families`) and therefore
supplies no contraction factor; it acts by rotating the direction, and in the
`rho_t >> 1` limit that rotation happens to point at `-sign(W_t)`.

**B6. `thm:conv` (i) and the main-text display (lines 297-301) drop the
`gamma`-weights.** The source bounds `sum_i w_i E||grad_i f||^2` with
`w_i = gamma_i/((1/p) sum_l gamma_l)`; the two coincide only for a **common**
`gamma_i`, but `thm:conv` (i) says "tuned `(gamma_i, mu)`" and then writes the
unweighted quantity. `cor:smooth` states it correctly, and the appendix preamble
flags it — the headline statements do not.

**B7. `tab:dict` puts the sign downlink in `B_2(alpha_P)`.** Thm. 19 *as printed*
requires `C_i^k in B(alpha_P)` (layer norm); only Rem. 23 licenses the `B_2`
variant, at the `rhobar_i^2` price. The proof of `cor:smooth` says this; the
table and its caption do not.

**B8. `cor:smooth` cites [Cor. 2], which is stated for `p = 1`.** The layer-wise
momentum tuning is not literally in the source (their Cor. 1 only substitutes the
initialisation). The extension is routine, but cite "Cor. 1-2" or say "the same
tuning applied per layer".

**B9. `as:2`'s `(L0,L1)` sentence drops the subscripts.** It writes
`L^0 + L^1 ||grad_i g(X)||_*`, but their Assumptions 8/9 carry `L^0_i, L^1_i` and
`L^0_{i,j}, L^1_{i,j}` — and `cor:l0l1`'s own `L^1_{i,max}` presupposes the
subscripted form.

**B10. Fig. 2's caption cites the wrong lemma**
(`ef21_signmuon_divergence.tex:31`): mu/variant-independence of the trajectory
comes from `lem:efsm_realize` (each mu gets its own `f~` producing the same
target sequence), not from `lem:efsm_reduction` (which removes `L` and `eta`).

**B11. Line 469, `s_t^(j) = sign(A(M_t^(j)))` — still an ascent step.** Carried
over from sections 6a/15, re-confirmed against the current `.tex` and the code.
`A` is the argmin, so this is `-sign(polar(M))`. Should be `sign(D_t^(j))`.

## C. Overstatements and scope gaps — true-ish, but a referee will press

**C1. Thms. 1-3 are proved on a linear objective, which violates Assumption
`as:1`.** Every convergence guarantee in the field assumes `f >= f^*`, so as
stated the counterexamples sit outside the assumption set and a referee can call
them vacuous. The fix is one remark and costs nothing: the iterates move along a
fixed ray on which `f` increases, so replacing `f` by any `C^infty` function
agreeing with `<G,W>` on `{f >= f(X_0) - 1}` and constant far below leaves the
whole trajectory unchanged while making `f` bounded below — *exactly* the trick
already used in `rem:efsm_bdd` for Thm. 4. Worth doing, if only for consistency
of rigour between Thm. 4 and Thms. 1-3.

**C2. "The matrices involved have condition number 16/9" (line 425 and
`ef21_signmuon_divergence.tex:56`) is not literally true.** It holds for the tail
`Mbar^+-`/`Dbar^+-`, which is where the divergence lives. But the preamble target
`S_2` is **rank one** (cond = infinity), and in the Nesterov branch
`M~_1 = S_1 H_1` has cond `= 1+T -> infinity` as `mu -> 0+` (`8.2e7` at
`mu = 1e-8`). Rephrase as "the matrices driving the divergent tail".

**C3. Thm. 4 tacitly fixes the rank-truncated LMO selection.** At `t = 2` the
input `M~_2 = S_2` is rank one, where the spectral-ball LMO is genuinely
non-unique — and the reduction appendix itself says so ("At rank-deficient `G`
the LMO is non-unique, but any selection works"). For the *convergence* side any
selection does work; for a *counterexample* the selection is part of the claim.
Line 343 does define `polar` via the rank-truncated SVD, so the paper is
internally consistent, but the theorem should say it. Relatedly, line 343 calls
that object "the orthogonal polar factor" — for rank-deficient `Y` it is a
partial isometry, not orthogonal.

**C4. "An exhaustive numerical search found no such `G` for `N_dim <= 3`"
(line 673) is not exhaustive, and no code in the repo produces it.** `grep` finds
no script backing this claim anywhere under `code/`. It is also weaker than what
is provable. Write `G = QH` (polar, `H >= 0`); then

    <G, sign(polar(G))> = <H, sym(Q^T sign(Q))>,

so **a counterexample with polar factor `Q` exists iff
`lambda_min(sym(Q^T sign(Q))) < 0`**, and the whole question is a property of
`O(n)` alone. For `n = 2` this *proves* the claim: writing `Q` as a rotation
`[[c,-s],[s,c]]` or a reflection `[[c,s],[s,-c]]`, in both cases
`Q^T sign(Q) = (|c|+|s|) I + (antisymmetric)`, and `H` symmetric kills the
antisymmetric part, so `<G, sign(polar(G))> = (|c|+|s|) tr(H) > 0`. (Numerics:
`min lambda_min = 1.000` over `2e5` Haar `Q`.) For `n = 3` several independent
searches — `5e5` Haar draws, Nelder-Mead/Powell multistart, and a vectorised
annealed hill-climb — bottom out at `+0.005 ... +0.046`, never negative. For
`n = 4` the same search reaches `-0.72`, and the paper's own `O` gives
`spec sym(O^T sign(O)) = (-0.4175, 1.5825, 2, 2)` — i.e. the published instance
is not even the worst one. **Recommend:** replace "exhaustive numerical search"
with the `n = 2` proof plus "no `3x3` instance was found in a search over
`O(3)`", and state the `lambda_min(sym(Q^T sign(Q)))` characterisation — it is
two lines and it turns a soft claim into a sharp one.

**C5. The unit-gain rule's premise fails for the oracle actually used.**
`eq:unit_gain_closed` is derived from `||U V^T||_F = sqrt(r)` *exactly*. The
5-step Newton-Schulz step used in every reported experiment is **6-22% shorter**
than `sqrt(min(m,n))`
(`test_newton_schulz_step_is_measurably_shorter_than_the_exact_lmo`; the drivers
print a `gain spread` warning past 1.15x). So the realized per-layer gain of the
LMO family is not flat, and `eta_0` is not exactly the per-step RMS gain for
those methods. Worth one sentence in `app:lrscale`.

**C6. The `{+-1}` / "1 bit" / "32x" framing is still unqualified.**
`sign(0) = 0`, so the alphabet is ternary on both the majority-vote uplink and
the EF21 uplink; the repo measures **8-17% zeros** on CNN2, `~1.37` bits/symbol
and a `~23x` (not `32x`) uplink reduction. Line 476's "the resulting sign is
equal to `+1` or `-1` in each component" is false as implemented — the code's
majority vote abstains on ties (`mv_ties="zero"`) and the paper never mentions
ties. Carried over from section 12; still open.

**C7. Fig. 1's caption over-generalises mu-independence (line 404).** The
sentence follows the list of baselines and reads as covering the whole figure,
but `prop:reduction` covers only the three sign placements. Muon is also
mu-independent (`polar(gamma_t G) = polar(G)`), but **SGD and the two EF21
baselines are not**: under EMA momentum `M_t = (1-mu^t)G`, so their steps depend
on mu (only asymptotically not). Restrict the sentence to the three methods it is
about.

**C8. "...is always a descent direction, since `<grad f, U_t V_t^T> ~=
||grad f||_*`" (lines 445-447).** `U_t V_t^T` is the polar factor of the
*momentum* (indeed of the EF21 estimator of it), not of `grad f`; the identity
`<grad f, polar(grad f)> = ||grad f||_*` holds only when the two coincide. The
`~=` is doing a lot of work, and this is the one place where the paper's
rhetorical claim about EF21-MuonUSign is not backed by the appendix. Either drop
it or point to `cor:smooth`.

**C9. "...transfer without any new analysis" (line 297)** — `lem:signcontr` and
`prop:instance` are new analysis. "...transfer without re-deriving the rates"
would be accurate.

**C10. Rem. `rem:dimension`'s "the framework admits iteration-dependent
`alpha`"** is a shade firmer than the source's Rem. 12, which says it "admits a
straightforward extension to iteration-dependent compression parameters" — i.e.
asserts extensibility rather than proving it.

**C11. Assumptions `as:1`/`as:2` are stated after their first use.** They live in
`ef21_musign_reduction.tex`, which is `\input` at line 764 — *after*
`ef21_signmuon_divergence.tex` (line 762), whose Thm. 4 invokes both, and long
after the main text's first reference at line 260. Consider hoisting the two
assumptions into the main text.

**C12. Notation clashes.** (a) Line 788 uses `mu_j` for eigenvalues of `B` while
`mu` is the momentum coefficient everywhere else. (b) Line 240 writes S-Muon's
step as `alpha U V^T + (1-alpha) eta sign(M_t)` — the `eta` is the paper's
learning-rate symbol appearing inside a conic combination. (c) `alpha` is the
compressor contraction (`lem:signcontr`), the exponent in
`lambda_l^sign = n^-alpha`, *and* the S-Muon mixing weight.

**C13. `app:images_task`'s "the nonconvex bound our theorems prove gives
`p = q = 1/2`" is only true if `err` is the *squared* gradient norm.** The
theorems give `min_t E||grad f||_*^2 = O(T^-1/2)`; if `err` is the norm itself
the prediction is `p = 1/4`. `err` is never defined. (Also: the stability column
reports `eta_max ||S||_F`, but the "predicted `2/L`" row is SGD's `eta_max`, and
for SGD `||S||_F = ||grad F||_F` is not a constant.)

**C14. "Both converge on the counterexamples above" (line 223)** — on a linear
objective there is nothing to converge to. They *descend*.

## D. Not checkable from this repo

* **`app:mishra` depends on quoting `\citet{mishra2026signmuon}`'s inequality
  (`eq:mishra_generic`) and their `R_T` estimate faithfully.** The internal
  algebra is right, but no copy of their paper is in the repo, so the
  transcription of their bound — and hence the fairly strong claim that "their
  finalized rate in fact covers only the gradient-sign instantiation" — could not
  be verified here. Given it is an assertion about a concurrent submission, it is
  worth one more read against their actual text before submission.
* The `330`-configuration / top-ten weight-decay observation attributed to them
  (line 1192), same reason.

## E. Code/paper mismatches with theoretical bite

(From the code cross-check; see also B11.)

1. `run_federated`'s Python default is `lr_scaling="legacy"` while the CLI
   default is `unit-gain` — a programmatic caller silently gets the old
   convention.
2. The synthetic arm applies **no** per-layer rule (`benchmark.py:349-359`), an
   exception to `app:lrscale`'s "for both families and every experiment reported
   here". Harmless (square problem, per-method eta) but stated too broadly.
3. `lmo_dtype` defaults to `bfloat16`, which can flip near-zero entries of
   `polar(M)` for the two sign-after methods. Documented in code, absent from the
   paper.
4. Federated runs use `freeze_bn_stats=True` for the whole run; not stated.
5. `centralized/train.py` reports EF21-MuonSign's *training* loss at `W` but its
   test/val at `X`, so Fig. 2's bottom row mixes iterates for that one method.
6. `code/counterexamples/problems.py` builds `psi_i` with half-period plateaus
   and sine ramps rather than the paper's `delta`-bands, and its cutoff
   `cos^2(pi s/2)` equals 1 only at `0` whereas `eq:bump` wants `phi == 1` on
   `[0,1/2]`; the resulting `f` is `C^1`, not `C^infty`. Trajectories coincide,
   and the module docstring flags most of it — but the claim that code and LaTeX
   "describe one object" is not literally true.
7. `REPRODUCE.md` section 4a still documents the Table-2 commands under
   `--lr-scaling legacy`; the current results are unit-gain. Stale.

---

# Code debugging pass — 2026-07-28

Six parallel audits (counterexamples vs. the theorems; federated vs. centralized
vs. the algorithm boxes; synthetic + per-layer LR + the centralized loop; every
`.md` against the argparse definitions; the test suite and every entry point;
nanoGPT and the support modules), then fixes. Complements the theory vetting
above, which covered the `.tex`; this one covers `code/`.

**Environment note.** `torch` was not installed and `C:` had 0.6 GB free, so the
test suite and every torch-dependent check had been unrunnable. Installed CPU
torch 2.8.0 into a venv on `D:` (`D:\signmuon-venv`), outside the repo. Three of
the six audits fell back to NumPy replicas because of this; their conclusions
were re-confirmed afterwards against the real code.

## A. Bugs fixed — these changed results

1. **`problems.py:219,226` — `Phi` was not the antiderivative of `psi`.** Both
   down-ramp branches integrated the wrong function: branch 3 should be
   `(2τ/π)·cos(π(t−p/2)/(2τ))`, branch 5 should carry a minus sign. Consequences:
   `loss_fn` was *not* the potential of `grad_fn` (max FD error 2.0 at μ=0, 398 at
   μ=0.99), and `Phi` jumped by `8τ/π` at every period boundary — so the "single
   smooth `f`" the proof promises was not the object the code evaluated, and every
   plotted `f` curve used it. **Theorem 4 is unaffected**: `grad_fn` was always
   correct and `Phi` was exactly `p`-periodic, so the per-period objective change
   never moved (`f(X_{t+2})−f(X_t) = 0.204166666666667` vs `49/240`). After the
   fix `max|grad_fn − FD(loss_fn)|` is 1.6e-08 / 1.1e-07 / 3.3e-06 at
   μ = 0 / 0.9-Nesterov / 0.99, i.e. finite-difference truncation error.
2. **Parity-mismatched slope window** in `plot_ef21_momentum.py:112` and
   `run_counterexamples.py:125`. `f` is period-two, and the window compared
   `f[999]` (odd) with `f[500]` (even), leaking half an oscillation into the
   slope — so the script the appendix cites as *confirming the exact rate* never
   printed it (0.0963–0.1003 instead of 0.1020833). The denominator was also off
   by one. Both now report exactly `49/480` for all ten (μ, variant) settings, and
   `run_counterexamples` shows `0.0000` for every non-diverging method instead of
   noise.
3. **`anonymize.py:197` — one ALLOW substring silenced an entire line, all
   rules.** 144 lines in `code/` contained an ALLOW token and were exempt from the
   whole scan. The exempt shape ("ported from KellerJordan's repo by \<author>
   \<email>") is exactly the shape a real leak takes here. ALLOW is now applied
   per *match*. Also: personal-fork repo URLs (`github.com/<handle>/SignMuon`) and
   names glued to other letters (`AlexKravatsky`) were missed — identifiers are
   now split into a bare-match list and a word-boundary list. Nine leak shapes
   that previously passed are now caught; `--check` is still clean on the tree.
4. **`--scale-baselines` was a silent no-op for `adam`** (`algorithms.py:789`
   `continue`s past the only place `lam` is applied). The overnight driver runs a
   whole `(adam, scale_baselines=True)` track — a tuning grid plus three final
   seeds — and reported "the better of the two" as evidence Adam was not
   handicapped. `server_adam` now gets one param group per tensor with `lr·λ`
   folded in, annealed from each group's own `initial_lr`.
5. **`grid-paper` overwrote `grid.json`.** `--grid-preset paper` shares
   `mode="grid"`, so it wrote the same file as the tuned grid — and `--mode final`
   reads that file. The "checkable discrepancy" could only be checked by
   destroying the result it was compared against. Now writes `grid-paper.json`.
6. **`aggregate.py` crashed** (`TypeError: '<' not supported between int and
   NoneType`) on any run without a `seed` key — i.e. exactly the pre-refactor
   files `load_runs` goes out of its way to support. Also `n_seeds` counted runs,
   not distinct seeds, and counted runs that lacked the metric. Now reports
   `n_runs` / `n_distinct_seeds`, prints `n/a` rather than `± 0.0000` for a single
   run, and warns on repeated seeds and on runs missing the metric.
7. **`--stage lr`'s "fine" pass re-ran configurations it had already measured.**
   `refine_grid` returns lattice *neighbours* and `coarse` is a run of consecutive
   lattice points, so for an interior winner all three were duplicates — ~30% of a
   ~10 GPU-h stage, and the reported config count was wrong. Now deduped.
8. **`run_federated(decoupled_weight_decay=…)` was dead** for all eight matrix
   methods (the server step shrank `X` unconditionally); no CLI ever set it. The
   federated coupled-decay ablation was unreachable. Now wired to a real
   `--weight-decay-mode`, matching `centralized.main`.
9. **`sgd`'s decay convention differed between the drivers** — coupled centrally
   (deliberately: SGD's step is not scale-invariant), decoupled federated. Now
   coupled in both.
10. **`run_federated`'s library default was `lr_scaling="legacy"`** while the CLI
    default is `unit-gain`. Aligned; the equivalence tests now pass `legacy`
    explicitly, since that is genuinely the convention they compare against.

## B. Test coverage — the load-bearing check did not pin the theory at all

This is the most important finding of the pass, and it was established by
**mutation testing**: injecting a bug in memory and seeing whether the suite
notices.

**Nothing pinned any step rule to its algorithm box.** The
federated↔centralized equality tests pin the two *implementations to each other*,
so a bug applied consistently to both is invisible. Measured, before the fix:

| injected bug | old suite |
| :--- | :--- |
| `SignMuon`→`MuonSign`'s formula, centralized only | caught |
| **`SignMuon` ↔ `MuonSign` swapped consistently in both** | **all 9 federated tests pass** |
| **EF21-before-oracle ↔ EF21-after-oracle swapped consistently** — literally the Theorem 4 distinction | **all 9 federated tests pass** |

The second and third are the ones that matter: the entire paper turns on *where*
the sign and the EF21 recursion sit relative to the LMO, and the suite could not
tell. The only tests that fired at all on the EF21 swap did so on incidental
artifacts — a state-dict key name, and a hard-coded `name != "ef21signmuon"`
exclusion elsewhere. Rename the key and the suite goes silent.

Added **`test_each_direction_is_the_documented_formula`**: a `_reference_trajectory`
helper transcribing each algorithm box literally and independently (sharing no
code with `common/optimizers.py`), driven by a fixed iterate-independent gradient
sequence over four momentum steps, plus a pairwise-distinctness guard so a
consistent swap cannot satisfy it by accident. Re-ran the mutations against it:

```
CAUGHT   SignMuon <-> MuonSign swapped consistently
CAUGHT   EF21-SignMuon <-> EF21-MuonUSign swapped consistently
CAUGHT   SignMuon signs before the LMO
CAUGHT   MuonSign drops the downlink sign
CAUGHT   EF21 scale = max|delta| instead of mean|delta|
```

**Weight decay was pinned only at `lambda == 1`.** `TinyNet`'s only matrix is
`(4, 6)` at the default `legacy` rule, so every multiplier is exactly 1 and
"decay is *not* scaled by the per-layer factor" was untestable. Scaling decoupled
decay by `lambda_mult` passed all 9 federated and all 4 decay tests. Added
`test_decoupled_decay_is_not_scaled_by_the_per_layer_multiplier` (direct, zero
gradient, `lambda = 0.25`) and
`test_the_two_drivers_agree_on_weight_decay_at_a_nontrivial_multiplier`; both
catch that mutation, and the pre-existing test still cannot — which is the point.

**The aspect factor was vacuous too.** Same `(4, 6)` cause: `sqrt(max(1,m/n)) == 1`,
so the bit-for-bit claim the READMEs make ("the factor lives in `lambda`
federated and in `scale_aspect` centrally") held trivially. Added `TallNet`
(`(12, 4)`, factor `sqrt(3)`), generalized the rule comparison into
`_compare_drivers_under_rule(rule, net, loader, weight_decay)`, and ran the
`unit-gain` equivalence on **both** fixtures.

**`muonserver` was absent from `CENTRAL_CLASSES`** with no explanation. Pinned by
`test_federated_muonserver_equals_centralized_muon_at_one_client` — at N=1 there
is nothing to average, so centralized `Muon` is the right reference. No federated
matrix method is now unpinned.

**`counterexamples/` was imported by no test** — the package that prints the
published Theorem 1–3 numbers. The suite re-implemented the instances in torch
inside `test_code.py`, so the two could drift apart silently. Added
`test_the_counterexample_package_reproduces_the_theorem_constants` and
`test_the_exact_svd_lmo_truncates_to_the_rank` (the rank-truncation property is
load-bearing — `sign(G)` is often low-rank and untruncated `U @ Vt` is non-unique
there, but the suite's local `exact_polar` does not truncate and only ever saw
full-rank Gaussians).

Also: `common/utils.seed_worker` existed, was documented, and was **wired into no
`DataLoader`** — dead code backing a promise. Now passed to the training loader.
`pytest` was documented as a supported runner and absent from `requirements.txt`.

## C. Removed (all recoverable from git)

`scrap/` — superseded by `counterexamples/`, and two of its scripts wrote
`ef21_signmuon_momentum.pdf` straight into `aaai_article/images/counterexamples/`
from the **superseded quadratic-valley** construction, overwriting the live
appendix figure. `nanogpt/train_nanogpt{,_classic}.py` — stale forks;
`train_nanogpt.py` imported only 7 of 8 methods (EF21-SignMuon, the Theorem 4
method, was missing) and used the retired legacy scaling. Three pre-fix nanoGPT
EF21 logs. `plot_nano.ipynb`, `log_synt.txt`, and `notebooks/` (all four read
`saves*/` and the retired `EF-UDSignMuon`). Three pre-rewrite shims in
`federated/data.py` that were exported and called from nowhere.

## D. Added

* `common/plotting.py` — one palette, one method→colour map, one save helper, so
  a method keeps its colour across every figure in the repo.
* `synthetic/plot_synthetic.py` — `loss`, `GN` (`fig:synthetic_results`), `floor`,
  `horizon` (`fig:synthetic_dynamics`), `kappa` (`fig:synthetic_kappa`). **The
  last three `\includegraphics` are commented out in the paper** — those files had
  never been produced by anything.
* `federated/plot_federated.py` — per-method curves with a ±1 sample-std band over
  seeds, grouped by `aggregate.group_key`.
* `synthetic/run_gpu --m/--n` — override the problem size for every stage, keeping
  the real grids, writing to `results/synthetic_<M>x<N>/` so a small pass cannot
  land where the reported numbers belong.
* `federated.tune --stage votes` — the majority-vote alignment table that
  justifies `--n_parties 11`. It is quoted in three places and was produced by
  nothing; it is now a command, and reproduces to Monte-Carlo error.
* `--stage anchors` now prints the per-layer multipliers and spreads for both
  families and the per-layer step-length ratio, which three docs claimed it
  already did.
* `centralized/train.py`'s summary now prints train accuracy and test loss, which
  two READMEs claimed it did.

## E. Documentation corrections of record

* **`REPRODUCE.md` §4a documented a table the paper no longer contains** — the
  `legacy`/`--head-adamw auto` seven-row block. Not one `lr` matched
  `tab:cifar_central`, which is now ten rows, three seeds, `unit-gain`. Rewritten.
* Every `Table N` / `Figure N` in `REPRODUCE.md` was stale against the current
  float order, in a file that already warned against numbering. All now labels.
* **"12 of its 20 conv layers … ~63% of the parameters"** (in `REPRODUCE.md` and
  `centralized/README.md`) is wrong twice over. Measured: **13 of 20**, holding
  **84.5%** of all parameters. The 63% is a different fact — the single shape
  `(512, 4608)`, appearing three times, is 63.3% of the model.
* **"SignSGD reaches only ~93% train"** is contradicted by the paper's own table
  (99.97%, and every method within 0.05 points).
* **"the test set is never loaded"** (six sites) is false — `load_raw` reads both
  splits. What is true, and is what matters, is that no test image is ever scored
  or ranked on under `--split tune`. Reworded.
* `counterexamples/README.md` — every command in it failed (wrong cwd, not `-m`);
  the stated default `μ = 0.8` is `μ = 0.0`; and it still asserted
  `MuonUSign = MuonSign`, the retired convention, in a repo whose naming section
  says otherwise.
* `verify_ns_oracle --mu/--nesterov` do not exist. `plot_ef21_momentum`'s μ set is
  `{0, 0.5, 0.9, 0.95, 0.99}`, not `{0, 0.25, …}`. `tests/README.md` said 43
  checks. "ten methods" → eleven in four places. `--verify-horizon` is documented
  in `tune.py` and does not exist.
* `63.7×` for `fc1`'s step-length ratio is not derivable from the stated formula;
  it is **67.9×** (`√(mn)/√min(m,n)`), now printed by `--stage anchors`.
* nanoGPT is **not** "not part of the paper's tables" — `tab:nanogpt` and
  `fig:nanogpt` are main text.
* `--live-bn-stats` does not make BatchNorm statistics live: the local model is
  rebuilt each round and never written back, so the flag *introduces* a train/eval
  mismatch rather than removing one. Help text now says so.
* EF21-MuonSign's train metrics are at `W` and its test/val at `X` — unavoidable
  (the gradient must be at `W`) but now stated in the summary and at the recorder.

## F. Still open

* **`--grid-preset paper` is ~5× the cost it is billed at**: the paper grids are
  linear and fine, so the preset is ~3850 configurations against the default's
  ~733, while `run_gpu` estimates it at "~2 h" next to `grid`'s "~3 h".
* **The two `overnight.py` drivers duplicate ~430 lines** of scheduler (19 of 23
  top-level names shared; five functions byte-identical). Worth lifting the
  state/report/schedule/sigint core into one module.
* **Unsourced measurements of record**: the 8–17% uplink-zero range, the
  `ns_steps`/gain-spread table, the polar-averaging shrinkage table, and the
  floor/iteration table (quoted at momentum 0.2 and `η = 2e-4`, neither of them a
  default). The floor table now carries the exact command that would reproduce it;
  the others still need either a script or a "measured once, on this hardware"
  label.
* The nanoGPT sharded path has **never been exercised with real collectives** on
  this machine — the gloo leg skips on Windows (no libuv). The passing result is
  the single-process simulation. Run it on Linux before the next LM runs.
* `train_gpt.py` is CRLF while `train_gpt_rec40_reference.py` is LF, so a plain
  `diff` against the reference shows the whole file as changed, defeating the
  point of keeping it. Its diffs are also unmarked, unlike `train_gpt_a100.py`'s.
