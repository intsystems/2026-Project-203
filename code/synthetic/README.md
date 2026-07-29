# `synthetic/` — the smooth convex benchmark

`F(X) = ½⟨X−C, A(X−C)B⟩` on `m × n` matrices. [`benchmark.py`](benchmark.py)
has seven modes; [`batched.py`](batched.py) is how they are executed.

The point of this problem is that **nothing has to be estimated**. `A` and `B` are
symmetric with a prescribed spectrum, so the Hessian is the Kronecker product
`B ⊗ A`, the smoothness constant is exactly `L = maxᵢⱼ λᵢμⱼ` and the strong-convexity
constant exactly `σ = minᵢⱼ λᵢμⱼ`, both in closed form. `∇F` and `F` are closed-form
too, so there is no autograd graph — which is what makes the bidirectional method
tractable here: its gradient must be taken at the broadcast model `W` while its
metrics are scored at the exact model `X`, and with a closed-form gradient either
point is one matmul away.

```bash
python3 -m synthetic.run_gpu --force      # every stage, ~1.6 h on one GPU
python3 -m synthetic.run_gpu --archive    # rebuild SUMMARY.md + the .zip
```

Everything is at `100 × 100` and averaged over three independent draws of
`(A, B)`.

`--force` is what makes the first command a *rerun*: a stage whose
`<method>/<mode>.json` is already on disk is skipped, so without it a box that
has run before skips all seven stages and exits in seconds looking like a
success. It overwrites `results/synthetic/` and the `.zip` beside it. Drop it
only on a fresh tree, or to resume a run that died part-way.

## The modes

Each is a reading of the descent lemma

`F(X_{t+1}) ≤ F(X_t) − η⟨∇F(X_t), D_t⟩ + (η²L/2)‖D_t‖²_F`,

whose first term is the rate and whose second is the floor.

| mode | what it measures |
| :--- | :--- |
| `alignment` | `ρ_t = ⟨∇F, D_t⟩ / (‖∇F‖_F‖D_t‖_F)` along the tuned trajectory, against the closed forms |
| `floor` | The plateau `F∞(η)`, `‖∇F‖∞(η)` of a constant step and their exponents in `η` |
| `horizon` | `err ~ T^-p`, `η* ~ T^-q` with `err = min_t ‖∇F‖²_*`, tuned separately at each budget |
| `stability` | The largest stable `η`, with SGD as the control that must return `2/L` |
| `kappa` | The tuned comparison at a controlled condition number, swept over decades |
| `grid` / `final` | Iterations to `F ≤ 1e-3`, `η` and `μ` tuned per method, and the curves at those optima |

`horizon` fits `p` on the **dual** norm the theorems bound — `ℓ1` for the sign
family, whose LMO minimizes over the `ℓ∞` ball, and nuclear for the LMO family,
whose ball is spectral (`DUAL_NORM` in `benchmark.py`). It reports the exponent
of the *squared* dual norm, which is the quantity in the theorem statements, and
carries the Frobenius exponent alongside so rows stay comparable across families.

## The grid is a batch

Every configuration in a sweep is an independent trajectory over the same
problem, so `batched.py` runs the whole `(η, μ, schedule)` grid at once: the
iterate is a `[B, m, n]` stack, the hyperparameters are `[B, 1, 1]` tensors, and
one step advances all `B` runs. This is worth doing because the sequential loop
was **latency bound, not compute bound** — at these sizes a step is ~50 tiny
CUDA kernels plus two device-to-host syncs, so an A4000 got ~1450 steps/second
whether the matrices were 64×64 or 100×100 and whether the batch was 1 run or,
now, 100. The corollary is that `--m` is not a cost knob: shrinking the problem
does not make a sweep cheaper, and only the grid does.

Two things had to be got right. The EF21 compressor's magnitude
`α_t = mean|Δ_t|` reduces **per slice** — a global `.mean()` over the stacked
residual broadcasts perfectly well and silently couples every run in the batch
to every other. And the metrics accumulate on the device: `best_f`, `best_‖∇F‖`,
the convergence step and the divergence flag are reduced with masked tensor ops
and read back once, at the end, rather than pulled to the host every iteration.
A configuration that has converged, diverged or run out of budget is masked out
where `run_one` would have broken out of its loop, and dropped from the batch at
the next compaction.

`--runner sequential` is the original one-at-a-time loop. It is the reference
implementation, not a fallback:
`tests.test_code.test_batched_runner_reproduces_the_sequential_one` checks the
two agree on all ten methods over a grid mixing all three schedules, a per-config
budget and an oversized step, and `run_gpu` runs it as a preflight before
committing the box to a sweep.

It compares a short horizon deliberately. A batched matmul reduces in a different
order from a single one, so the two runners see gradients differing at the last
float32 bit; that stays at `1e-7` for ~20 steps and then amplifies, because `sign`
is discontinuous and an entry within rounding of zero flips. That is a property
of these methods — the one this paper is about — and any change of GPU or BLAS
does the same. What it costs the reported numbers, measured at `100 × 100` over
800 steps: SignMuon, MuonSign, Muon, SignSGD and SGD agree exactly, the EF21
methods to `1e-4`, MuonUSign and Adam to `3e-3`, and the alignment statistics to
`1e-5`. Every table entry is a statistic over a trajectory — a plateau level, a
fitted slope, a distribution of `ρ` — and those are stable; a single late iterate
is not, and none is reported.

## Read the iteration count for what it is

For every sign-family method the iteration count on this problem is *exactly*
inversely proportional to `η`, and there is a hard `η` above which the method never
reaches the target at all. That is the signature of a constant step size on a
problem whose minimizer it cannot reach: a `±1` step has fixed length `η√(mn)`, so
the iterate settles into a ball of that radius and `F` plateaus. The tuner returns
the largest `η` whose plateau still fits under the target, so "iterations to target"
is `const/η_max` — a measurement of the **accuracy floor**, not of the descent rate.

At matched `η` the ranking can therefore invert: SignMuon's advantage over
SignSGD is a lower floor, while SignSGD descends faster per step. The two effects
pull in opposite directions and belong in the write-up separately — which is what
`floor` and `horizon` are for. The current values are in `SUMMARY.md`, floor and
budget-scaling sections.

`alignment` is the mode that is about the methods rather than the protocol. The
descent lemma needs `ρ_t > 0` and the divergence theorems drive it negative, so its
distribution along the trajectory is the quantity of interest. The closed-form
references it is read against are `ρ = 1` (SGD), `ρ = ‖G‖₁/(‖G‖_F√(mn)) → √(2/π)`
(SignSGD) and `ρ = ‖G‖_*/(‖G‖_F√r)` (Muon).

## Two conventions that move numbers

* **Grids are logarithmic, five decades wide, and set per step-norm family.**
  A sign step has length `η√(mn)` and an LMO step `η√r`, so at `100×100` their
  optimal `η` are a decade apart and they need different windows, `1e-5:1e0` and
  `1e-4:1e1`. Each ends past the largest stability edge `--mode stability`
  measures for its family (`η_max` reaches `0.134` for a sign step and `1.89`
  for an LMO one), so no stable `η` falls outside the search — a stable step
  size the grid cannot reach is a censored optimum waiting to happen. The points
  above the edge are nearly free: they diverge within a few steps and the runner
  retires a diverged trajectory rather than running it out.
  Override a whole family at once with `--lr-grid sign=lo:hi:xN`
  or `--lr-grid lmo=...`; a method name still overrides one method. Enumerating
  the ten grids by hand instead is what put `muonusign` and `ef21signmuon` — both
  of which take an LMO-length step — on the sign window, so keep them derived
  from `SIGN_FAMILY`/`LMO_FAMILY` as `family_lr_grids` does.
  An optimum landing on an edge is an upper bound rather than a tuned value; the
  tuner prints `[BOUNDARY]`, records `on_grid_boundary` in the JSON, and
  `SUMMARY.md` daggers the censored cell. Such a row should have its grid
  widened before it is reported.
* **`--spectrum uniform`** (the default) leaves the condition number to the draw
  — near `1e4` at `m = n = 100`. `--spectrum logspace --kappa K` fixes `L = 1`
  and `L/σ = K` exactly, which is what `kappa` sweeps.

Results land in `results/synthetic/`, with `SUMMARY.md` collecting every table
and `MANIFEST.json` the commit, GPU, CPU, torch/CUDA versions and wall time per
stage. See [`../REPRODUCE.md`](../REPRODUCE.md) §3.
