# Divergence counterexamples

Self-contained, exact-SVD implementations of the eight optimizers from
*"SignMuon, MuonSign, and the Role of Error Feedback"*, run on the two linear
divergence counterexamples of Theorems 1-3 and on the universal 2x2 instance of
Theorem 4 (`th:ef_div`).

Every method reproduces the corresponding pseudocode box in the paper,
with two deliberate choices:

* **Exact SVD LMO.** The Muon LMO is the rank-truncated polar factor `U Vᵀ`
  computed from an exact SVD, not the Newton–Schulz polynomial used in
  practice. Only nonzero singular directions are kept (`r = rank`), because
  `U @ Vᵀ` from the full SVD is non-unique on rank-deficient inputs — and
  `sign(G)` is often low-rank.
* **Momentum.** EMA / heavy-ball momentum `Mₜ = μ·Mₜ₋₁ + (1−μ)·Gₜ` with an
  optional Nesterov look-ahead `M̃ₜ = (1−μ)·Gₜ + μ·Mₜ`. The runner defaults to
  `μ = 0.0`, standard (non-Nesterov) momentum — the `Optimizer` class default of
  `0.8` is always overridden by `run_counterexamples.py`. Both are parameters, and
  by Proposition 1 neither can change a verdict on the linear problems.

## Files

| File | Contents |
|------|----------|
| [`optimizers.py`](optimizers.py) | `muon_lmo`, `scaled_sign`, and the 8 optimizers (`OPTIMIZERS` registry) |
| [`problems.py`](problems.py) | The 4×4 and 5×5 linear counterexamples, the universal 2×2 EF21-SignMuon counterexample (exact theorem construction, per `(μ, variant)`), and a `_self_check` reproducing the paper's exact constants |
| [`run_counterexamples.py`](run_counterexamples.py) | Runs every method on all three problems, prints verdict tables, saves the figures |
| [`plot_ef21_momentum.py`](plot_ef21_momentum.py) | Runs EF21-SignMuon on the `(μ, variant)`-specific universal instance for `μ ∈ {0, 0.5, 0.9, 0.95, 0.99}`, both variants; shows divergence at the common slope `49/480` (momentum does not restore convergence) |
| [`verify_ns_oracle.py`](verify_ns_oracle.py) | The same instances under the *implemented* Newton–Schulz oracle instead of the exact SVD, over step counts and dtypes — which choices of `σ₁`/`M` survive the approximation |
| `figures/` | Generated plots (`*.png`, `*.pdf`) |

## Methods

Six paper algorithms + two references (`SignSGD`, `Muon`). `D = MuonLMO(·)`
is the exact polar factor; `sign` is elementwise; `scaled_sign(Y) = mean|Y|·sign(Y)`.

| Method | Update direction `d` (`X ← X − η·d`) |
|--------|--------------------------------------|
| `SignMuon` | `sign(LMO(M̃))` — sign **after** LMO |
| `EF21-SignMuon` | EF21 estimator of `LMO(M̃)` (scaled-sign on the residual) |
| `MuonUSign` | `LMO(sign(M̃))` — sign **before** LMO |
| `MuonSign` | `sign(LMO(sign(M̃)))` — sign before *and* after LMO |
| `EF21-MuonUSign` | `LMO(g_est)`, where `g_est` is a scaled-sign EF21 estimate of `M̃` |
| `EF21-MuonSign` | as above + a second EF21-P loop compressing the downlink model shift |
| `SignSGD` (ref) | `sign(M̃)` |
| `Muon` (ref) | `LMO(M̃)` |

## Running

Run from `code/`, the package root — not from this directory.

```bash
cd code
python3 -m counterexamples.problems             # reproduces −412.311, −13.888 and the 49/480 table
python3 -m counterexamples.run_counterexamples  # μ=0 by default; also prints −76; writes figures/
python3 -m counterexamples.run_counterexamples --nesterov
python3 -m counterexamples.run_counterexamples --mu 0.9 --eta 2e-3 --T 120
python3 -m counterexamples.plot_ef21_momentum   # momentum sweep for Counterexample 3
python3 -m counterexamples.verify_ns_oracle     # exact vs Newton–Schulz oracle
```

`run_counterexamples.py` writes **three** figures, to both `figures/` (PNG + PDF)
and `../aaai_article/images/counterexamples/` (PDF, which is what LaTeX includes).

For the linear objectives `f(W) = Tr(Gᵀ W)` the gradient is constant, so
`f` decreases **iff** the per-step descent inner product `⟨G, dₜ⟩` is positive.
The verdict column therefore tests the sign of `mean ⟨G, dₜ⟩` — the exact
divergence criterion, independent of `η` and `T` (`verdict_mode="inner"`).
Counterexample 3 uses `verdict_mode="slope"` instead: its ascent is
**second-order** (the compressor overshoot, not a downhill step), so `⟨G, dₜ⟩`
stays positive while `f` rises, and divergence is read off the positive tail
slope of `f`. It is rebuilt for the run's `(μ, variant)` and diverges for every
choice.

> At `μ = 0` (the default) the slope test cleanly separates EF21-SignMuon from
> everything else. At large `μ` it does **not**: the periodic term of the
> Counterexample-3 landscape scales with `A(μ)` (199 at `μ = 0.99`), so the
> bounded methods wander over a range that a *fixed absolute* `SLOPE_TOL` reads
> as ascent. The divergence of EF21-SignMuon is `μ`-independent — its exact rate
> is `49/480` for every `μ` — but the *verdict column* is only meaningful at
> small `μ`. Use `counterexamples.problems`, which measures over whole periods,
> to check the rate at any `μ`.

## Results (defaults: μ = 0, standard momentum)

**Counterexample 1 — SignMuon (Theorem 1, 4×4).** `G = 1000·u₁v₁ᵀ + O`, so
`LMO(G) = O` but `⟨G, sign(O)⟩ = −412.311 < 0`. Only **SignMuon** diverges;
every other method (including the EF21 variants) descends.

**Counterexample 2 — MuonSign / MuonUSign (Theorem 2, 5×5).** `sign(G) = S`,
and `polar(S)` disagrees with `S` at exactly the one inflated entry, giving
`⟨G, LMO(sign(G))⟩ = −13.888 < 0`. **MuonUSign** diverges — and so does
**MuonSign** here (`−76`), since its downlink sign flips that same entry.
All EF21 variants, `SignMuon`, `SignSGD`, and `Muon` descend.

In both linear cases the Error-Feedback methods restore convergence, matching
the paper's claims. Because the linear-problem step direction is invariant to
the momentum scaling, the divergence verdicts are identical under `μ ∈ [0,1)`
and under both momentum variants.

**Counterexample 3 — EF21-SignMuon (universal, Theorem `th:ef_div`).**
`ef21_signmuon_counterexample(mu, nesterov)` returns the *exact* function the
appendix theorem builds for the given momentum coefficient and variant — code
and proof describe one object. On a *linear* objective EF21-SignMuon cannot be
broken (its estimator converges to `polar(G)` and the step is genuine descent),
and a plain quadratic valley diverges only at `μ=0`; so the universal instance
instead *forces* a fixed sequence of LMO targets — a rotation `S₁`, a rank-one
map `S₂`, then alternating reflections `D̄^±` — regardless of `(μ, variant)`.
The reflections share a small diagonal while their `O(1)` off-diagonal flips
sign each step, so the shared scaled-sign magnitude `αₜ = mean|Δₜ| = 24/25`
overshoots the diagonal: the estimator's `(2,2)` entry locks into the period-2
cycle `{−61/200, +131/200}` of mean `+7/40 > 0`, opposite to its target
`−7/25`, and one coordinate of `X` marches off, so `f → +∞` at the exact rate
`49/480` per step. The function is
`f(W) = −γW₂₂ + A(Φ₁(W₁₂)+Φ₂(W₂₁)) + Σₖ bₖ(W)` (`γ=7/12`): a linear divergence
slope, two periodic ramps `Φᵢ` sustaining the alternating off-diagonal
gradient, and three compactly supported corrections `bₖ` seeding the first
three gradients; `A` and the `bₖ` depend on `(μ, variant)` (momentum is a
positive linear filter of the gradients, which we invert). Divergence holds for
**every** `L>0`, `η>0`, `μ∈[0,1)` and both variants — the iterate trajectory is
identical across them, and only **EF21-SignMuon** diverges while `Muon`,
`EF21-MuonUSign`, `SignSGD` and the others descend. It shows the `Θ(σ_min/L)`
step-size restriction of the conditional convergence theorem cannot be replaced
by any `(L, μ)`-only rule. Figures: `run_counterexamples.py` (the two-scale
`ef21_signmuon_counterexample`) and [`plot_ef21_momentum.py`](plot_ef21_momentum.py)
(the momentum sweep `ef21_signmuon_momentum`).
