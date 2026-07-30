# Divergence counterexamples

Exact-SVD implementations of the eight optimizers of *"SignMuon, MuonSign, and the
Role of Error Feedback"*, run on the two linear instances of Theorems 1–3 and on
the universal 2×2 instance of Theorem 4 (`th:ef_div`). Everything here is CPU,
numpy only, and finishes in under a minute.

```bash
cd code                                          # the package root
python3 -m counterexamples.problems              # the exact constants
python3 -m counterexamples.run_counterexamples   # fig:divergence_plot
python3 -m counterexamples.enumerate_minimality  # the Thm 2-3 minimality claims
python3 -m counterexamples.verify_ns_oracle      # exact vs Newton-Schulz oracle
```

`run_counterexamples` and `plot_ef21_momentum` are the only scripts in the
repository that write into `aaai_article/`: each figure goes to `figures/` as
PNG + PDF and to `../aaai_article/images/counterexamples/` as the PDF LaTeX
includes.

## Files

| File | Contents |
|------|----------|
| [`optimizers.py`](optimizers.py) | `muon_lmo`, `sign_pm1`, `scaled_sign`, the 8 optimizers (`OPTIMIZERS`) |
| [`problems.py`](problems.py) | the 4×4 and 5×5 linear instances, the universal 2×2 EF21-SignMuon instance built per `(mu, variant)`, and a `_self_check` reproducing the paper's constants |
| [`run_counterexamples.py`](run_counterexamples.py) | every method on all three problems: verdict tables and `counterexamples_main` |
| [`plot_ef21_momentum.py`](plot_ef21_momentum.py) | EF21-SignMuon at `mu ∈ {0, 0.5, 0.9, 0.95, 0.99}`, both variants, against the common slope `49/480` |
| [`enumerate_minimality.py`](enumerate_minimality.py) | which shapes admit a sign/polar mismatch at all |
| [`verify_ns_oracle.py`](verify_ns_oracle.py) | the same instances under the *implemented* Newton–Schulz oracle, over step counts and dtypes |

## Three conventions

* **Exact SVD LMO.** `muon_lmo` is the rank-truncated polar factor `U Vᵀ` of an
  exact SVD, which is what the theorems are stated about — not the Newton–Schulz
  polynomial used in practice. Only nonzero singular directions are kept, because
  `U @ Vᵀ` from a full SVD is non-unique on rank-deficient input, and `sign(G)` is
  often low-rank. `verify_ns_oracle` measures what the practical oracle does
  instead; the two are **not** interchangeable here.
* **Randomized `sign(0)`.** Zeros map to an independent random `±1`, as in the
  paper, so every sign channel is a strict bit. Each optimizer owns its generator
  (`sign_seed`, default 0), so a run depends only on its own seed. No constant in
  Theorems 1–3 is affected — none of those matrices, or their oracle outputs, has
  a zero entry. What the convention does reach is the seven bounded methods on the
  Theorem 4 instance, whose field gradient has an exactly-zero `(1,1)` entry by
  construction: they stay bounded either way, and EF21-SignMuon's own trajectory
  is untouched, every residual along it being strictly nonzero.
* **Momentum** is the EMA form `Mₜ = μ·Mₜ₋₁ + (1−μ)·Gₜ` with an optional Nesterov
  look-ahead `M̃ₜ = (1−μ)·Gₜ + μ·Mₜ`, `μ = 0` by default. That default costs
  nothing: on the linear objectives the step is the same matrix for every
  `μ ∈ [0,1)` and either variant (Proposition 1), and on the Theorem 4 instance so
  is the entire trajectory.

## Methods

`D = muon_lmo(·)` is the exact polar factor, `sign` is elementwise,
`scaled_sign(Y) = mean|Y|·sign(Y)`.

| Method | Update direction `d` (`X ← X − η·d`) |
|--------|--------------------------------------|
| `SignMuon` | `sign(LMO(M̃))` — sign **after** the LMO |
| `MuonUSign` | `LMO(sign(M̃))` — sign **before** |
| `MuonSign` | `sign(LMO(sign(M̃)))` — **both** sides |
| `EF21-SignMuon` | EF21 estimator of `LMO(M̃)`, refreshed by a scaled sign of the residual |
| `EF21-MuonUSign` | `LMO(g_est)`, `g_est` a scaled-sign EF21 estimate of `M̃` |
| `EF21-MuonSign` | as above, plus an EF21-P loop compressing the downlink model shift |
| `SignSGD` (ref) | `sign(M̃)` |
| `Muon` (ref) | `LMO(M̃)` |

## How a verdict is reached

On a linear objective `f(W) = Tr(GᵀW)` the gradient is constant, so `f` decreases
**iff** the per-step inner product `⟨G, dₜ⟩` is positive. The verdict for
Counterexamples 1–2 is therefore the sign of `mean ⟨G, dₜ⟩` — the exact criterion,
free of `η` and `T` (`verdict_mode="inner"`).

Counterexample 3 needs a different test: its ascent is second-order, driven by the
compressor overshooting rather than by a downhill step, so `⟨G, dₜ⟩` stays positive
while `f` rises. It is read as the theorem states it — `f(X_{t+2}) − f(X_t) = c > 0`
for every `t` past the transient — and a method is called divergent when that
period-two increment is positive throughout the tail and constant to 5%
(`verdict_mode="period"`). An absolute tolerance on the tail *slope*, which this
script used until 2026-07-30, does not work: the bounded methods oscillate over a
range that scales with the instance's periodic constant `A(μ)` (199 at `μ = 0.99`),
so over a fixed window their apparent slope scales with it too, and a threshold
tight enough to catch the ascent at `μ = 0` reports every bounded method as
ascending at `μ = 0.9`. The per-period increment has no such failure mode — a
bounded trajectory cannot hold a constant positive one — and separates the eight
methods correctly at every `μ ∈ {0, 0.25, 0.5, 0.9, 0.95, 0.99}`, both variants, at
`T = 60` and at `T = 400`.

## Results (defaults: μ = 0, standard momentum)

**Counterexample 1 — SignMuon (Theorem 1, 4×4).** `G = 1000·u₁v₁ᵀ + O` with `O`
orthogonal and `Ov₁ = u₁`, so `LMO(G) = O` exactly while `⟨G, sign(O)⟩ =
−42468/103 ≈ −412.311 < 0`. Only **SignMuon** diverges; every other method,
including both EF21 variants, descends.

**Counterexample 2 — MuonUSign and MuonSign (Theorems 2–3, 5×5).** `sign(G) = S`,
and `polar(S)` disagrees with `S` at exactly one entry, the deepest mismatch any
5×5 sign matrix admits (`D₄₂ = −1/√17`). Loading that entry gives
`⟨G, LMO(sign(G))⟩ ≈ −13.888` and `⟨G, sign(LMO(sign(G)))⟩ = −76`, so
**MuonUSign** and **MuonSign** both diverge on the *same* instance. `SignMuon`,
`SignSGD`, `Muon` and all three EF21 methods descend.

In both linear cases error feedback restores convergence, matching the paper.

**Counterexample 3 — EF21-SignMuon (Theorem 4, universal 2×2).**
`ef21_signmuon_counterexample(mu, nesterov)` returns the *exact* function the
appendix builds for that momentum coefficient and variant, so code and proof
describe one object. EF21-SignMuon cannot be broken on a linear objective — its
estimator converges to `polar(G)` and the step is genuine descent — and a plain
quadratic valley breaks it only at `μ = 0`; so the universal instance instead
*forces* a fixed sequence of LMO targets: a rotation `S₁`, a rank-one map `S₂`,
then alternating reflections `D̄^±`, whatever `(μ, variant)` is. The reflections
share a small diagonal while their `O(1)` off-diagonal flips sign every step, so
the shared magnitude `αₜ = mean|Δₜ| = 24/25` overshoots the diagonal: the
estimator's `(2,2)` entry locks into the period-two cycle `{−61/200, +131/200}`,
of mean `+7/40 > 0` against its target `−7/25`, and the coordinate it drives runs
off, so `f → +∞` at exactly `49/480` per step. The function is
`f(W) = γ·h(W₂₂) + A(Φ₁(W₁₂) + Φ₂(W₂₁)) + Σₖ bₖ(W)` with `γ = 7/12`: a linear
divergence slope (levelled off above `W₂₂ = 1`, which no iterate reaches, so that
`f` is bounded below), two periodic ramps `Φᵢ` sustaining the alternating
off-diagonal gradient, and three compactly supported corrections `bₖ` seeding the
first three gradients. `A` and the `bₖ` depend on `(μ, variant)` — momentum is a
positive linear filter of the gradients, which we invert. Divergence holds for
**every** `L > 0`, `η > 0`, `μ ∈ [0,1)` and both variants, the iterate trajectory
being identical across them, and only **EF21-SignMuon** diverges while `Muon`,
`EF21-MuonUSign`, `SignSGD` and the rest stay bounded. It shows that the
`Θ(σ_min/L)` step-size restriction of the conditional convergence theorem cannot
be replaced by any `(L, μ)`-only rule.

Figures: the right-hand panel of `counterexamples_main`, framed so the linear
ascent is visible above the band the seven bounded methods sit in, and
`ef21_signmuon_momentum` from [`plot_ef21_momentum.py`](plot_ef21_momentum.py).
