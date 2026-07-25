# scrap/ — archived plotting scripts (not part of the submission)

These three scripts build the EF21-SignMuon divergence figures from an **earlier,
superseded construction**: a fixed *quadratic-valley* objective
`f(W) = c₁W₀₀ + c₂W₁₁ + (K/2)[(W₀₁−a)² + (W₁₀−b)²]` on which EF21-SignMuon
ascends at rate ≈ `0.0716` per step.

They were replaced by the **universal divergence theorem** (one exact-rational
`L`-smooth `f` per `(L, η, μ, variant)`, rate `49/480`), whose canonical
implementation lives in [`../counterexamples/`](../counterexamples/)
(`problems.py`, `optimizers.py`, `run_counterexamples.py`, `plot_ef21_momentum.py`).
The current paper's figures come from there, **not** from these scripts.

They are kept here only so the earlier construction and its plots are not lost.
Nothing in `../counterexamples/` or the rest of `code/` imports them, and they
should be excluded when packaging the supplementary material.

> ⚠️ Naming clash: `plot_ef21_signmuon_momentum.py` below and the canonical
> `../counterexamples/plot_ef21_momentum.py` both emit a file named
> `ef21_signmuon_momentum.{pdf,png}` — from *different* constructions. Do not run
> the two into the same output directory. (These scrap scripts write to a local
> `im_dif/` folder and only touch the paper tree if it already exists.)

## Files

| File | What it does |
|------|--------------|
| `plot_six_methods_counterexample.py` | Six-method comparison on the quadratic valley (`μ=0`, `η=1`); only EF21-SignMuon ascends. Emits `six_methods_counterexample.{pdf,png}`. |
| `plot_ef21_signmuon_divergence.py` | Two panels: (a) one instance (`L=η=1`), method comparison; (b) scale invariance — `f/(Lη²)` collapses onto slope `≈0.0716` for every `(L, η)`. Emits `ef21_signmuon_divergence.{pdf,png}`. |
| `plot_ef21_signmuon_momentum.py` | Divergence *under momentum* on general-quadratic constructions A (standard) and B (Nesterov); long-horizon trajectories for several `μ`. Emits `ef21_signmuon_momentum.{pdf,png}` (superseded — see the clash note above). |

## Provenance note

The docstrings in these files cite `ef21_signmuon_divergence.tex`. That appendix
file has since been rewritten around the universal construction, so the citation
no longer matches — treat the scripts as a historical snapshot of the older
quadratic-valley argument.
