# `synthetic/` — the smooth convex benchmark

`F(X) = ½⟨X−C, A(X−C)B⟩` on `m × n` matrices. One file,
[`benchmark.py`](benchmark.py), with seven modes.

The point of this problem is that **nothing has to be estimated**. `A` and `B` are
symmetric with a prescribed spectrum, so the Hessian is the Kronecker product
`B ⊗ A`, the smoothness constant is exactly `L = maxᵢⱼ λᵢμⱼ` and the strong-convexity
constant exactly `σ = minᵢⱼ λᵢμⱼ`, both in closed form. `∇F` and `F` are closed-form
too, so there is no autograd graph — which is what makes the bidirectional method
tractable here: its gradient must be taken at the broadcast model `W` while its
metrics are scored at the exact model `X`, and with a closed-form gradient either
point is one matmul away.

```bash
python3 -m synthetic.benchmark --mode grid  --device cuda:0   # the paper's protocol
python3 -m synthetic.benchmark --mode final --device cuda:0 --save-histories
```

## The modes

| mode | what it measures |
| :--- | :--- |
| `grid` / `final` | Iterations to `F ≤ 1e-3`, `η` and `μ` tuned per method — the paper's table |
| `alignment` | `ρ_t = ⟨∇F, d_t⟩ / (‖∇F‖‖d_t‖)` along the trajectory, against closed-form references |
| `floor` | The accuracy floor `F∞(η)` and its exponent |
| `horizon` | `err ~ T^-p` and `η* ~ T^-q`, tuned separately at each budget |
| `stability` | The largest stable `η` against the Frobenius trust region |
| `kappa` | The same at a controlled condition number |

## Read the iteration count for what it is

For every sign-family method the iteration count on this problem is *exactly*
inversely proportional to `η`, and there is a hard `η` above which the method never
reaches the target at all. That is the signature of a constant step size on a
problem whose minimizer it cannot reach: a `±1` step has fixed length `η√(mn)`, so
the iterate settles into a ball of that radius and `F` plateaus. The tuner returns
the largest `η` whose plateau still fits under the target, so "iterations to target"
is `const/η_max` — a measurement of the **accuracy floor**, not of the descent rate.

At matched `η` the ranking can invert. Measured at `m=n=100`, momentum 0.2:

| | SignMuon | SignSGD |
| :--- | ---: | ---: |
| floor `F∞` at `η = 2e-4` | `2.57e-5` | `1.72e-4` |
| iterations to `1e-3` at matched `η = 2e-4` | 1532 | **1190** |
| iterations to `1e-3` at each method's tuned `η` | **~1830** | 2036 |

SignMuon's advantage is a ~2× lower floor; it *loses* ~28% on per-step descent. The
two effects pull in opposite directions and belong in the paper separately — which
is what `floor` and `horizon` are for.

`alignment` is the mode that is about the methods rather than the protocol. The
descent lemma needs `ρ_t > 0` and the divergence theorems drive it negative, so its
distribution along the trajectory is the quantity of interest. The closed-form
references it is read against are `ρ = 1` (SGD), `ρ = ‖G‖₁/(‖G‖_F√(mn)) → √(2/π)`
(SignSGD) and `ρ = ‖G‖_*/(‖G‖_F√r)` (Muon).

## Two caveats carried in the code

* **The default grids are no longer the paper's.** The published grids are linear
  and one decade wide, which left SGD censored at its own boundary — its reported
  optimum is the *top* of both grids. The defaults are now logarithmic and 3–4
  decades wide, and the tuner prints `[BOUNDARY]` and records `on_grid_boundary`
  in the JSON whenever an optimum still lands on an edge. `--grid-preset paper`
  selects the published grids verbatim, so the discrepancy is checkable.
* **`--spectrum uniform`** (the default, and what the published table used) leaves
  the condition number uncontrolled — it comes out near `1e4` at `m=n=100` and
  `2.5e5` at `m=n=500`, and the paper never reports it. `--spectrum logspace
  --kappa K` fixes `L = 1` and `L/σ = K` exactly instead.

Results land in `results/synthetic/<method>/{grid,final}.json`; `--save-histories`
adds the loss and gradient-norm curves the paper's figure plots. See
[`../REPRODUCE.md`](../REPRODUCE.md) §3.
