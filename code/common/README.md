# `common/` — the shared library

Everything that more than one experiment needs. Nothing here trains anything; the
four experiment packages import from it, which is what stops them describing
different algorithms under the same name.

| File | What it holds |
| :--- | :--- |
| [`optimizers.py`](optimizers.py) | The eight methods as `torch.optim.Optimizer` subclasses, plus `muon_lmo` and the Newton–Schulz iteration |
| [`lr_scaling.py`](lr_scaling.py) | Per-layer learning-rate rules, derived rather than tuned |
| [`models.py`](models.py) | `CNN2`, `ResNet9`, `ResNet18` |
| [`utils.py`](utils.py) | Seeding, run directories, the metrics schema, parameter routing, the cosine schedule |

## `optimizers.py`

One base class with a `_direction` hook; each method is the ~5 lines that differ.
`M` is the effective momentum direction and `polar(Y) = U Vᵀ` is the Muon LMO.

| Class | `d_t` (the step is `X ← X − η·λ·d_t`) | Family |
| :--- | :--- | :--- |
| `Muon` | `polar(M)` | `lmo` |
| `SignSGD` | `sign(M)` | `sign` |
| `SignMuon` | `sign(polar(M))` | `sign` |
| `MuonUSign` | `polar(sign(M))` | `lmo` |
| `MuonSign` | `sign(polar(sign(M)))` | `sign` |
| `EF21SignMuon` | EF21 estimator of `polar(M)` | `lmo` |
| `EF21MuonUSign` | `polar(g_est)`, `g_est → M` | `lmo` |
| `EF21MuonSign` | as above + downlink EF21-P on `X` | `lmo` |

Three things worth knowing before you read a step:

* **Momentum is the EMA form** `M = μM + (1−μ)G` of the paper's algorithm boxes.
  It is trajectory-identical to the heavy-ball form of the main text: the buffers
  differ by the constant `(1−μ)`, and `sign`, `polar` and the EF21 recursion are all
  positively homogeneous, so the factor cancels. `test_gradient_scale_invariance`
  pins it.
* **Weight decay is decoupled** (`p *= 1 − lr·wd`). This is not a style preference.
  Every `_direction` here is positively homogeneous of degree *zero*, so folding
  `wd·p` into the gradient cannot shorten the step by a single percent — it only
  rotates it, by an amount set by the drifting ratio `wd‖p‖/‖g‖`.
  `decoupled_weight_decay=False` reproduces the coupled convention for the ablation.
* **`EF21MuonSign` keeps two models.** `p.data` is the broadcast model `W`, so
  `p.grad` really is the gradient at `W`; the exact server model `X` — the iterate
  the theory bounds — lives in optimizer state. Use `using_exact()` to evaluate on
  it, or `restore_exact()` to install it permanently.

`zeropower_via_newtonschulz5` approximates the polar factor's *direction* well but
its *norm* only to a few percent, and it oscillates rather than converging — see
`test_newton_schulz_gets_the_direction_but_not_the_scale`, which measures the band
rather than asserting a tolerance someone picked. That is precisely why magnitude
is handled separately, by `lr_scaling`.

## `lr_scaling.py`

The two families produce step matrices whose Frobenius norms scale differently with
shape (`√min(m,n)` against `√(mn)`), so one global rate cannot be right for both,
nor across layers. The rule is *derived* from a single criterion — an update's RMS
gain `γ(A) = ‖A‖_F/√fan_out` should be a fixed fraction of the initialization's:

```
λ = √fan_out / ‖s‖_F     ⟹     lmo: η₀·√max(1, m/n)      sign: η₀/√fan_in
```

The strongest evidence that this is the right criterion is that it *re-derives* the
`√max(1, m/n)` factor already shipped in reference Muon — which is also why Muon's
learning rate is known to transfer across widths. The sign family has simply never
been given its counterpart.

```bash
python3 -m common.lr_scaling                     # list the rules and what each assumes
python3 -m common.lr_scaling --compare           # per-layer multiplier profiles
python3 -m common.lr_scaling --measure           # is a single sign step incoherent or aligned?
```

`--measure` answers the one modelling question the derivation leaves open. `mup`
applies the same criterion to the *accumulated* update assuming successive sign
steps align with the activations; `unit-gain` assumes they do not. The measurement
says `‖sign(polar(M))‖_op ≈ 0.93(√m + √n)` to within 8% at every ResNet-18 shape —
spectrally incoherent, not rank-one aligned. The complementary check is
`centralized.main --log-gain`, which fits the growth exponent of the accumulated
update directly.

## `utils.py`

`History` is the piece to know. It records the x-axis **explicitly** and stores only
evaluated points:

```json
{"steps": [0, 100, 200], "test_acc": [10.0, 61.2, 74.8], "test_loss": [...]}
```

The pre-refactor format wrote one entry per round and forward-filled whenever
`eval_freq > 1`, which turns unmeasured rounds into flat plateaus and makes a
cross-seed mean meaningless. `split_param_names` is the other load-bearing one:
both drivers call it, so the centralized and federated settings cannot drift apart
in which parameters count as matrices.
