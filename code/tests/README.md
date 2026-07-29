# `tests/` — the CPU test suite

```bash
cd code
python3 -m tests.test_code      # ~1 min, no GPU, no downloads
pytest tests/test_code.py       # or under pytest
```

One file, [`test_code.py`](test_code.py). No GPU, no dataset, no network, about a
minute. Both `overnight.py` drivers run it as their first preflight step and refuse
to start a night if it fails, printing the failing assertions rather than just the
test names.

## What it is actually for

Most of these do not test "does the code run". They pin **claims the paper makes**,
so that a change to the code cannot silently invalidate a sentence in the paper.

| Group | The claim being pinned |
| :--- | :--- |
| Newton–Schulz | It approximates the polar factor's *direction*, and its *norm* only to a measured band — which is why magnitude is handled separately by `lr_scaling` |
| Scale invariance | Every method's iterates are unchanged under `G → cG`, which is exactly why the paper's heavy-ball main text and its EMA algorithm boxes describe the same trajectories |
| Theorems 1–3 | The published descent inner products, recomputed in torch |
| Exact vs Newton–Schulz | The two oracles genuinely disagree on the published instances — a measured fact, pinned so it cannot drift |
| Weight decay | Coupled decay cannot shorten a scale-invariant step; it only rotates it |
| Per-layer scaling | `unit-gain` re-derives the shipped Muon aspect factor, and equalizes the per-step gain exactly |
| Federated ↔ centralized | **The load-bearing one**, below |
| Batched ↔ sequential | The synthetic sweeps' fast path reproduces `run_one` on all ten methods — the same kind of two-implementations problem, below |
| Federated protocol | The validation split is held out before partitioning; the GPU augmentation matches torchvision; the uplink alphabet is ternary |
| Centralized export | The paper's table aggregates per-seed tail means, in that order; every driver phase lands in the right bucket; the figures are a function of the export bundle alone |
| Plumbing | The metrics schema, multi-seed aggregation, the anonymity scan |

## The load-bearing test

`test_federated_one_client_equals_centralized` and
`test_federated_per_layer_scaling_matches_centralized`: the federated driver with
`N = 1` must reproduce the corresponding centralized optimizer **exactly**, for all
eight matrix rules, under both the `legacy` and `unit-gain` conventions.

`federated/algorithms.py` and `common/optimizers.py` are two implementations of the
same eight algorithms. Nothing but this test stops them drifting apart — and they
had drifted apart before it existed, in the learning-rate schedule, in the routing
of biases and BatchNorm, and in the weight-decay convention.

`test_batched_runner_reproduces_the_sequential_one` is the same situation a third
time. `synthetic/batched.py` runs a whole hyperparameter grid as one `[B, m, n]`
trajectory, which is what makes the synthetic sweeps finish in minutes instead of
hours, but it is a second implementation of all ten update rules — so it is checked
against `benchmark.run_one`, which drives the real optimizer classes, over a grid
containing a diverging run, a per-config budget and all three schedules, with
`stop_at_target` both off and on. Run it in float32: in bfloat16 a batched matmul
can pick a different cuBLAS kernel than a single one, and the methods that sign the
LMO output are sensitive to the last bit.

A related trap, worth knowing if you edit these: a test whose *reference* is
hand-rolled inside the test file can pin the wrong convention. This one did — its
reference decayed the auxiliary AdamW group, which the real centralized path never
does — so it certified an agreement that did not hold against production code.
If you change a convention, change it in `train.py`, not in the reference.

## Tests that measure rather than assert

Several checks derive their own tolerance instead of hard-coding one, so they cannot
go stale when the implementation changes:

* `test_coupled_decay_does_not_shrink_the_implemented_step` measures the
  Newton–Schulz norm band over the three input distributions the methods actually
  feed it, then asserts against *that*.
* `test_gpu_crop_and_flip_match_torchvision_distributionally` compares the per-row
  probability that an output row came from the zero padding against
  `RandomCrop(32, padding=4)` itself, over 2000 draws. A shape check would pass on
  an off-by-one in the padding; this does not.

## Adding one

Any module-level function named `test_*` is picked up by both the built-in runner
and pytest. Keep it CPU-only and under a second or two — the value of this suite is
that it runs before every night, and that stops being true if it takes ten minutes.
