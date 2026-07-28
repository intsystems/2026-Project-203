# `centralized/` — single-node CIFAR-10 / MNIST

ResNet-18 on CIFAR-10, 75 epochs, cosine-annealed. This is the paper's centralized
table and Figure 2.

| File | What it does |
| :--- | :--- |
| [`main.py`](main.py) | Entry point: one run, one config, one `metrics.json` |
| [`train.py`](train.py) | The training loop and optimizer construction |
| [`data.py`](data.py) | Loaders, including the fixed 45k/5k tuning split |
| [`tune.py`](tune.py) | Equal-budget, validation-only learning-rate search |
| [`overnight.py`](overnight.py) | The whole protocol as one resumable command |
| [`export_article.py`](export_article.py) | Collects finished runs into a table/figure bundle |

## Start here

```bash
cd code
python3 -m centralized.overnight --device cuda:0 --budget-hours 0 \
    --final-seeds 0 1 2 --download
```

Watch the first ~6 minutes: it runs the CPU test suite, prints the per-layer
learning-rate table, times two real epochs on *your* GPU, then prints a schedule
with a finish time per phase. After that you can leave it. `results/overnight/REPORT.md`
is rewritten after every phase and every final run, so you can read it mid-run
without stopping anything; Ctrl-C stops cleanly and writes it.

Phases, ordered so that stopping early costs the least:

| Phase | What it establishes |
| :--- | :--- |
| `gain` | Does the accumulated update grow like `√t` (incoherent → `unit-gain`) or like `t` (aligned → `mup`)? Run at a **constant** rate — with annealing the accumulation saturates and the slope would measure the schedule |
| `lr` | `η₀` per method, equal budget, 1–2–5 lattice, grid widened when an optimum lands on an endpoint |
| `verify` | Is the short-horizon ranking horizon-stable? That is the assumption a short proxy makes, and it is checkable |
| `final` | Full-50k runs at the tuned rates, **seed-major** — every method at seed 0 before any reaches seed 1, so a cut night leaves a complete table rather than fragments |
| `wd` | The same methods with weight decay on, to see whether the *ordering* moves |

## The protocol, and why it is shaped this way

Four properties are enforced by the code rather than by discipline.

1. **No test-set tuning.** Selection runs use `--split tune`, a fixed 45k/5k
   partition with `--val-seed` independent of `--seed`, so every method and every
   seed tunes against the identical split and the split is not a confounder.
   Selection reads `val_acc` averaged over the last `--last-k` epochs — a tail mean,
   not one noisy evaluation.
2. **Equal budget.** Every method gets the same number of configurations on a
   *multiplicatively* anchored grid. A shared absolute grid would be unfair, because
   the two families have genuinely different natural scales.
3. **An optimum on a grid endpoint is a failure, not a result.** The grid is widened
   along the same lattice and the method re-run, up to four times.
4. **Per-layer rates are derived, not tuned.** `--lr-scaling` sets
   `η_layer = η₀·λ(family, shape)` analytically; only the shape-free `η₀` is searched.

The falsifiable prediction that comes with (4): once the shape dependence lives in
`λ`, the tuned `η₀` should agree *within* each family. `tune.py` reports that
agreement, and if it fails that is a real finding rather than a protocol note.

## Which `--lr-scaling`?

| rule | sign family | LMO family | note |
| :--- | :--- | :--- | :--- |
| `legacy` | `η₀` | `η₀·√max(1,m/n)` | what the paper's published numbers used |
| `none` | `η₀` | `η₀` | also what the concurrent Sign-Muon implementation runs |
| **`unit-gain`** | `η₀/√fan_in` | `η₀·√max(1,m/n)` | **derived; the default** |
| `mup` | `η₀/fan_in` | `η₀·√max(1,m/n)` | assumes accumulated steps align |
| `mishra-analysis` | `η₀/√(mn)` | `η₀/√min(m,n)` | the normalization used in their *proof* |

> **ResNet-18 is a weak instrument for the exponent.** **Thirteen** of its twenty
> conv weight tensors have `fan_in/fan_out = 9` exactly and hold **84.5%** of all
> parameters — and one shape alone, `(512, 4608)` appearing three times, is 63% of
> the model. So `α`
> is identified only through the transition and 1×1-downsample layers. CNN2 in the
> federated setting has a 7.8× multiplier spread and no such dominant shape, so it
> is the better instrument; the `--log-gain` diagnostic measures the exponent
> directly either way.

## The reported metric

| metric | role |
| :--- | :--- |
| test accuracy, mean of the last `--last-k` epochs | **primary** |
| test accuracy at the best-`val_acc` epoch | early stopping, done properly |
| train accuracy | underfitting diagnostic |
| epochs to `--target-acc` | separates *speed* from final quality |

All of them are printed in a `--- summary ---` block at the end of every run, so a
log alone fills a table row.

**On the test-loss/accuracy divergence:** test cross-entropy rises after epoch ≈40
while test accuracy keeps improving. That is the standard overconfidence regime once
train accuracy saturates — the loss on misclassified points grows while the decision
boundary still improves. It is not a bug and not a reason to early-stop on test.

See [`../REPRODUCE.md`](../REPRODUCE.md) §4 for the exact commands.
