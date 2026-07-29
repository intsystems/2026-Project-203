# `centralized/` — single-node CIFAR-10 / MNIST

ResNet-18 on CIFAR-10, 75 epochs, cosine-annealed. This is the paper's
`tab:cifar_main`, `tab:cifar_central` and `fig:cifar_*`.

| File | What it does |
| :--- | :--- |
| [`main.py`](main.py) | Entry point: one run, one config, one `metrics.json` |
| [`train.py`](train.py) | The training loop and optimizer construction |
| [`data.py`](data.py) | Loaders, including the fixed 45k/5k tuning split |
| [`tune.py`](tune.py) | Equal-budget, validation-only learning-rate search |
| [`overnight.py`](overnight.py) | **Command 1**: the whole protocol, resumable |
| [`export_article.py`](export_article.py) | **Command 2**: pack the results into one archive |
| [`plot_analysis.py`](plot_analysis.py) | Redraw every figure from that archive |

## Two commands

On the GPU box:

```bash
cd code
python3 -m centralized.overnight --device cuda:0 --download   # ~1.5 days, resumable
python3 -m centralized.export_article                         # -> results/article_export.tar.gz
```

Download that one `.tar.gz` (~1 MB), unpack it here, and:

```bash
python3 -m centralized.plot_analysis --bundle article_export  # -> results/analysis/
```

The run tree itself never has to move: it is ~1.5 GB of `model.pt` per sweep, and
the archive holds every number the paper quotes.

Until that re-run lands, the **submitted** figures are redrawn from the committed
outputs of the 2026-07-27 run:

```bash
python3 -m centralized.plot_analysis --legacy   # reads table2_full.csv + curves*.json
```

byte for byte, pinned by `test_the_submitted_figures_can_still_be_redrawn`. Those
four files stay until the new run replaces the figures.

Watch the first ~6 minutes of the run. It executes the CPU test suite, prints the
per-layer learning-rate table, records the exact GPU / driver / CUDA / Python /
PyTorch, times two real epochs on *your* GPU, then prints a schedule with a finish
time per phase. After that you can leave it. `results/overnight/REPORT.md` is
rewritten after every phase and every final run, so you can read it mid-run
without stopping anything; Ctrl-C stops cleanly and writes it.

Phases, ordered so that stopping early costs the least:

| Phase | What it establishes |
| :--- | :--- |
| `gain` | Does the accumulated update grow like `√t` (incoherent → `unit-gain`) or like `t` (aligned → `mup`)? Run at a **constant** rate — with annealing the accumulation saturates and the slope would measure the schedule |
| `aux` | Is the optimal `lr_aux` the same for two methods an order of magnitude apart in `η₀`? If so, one value may be held fixed for every method and *reported as verified* |
| `lr` | `η₀` per method, equal budget, 1–2–5 lattice, grid widened when an optimum lands on an endpoint — **at the reporting horizon**, see below |
| `final` | Full-50k runs at the tuned rates, **seed-major** — every method at seed 0 before any reaches seed 1, so a cut night leaves a complete table rather than fragments |
| `wd` | The same methods with weight decay on, to see whether the *ordering* moves |

## Why `η₀` is selected at 75 epochs, not at a proxy

The driver used to rank learning rates at 15 epochs and re-check the top few at
75. The check failed. On the 2026-07-27 run the ranking **reversed** for both
methods probed:

| method | best @ 15 ep | best @ 75 ep |
| :--- | ---: | ---: |
| SignMuon | `0.02` (93.68) | `0.2` (94.54) |
| Muon | `0.05` (93.46) | `0.01` (94.60) |

That is not seed noise. Both runs anneal cosinally to zero over *their own*
horizon, so a 15-epoch run spends nearly its whole budget at a decayed rate and a
75-epoch run does not — the proxy measures a different schedule. Worse for a
patch, the bias is not even in a consistent direction: SignMuon's optimum moved
up three lattice steps and Muon's down two.

So the `lr` phase runs at `--final-epochs`. It costs more than a proxy and buys
the one thing a proxy cannot: the rate in the table is the rate that won at the
horizon the table reports. Selection is still `val_acc` on the 45k/5k split, so
the test set enters nothing.

`lr_aux` is exempt, and legitimately: the `aux` phase asks whether the *argmax
over `lr_aux`* agrees between two methods measured at the **same** horizon, and a
shared horizon bias cancels in that comparison. It runs at `--aux-epochs 15`.

## The protocol, and why it is shaped this way

Four properties are enforced by the code rather than by discipline.

1. **No test-set tuning.** Selection runs use `--split tune`, a fixed 45k/5k
   partition with `--val-seed` independent of `--seed`, so every method and every
   seed tunes against the identical split and the split is not a confounder.
   Selection reads `val_acc` averaged over the last `--last-k` epochs — a tail mean,
   not one noisy evaluation.
2. **Equal budget.** Every method gets the same number of configurations on a
   *multiplicatively* anchored grid (`tune.anchor_for`). A shared absolute grid
   would be unfair, because the two families have genuinely different natural scales.
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
| `legacy` | `η₀` | `η₀·√max(1,m/n)` | one global rate for the sign family |
| `none` | `η₀` | `η₀` | also what the concurrent Sign-Muon implementation runs |
| **`unit-gain`** | `η₀/√fan_in` | `η₀·√max(1,m/n)` | **derived; the default** |
| `mup` | `η₀/fan_in` | `η₀·√max(1,m/n)` | assumes accumulated steps align |
| `mishra-analysis` | `η₀/√(mn)` | `η₀/√min(m,n)` | the normalization used in their *proof* |

> **ResNet-18 is a weak instrument for the exponent.** **Thirteen** of its twenty
> conv weight tensors have `fan_in/fan_out = 9` exactly and hold **84.5%** of all
> parameters — and one shape alone, `(512, 4608)` appearing three times, is 63% of
> the model. So `α` is identified only through the transition and 1×1-downsample
> layers. CNN2 in the federated setting has a 7.8× multiplier spread and no such
> dominant shape, so it is the better instrument; the `gain` phase measures the
> exponent directly either way.

## The reported metric

| metric | role |
| :--- | :--- |
| test accuracy, mean of the last `--last-k` epochs | **primary** |
| train accuracy | underfitting diagnostic |
| epochs to `--target-acc` | separates *speed* from final quality |
| median epoch time | cost of the method |

All of them are printed in a `--- summary ---` block at the end of every run, so a
log alone fills a table row — and `table_cifar.csv` in the export bundle carries
them aggregated over seeds exactly as the paper defines them, so the table is
never retyped by hand.

**On the test-loss/accuracy divergence:** test cross-entropy rises after epoch ≈40
while test accuracy keeps improving. That is the standard overconfidence regime once
train accuracy saturates — the loss on misclassified points grows while the decision
boundary still improves. It is not a bug and not a reason to early-stop on test.

See [`../REPRODUCE.md`](../REPRODUCE.md) §4 for the exact commands.
