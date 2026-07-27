# `notebooks/` — plotting only

| Notebook | Figure |
| :--- | :--- |
| [`plot_centralized.ipynb`](plot_centralized.ipynb) | Centralized train loss / test accuracy |
| [`plot_federated.ipynb`](plot_federated.ipynb) | Federated loss / accuracy per federation scale |
| [`plot_synthetic.ipynb`](plot_synthetic.ipynb) | Convex-benchmark loss and gradient-norm curves |
| [`plot_counterexample.ipynb`](plot_counterexample.ipynb) | Divergence trajectories |

No notebook computes a result. They read `metrics.json` files out of `results/` and
draw them. Anything a notebook shows should be reproducible from a
[`../REPRODUCE.md`](../REPRODUCE.md) command; if it is not, the notebook is doing
something it should not.

> **They still read the pre-reorganization paths.** All four were written against
> `saves/`, `saves_federated/` and `saves_synthetic*/`, and the counterexample one
> against the old `EF-UDSignMuon/` directory names. Repoint them at `results/`
> before the next round of figures. The `counterexamples/` figures do not need a
> notebook at all — `python3 -m counterexamples.run_counterexamples` writes them
> directly, to both `counterexamples/figures/` and the paper's image directory.

For multi-seed curves, prefer [`../aggregate.py`](../aggregate.py) over ad-hoc
notebook averaging:

```bash
python3 -m aggregate --root results/federated --metric test_acc \
    --csv summary.csv --curves curves.json
```

It groups runs by configuration-minus-seed, averages pointwise on the common step
indices, and reports mean ± sample std — and it flags unequal seed counts and
single-seed groups instead of printing a misleading `± 0`. `curves.json` holds the
pointwise mean/std curves for error-band plots.

## Before committing a notebook

Clear the outputs, or let the bundler do it:

```bash
python3 -m anonymize --check     # reports which notebooks still carry outputs
```

Notebook outputs are the most common double-blind leak in supplementary material —
they carry absolute paths, usernames and hostnames from the machine that ran them.
One of these notebooks did. `python3 -m anonymize --build` strips every output when
it packages the supplement, leaving the working copies untouched.
