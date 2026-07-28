# Re-run plan — federated + synthetic

Written 2026-07-28, after the randomized-sign convention landed. Every command
runs from `code/`. The two studies answer different questions and should be read
that way:

* **synthetic** — does the method behave the way the theory says it does?
* **federated** — can these methods actually do federated learning?

Nothing below needs a code change. The convention change (`sign_pm1`) is already
in and the test suite is 66/66.

---

## 0. Before starting (10 min)

```bash
python3 -m tests.test_code                    # must be 66/66
python3 -m federated.tune --stage anchors     # per-layer multipliers + transported grid anchors
python3 -m counterexamples.enumerate_minimality   # the Thm 2-3 minimality claims
```

`--stage anchors` is the sanity check that matters: it prints λ_ℓ per layer per
family and the transported grid anchors. If those look wrong, stop — every
tuned rate downstream inherits them.

---

## 1. Synthetic — alignment with theory (~2.5 h, or ~1 h without `grid`)

The paper's claims here are about *exponents and signs*, not about winning.
Run in this order; the first stage is the control.

```bash
python3 -m synthetic.run_gpu --stages stability   # ~5 min  CONTROL: SGD must give eta_max = 2/L
python3 -m synthetic.run_gpu --stages alignment   # ~20 min rho_t > 0 on random instances (Table 6)
python3 -m synthetic.run_gpu --stages floor       # ~25 min floor slope = 1 (Table 7)
python3 -m synthetic.run_gpu --stages horizon     # ~45 min p, q exponents (Table 7)
python3 -m synthetic.run_gpu --stages kappa       # ~30 min conditioning sweep (Fig. 7)
python3 -m synthetic.run_gpu --stages grid,final  # ~3 h   Tables 3/8 re-run + Fig. 6 curves
python3 -m synthetic.plot_synthetic               # all five figures
```

**Do not skip `stability`.** SGD's η_max reproducing the textbook `2/L` is the
only end-to-end check that the harness is measuring what it claims; if that
fails, every other number in the study is suspect.

**What fills what.** `results/synthetic/SUMMARY.md` fills the four `[fill]`
slots in `app:images_task` and Tables `tab:synthetic_alignment`,
`tab:synthetic_dynamics`. The predicted values are printed alongside the
measured ones, so a disagreement is visible without cross-referencing the paper.

**Expect `p` to disagree with 1/2, and say so.** The instance is strongly convex
(σ > 0 by construction), so `p = q = 1` is the *correct* answer there and
`p = 1/2` is the nonconvex bound. The paper already says the fit reports which
regime the instance is in; do not "fix" a p ≈ 1.

**Any row flagged `[BOUNDARY]`** in the grid stage is not a tuned value — widen
the grid and re-run that method before publishing it.

---

## 2. Federated — can these methods do federated learning? (one night)

```bash
python3 -m federated.overnight --device cuda:0 --budget-hours 12 --download
```

Watch the first few minutes: it runs the test suite, prints the per-layer table
and anchors, times a real 40-round job on *your* GPU, then prints a schedule
with a finish time per phase. After the schedule appears it is unattended, and
`results/federated_overnight/REPORT.md` is rewritten after every phase so it can
be read mid-run. `--resume` picks up after a crash.

Phase order is already the right one for this goal: `lr` → `verify` → `final`
→ `wd`, with `wd` and `verify` the first to be dropped under budget pressure.

**Settings that are now defaults and should stay:**

| setting | value | why |
| :--- | :--- | :--- |
| `--uplink-zeros` | `random` | the paper's convention: a strict 1-bit channel |
| `--mv-ties` | `random` | ditto; unreachable at odd N anyway |
| `--n_parties` | 11 | odd ⇒ the majority vote cannot tie |
| `--lr-scaling` | `unit-gain` | the rule the paper states |
| `--split tune` | during `lr` | the test set is never ranked on |
| weight decay | 0 primary | matches the theorems; `wd` phase is the ablation |

**Eleven methods, and EF21-SignMuon belongs in the table.** It is the method
Theorem 4 predicts can diverge, and showing it training fine on CIFAR while
diverging on the constructed instance is evidence *for* the paper's framing, not
against it. `muonserver` is the uncompressed control for the server-LMO family —
do not compare the server-LMO methods against worker-LMO `muon` alone.

### 2b. The per-layer rule ablation (fills `tab:lr_ablation`)

Three tuning passes, sign family only, then a couple of final runs at each
selected rate:

```bash
for rule in none unit-gain mup; do
  python3 -m federated.tune --stage lr --lr-scaling $rule \
      --methods signmuon muonsign signsgd --split tune --out results/federated/rules_$rule
done
```

Then re-run the selected rate of each (method, rule) pair at the final horizon,
seed 0, and write `results/federated/scaling_compare.csv`. The claim being
defended is narrow and should be stated narrowly: **the ordering of the methods
does not change with the rule.** The selected η₀ *does* change, by roughly the
prescribed multiplier — that is the rule working, not a problem.

---

## 3. What each experiment is allowed to claim

Keep these separate when writing the results; conflating them is what invites a
reviewer to press.

* **Synthetic** — exponents, signs, and floors against closed-form predictions.
  Not a horse race: on a benign convex quadratic the EF21 methods *should* cost
  something, because their purpose is a guarantee on problems where the plain
  methods have none.
* **Federated** — accuracy at a fixed round budget under a real 1-bit uplink,
  across two federation scales. The headline is that the compressed methods keep
  up with Muon and beat SignSGD; differences within a seed spread are not
  results.
* **Counterexamples** — worst case, exact oracle. Already done and unchanged by
  the convention (verified: no instance has a zero entry).

---

## 4. Numbers that must be regenerated, not copied

These are quoted in the paper or the notes and were produced before the current
code; do not carry them over:

* Tables 4–5 (federated) — protocol changed (validation split, per-layer rule,
  uniform decay/AdamW head, N = 11).
* Table 3 / Table 8 (synthetic) — grids were one-decade linear and left optima
  censored; two rows were off their stated grid.
* The `8–17%` uplink-zero range — still worth recording as a diagnostic, but it
  no longer feeds any bit-accounting claim, since zeros are randomized.
* The round-trip communication table (`1.9×` vs `25×`) — recompute with
  `communication_bits` under the current alphabet (now a genuine 1 bit).

---

## 5. Known gaps to close before submission (not experiments)

* `app:mishra` — the transcription of Mishra et al.'s bound has never been
  checked against their actual paper; it is not in the repo.
* nanoGPT `η₀ = 0.03` for the sign family: the unit-gain rule fixes λ_ℓ, not the
  0.06 → 0.03 halving across families. Document its origin or re-derive it.
* The nanoGPT sharded path has never run with real collectives on this machine
  (gloo skips on Windows). Run it on Linux before any further LM runs.
