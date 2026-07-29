# Re-run plan — federated + synthetic

Written 2026-07-28, after the randomized-sign convention landed. Every command
runs from `code/`. The two studies answer different questions and should be read
that way:

* **synthetic** — does the method behave the way the theory says it does?
* **federated** — can these methods actually do federated learning?

Nothing below needs a code change. The convention change (`sign_pm1`) is already
in and the test suite passes.

Updated 2026-07-30 after the federated proofread: `communication_bits` now takes
the run's alphabet (§4), `--final-seeds` defaults to five, and §2b is restated as
the one outstanding federated claim rather than a table that no longer exists.
The N = 11 five-seed federated table itself is **done** — §2 is a re-run
recipe, not outstanding work.

---

## 0. Before starting (10 min)

```bash
python3 -m tests.test_code                    # must pass with no failures
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
python3 -m synthetic.run_gpu --stages grid final  # ~3 h   Tables 3/8 re-run + Fig. 6 curves
python3 -m synthetic.plot_synthetic               # all five figures
```

### Running at 100x100 to save time

Only `grid` and `final` use the paper's 500x500; the five sweep stages above
already run at **100x100** by default, so most of this list is cheap already.
To put everything at 100x100:

```bash
python3 -m synthetic.run_gpu --m 100 --n 100                       # all stages
python3 -m synthetic.run_gpu --stages grid final --m 100 --n 100   # just the expensive two
```

That writes to `results/synthetic_100x100/`, a separate tree, so a small pass
can never land where 500x500 numbers belong.

**Pass `--m/--n` to every invocation of a given pass, or to none.** The tree name
is chosen by whether the flag was given, not by the size that actually ran, so
sweeps without it (`results/synthetic/`) and `grid` with it
(`results/synthetic_100x100/`) end up split across two trees despite being the
same size. The `--list` time estimates assume the default sizes and are far too
pessimistic at 100x100.

**One paper consequence, worth deciding before you run.** The appendix states
500x500 in three places, and the superseded Table in `app:synthetic_v1` is a
500x500 result. New numbers at 100x100 make the new-vs-superseded comparison
apples-to-oranges, and the condition number changes with it (3.65e4 at 100x100
against 1.645e7 for the 500x500 seed-1337 draw). Two clean options: keep `grid`
at 500x500 purely for comparability with the published table, or run everything
at 100x100 and relabel the superseded section as the historical 500x500 record.
The sweeps are unaffected either way -- they measure exponents, which is the
size-robust part.

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
| `--final-seeds` | `0 1 2 3 4` | five is what `tab:exp_3` reports; rebalancing trims from the end under budget pressure and says so |

**Eleven methods, and EF21-SignMuon belongs in the table.** It is the method
Theorem 4 predicts can diverge, and showing it training fine on CIFAR while
diverging on the constructed instance is evidence *for* the paper's framing, not
against it. `muonserver` is the uncompressed control for the server-LMO family —
do not compare the server-LMO methods against worker-LMO `muon` alone.

### 2b. The per-layer rule ablation — the one outstanding federated claim

This is the highest-value remaining run. The appendix's "Sensitivity: what the
rule can and cannot affect" paragraph in `app:lrscale` now states outright that
the sign family has **not** been re-tuned under the competing conventions, and
bounds the exposure instead. This closes that gap. (There is no
`tab:lr_ablation` any more — the table was cut with the N = 3 experiment and
survives only under `old/`; a new one would go in `app:lrscale`.)

Three tuning passes, sign family only, then a final run at each selected rate:

```bash
for rule in none unit-gain mup; do
  python3 -m federated.tune --stage lr --lr-scaling $rule \
      --methods signmuon muonsign signsgd --out results/federated/rules_$rule
done
```

~45 tuning jobs at 400 rounds plus 9 finals at 2000 ≈ 5–6 h on an A100. Then
re-run the selected rate of each (method, rule) pair at the final horizon, seed 0,
and write `results/federated/scaling_compare.csv`. The claim being defended is
narrow and should be stated narrowly: **the ordering of the methods does not
change with the rule.** The selected η₀ *does* change, by roughly the prescribed
multiplier — that is the rule working, not a problem.

### 2c. Two cheap checks on the published table

* **Was SGD's η₀ = 0.1 a censored endpoint?** Every other selected rate in
  `tab:exp_3` is interior to its method's five-point grid; SGD's transported
  anchor is 0.02, so 0.1 is the *top* of the initial window and only survives as a
  tuned value if the boundary extension fired and 0.1 still won. Read the `lr`
  phase's boundary column in `REPORT.md` — no rerun needed unless it was censored.
* **`lr_aux = 0.001` is held fixed, not verified, in the federated arm.**
  `python3 -m federated.tune --stage aux --rounds 400` (~30 configs, ~2 h) would
  let the appendix say "verified method-independent", which is what the
  centralized arm can already claim from its own `aux` phase.

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
* The round-trip communication table (now `~1.9×` vs `29.4×`) — `communication_bits`
  now takes the run's alphabet as an argument and no longer charges the ternary
  entropy to a channel that transmits `±1`, so `tab:commacct` and a fresh run log
  agree. **Any `round_trip_reduction` already in `state.json` or `REPORT.md` from a
  pre-2026-07-30 run is the old, inflated figure** — regenerate the report from a
  new run rather than quoting it. The paper's table was always the correct one.

---

## 5. Known gaps to close before submission (not experiments)

* `app:mishra` — the transcription of Mishra et al.'s bound has never been
  checked against their actual paper; it is not in the repo.
* nanoGPT `η₀ = 0.03` for the sign family: the unit-gain rule fixes λ_ℓ, not the
  0.06 → 0.03 halving across families. Document its origin or re-derive it.
* The nanoGPT sharded path has never run with real collectives on this machine
  (gloo skips on Windows). Run it on Linux before any further LM runs.
