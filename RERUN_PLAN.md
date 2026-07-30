# Re-run plan — federated + synthetic

Written 2026-07-28, after the randomized-sign convention landed. Every command
runs from `code/`. The two studies answer different questions and should be read
that way:

* **synthetic** — does the method behave the way the theory says it does?
* **federated** — can these methods actually do federated learning?

The convention change (`sign_pm1`) is in on the torch side and the test suite
passes.

Updated 2026-07-30 after the federated proofread: `communication_bits` now takes
the run's alphabet (§4), `--final-seeds` defaults to five, and §2b is restated as
the one outstanding federated claim rather than a table that no longer exists.
The N = 11 five-seed federated table itself is **done** — §2 is a re-run
recipe, not outstanding work.

Updated again 2026-07-30 after the synthetic and counterexample proofread. Both are
**done**; §1 is likewise a re-run recipe rather than outstanding work. Two things in
this section had gone stale and are corrected below: every stage now runs at
100 × 100 (there is no 500 × 500 anywhere, in the code or in the appendix), and the
budget exponent comes out at `p ≈ 2`, not the `p ≈ 1` this file used to predict. The
counterexample code did need one change after all — it followed `sign(0) = 0` where
the paper randomizes — for which see §3.

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

## 1. Synthetic — alignment with theory (~1.6 h, all seven stages)

**Status: done.** The reported numbers come from the 2026-07-29 run at commit
`cf18382` on an RTX A4000, whose `SUMMARY.md` fills `tab:synthetic_tuned`,
`tab:synthetic_alignment`, `tab:synthetic_dynamics`, both synthetic figures and every
number in `app:images_task`. The 2026-07-30 proofread re-checked all of them against
that file and found no discrepancy, no censored optimum and no learning rate on a
grid edge. What follows is the recipe for re-running it, not a list of gaps.

The paper's claims here are about *exponents and signs*, not about winning. One
command does the lot; run the stages separately only to split the wall clock.

```bash
python3 -m synthetic.run_gpu --force              # all seven, ~1.6 h, writes the .zip
python3 -m synthetic.plot_synthetic               # both figures, from results/synthetic/
```

```bash
python3 -m synthetic.run_gpu --stages stability   # ~2 min  CONTROL: SGD must give eta_max = 2/L
python3 -m synthetic.run_gpu --stages alignment   # ~3 min  rho_t > 0 on random instances
python3 -m synthetic.run_gpu --stages floor       # ~35 min floor slope = 1
python3 -m synthetic.run_gpu --stages horizon     # ~18 min p, q exponents
python3 -m synthetic.run_gpu --stages kappa       # ~22 min conditioning sweep
python3 -m synthetic.run_gpu --stages grid final  # ~15 min the fixed-target table + curves
```

**Everything is 100 × 100, every stage, and `--m` is not a cost knob.** There is no
500 × 500 left in the code or in the appendix, and no separate size for `grid`. At
these sizes a step is kernel-launch bound, so `--m 20` and `--m 100` cost about the
same; what a sweep costs is (configurations × iterations). `--m N` writes to
`results/synthetic_NxN/` rather than `results/synthetic/`, so pass it to every
invocation of a pass or to none — the tree name is chosen by whether the flag was
given, not by the size that ran.

**`--force` is what makes it a re-run.** Without it a box that has run before skips
every stage and exits in seconds looking like a success.

**Do not skip `stability`.** SGD's η_max reproducing the textbook `2/L` is the only
end-to-end check that the harness is measuring what it claims; if that fails, every
other number in the study is suspect. The 2026-07-29 run gives `2.063` against
`2/L = 2.019`, a ratio of `1.02`.

**What to expect from the exponents, so a correct result is not "fixed".** The floor
slope is `1.000` for eight of the nine methods that have a floor and `0.989` for
SignSGD. The budget pair does *not* come out at either textbook value: the eight
norm-fixed methods tune `η*` as the nonconvex bound prescribes (`q ∈ [0.39, 0.55]`,
about `1/2`) while the error they attain falls at `p ∈ [1.76, 2.14]`, about twice
what strong convexity would give, because a quadratic is easier than the worst case
of its smoothness class. An earlier version of this file predicted `p ≈ 1`; that was
wrong, and the appendix now states the measured pair and what it means.

**Any row flagged `[BOUNDARY]`**, or any `†` in a `SUMMARY.md` table row, is not a
tuned value — widen the grid and re-run that method before publishing it.
`grep '†' results/synthetic/SUMMARY.md | grep '^|'` lists them (a plain `grep -c`
also counts the three legend lines). The 2026-07-29 run has five, every one of them a
*momentum* of `0.99` at the top of its grid, for SGD and SignSGD at the largest
condition numbers, which the appendix states as a one-sided bound rather than a
measurement. No learning rate is censored.

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
* **Counterexamples** — worst case, exact oracle. Done, and the published constants
  are unchanged by the randomized-sign convention: no Theorem 1–3 instance, and no
  oracle output of one, has a zero entry, which
  `test_the_counterexample_constants_do_not_depend_on_the_sign_convention` now pins.
  The numpy module itself did have to change, having used `sign(0) = 0` where the
  paper randomizes; `fig:divergence_plot`'s third panel is redrawn as a result, its
  band of bounded methods slightly noisier. Nothing else in the figure moves, and no
  verdict does.

---

## 4. Numbers that must be regenerated, not copied

These are quoted in the paper or the notes and were produced before the current
code; do not carry them over:

* Tables 4–5 (federated) — protocol changed (validation split, per-layer rule,
  uniform decay/AdamW head, N = 11).
* ~~Table 3 / Table 8 (synthetic) — grids were one-decade linear and left optima
  censored; two rows were off their stated grid.~~ **Regenerated** by the 2026-07-29
  run on five-decade per-family grids; no optimum is censored now.
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
