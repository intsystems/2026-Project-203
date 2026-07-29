# SignMuon, MuonSign, and the Role of Error Feedback

**Sign compression around the Muon LMO: neither order converges, and EF21 is what fixes it.**

[![License](https://img.shields.io/github/license/intsystems/2026-Project-203?color=green)](LICENSE)
[![Contributors](https://img.shields.io/github/contributors/intsystems/2026-Project-203)](https://github.com/intsystems/2026-Project-203/graphs/contributors)
[![Issues](https://img.shields.io/github/issues-closed/intsystems/2026-Project-203.svg?color=0088ff)](https://github.com/intsystems/2026-Project-203/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr-closed/intsystems/2026-Project-203.svg?color=7f29d6)](https://github.com/intsystems/2026-Project-203/pulls)

<table>
    <tr>
        <td align="left"> <b> Author </b> </td>
        <td> Maria Smirnova </td>
    </tr>
    <tr>
        <td align="left"> <b> Consultant </b> </td>
        <td> Alexey Kravatskiy </td>
    </tr>
</table>

## The question

Muon takes a steepest-descent step in the spectral norm: for a matrix parameter it
sends the full orthogonalized update `UVᵀ`. Under a federated communication budget
you cannot afford to send that, so you compress it to one bit per entry with a
sign. There are two obvious places to put the sign, and one obvious question:

| | | |
| :--- | :--- | :--- |
| **SignMuon** | `sign(LMO(·))` | sign **after** the oracle |
| **MuonUSign** | `LMO(sign(·))` | sign **before** the oracle |
| **MuonSign** | `sign(LMO(sign(·)))` | sign on **both** sides |

**Neither placement converges.** We exhibit an *L*-smooth function on which each of
the three provably ascends — and momentum cannot save them, since every one of these
step maps is positively homogeneous of degree zero. Error feedback is what restores
the guarantee: EF21-MuonSign attains the standard `O(T^{-1/2})` rate for smooth
nonconvex problems, at one bit per parameter on both channels.

The catch is that error feedback is not uniformly free. Compressing the *uplink*
costs almost nothing. Compressing the *downlink* costs whatever the layer's rank
makes it cost — on nanoGPT the broadcast model `W` is indistinguishable from the
uncompressed method while the exact server model `X`, the iterate the theory bounds,
diverges. We localize that to a single layer type and explain the mechanism.

## Assets

| | |
| :--- | :--- |
| 📄 **Paper (AAAI)** | [`aaai_article/v2_SignMuon_AAAI.pdf`](aaai_article/v2_SignMuon_AAAI.pdf) |
| 📄 Paper (RU) | [`paper/main_ru.pdf`](paper/main_ru.pdf) |
| 💻 **Code** | [`code/`](code) — start at [`code/REPRODUCE.md`](code/REPRODUCE.md) |
| 📊 Slides | [`slides/Final_talk_Smirnova.pdf`](slides/Final_talk_Smirnova.pdf) |
| 🖼️ Poster | [`slides/Poster_SignMuon.pdf`](slides/Poster_SignMuon.pdf) |
| 🔗 Literature review | [`LINKREVIEW.md`](LINKREVIEW.md) |
| 🔍 Code-review notes | [`REVIEW_NOTES.md`](REVIEW_NOTES.md) |

## Abstract

SignMuon — sign compression of the Muon update to one bit per parameter — is the
most straightforward adaptation of Muon to an extremely low communication budget.
It outperforms SignSGD in practice, yet it can diverge on a linear function. The
natural conjecture is that the order of operations is to blame, and that signing
*before* rather than after the Linear Minimization Oracle (LMO) would fix it. It
does not: sign-before (**MuonUSign**) diverges too, as does signing on both sides
(**MuonSign**), so no naive placement of the sign around the LMO converges. Error
feedback repairs one of them: **EF21-MuonUSign** attains the standard `O(T^{-1/2})`
rate for smooth nonconvex optimization and decisively outperforms SignSGD, and
compressing the downlink as well costs little beyond that. What the guarantee
costs is the placement. Across centralized, federated and language-modelling
experiments the best placement is consistently sign-*after*-the-LMO — precisely
the one our counterexamples break and error feedback fails to repair.

## The eight methods

Every method is one line of a single template: where the LMO is evaluated, and what
compresses each channel.

| Method | Matrix step `d_t` | Uplink | Downlink | Guarantee |
| :--- | :--- | :---: | :---: | :--- |
| `muon` | `polar(M)` | 32 bit | 32 bit | reference |
| `signsgd` | `sign(M)` | 1 bit | 32 bit | reference |
| `signmuon` | `sign(polar(M))` | 1 bit | 32 bit | ✗ diverges (Thm 1) |
| `muonusign` | `polar(sign(M))` | 1 bit | 32 bit | ✗ diverges (Thm 2) |
| `muonsign` | `sign(polar(sign(M)))` | 1 bit | 1 bit | ✗ diverges (Thm 3) |
| `ef21signmuon` | EF21 on `polar(M)` | 1 bit | 32 bit | ✗ diverges (Thm 4) |
| `ef21muonusign` | `polar(g_est)`, `g_est → M` | 1 bit | 32 bit | ✓ `O(T^{-1/2})` |
| `ef21muonsign` | as above + downlink EF21-P | 1 bit | 1 bit | ✓ `O(T^{-1/2})` |

`M` is the momentum direction and `polar(Y) = UVᵀ` is the Muon LMO. The divergence
theorems are stated for the **exact** oracle and reproduced with an exact
rank-truncated SVD; networks are trained with the Newton–Schulz approximation, as
practitioners do.

> **Naming.** `MuonSign` changed meaning during the project: it used to name the
> sign-*before* method, which the paper now calls `MuonUSign`. Old CLI spellings
> still resolve, but there is deliberately **no** alias for the old `MuonSign`,
> because resolving it silently would swap the algorithm rather than the label.

## Where the evidence comes from

| Experiment | Setting | What it decides |
| :--- | :--- | :--- |
| Counterexamples | exact LMO, closed form | Theorems 1–3 reproduce in **< 10 s** on CPU; Theorem 4's figure ~20 s |
| Synthetic quadratic | `F(X) = ½⟨X, AXB⟩`, 100×100 | descent rate vs the sign floor, at a matched 1-bit budget |
| Centralized CIFAR-10 | ResNet-18, 75 epochs | does 1-bit cost accuracy? |
| Federated CIFAR-10 | CNN2, 11 clients, 2000 rounds | does it survive a real federation? |
| nanoGPT speedrun | 12-layer transformer, 8×H100 | does it survive matrices wide enough for the rank term to bite? |

## Quick start

Everything runs from `code/`, which is the Python package root.

```bash
cd code
pip install -r requirements.txt
python3 -m tests.test_code            # CPU only, no downloads, ~1 min
```

Reproduce the theory in ten seconds, no GPU:

```bash
python3 -m counterexamples.problems             # the Theorem 1-4 constants
python3 -m counterexamples.run_counterexamples  # Figure 1
```

Run the full centralized or federated protocol as one resumable, budget-aware
command — it self-checks, times your GPU, prints a schedule, and writes a report
you can read mid-run:

```bash
python3 -m centralized.overnight --device cuda:0 --budget-hours 12 --download
python3 -m federated.overnight   --device cuda:0 --budget-hours 12 --download
```

Or a single run:

```bash
python3 -m centralized.main --dataset cifar10 --model resnet18 --optimizer signmuon \
    --lr-scaling unit-gain --epochs 75 --lr 0.03 --device cuda:0

python3 -m federated.main --dataset cifar10 --model cnn2 --algorithm ef21muonsign \
    --lr-scaling unit-gain --rounds 2000 --n_parties 11 --n_steps 3 --lr 0.05 \
    --device cuda:0
```

**[`code/REPRODUCE.md`](code/REPRODUCE.md) has the exact command for every table and
figure in the paper**, with the published hyperparameters filled in.

## Repository layout

```
├── aaai_article/     the AAAI submission (LaTeX + figures)
├── paper/            the Russian-language version
├── slides/           talk and poster
├── code/             everything runnable  ->  code/README.md
│   ├── common/         optimizers, models, per-layer LR rules
│   ├── centralized/    ResNet-18 / CIFAR-10
│   ├── federated/      all eleven methods, one parameterized driver
│   ├── synthetic/      the convex quadratic benchmark
│   ├── counterexamples/  Theorems 1-4, exact LMO
│   └── nanogpt/        modded-nanogpt speedrun port
├── LINKREVIEW.md     literature review
└── REVIEW_NOTES.md   code-review findings, and what is still open
```

## Three things worth knowing before reading the numbers

* **Only `η₀` is tuned.** The per-layer rate is *derived* from the layer shape by
  the unit-gain rule (`code/common/lr_scaling.py`), whose main evidence is that the
  same criterion re-derives the `√max(1, m/n)` factor already shipped in reference
  Muon. Selection is on a held-out validation split; the test set is never read
  during tuning.
* **1 bit per parameter, exactly.** Exact zeros are resolved to a random ±1
  (`--uplink-zeros random`, the default), so every transmitted symbol is a genuine
  bit. Counting the uncompressed auxiliary group, the realized cost is 1.087 bits
  per parameter, a 29.4× reduction rather than 32×. Pass `--uplink-zeros keep` to
  recover the old ternary alphabet for diagnostics.
* **BatchNorm statistics never update in the federated setting.** Local models are
  discarded each round and BatchNorm runs in inference mode during accumulation, so
  the running statistics stay at their initialization for the whole run. This is
  self-consistent — no train/test mismatch — but it is why the federated CNN2
  accuracies sit where they do.
