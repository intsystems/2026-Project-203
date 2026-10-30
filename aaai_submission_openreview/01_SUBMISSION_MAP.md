# Submission and current-source map

Prepared 29 October 2026. The reviewers saw the downloaded submission, not the updated repository article. The rebuttals therefore state the actual corrections and new results directly. Nothing was submitted, committed or pushed.

## Authority and immutable inputs

The downloaded main paper (9 pages), technical supplement (35 pages), checklist (2 pages), code/data archive, and review export are the record of what was submitted. All five original hashes were verified against `inputs_manifest.json`, then reverified by `evidence/verify_rebuttal_evidence.py`. Preserve these files unchanged.

| Original file | Bytes | Verified SHA-256 |
| --- | ---: | --- |
| `48060_SignMuon_MuonSign_and_th.pdf` | 537728 | `21b59932be1126ea6fb14a7692552e857835379e76df49c9a5f15df97131e5cd` |
| `48060_SignMuon_MuonSign_and_th_Code And Data Supplement.zip` | 5769984 | `71949a629127440afc0644cd0c15c6c8c56d98a7b446fdd9dd29fd6b9e945379` |
| `48060_SignMuon_MuonSign_and_th_Reproducibility Checklist.pdf` | 99062 | `6aeef05eb5f0b1483a2472414678f1c3783859810daf07b711b5c396652a0a36` |
| `48060_SignMuon_MuonSign_and_th_Technical Supplement.pdf` | 1692505 | `c99343a41e01a2a571bdb9e58b8a011488534c3d52c6f8edd5e09035171fd11c` |
| `signmuon_openreview_page.json` | 54034 | `0cc0650775acecc94b3c7da35198ad3132ece81c54bf64d1b694d2eb14122345` |

`evidence/extracted/` contains page-labelled derived text, not replacement source documents. Two-column extraction sometimes interleaves text; PDF pages and TeX were used to resolve formulas/tables.

## Current article correspondence

| Current source | Role | Relevant change / limitation |
| --- | --- | --- |
| `aaai_article/v2_SignMuon_AAAI.tex` → `signmuon_body.tex` | Updated main wrapper/body | Main CIFAR and federated tables now use the values already in submitted A5/A8. Some overbroad claims remain. |
| `v2_SignMuon_AAAI_supp.tex` → `signmuon_appendix.tex` | Updated supplement | Includes expanded convergence material; numbering can differ from the downloaded supplement. |
| `v2_SignMuon_AAAI_full.tex` | Combined wrapper | Same body/appendix; do not confuse it with the reviewers' downloaded main PDF. |
| `ef21_nesterov_convergence.tex` | New positive analysis | Fixed Nesterov tracking; not nanoGPT's changing momentum. |
| `ef21_fixed_radius.tex` | New deterministic-schedule analysis | Constant/cosine first-moment residual/rate statements; fixed peak LR and momentum do not imply vanishing stationarity. |
| `ef21_rate_takeaways.tex` | Separate second-moment argument | Gradient-scaled steps and horizon tuning; no squaring of a first-moment expectation. |
| `signmuon_addendum.tex` / `_body.tex` / `.bib` | Existing author working addendum | Conditional SignMuon, bounded-below plain witnesses and EF gradient-floor scaling. Its deferred experiments are not completed results. Preserved unchanged. |
| `ef21_signmuon_divergence.tex` | Complete Theorem 4 proof | Parameter-dependent objective for fixed eta/mu, exact polar selection and rational limit cycle. |
| `signmuon_mishra_comparison.tex` and main comparison | Still needs correction | Mishra v2 (17 October) now gives conditional actual-vote stochastic theory. The v1-only dismissal is stale. |
| `evidence/EF21_JOINT_SMOOTHNESS.md` | Separate candidate derivation made during preparation | Not in the manuscript and not relied on by revision 2 replies. The manuscript’s EF21 proofs retain separability; its conditional SignMuon theorem permits interacting layers. |

Current PDFs are separately versioned artifacts. Neither a newer filename nor a timestamp establishes that a PDF contains all current TeX changes. Their exact hashes/pages and all current TeX/Bib hashes are recorded in `evidence/verified_claims.json`.

| Current PDF | Pages | SHA-256 at audit |
| --- | ---: | --- |
| `aaai_article/signmuon_addendum.pdf` | 5 | `9f9b854694d00a73356031eff106833eadd54de529f09be6a3c4834ec98b0d6b` |
| `aaai_article/v2_SignMuon_AAAI.pdf` | 9 | `765703175c4140367f541529c3f12097d47225e464c40d52546cc4d82a592149` |
| `aaai_article/v2_SignMuon_AAAI_full.pdf` | 40 | `573679a153eb1c3ffcec3446826e0661508115106f1b61dd912ac0b28b8fe552` |
| `aaai_article/v2_SignMuon_AAAI_full_blue.pdf` | 41 | `ef32695c3eb88f6850213fe12a44a62b51d54a3ae1b84e31c9373bbca2762753` |
| `aaai_article/v2_SignMuon_AAAI_full_codex.pdf` | 40 | `599823a1da9aa57c735ebf263683c774d6e15c7575d6cb5b0a60114c57ad8567` |
| `aaai_article/v2_SignMuon_AAAI_supp.pdf` | 32 | `f26493bea31adea1aacee081e6ce29f829f92986ebbc0e1f8270ea08c93dff9f` |

## Submitted code versus repository

The submitted archive contains 109 non-directory files: 91 byte-identical, 17 different, and 1 absent in the audited repository snapshot. `evidence/source_audit.json` records every member's original/current hashes. `evidence/submitted_to_repo.diff` contains the text differences. The archive is authoritative for whether evidence was already submitted.

- `code/centralized/main.py` — different
- `code/centralized/train.py` — different
- `code/common/optimizers.py` — different
- `code/common/README.md` — different
- `code/federated/algorithms.py` — different
- `code/federated/main.py` — different
- `code/federated/README.md` — different
- `code/nanogpt/README.md` — different
- `code/nanogpt/signmuon_optimizers.py` — different
- `code/nanogpt/test_signmuon_optimizers.py` — different
- `code/nanogpt/train_gpt.py` — different
- `code/README.md` — different
- `code/results/article_export.tar.gz` — different
- `code/results/federated_export_results.zip` — different
- `code/results/synthetic_results.zip` — different
- `code/tests/README.md` — different
- `code/tests/test_code.py` — different
- `code/MANIFEST.md` — missing in repo

Core algorithms were read in the archive and repository. Changes include optional compression diagnostics, tests/docs and result-container metadata; do not infer source identity from matching aggregate tables. More strongly, all **175 federated metrics training histories** in the original nested archive are semantically identical to their current counterparts. This was checked independently of metadata, line endings and ZIP bytes.

Live repository work appeared during preparation in `code/federated/heterogeneous.py`, its test and README files. Those files were left intact. A runner and tests do not constitute a completed heterogeneous experiment; no such new result was available in the audited result directory.

## Which values to use

| Setting | Stale submitted main | Submitted supporting evidence / updated main | Treatment |
| --- | --- | --- | --- |
| Federated MuonSign vs SignSGD | Table 2: 80.68±1.88 vs 81.83±0.43 | A8 + submitted five-seed metrics: **82.94±0.19 vs 81.44±0.15** | Own the reporting error; reverse the affected ranking sentence. |
| Federated SignMuon vs Muon | Table 2: 85.81?0.25 vs 85.99?0.26 | A8/archive: **85.72±0.24 vs 85.98±0.26** | Comparable observed accuracy; no equivalence/significance proof. |
| Centralized SignMuon vs Muon | Table 1: 94.53±0.22 vs 94.49±0.09, with EF21-SignMuon 94.64±0.24 on top and SignSGD 93.27±0.16 (corrected in revision 11; the earlier 94.26 entry was wrong) | A5 + submitted `article_export/table_cifar.csv`: **94.60±0.15 vs 94.35±0.27**; EF21-SignMuon 94.31±0.11, SignSGD 93.37±0.26 | Replace the whole main table coherently, not just a convenient cell. |
| nanoGPT | Table 3 | Unchanged single-run values; SignMuon3.2881, EF21-SignMuon3.2860, SignSGD3.4049 | No new seeds or statistical ranking. |

Federated values are means across five independently trained seeds of each seed's final-five-evaluation average; SD is the sample SD across those five means. Repeated checkpoints are not independent seeds. The central submitted export has three independent final seeds and rounded aggregate values; its extracted `runs.csv`/`table_cifar.csv` are in `evidence/`. nanoGPT has one independent, unpinned run per compressed method. X/W are two evaluations of the same EF21-MuonSign run.

## Reviews and conference settings

| Workspace | Reviewer | Note ID | Rating / confidence |
| --- | --- | --- | --- |
| R1 | 7S1b | smHroyG4g5 | 6 / 4 |
| R2 | D4r2 | UC752MnbRo | 7 / 4 |
| R3 | Ueq5 | xHT5WIiVi7 | 6 / 2 |
| R4 | pqje | kkv2ydIYff | 5 / 4 |
| R5 | Tx34 | EDZy4k8BK7 | 4 / 3 |
| AI | AI_Reviewer | LKOb5GFqWI | AI review; no score request |

Configured author-supplied rules: **1500 UTF-16 source characters per review**, headings/newlines/math included; no external links; deadline **31 October 2026 AoE**. These settings were retained. The confidential field's limit/readers remain unknown and must be checked in the actual form. No official field rules were inferred from a different venue.

## Preservation and summary migration

Late source update: the live author addendum body/PDF changed after the initial article hash snapshot. The latest 479-line body was read again: its initial heterogeneous plan had pending results; the latest update now includes completed transfer results while LR retuning remains pending, and its EF21 cross-layer extension remains marked open. Revision 2 corrects the earlier rebuttal’s conflation: the manuscript retains EF21 separability. The independent J1 candidate is separate and is not used to claim a completed manuscript extension. `evidence/final_source_drift.json` records the later hashes and the concurrently changed code README files. Those source files were not edited by this preparation.

`response_history/preparation_baseline_20261029T134259Z/` preserves all pre-preparation JSON, summaries and strategy files. The imported fragments had revision0 and empty answers. All original concern IDs are retained, but placeholder headings/praise/reference entries were reassigned to genuine, fine-grained concerns. The migration affects empty scaffolding, not authored scientific answers. All appended **Original review** blocks remain byte-for-byte equivalent after normal newline decoding; original JSON bytes remain unchanged. Each new concern has one response fragment and exactly one topic. Duplicate mentions link to the same concern. `evidence/concern_evidence.json` records all 64 concrete concerns, evidence, intended answers and pending status.

## Revision 2 proof audit

The EF21 fixed-radius proof explicitly invokes block independence. The addendum’s “EF21 cross-layer coupling: open target” paragraph confirms that extension remains open in the manuscript. The conditional plain SignMuon theorem uses joint Frobenius smoothness and allows interacting layers. See `evidence/revision_2_audit.md`. All downloaded originals and source review passages are preserved.

## New transfer archive supplied during revision 2

`code/results/rebuttal_results.zip` was supplied while the audit was running. Its immutable copy is `evidence/heterogeneous_transfer_results.zip`; this is new rebuttal evidence, not a submitted input. The independent audit reproduces all 50 transfer runs (10 methods, 5 seeds) and their means/sample SDs. The current addendum now reports these results. Its EF21 cross-layer target remains open. LR retuning, local updates and partial participation remain outstanding.

## Lean evidence used in the readability revision

`lean_proofs/comparator/VERIFICATION.json` records 45 passed statement comparisons and kernel replay on 23 October 2026. Its reference and dependency hashes still match. `lean_proofs/PAPER_AUDIT.md` and `README.md` explain the original failure theorems, generalized EF21 tracking/convergence proofs, approximation-error terms and manual cosine-sum specialization. The conditional plain SignMuon theorem is an analytic addendum result and is not described as Lean-verified. See `evidence/revision_3_readability.md`.

## Revision 4: timing of additions and heterogeneous table

The completed momentum/schedule extensions, conditional SignMuon theorem and heterogeneous tests are now explicitly distinguished from submitted results. The original failure theorems and submitted vision measurements remain labeled accordingly. Public replies omit credit-only sentences; the source/bibliography audit above is retained.

The full 10-method transfer table is exported in the reply to **7S1b, Q2** and referenced from the five other replies and the AC comment. It is generated directly from `evidence/heterogeneous_transfer_verified.json`; the frozen archive hash was rechecked and matches that verification record. No new training or numerical result was produced during this prose edit.

## Separate quadratic construction added during revision 4

The live `aaai_article/signmuon_addendum_body.tex` changed during preparation to include `ef21_signmuon_vanishing_steps.tex`. The preparation did not edit these scientific sources. The new theorem and its exact rational/algebraic certificate were read, checked and frozen in `evidence/ef21_vanishing_snapshot/`. Hashes and the 20 successful trajectory checks are recorded in `evidence/new_quadratic_certificate.json`. Tx34 Q2 and the AC comment now use this stronger completed result. The existing Lean audit does not cover N4.

## Revision 5: explicit benchmark identities and constructive lower bounding

The exported replies distinguish federated CNN2/CIFAR-10 (85.72% SignMuon; 11 IID clients, 5 seeds), centralized ResNet18/CIFAR-10 (94.60%; 3 seeds) and nanoGPT validation loss (1 run/method). All values retain their previously verified sources.

The shared heterogeneous table highlights both gradient-side EF variants. The only-success statement is scoped to Muon variants above 80% in the specified reused-IID-rate test. The softplus construction is read directly from `signmuon_addendum_body.tex`, Proposition `add:prop:softplus`: positive gradient rescaling preserves the ascent directions and gives smooth convex objectives bounded below by 0. No new training or proof-source edit was performed in this revision.

## Revision 6: review-specific theory wording

Ueq5 discusses schedules, layer separability and the first/second-moment issue; it does not ask about nanoGPT warmup or auxiliary Adam. Those unsolicited remarks are removed from that reply. pqje Q1 explicitly asks about nanoGPT Nesterov, so its short fixed-momentum/warmup answer is retained. The new rate statement identifies both gradient-side EF21 algorithms and the nuclear gradient quantity from `ef21_fixed_radius.tex`, `cor:radius_schedules`. The conditional plain SignMuon theorem likewise controls the sum of nuclear block-gradient norms in `add:eq:rate`; joint Frobenius smoothness is its smoothness assumption. The existing Lean proof map supports tracking and convergence proof verification without a “core” qualifier; manual specialization details remain in the internal audit.

## Revision 7: completed heterogeneous LR tuning

The supplied cumulative `rebuttal_results4.zip` is frozen as `evidence/heterogeneous_tuning_results.zip` (SHA-256 `9d7c55c609b9a7fff8761c70b7d596bda13d972803e4dd7f36dfd3956f001bec`). Its audit independently reproduces 72 validation trials, all ten interior LR winners and 20 five-seed final rows from 167 unique completed runs. The five Muon finals are reused across transfer/tuned rows. All original fifty transfer records are unchanged. The public table now reports the tuned comparison; the addendum shows IID, transfer and tuned values, selected rates, validation protocol and the two-trial MuonSign refinement. Existing addendum author changes were backed up before editing; its mathematical proof sections are preserved. The PDF is rebuilt. LR tuning is complete; local parameter steps and partial participation remain untested.
