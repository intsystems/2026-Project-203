# Rebuttal workspace — completed heterogeneous tuning, revision 7

All six replies and the AC comment now use the validation-LR-tuned heterogeneous results. The full ten-method table is in **7S1b Q2**, with references from the other five replies and AC comment. Both EF rows remain bold. Each human reply retains its conditional request to raise the score.

## Current drafts

Open `start_workspace.cmd`, or read `rebuttals/rebuttal_*.md`.

| Reviewer | UTF-16 characters /1500 | Selected concerns | Missing evidence |
| --- | ---: | ---: | --- |
| 7S1b | 1495 | 7 | R1.6, R1.7, R1.8 |
| D4r2 | 1451 | 2 | R2.2, R2.8 |
| Ueq5 | 1472 | 5 | None |
| pqje | 1464 | 14 | R4.15 |
| Tx34 | 1493 | 6 | None |
| AI_Reviewer | 1478 | 15 | AI.6, AI.20 |

## New evidence and addendum

**EF21-MuonSign: 84.03±0.52%; EF21-MuonUSign: 83.84±0.32%.** They are the only methods above 80% after LR tuning in this ten-method comparison. SGD reaches 79.71±2.03%, Adam 75.16±1.07%, Muon 46.02±8.55%, and SignMuon 42.96±5.94%. This preserves the practical recommendation: SignMuon is strong on the submitted IID/centralized vision tasks; gradient-side EF leads under this label skew.

CNN2/CIFAR-10, 11 Dirichlet(0.1) clients, 2000 rounds, no local parameter updates, cosine schedule, fixed momentum 0.9. LR selection uses seed 0 and a 5000-image validation holdout; five final seeds retrain on all 50000 images. Mean±sample SD is over per-run final-five test means. All ten LR winners are interior after MuonSign's two additional validation trials; its selected 5e-5 is unchanged.

`evidence/heterogeneous_tuning_results.zip` freezes the supplied fourth archive. The independent audit verifies all 167 completed runs: 72 validation trials and 95 unique final runs. The two comparisons have 100 final-seed memberships because Muon's five finals are shared. All original fifty transfer runs are unchanged. `heterogeneous_tuning_verified.json` records each selection and result.

The addendum now contains all ten selected rates and tuned means/SDs beside the homogeneous and transfer values, explains the completed search, and reports the persistent EF advantage. Its PDF is rebuilt; mathematical sections and prior author edits are preserved. The existing N4 input file is a build dependency and matches the frozen theorem source.

## Remaining evidence and preservation

LR tuning is completed. Local parameter updates, partial participation, network alignment measurements, extra nanoGPT seeds and a matched-query SSVR comparison remain untested. The eight pending concern flags and seven author-directed presentation omissions are unchanged. All 64 concern IDs and 79 exact passage links are preserved. Original review wording, inputs and conference rules are unchanged.

The pre-update JSON, exported replies and current author addendum body/PDF are backed up in `response_history/before_tuned_results_20261030T065830Z/`. Complete EF21 tracking/convergence proof wording remains unchanged; no new Lean claim is attached to the plain SignMuon, softplus or N4 analytic results.

The independently editable confidential draft has 2478 UTF-16 characters. Its destination limit/readers remain unknown; the configured limit stays `null`.

## Checks and collaboration

Build with `python -X utf8 aaai_submission_openreview/primer/workspace.py build`, inspect the editor pages, then run the same command with `check`. The run audit, browser inspection, source/selection integrity, addendum table/PDF inspection and character counts are recorded in `evidence/revision_7_validation.json`. The primer itself is unchanged; no training or full Lean build was needed for this results update.

The build/check, all seven editor pages, all six public limits and the six-page addendum PDF passed verification.

Save browser edits before pulling or merging, coordinate changes to `response_fragments.json`, and reload after rebuilding. Saves are local. The configured deadline remains 31 October 2026 AoE.
