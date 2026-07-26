# Centralized results export

* source: `/home/legeartis/SignMuon/code/results/centralized`
* runs found: 156
* unreadable: 0
* targets for `epochs_to_*`: 90%, 93%, 94%

## Files

| file | bytes | contents |
| :--- | ---: | :--- |
| `runs.csv` | 43322 | one row per run: config + derived summary metrics |
| `curves.csv` | 865530 | tidy per-epoch series for the figures |
| `gain.csv` | 8824 | accumulated-gain series (`--log-gain` runs only) |
| `gain_fits.csv` | 433 | log-log slope of gain_median vs epoch |
| `scaling_compare.csv` | 1225 | final runs pivoted optimizer x lr_scaling |
| `configs.json` | 168231 | full config of every run |
| `overnight/REPORT.md` | 4891 | copied verbatim |
| `overnight/REPORT_backup.md` | 5499 | copied verbatim |
| `overnight/state.json` | 92518 | copied verbatim |

## Notes

* Accuracies are percentages. `*_tail` is the mean of the final `last_k`
  recorded epochs -- the paper's primary metric -- not a single epoch.
* `val_*` is empty for `split=full` runs, which have no validation set;
  `test_acc` on `split=tune` runs comes from a 45k-trained model and was
  never used for selection.
* `epochs_to_*` is recomputed from the history here, so it does not depend
  on the training logs having been kept.
* Single seed per configuration unless `runs.csv` shows otherwise: check
  `seed` before quoting any difference as a result.
