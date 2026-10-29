Dear AC/SPC, our contribution is a systematic comparison of signing before, after and on both sides of Muon’s orthogonalization, with and without error feedback. In the submitted federated study, SignMuon reaches 85.72% using 1 bit per matrix entry, versus Muon 85.98% and SignSGD 81.44% (5 seeds). Its centralized accuracy is 94.60% versus SignSGD 93.37% (3 seeds); nanoGPT loss is 3.2881 versus 3.4049 (1 run/method).

For Ueq5 and pqje, the submitted logs reproduce A8: MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%. We will correct the main table and its reversed comparison.

We prove that each sign placement can produce ascent directions and that error feedback on the orthogonalized output can diverge on a smooth, lower-bounded objective. These results are verified in Lean. Theorem 4 concerns fixed parameters, so it does not exclude decreasing schedules or parameters chosen for a finite budget. We also generalized the EF21 analysis to EMA/Nesterov and deterministic learning rates, with core proofs verified in Lean; layer separability remains. Our separate SignMuon theorem permits interacting layers under a positive-alignment condition.

New Dirichlet(0.1) transfer at IID rates gives EF21-MuonSign 83.23% versus SignMuon 35.92%; retuning is pending. We ask that the measured sign-based performance and the precise theoretical results be considered together.
