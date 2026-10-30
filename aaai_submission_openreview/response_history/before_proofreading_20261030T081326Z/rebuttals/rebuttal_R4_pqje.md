Thank you for the questions.

1. Re the extended guarantees: After your review, we generalized EF21-Muon(U)Sign convergence to fixed EMA/Nesterov and constant/cosine rates, and verified the proofs in Lean. This gives expected nuclear gradient norm bounds, with explicit noise and polar approximation terms. We also proved the squared-gradient rate for gradient-dependent steps.
2. Re Table 2: The submitted federated CNN2/CIFAR-10 logs (11 IID clients, 5 seeds) give MuonSign 82.94±0.19% versus SignSGD 81.44±0.15%, agreeing with A8. We corrected Table 2 and the reversed comparison. SignMuon reaches 85.72% versus Muon 85.98%.
3. Re our contribution: After submission, we proved SignMuon converges if signed updates retain enough alignment with momentum, under joint smoothness and unbiased bounded-variance noise. This permits interacting layers; network alignment remains unmeasured. On label-skewed CNN2/CIFAR-10, **only EF21-Muon(U)Sign exceed 80%** after validation LR tuning (7S1b, Q2).
4. Re nanoGPT: nanoGPT tests centralized updates after full-precision gradient averaging: validation loss 3.2881 for SignMuon versus 3.4049 for SignSGD. Our extension covers fixed Nesterov momentum; nanoGPT uses momentum warmup. Federated CNN2 tests worker voting.
5. Re the two models: EF21-MuonSign updates server model X and queries broadcast model W; its theorem controls X.

If these answers and revisions resolve your concerns, please consider raising your score.
